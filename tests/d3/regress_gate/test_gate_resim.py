"""D3 回归门（D3-AC-01）的 resim 帧摘要对拍与 K9 的 D3 扩展扫描：

- resim：两种开关条件（`AWR_D3_ENABLED = 0 / 1`）下按 `gate.json` 固定剧本录制输入日志并重仿真，帧摘要序列与输入条目序列
  两两相同，且与 `baseline.json` 的 `resim` 节（d2@928c57b 记录）相同；剧本改动使基线失效；分歧段可定位；
- K9 扩展：D2 公开演示构建产物中 `/lab/`、`/api/bench/v1`（及 runs、events、session 路径）出现 0 次，出现即不通过；
- 门入口实际执行：经 `main()` 跑一个轻量门步骤（`k9-d3`，临时产物目录）并检查报告与退出码。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _gate():
    spec = importlib.util.spec_from_file_location("regress_gate", ROOT / "tools/bench/d3/regress_gate.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _dist(root: Path, body: str = "export const a = 1") -> Path:
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text('<!doctype html><script type="module" src="/assets/main-x.js"></script>', encoding="utf-8")
    (root / "assets" / "main-x.js").write_text(body, encoding="utf-8")
    (root / "assets" / "logo.png").write_bytes(b"\x89PNG/lab/")  # 非文本扩展名不扫描
    (root / "build-info.json").write_text('{"channel": "demo", "commit": "928c57b"}', encoding="utf-8")
    return root


def test_gate_registers_resim_and_k9_d3() -> None:
    m = _gate()
    g = m.load_gate()
    by_id = {s["id"]: s for s in g["steps"]}
    default = [s["id"] for s in m.select_steps(g, None)]
    assert by_id["resim"]["item"] == 2 and by_id["resim"]["kind"] == "builtin" and "resim" in default
    assert by_id["k9-d3"]["item"] == 6 and "k9-d3" in default
    assert default.index("public-build") < default.index("k9-d3")                       # 先构建再扫描
    assert set(m.BUILTIN) == {s["id"] for s in g["steps"] if s["kind"] == "builtin"}
    k = g["k9_d3"]
    assert {"/lab/", "/api/bench/v1/runs", "/api/bench/v1/events", "/api/bench/v1/session", "/api/bench/v1"} <= set(k["forbidden"])
    assert k["dist"] == ".cache/deploy/emax/web-dist"                                      # make demo-build 的输出目录
    r = g["resim"]
    assert r["n_vehicles"] >= 2 and r["load_world"] and {c["op"] for c in r["script"]} >= {"takeoff", "goto", "orbit", "rtl"}


def test_resim_summary_and_spec_hash() -> None:
    m = _gate()
    d = [f"{i:064x}" for i in range(1201)]
    e = [{"tick": 1, "kind": "lease", "cid": "seat", "payload_sha256": "aa"}, {"tick": 501, "kind": "cmd", "cid": "c", "payload_sha256": "bb"}]
    s = m.resim_summary(d, e, 7500)
    assert s["frames"] == 1201 and s["distinct"] == 1201 and s["inputs"] == 2 and s["end_tick"] == 7500
    assert len(s["chunks"]) == 3                                                          # 每 500 帧一段
    assert s["inputs_sha256"] != m.resim_summary(d, [dict(e[0]), {**e[1], "tick": 502}], 7500)["inputs_sha256"]
    assert m.resim_digests(s) == {"frames": "1201", "frames_sha256": s["frames_sha256"], "inputs": "2", "inputs_sha256": s["inputs_sha256"]}
    g = m.load_gate()["resim"]
    assert m.spec_sha256(g) == m.spec_sha256({**g, "doc": "x", "timeout_s": 1})              # 说明与超时不影响基线
    assert m.spec_sha256(g) != m.spec_sha256({**g, "end_s": g["end_s"] + 1})


def test_first_diverging_chunk() -> None:
    m = _gate()
    base = {"chunks": ["a", "b", "c"]}
    same = [{"summary": {"chunks": ["a", "b", "c"]}}, {"summary": {"chunks": ["a", "b", "c"]}}]
    assert m.first_diverging_chunk(same, base) is None
    assert m.first_diverging_chunk([same[0], {"summary": {"chunks": ["a", "x", "c"]}}], base) == 1
    assert m.first_diverging_chunk([{"summary": {"chunks": ["a", "b"]}}], base) == 2         # 帧数变少
    assert m.first_diverging_chunk([{"summary": {}}], None) is None


def test_write_baseline_keeps_other_sections(tmp_path, monkeypatch) -> None:
    m = _gate()
    b = tmp_path / "baseline.json"
    b.write_text(json.dumps({"schema": "awr.d3.regress_gate.baseline.v1", "digests": {"events.jsonl": "1"}, "source": {"commit": "928c57b"}}))
    monkeypatch.setattr(m, "BASELINE", b)
    m.write_baseline("resim", {"frames": "3"})
    got = json.loads(b.read_text())
    assert got["digests"] == {"events.jsonl": "1"} and got["source"] == {"commit": "928c57b"} and got["resim"] == {"frames": "3"}


def test_committed_resim_baseline_matches_spec() -> None:
    m = _gate()
    base = m.load_baseline()["resim"]
    assert base["spec_sha256"] == m.spec_sha256(m.load_gate()["resim"])                   # 剧本未在基线之后改动
    assert base["source"]["tree"].startswith("d2@928c57b") and base["source"]["flags"] == {"AWR_D3_ENABLED": "0"}
    assert int(base["frames"]) > 3000 and int(base["inputs"]) >= 5 and set(m.RESIM_KEYS) <= set(base)


def test_scan_k9_d3(tmp_path) -> None:
    m = _gate()
    k = m.load_gate()["k9_d3"]
    clean = m.scan_k9_d3(_dist(tmp_path / "clean"), k["forbidden"], k["text_ext"])
    assert not clean["missing"] and clean["files"] == 3 and not any(clean["counts"].values()) and clean["hits"] == []
    dirty = _dist(tmp_path / "dirty", "const B=`/api/bench/v1`;go(`/lab/runs`);go(`/lab/login`);fetch(B+`/runs`)")
    r = m.scan_k9_d3(dirty, k["forbidden"], k["text_ext"])
    assert r["counts"]["/lab/"] == 2 and r["counts"]["/api/bench/v1"] == 1 and r["counts"]["/api/bench/v1/runs"] == 0
    assert [h[:2] for h in r["hits"]] == [["assets/main-x.js", "/lab/"], ["assets/main-x.js", "/lab/"], ["assets/main-x.js", "/api/bench/v1"]]
    (tmp_path / "html").mkdir()
    (tmp_path / "html" / "index.html").write_text('<a href="/api/bench/v1/session">x</a>', encoding="utf-8")
    assert m.scan_k9_d3(tmp_path / "html", k["forbidden"], k["text_ext"])["counts"]["/api/bench/v1/session"] == 1
    assert m.scan_k9_d3(tmp_path / "none", k["forbidden"], k["text_ext"])["missing"]


@pytest.mark.parametrize(("body", "code"), [("export const a = 1", 0), ("route(`/lab/board`)", 1)])
def test_main_runs_a_light_gate_step(tmp_path, body, code) -> None:
    """门入口实际执行一个轻量门步骤（k9-d3 指向临时产物目录），报告写入 --out，退出码与判定一致。"""
    m = _gate()
    g = m.load_gate()
    g["k9_d3"]["dist"] = str(_dist(tmp_path / "dist", body))
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps(g), encoding="utf-8")
    out = tmp_path / "out"
    assert m.main(["--gate", str(gate), "--steps", "k9-d3", "--out", str(out)]) == code
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    st = rep["steps"]["k9-d3"]
    assert rep["schema"] == "awr.d3.regress_gate.report.v1" and rep["pass"] is (code == 0) and st["item"] == 6
    assert st["build_info"] == {"channel": "demo", "commit": "928c57b"} and (out / "k9-d3.log").is_file()
    assert m.main(["--gate", str(gate), "--steps", "nope", "--out", str(out)]) == 2


def test_resim_step_matches_d2_baseline(tmp_path) -> None:
    """实际运行 resim 门步骤：两种开关条件下帧摘要序列与 D2 基线（d2@928c57b）逐字节一致（约 20 s）。"""
    from awr.sim.fleet.fleet import resolve_kernel

    m = _gate()
    base = m.load_baseline()["resim"]
    if resolve_kernel(base["source"]["kernel"])[0] != base["source"]["kernel"]:
        pytest.skip(f"kernel {base['source']['kernel']} unavailable; the D2 baseline was recorded with it")
    world = m.load_gate()["resim"]["world_id"]
    if not (Path(os.environ.get("AWR_WORLDS_DIR", ROOT / "worlds")) / world).is_dir():
        pytest.skip(f"needs worlds/{world} (make worlds): the D2 baseline was recorded with its geometry loaded")
    out = tmp_path / "out"
    assert m.main(["--steps", "resim", "--out", str(out)]) == 0
    st = json.loads((out / "report.json").read_text(encoding="utf-8"))["steps"]["resim"]
    assert st["pass"] and st["diffs"] == [] and st["problems"] == [] and st["first_diverging_chunk"] is None
    assert [r["flags"] for r in st["runs"]] == [{"AWR_D3_ENABLED": "0"}, {"AWR_D3_ENABLED": "1"}]
    for r in st["runs"]:
        s = r["summary"]
        assert s["frames_sha256"] == base["frames_sha256"] and s["chunks"] == base["chunks"]
        assert s["distinct"] > 0.5 * s["frames"]                                           # 机体在运动，不是静止帧
