"""世界米制门（M22-FR-016、M22-AC-008；M23 §6.3 `E-WORLD`；AWR-05 §5.1.2、ADR-130；D3-AC-02）。

`metric_ok()`：`ok = (scaleStatus ≠ relative) ∧ 标定证据完整`，`ranked_ok = ok ∧ scaleStatus ≠ assumed`；事件编译对
`ok = false` 的世界返回 600 `WORLD_UNCALIBRATED` 且不装载舞台，排名编译另拒 `assumed`。构造三种状态（landmark、
assumed、relative）与证据不完整的变体：前半部分只用 JSON 证据（crop 世界包的三个文件），后半部分在合成 GLB 世界上走编译。
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from awr.world.geometry.status import CALIBRATION_GATES, SCALE_STATUSES, metric_ok

ROOT = Path(__file__).resolve().parents[3]
CROP = ROOT / "worlds" / "oryahovo-crop1"
pytestmark = pytest.mark.skipif(not CROP.is_dir(), reason="needs worlds/oryahovo-crop1 (make worlds)")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "world"))
sys.path.insert(0, str(HERE))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rebind(d: Path) -> None:
    """改写 coordinate.json 后重新绑定（模拟一次合法的重新发布）：world.json 的 `coordinate.sha256` 与 `files[]` 条目，
    以及其余 JSON 图层元数据中记录的 `coordinate_sha256`（zones、raycast、appearance、env 等按旧值逐字替换）。"""
    w = json.loads((d / "world.json").read_text(encoding="utf-8"))
    old, sha = w["coordinate"]["sha256"], _sha(d / "coordinate.json")
    w["coordinate"]["sha256"] = sha
    for f in w.get("files", []):
        if f.get("path") == "coordinate.json":
            f["sha256"], f["bytes"] = sha, (d / "coordinate.json").stat().st_size
    (d / "world.json").write_text(json.dumps(w, ensure_ascii=False, indent=1), encoding="utf-8")
    for p in sorted(d.rglob("*")):
        if p.suffix in (".json", ".geojson") and p.name not in ("world.json", "coordinate.json") and p.is_file():
            t = p.read_text(encoding="utf-8")
            if old in t:
                p.write_text(t.replace(old, sha), encoding="utf-8")


def _set_scale(d: Path, status: str) -> None:
    for name in ("world.json", "coordinate.json"):
        doc = json.loads((d / name).read_text(encoding="utf-8"))
        doc["scaleStatus"] = status
        (d / name).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    _rebind(d)


def _edit(d: Path, name: str, fn) -> None:
    doc = json.loads((d / name).read_text(encoding="utf-8"))
    fn(doc)
    (d / name).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    if name == "coordinate.json":
        _rebind(d)


@pytest.fixture()
def crop_ev(tmp_path: Path) -> Path:
    """crop 世界包的标定证据（world.json、coordinate.json、qa/report.json）副本。"""
    d = tmp_path / CROP.name
    (d / "qa").mkdir(parents=True)
    for rel in ("world.json", "coordinate.json", "qa/report.json"):
        shutil.copyfile(CROP / rel, d / rel)
    return d


# ------------------------------------------------------------------------------------------------------- metric_ok（JSON 证据）
def test_committed_crop_world_is_metric_and_rankable() -> None:
    """D3-AC-02：crop 世界包 `scaleStatus = landmark` 且 `metric_ok()` 为真（M22-AC-008：crop 为 ok、ranked_ok）。"""
    ms = metric_ok(CROP)
    assert ms.ok and ms.ranked_ok and ms.reason is None and ms.missing == ()
    assert ms.scale_status == "landmark" and ms.world_id == "oryahovo-crop1"
    assert ms.content_version == json.loads((CROP / "world.json").read_text())["contentVersion"]
    assert metric_ok(str(CROP)) == ms


@pytest.mark.parametrize(("status", "ok", "ranked"), [("landmark", True, True), ("assumed", True, False), ("relative", False, False)])
def test_three_scale_states(crop_ev: Path, status: str, ok: bool, ranked: bool) -> None:
    """M22-AC-008：relative 世界 `ok = false`；assumed 世界 `ok = true`、`ranked_ok = false`；landmark 两者皆真。"""
    _set_scale(crop_ev, status)
    ms = metric_ok(crop_ev)
    assert (ms.ok, ms.ranked_ok, ms.scale_status) == (ok, ranked, status)
    assert ms.reason == {"landmark": None, "assumed": "scale_not_rankable", "relative": "scale_relative"}[status]
    assert ms.missing == (() if ok else ("scale_relative",))
    assert ms.to_json()["missing"] == list(ms.missing)


@pytest.mark.parametrize("status", [s for s in SCALE_STATUSES if s not in ("relative", "assumed")])
def test_stronger_scale_states_are_not_rejected(crop_ev: Path, status: str) -> None:
    """v1.2 修订：`gnss`、`lidar`、`rtk`、`survey` 不被误拒（原写法 `∈ {landmark, assumed}` 的缺陷）。"""
    _set_scale(crop_ev, status)
    ms = metric_ok(crop_ev)
    assert ms.ok and ms.ranked_ok and ms.scale_status == status


def _drop_evidence(c: dict) -> None:
    c["source"]["evidence"] = []


def _bad_units(c: dict) -> None:
    c["source"]["unitsToMeters"] = 0.0


def _north_unknown(c: dict) -> None:
    c["trueNorth"]["confidence"] = "unknown"


def _no_registration(c: dict) -> None:
    c["registration"] = None


def _registration_none(c: dict) -> None:
    c["registration"]["method"] = "none"


def _no_levelling(c: dict) -> None:
    del c["source"]["leveledDeg"]


def _coord_qa_fail(c: dict) -> None:
    c["qa"]["status"] = "fail"


@pytest.mark.parametrize(("fn", "code"), [
    (_drop_evidence, "scale_evidence_missing"), (_bad_units, "scale_evidence_missing"), (_north_unknown, "true_north_unknown"),
    (_no_registration, "registration_missing"), (_registration_none, "registration_missing"), (_no_levelling, "levelling_missing"),
    (_coord_qa_fail, "qa_failed")])
def test_landmark_with_incomplete_coordinate_evidence(crop_ev: Path, fn, code: str) -> None:
    """D3-AC-02：`scaleStatus = landmark` 但标定证据不完整时同样 `ok = false`。"""
    _edit(crop_ev, "coordinate.json", fn)
    ms = metric_ok(crop_ev)
    assert not ms.ok and not ms.ranked_ok and ms.scale_status == "landmark"
    assert ms.missing == (code,) and ms.reason == code


def _gate(gid: str, **kw):
    def fn(q: dict) -> None:
        g = next(g for g in q["gates"] if g["id"] == gid)
        if kw.get("drop"):
            q["gates"].remove(g)
        else:
            g["pass"] = False
    return fn


@pytest.mark.parametrize("gid", CALIBRATION_GATES)
def test_calibration_gates_must_be_present_and_pass(crop_ev: Path, gid: str) -> None:
    _edit(crop_ev, "qa/report.json", _gate(gid))
    assert metric_ok(crop_ev).missing == ("calibration_gate_failed",)
    _edit(crop_ev, "qa/report.json", _gate(gid, drop=True))
    assert metric_ok(crop_ev).missing == ("calibration_gate_missing",)


def test_non_calibration_gate_failure_does_not_block(crop_ev: Path) -> None:
    """标定之外的门禁（G-01 建筑高度告警等）不属于标定证据。"""
    _edit(crop_ev, "qa/report.json", _gate("G-01"))
    assert metric_ok(crop_ev).ok


def test_qa_report_binding(crop_ev: Path) -> None:
    def other_build(q: dict) -> None:
        q["content_version"] = "000000000000"

    def failed(q: dict) -> None:
        q["status"] = "fail"

    _edit(crop_ev, "qa/report.json", other_build)
    assert metric_ok(crop_ev).missing == ("qa_mismatch",)
    shutil.copyfile(CROP / "qa/report.json", crop_ev / "qa/report.json")
    _edit(crop_ev, "qa/report.json", failed)
    assert metric_ok(crop_ev).missing == ("qa_failed",)
    (crop_ev / "qa/report.json").unlink()
    assert metric_ok(crop_ev).missing == ("qa_missing",)


def test_coordinate_binding_and_consistency(crop_ev: Path) -> None:
    raw = (crop_ev / "coordinate.json").read_text(encoding="utf-8")
    (crop_ev / "coordinate.json").write_text(raw + "\n", encoding="utf-8")       # 未重新绑定 sha256
    assert metric_ok(crop_ev).missing == ("coordinate_sha_mismatch",)
    (crop_ev / "coordinate.json").write_text(raw, encoding="utf-8")
    _edit(crop_ev, "coordinate.json", lambda c: c.update(scaleStatus="survey"))
    assert metric_ok(crop_ev).missing == ("scale_status_inconsistent",)
    (crop_ev / "coordinate.json").unlink()
    ms = metric_ok(crop_ev)
    assert not ms.ok and ms.missing == ("coordinate_missing",)


def test_missing_or_unknown_scale_status(crop_ev: Path) -> None:
    """`scaleStatus` 缺失或取枚举外的值（例如旧写法的 unknown）一律视为未标定。"""
    _set_scale(crop_ev, "unknown")
    ms = metric_ok(crop_ev)
    assert not ms.ok and ms.scale_status is None and ms.missing[0] == "scale_status_missing"
    assert metric_ok(crop_ev.parent / "absent").missing == ("world_json_missing",)
    (crop_ev / "world.json").write_text("{", encoding="utf-8")
    assert metric_ok(crop_ev).missing == ("world_json_invalid",)


def test_in_memory_form_matches_directory_form() -> None:
    docs = {"world": json.loads((CROP / "world.json").read_text()), "coordinate": json.loads((CROP / "coordinate.json").read_text()),
            "qa_report": json.loads((CROP / "qa/report.json").read_text())}
    assert metric_ok(docs) == metric_ok(CROP)
    d2 = copy.deepcopy(docs)
    d2["world"]["scaleStatus"] = d2["coordinate"]["scaleStatus"] = "relative"
    assert metric_ok(d2).missing == ("scale_relative",)
    assert metric_ok({"world": docs["world"]}).missing == ("coordinate_missing", "qa_missing")


def test_d2_profile_does_not_require_glb_evidence() -> None:
    """D2 世界（无 `profile = d3-glb`）不要求 `registration` 与 `leveledDeg`；synthcity 为 assumed：ok 而不可排名。"""
    ms = metric_ok(ROOT / "worlds" / "synthcity")
    assert ms.ok and not ms.ranked_ok and ms.scale_status == "assumed" and ms.reason == "scale_not_rankable"


# ----------------------------------------------------------------------------------------------------------- 事件编译 600
@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory):
    pytest.importorskip("open3d")
    pytest.importorskip("PIL")
    pytest.importorskip("zstandard")
    from synthworld import build_synth_world

    tmp = tmp_path_factory.mktemp("wg")
    mp = pytest.MonkeyPatch()
    yield build_synth_world(tmp, mp), tmp
    mp.undo()


def _variant(synth, tmp: Path, status: str, fn=None) -> Path:
    w, _ = synth
    d = tmp / f"world-{status}"
    shutil.copytree(w, d)
    _set_scale(d, status)
    if fn is not None:
        _edit(d, "coordinate.json", fn)
    return d


def _compile(synth, world: Path, tmp: Path, **kw):
    from test_compile import EVENT, OVERLAY

    from awr.sim.events.compile import CompileOptions, compile_event

    d = tmp / "ev"
    d.mkdir(exist_ok=True)
    (d / "anchors.json").write_text(json.dumps(OVERLAY))
    (d / "event.json").write_text(json.dumps(EVENT))
    return compile_event(d / "event.json", world, opts=CompileOptions(out=d / "v1", cache_dir=synth[1] / "cache", **kw))


def _world_errors(r) -> list[dict]:
    return [e for e in r.gates[0]["errors"] if e["code"] == "E-WORLD"]


def test_compile_relative_world_returns_600(synth, tmp_path: Path) -> None:
    """D3-AC-02：对 `scaleStatus = relative` 的世界提交事件编译返回 600，不装载舞台、不写产物。"""
    r = _compile(synth, _variant(synth, tmp_path, "relative"), tmp_path)
    assert r.status == "failed" and r.reason == 600 and len(r.gates) == 1 and r.gates[0]["status"] == "fail"
    (e,) = _world_errors(r)
    assert e["reason"] == 600 and e["path"] == "/world" and e["detail"]["missing"] == ["scale_relative"]
    assert r.as_dict()["reason"] == {"code": 600, "name": "WORLD_UNCALIBRATED"}
    assert r.world_metric["ok"] is False and not (tmp_path / "ev" / "v1").exists()


def test_compile_landmark_without_evidence_returns_600(synth, tmp_path: Path) -> None:
    """D3-AC-02：标定证据不完整（`source.evidence[]` 为空）的 landmark 世界同样返回 600。"""
    r = _compile(synth, _variant(synth, tmp_path, "landmark", _drop_evidence), tmp_path, preflight_only=True)
    assert r.status == "failed" and r.reason == 600
    assert _world_errors(r)[0]["detail"]["missing"] == ["scale_evidence_missing"]


def test_compile_assumed_world_passes_unless_ranked(synth, tmp_path: Path) -> None:
    """assumed 世界可编译（`ok = true`）；排名编译另拒（`ranked_ok = false`），返回 600 且 `detail.ranked = true`。"""
    d = _variant(synth, tmp_path, "assumed")
    r = _compile(synth, d, tmp_path, preflight_only=True)
    assert r.status == "preflight" and r.reason is None and not _world_errors(r)
    assert r.world_metric["ok"] is True and r.world_metric["ranked_ok"] is False
    r = _compile(synth, d, tmp_path, preflight_only=True, ranked=True)
    assert r.status == "failed" and r.reason == 600
    (e,) = _world_errors(r)
    assert e["detail"]["ranked"] is True and e["detail"]["reason"] == "scale_not_rankable" and "assumed" in e["message_en"]


def test_compile_landmark_world_passes_ranked(synth, tmp_path: Path) -> None:
    w, _ = synth
    assert metric_ok(w).ok and metric_ok(w).ranked_ok
    r = _compile(synth, w, tmp_path, preflight_only=True, ranked=True)
    assert r.status == "preflight" and r.reason is None and r.world_metric["ranked_ok"] is True


def test_world_mismatch_keeps_precedence(synth, tmp_path: Path) -> None:
    """事件引用他世界（615）且世界未标定（600）时两条均报告，原因码取 615（M23 门 1 `E-WORLD` 次序）。"""
    from test_compile import EVENT, OVERLAY

    from awr.sim.events.compile import CompileOptions, compile_event

    ev = copy.deepcopy(EVENT)
    ev["world"] = {"id": "another-world"}
    d = tmp_path / "ev"
    d.mkdir()
    (d / "anchors.json").write_text(json.dumps(OVERLAY))
    (d / "event.json").write_text(json.dumps(ev))
    r = compile_event(d / "event.json", _variant(synth, tmp_path, "relative"), opts=CompileOptions(out=d / "v1", cache_dir=synth[1] / "cache"))
    assert r.reason == 615 and sorted(e["reason"] for e in _world_errors(r)) == [600, 615]


def test_compile_missing_world_json_returns_600(tmp_path: Path) -> None:
    pytest.importorskip("open3d")
    from test_compile import EVENT

    from awr.sim.events.compile import compile_event

    (tmp_path / "event.json").write_text(json.dumps(EVENT))
    r = compile_event(tmp_path / "event.json", tmp_path / "no-world")
    assert r.status == "failed" and r.reason == 600 and r.world_metric["missing"] == ["world_json_missing"]
