"""D3 回归门第 3 条（D2 性能子集）与 relay 场景质量检查差分口径的自检（D3-AC-01；AWR-05 §4.8；封板任务 G04）。

- 配置：`perf` 为缺省步骤（第 3 条），子集用例与比较指标登记在 gate.json，指标所指用例真实存在；
- 判据：逐指标中位与 D2 基线差 ≤ 5%；D2 通过的用例须仍通过，D2 未达成的预算用例结果相同即可；主机指纹不同即不可比；
- worker：真实运行一次 tests/targets/test_budget.py 的 perf 用例，插件记录 call 阶段 CPU 时间与 record_property 指标；
- 入库基线：`perf` 节与 gate.json 一致（用例列表、指标、主机指纹）；`relay_checks` 节（D2 的场景质量检查）自身差分为空；
- relay 质量检查："与 D2 基线相同即通过"——本身通过、或与 D2 同样不通过且取值相同即通过，任何新增或变化的不通过即回退。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _gate():
    spec = importlib.util.spec_from_file_location("regress_gate_g04", ROOT / "tools/bench/d3/regress_gate.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


G = _gate()
HOST = {"cpu_model": "X", "python": "3.12", "numpy": "2.0", "numba": "0.60"}
NID = "tests/targets/test_budget.py::test_stage_budget_30x12"


# ==================================================================== 配置


def test_perf_step_is_default_and_registered() -> None:
    g = G.load_gate()
    step = next(s for s in g["steps"] if s["id"] == "perf")
    assert step["item"] == 3 and step["kind"] == "builtin" and step.get("default", True)
    assert "perf" in [s["id"] for s in G.select_steps(g, None)]
    p = g["perf"]
    assert p["max_regression"] == 0.05 and p["repeats"] >= 3 and p["warmup"] >= 1          # D3-AC-01："与基线差 ≤ 5%"；ADR-033 三次取中位
    assert set(p["metrics"]) and all(nid.split("::")[0] in p["pytest"] for nid in p["metrics"])
    for nid in p["metrics"]:                                                               # 登记的用例真实存在且带 perf 标记
        f, name = nid.split("::")
        text = (ROOT / f).read_text(encoding="utf-8")
        assert f"def {name}(" in text and "pytest.mark.perf" in text, nid
    assert "stage_cpu_ms_per_sim_s" in p["metrics"][NID]
    assert all(v == "1" for v in p["env"].values())                                        # 数值库单线程：CPU 时间只计本线程


# ==================================================================== 汇总与判据


def _run(outcome: str, **metrics: float) -> dict:
    return {"tests": {NID: {"outcome": outcome, "metrics": metrics}}}


def test_perf_summary_median_outcome_and_spread() -> None:
    s = G.perf_summary([_run("failed", call_cpu_ms=100.0), _run("failed", call_cpu_ms=104.0), _run("failed", call_cpu_ms=99.0)])[NID]
    assert s["outcome"] == "failed" and s["n"] == 3 and s["metrics"]["call_cpu_ms"] == 100.0 and s["spread"]["call_cpu_ms"] == 0.05
    assert G.perf_summary([_run("failed", a=1.0), _run("passed", a=1.0)])[NID]["outcome"] == "mixed"


def _base(outcome: str = "failed", **metrics: float) -> dict:
    return {"pytest": ["tests/targets/test_budget.py"], "host": HOST, "tests": {NID: {"outcome": outcome, "metrics": metrics}}}


CFG = {"pytest": ["tests/targets/test_budget.py"], "metrics": {NID: ["call_cpu_ms"]}, "max_regression": 0.05}


def _judge(outcome: str, value: float, base: dict | None, host: dict = HOST) -> list[str]:
    summ = G.perf_summary([_run(outcome, call_cpu_ms=value)])
    return G.judge_perf(summ, base, CFG, host)[0]


def test_judge_perf_regression_threshold() -> None:
    assert _judge("failed", 104.9, _base(call_cpu_ms=100.0)) == []                        # D2 预算未达成、结果相同、差 ≤ 5%
    assert _judge("failed", 90.0, _base(call_cpu_ms=100.0)) == []                         # 变快
    p = _judge("failed", 105.2, _base(call_cpu_ms=100.0))
    assert len(p) == 1 and "call_cpu_ms 105.2 > D2 baseline 100 x 1.05" in p[0]
    rows = G.judge_perf(G.perf_summary([_run("failed", call_cpu_ms=105.2)]), _base(call_cpu_ms=100.0), CFG, HOST)[1]
    assert rows == [{"test": NID, "metric": "call_cpu_ms", "value": 105.2, "d2": 100.0, "ratio": 1.052, "spread": 0.0}]


def test_judge_perf_outcomes() -> None:
    assert "D2 baseline passed" in _judge("failed", 100.0, _base("passed", call_cpu_ms=100.0))[0]   # D2 通过的须仍通过
    assert "outcome skipped != D2 baseline failed" in _judge("skipped", 100.0, _base(call_cpu_ms=100.0))[0]
    assert _judge("passed", 300.0, _base("failed", call_cpu_ms=100.0)) == []              # D2 未达成、现在达成：只记录不比较
    assert _judge("passed", 100.0, _base("passed", call_cpu_ms=100.0)) == []


def test_judge_perf_not_comparable() -> None:
    assert "no perf baseline" in _judge("failed", 100.0, None)[0]
    p = _judge("failed", 100.0, _base(call_cpu_ms=100.0), host={**HOST, "cpu_model": "Y"})
    assert len(p) == 1 and "not comparable" in p[0]                                       # 主机不同：不可比
    b = _base(call_cpu_ms=100.0)
    b["pytest"] = ["tests/perception"]
    assert "perf.pytest changed" in _judge("failed", 100.0, b)[0]
    assert "metric call_cpu_ms missing (D2 baseline)" in _judge("failed", 100.0, _base(other=1.0))[0]
    assert G.judge_perf({}, _base(call_cpu_ms=1.0), CFG, HOST)[0] == [f"{NID}: not run"]
    assert G.judge_perf(G.perf_summary([_run("failed", call_cpu_ms=1.0)]), {**_base(), "tests": {}}, CFG, HOST)[0] == [f"{NID}: no D2 baseline"]


def test_load_precondition_is_per_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(G, "_host_load", lambda: (10.0, 64))
    assert G.wait_perf_load({"load_max_per_cpu": 0.5, "load_wait_s": 0}) == (True, [0.156])
    monkeypatch.setattr(G, "_host_load", lambda: (40.0, 64))
    assert G.wait_perf_load({"load_max_per_cpu": 0.5, "load_wait_s": 0}) == (False, [0.625])


# ==================================================================== worker 与步骤（真实运行 M18 台架）


def test_perf_worker_records_cpu_time_and_properties(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    out = tmp_path / "run.json"
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"steps": [], "perf": {"pytest": ["tests/targets/test_budget.py"]}}), encoding="utf-8")
    argv = [sys.executable, str(ROOT / "tools/bench/d3/regress_gate.py"), "--perf-worker", str(out), "--gate", str(gate)]
    assert subprocess.run(argv, cwd=ROOT, capture_output=True, timeout=600, check=False).returncode == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["exit"] in (0, 1) and r["host"] == G.host_fingerprint()
    t = r["tests"][NID]
    assert t["outcome"] in ("passed", "failed")
    m = t["metrics"]
    assert {"call_cpu_ms", "call_wall_ms", "stage_cpu_ms_per_sim_s", "step_cpu_p99_ms", "stage_wall_ms_per_sim_s", "eff_cache"} <= set(m)
    assert m["stage_cpu_ms_per_sim_s"] > 0 and m["call_cpu_ms"] <= m["call_wall_ms"] * 1.01 + 1.0 and m["eff_cache"] <= 256
    assert "tests/targets/test_budget.py::test_measure_stage_reports_all_metrics" not in r["tests"]   # 只运行 perf 用例


def test_step_perf_records_and_compares(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """经子进程完整执行 perf 步骤：记录基线到临时 baseline.json，再与之比较（放宽阈值只验证管线，不判性能）。"""
    pytest.importorskip("numpy")
    # make 把 pytest 包在全局性能锁（flock -s $AWR_PERF_LOCK）里运行；这里验证的是管线，不是锁，否则 step_perf 的
    # LOCK_EX 等自己的父进程放锁，永远等不到（托管 CI shard 2 曾因此卡到 45 min 超时）
    monkeypatch.delenv("AWR_PERF_LOCK", raising=False)
    monkeypatch.delenv("AWR_PERF_LOCK_HELD", raising=False)
    base = tmp_path / "baseline.json"
    base.write_text(json.dumps({"schema": "awr.d3.regress_gate.baseline.v1", "digests": {"k": "v"}}) + "\n", encoding="utf-8")
    monkeypatch.setattr(G, "BASELINE", base)
    monkeypatch.setattr(G, "ROOT", ROOT)
    p = {"pytest": ["tests/targets/test_budget.py"], "metrics": {NID: ["stage_cpu_ms_per_sim_s"]}, "warmup": 0, "repeats": 1, "baseline_repeats": 2,
         "max_regression": 10.0, "load_max_per_cpu": 1e9, "load_wait_s": 0, "env": {"OMP_NUM_THREADS": "1"}}
    g = {"perf": p}
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"steps": [], **g}), encoding="utf-8")
    monkeypatch.setattr(G, "PY", sys.executable)
    real_run = G._run
    monkeypatch.setattr(G, "_run", lambda argv, *a, **k: real_run([*argv, "--gate", str(gate)] if "--perf-worker" in argv else argv, *a, **k))
    r = G.step_perf(g, tmp_path, True)
    b = json.loads(base.read_text(encoding="utf-8"))
    assert b["digests"] == {"k": "v"}                                                    # 只改写 perf 节
    assert b["perf"]["host"] == G.host_fingerprint() and NID in b["perf"]["tests"] and b["perf"]["repeats"] == 2   # baseline_repeats
    assert b["perf"]["tests"][NID]["n"] == 2
    assert r["pass"], r["problems"]
    assert r["compare"][0]["metric"] == "stage_cpu_ms_per_sim_s" and r["compare"][0]["ratio"] is not None
    r2 = G.step_perf(g, tmp_path, False)
    assert r2["pass"], r2["problems"]
    assert r2["tests"][NID]["n"] == 1                                                    # 门本身取 repeats 次


# ==================================================================== 入库基线


def test_committed_perf_baseline_matches_gate() -> None:
    g, b = G.load_gate(), G.load_baseline()
    perf = b.get("perf")
    assert perf, "baseline.json has no perf section (record on dmax with the d2 awr package)"
    assert sorted(perf["pytest"]) == sorted(g["perf"]["pytest"])
    assert set(perf["host"]) == {"cpu_model", "python", "numpy", "numba"} and "d2@928c57b" in perf["source"]["tree"]
    for nid, keys in g["perf"]["metrics"].items():
        assert nid in perf["tests"], nid
        assert all(k in perf["tests"][nid]["metrics"] for k in keys), nid
    assert perf["repeats"] == g["perf"]["baseline_repeats"] >= g["perf"]["repeats"] and perf["warmup"] == g["perf"]["warmup"]
    for nid in g["perf"]["metrics"]:
        assert perf["tests"][nid]["n"] == perf["repeats"], nid                            # 每次重复都产出了该用例


# ==================================================================== relay 场景质量检查差分


def _relay(checks: list[tuple[str, bool, object]]) -> dict:
    cs = [{"check": "summary.exit_code", "ok": True, "value": 0, "expect": 0}] + [{"check": c, "ok": ok, "value": v, "expect": "x"}
                                                                                   for c, ok, v in checks]
    return {"cases": [{"case": f"relay-b-1h-r{k}", "checks": cs} for k in (1, 2)] + [{"case": "relay-b-1h-determinism", "ok": True}]}


GAPS = [["other", 6.0], ["other", 33.6]]
D2 = {"checks": {"gaps.non_exempt": {"ok": False, "value": GAPS}, "dispatch_delays.battery": {"ok": True, "value": 0},
                 "handover.overlap_min": {"ok": True, "value": 15.2}}}


def test_relay_quality_same_as_d2_passes() -> None:
    rep = _relay([("gaps.non_exempt", False, [list(x) for x in GAPS]), ("dispatch_delays.battery", True, 0),
                  ("handover.overlap_min", True, 18.8)])                                 # 通过项取值变化不影响
    problems, rows = G.relay_quality_diff(rep, "b", 1.0, D2)
    assert problems == []
    v = {(r["case"], r["check"]): r["verdict"] for r in rows}
    assert v[("relay-b-1h-r1", "gaps.non_exempt")] == "same-as-d2" and v[("relay-b-1h-r2", "handover.overlap_min")] == "pass"
    assert not any(r["check"] == "summary.exit_code" for r in rows)                       # 运行性检查另由 relay_verdict 严格判定
    assert G.relay_verdict(rep, "b", 1.0)[1] == []


def test_relay_quality_changes_regress() -> None:
    changed = [["other", 6.0], ["other", 13.8]]                                          # 同样不通过，但间隙变了
    rep = _relay([("gaps.non_exempt", False, changed), ("dispatch_delays.battery", False, 2), ("handover.overlap_min", True, 15.2)])
    problems, _ = G.relay_quality_diff(rep, "b", 1.0, D2)
    assert len(problems) == 4                                                            # r1、r2 各两项
    assert any("gaps.non_exempt fails" in p and "13.8" in p for p in problems)
    assert any("dispatch_delays.battery fails (2 != D2 0)" in p for p in problems)
    p, _ = G.relay_quality_diff(_relay([("new.check", False, 1)]), "b", 1.0, D2)
    assert p == ["relay-b-1h-r1: new.check fails (no D2 baseline)", "relay-b-1h-r2: new.check fails (no D2 baseline)"]
    p, _ = G.relay_quality_diff(_relay([("gaps.non_exempt", False, GAPS)]), "b", 1.0, None)  # 无基线：任何不通过都回退
    assert len(p) == 2


def test_committed_relay_checks_baseline_is_self_consistent() -> None:
    b = G.load_baseline()
    rc = b.get("relay_checks")
    assert rc, "baseline.json has no relay_checks section (record zero-impact on d2)"
    assert rc["case"] == "relay-b-1h-r1" and "d2@928c57b" in rc["source"]["tree"]
    assert not any(k in rc["checks"] for k in G.RELAY_RUN_CHECKS)
    assert rc["checks"]["gaps.non_exempt"]["ok"] is False                                 # D2 既有的间隙缺陷（审计 AC-01 缺口 6）
    rep = {"cases": [{"case": f"relay-b-1h-r{k}", "checks": [{"check": c, **v} for c, v in rc["checks"].items()]} for k in (1, 2)]}
    assert G.relay_quality_diff(rep, "b", 1.0, rc)[0] == []


def _fake_relay_run(calls: list, checks: list[tuple[str, bool, object]]):
    """替代 G._run：relay_batch 调用写出 report.json 与两个对拍文件（内容固定），pytest 调用直接返回 0；记录 (argv, cwd)。"""

    def run(argv, cwd, timeout_s, env=None, log=None):
        calls.append((list(argv), Path(cwd), dict(env or {})))
        if "--out" in argv:
            d = Path(argv[argv.index("--out") + 1])
            rep = _relay(checks)
            for r in ("-r1", "-r2"):
                (d / f"relay-b-1h{r}").mkdir(parents=True, exist_ok=True)
            for f in ("relay_stats.json", "events.jsonl"):
                (d / "relay-b-1h-r1" / f).write_text(f, encoding="utf-8")
            (d / "report.json").write_text(json.dumps(rep), encoding="utf-8")
        return 0, 0.1

    return run


def test_zero_impact_runs_relay_batch_from_the_stack_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """记录 D2 基线时 relay_batch 取自 d2 检出（--stack-root）：relay_batch 把自身所在树的 python/ 置于 worker PYTHONPATH 之首，
    只设 PYTHONPATH 时 worker 实际运行的是本检出的代码。记录写 digests 与 relay_checks（附 source），再按差分口径判定。"""
    assert G.relay_root() == G.ROOT                                                      # 缺省：本检出
    d2 = tmp_path / "d2"
    d2.mkdir()
    base = tmp_path / "baseline.json"
    base.write_text(json.dumps({"schema": "awr.d3.regress_gate.baseline.v1", "resim": {"frames": "1"}}) + "\n", encoding="utf-8")
    monkeypatch.setattr(G, "BASELINE", base)
    monkeypatch.setattr(G, "ROOT", ROOT)
    monkeypatch.setattr(G, "PUBLIC_STACK_ROOT", [str(d2)])
    monkeypatch.setattr(G, "SOURCE_LABEL", ["d2@928c57b (test)"])
    calls: list = []
    monkeypatch.setattr(G, "_run", _fake_relay_run(calls, [("gaps.non_exempt", False, GAPS), ("handover.overlap_min", True, 18.8)]))
    g = {"zero_impact": {"relay_batch": {"case": "b", "hours": 1, "seed": 11}, "files": ["relay_stats.json", "events.jsonl"],
                         "flags": [{"AWR_D3_ENABLED": "0"}, {"AWR_D3_ENABLED": "1"}], "pytest": ["tests/targets/test_determinism.py"]}}
    r = G.step_zero_impact(g, tmp_path / "out", True)
    relay = [c for c in calls if "--out" in c[0]]
    assert len(relay) == 2 and all(c[0][1] == str(d2 / "tools/bench/sandbox/relay_batch.py") and c[1] == d2 for c in relay)
    assert [c[2] for c in relay] == g["zero_impact"]["flags"]
    b = json.loads(base.read_text(encoding="utf-8"))
    assert b["resim"] == {"frames": "1"}                                                  # 只改写本步骤的节
    assert set(b["digests"]) == {"relay_stats.json", "events.jsonl"} and b["relay_batch"]["case"] == "b"
    rc = b["relay_checks"]
    assert rc["case"] == "relay-b-1h-r1" and set(rc["checks"]) == {"gaps.non_exempt", "handover.overlap_min"}
    assert rc["source"]["tree"] == "d2@928c57b (test)" and rc["source"]["digests"] == b["digests"]
    assert r["pass"], r                                                                   # 与刚记录的基线相同：通过
    assert r["relay_root"] == str(d2) and r["runs"][0]["quality_problems"] == []
    verdicts = {(q["check"], q["verdict"]) for q in r["runs"][0]["quality"]}
    assert ("gaps.non_exempt", "same-as-d2") in verdicts and ("handover.overlap_min", "pass") in verdicts
    # 同一基线、d3 一侧间隙变化：判回退（字节对拍同时不一致时两类问题分别列出）
    monkeypatch.setattr(G, "PUBLIC_STACK_ROOT", [None])
    monkeypatch.setattr(G, "_run", _fake_relay_run([], [("gaps.non_exempt", False, [["other", 6.0]]), ("handover.overlap_min", True, 18.8)]))
    r2 = G.step_zero_impact(g, tmp_path / "out2", False)
    assert not r2["pass"] and r2["relay_root"] == str(ROOT)
    assert len(r2["runs"][0]["quality_problems"]) == 2 and r2["diffs"] == []
