#!/usr/bin/env python3
"""tools/bench/d3/regress_gate.py：D3 回归门（AWR-05 §4.8、D3-AC-01；WP-D3-00）。每个改动合入 `main` 之前运行（ADR-225 起只保留 `main`）。

步骤与命令登记在 `tests/d3/regress_gate/gate.json`（第 1–6 条）：
  1. D2 Python 测试全集（`pytest -m "not perf"`）、契约测试与前端单测；
  2. 零影响对拍（builtin `zero-impact`）：`AWR_D3_ENABLED = 0` 与 `= 1`（不加载事件）两种条件下各运行一次
     `relay_batch.py --case b --hours 1 --determinism`，`relay_stats.json` 与 `events.jsonl` 的 sha256 两两相同，且与
     `tests/d3/regress_gate/baseline.json`（D2 基线，`--record-baseline` 在 `d2` 分支上生成）相同；两种条件下
     `tests/targets/test_determinism.py` 都通过；relay_batch 的运行性检查（`RELAY_RUN_CHECKS`）与 determinism 为严格门槛，
     间隙等场景质量检查按差分口径判定：本身通过，或与 `baseline.json` 的 `relay_checks` 节（D2 基线）同样不通过且取值相同；
     resim 帧摘要对拍（builtin `resim`）：两种条件下各在独立子进程中按 `gate.json` 的固定剧本录制一次输入日志（假墙钟单步
     驱动，确定性），再以 `awr.sim.runtime.resim` 重仿真，帧摘要序列（每次 StateRing 发布的 Full64 sha256）与输入条目序列
     两两相同，且与 `baseline.json` 的 `resim` 节（D2 基线，d2@928c57b）相同；
  3. D2 性能用例子集（builtin `perf`，缺省运行；G04）：gate.json `perf.pytest` 的 perf 用例在独立子进程中预热 1 次、重复 7 次，
     逐用例 call 阶段的本线程 CPU 时间与用例写入的指标取中位，与 `baseline.json` 的 `perf` 节（D2 基线，dmax CPU 池中无其他
     D3 作业的窗口录制，`baseline_repeats` 次取中位）差 ≤ 5%；结果与 D2 相同（D2 通过的仍通过）；主机指纹须与基线相同，开跑前按核归一的负载前置条件（ADR-033）；
  4. `make lint`（含 lint-m06 与 import 方向规则）；5. import 方向；6. 公开演示构建与 K9 生产包扫描；K9 的 D3 扩展扫描
     （builtin `k9-d3`）：D2 公开演示构建产物的文本文件中 D3 入口字符串（`/lab/`、`/api/bench/v1/runs|events|session`，以及
     bench 客户端拼接路径所用的基址 `/api/bench/v1`）出现 0 次；报告附产物的 build-info（提交与构建时刻，判断是否陈旧）；
     D2 公开站 e2e 与 Tier S 首屏（builtin `public-e2e`）：本机 cgroup 模拟 emax 口径（AWR-04 §13）起后端栈，以仿 nginx 分流的
     站点提供公开演示构建（`/lab/` 构建存在时挂在 /lab/），运行 `tests/e2e/d2-public-gate.spec.ts`（落地页、World Hub、共享展示
     viewer、访客进入沙盒、D3 入口请求为 0、/lab/ 只发静态 GET），Tier S 首屏采样与 `baseline.json` 的 `public_e2e` 节（D2 基线）
     比较（判据见 `judge_tier_s`）。封板上线后在 emax 生产环境复测（ADR-194），本步骤是合入前的本机门。

用法：
    python tools/bench/d3/regress_gate.py                       # 缺省步骤
    python tools/bench/d3/regress_gate.py --steps zero-impact   # 只跑指定步骤（逗号分隔）
    PYTHONPATH=<d2 检出>/python python tools/bench/d3/regress_gate.py --steps zero-impact --record-baseline --stack-root <d2 检出> \
        --source-label "d2@928c57b ..."                         # 以 d2 检出的 relay_batch 与 awr 包记录 digests 与 relay_checks
                                                                # （d2 检出另需 tools/、scenarios/，worlds/ 链接到本检出）
    PYTHONPATH=<d2 检出>/python python tools/bench/d3/regress_gate.py --steps perf --record-baseline --source-label ...
                                                                # 以 d2 的 awr 包在基线主机（dmax CPU 池容器）记录性能基线
    PYTHONPATH=<d2 检出>/python python tools/bench/d3/regress_gate.py --steps resim --record-baseline
                                                                # 以 d2 检出的 awr 包记录 resim 基线（本脚本不随 d2 发布；
                                                                # d2 检出 = git archive 928c57b python packages vehicles configs，
                                                                # worlds/ 链接到本检出；--source-label 写入 source.tree）
    python tools/bench/d3/regress_gate.py --steps public-e2e --record-baseline --stack-root <d2 检出> --public-dist <D2 标签构建> \
        --source-label "d2@928c57b ..."                         # 以 d2 代码的栈与 D2 标签的演示构建记录 Tier S 基线
输出 `runs/perf/d3-gate-<UTC>/report.json`（`awr.d3.regress_gate.report.v1`）；退出码 0 通过、1 不通过、2 参数错误。
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GATE = ROOT / "tests" / "d3" / "regress_gate" / "gate.json"
BASELINE = ROOT / "tests" / "d3" / "regress_gate" / "baseline.json"
PY = sys.executable


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_gate(path: Path = GATE) -> dict:
    g = json.loads(path.read_text(encoding="utf-8"))
    ids = [s["id"] for s in g["steps"]]
    if len(ids) != len(set(ids)):
        raise ValueError("gate.json: duplicate step ids")
    return g


def select_steps(g: dict, only: list[str] | None) -> list[dict]:
    if only:
        known = {s["id"] for s in g["steps"]}
        bad = [x for x in only if x not in known]
        if bad:
            raise ValueError(f"unknown steps {bad}; known {sorted(known)}")
        return [s for s in g["steps"] if s["id"] in only]
    return [s for s in g["steps"] if s.get("default", True)]


def compare_digests(runs: list[dict[str, str]], baseline: dict[str, str] | None) -> list[str]:
    """runs: 每种开关条件下 {文件名: sha256}；返回不一致的说明（空表示一致）。"""
    out = []
    if not runs:
        return ["no runs"]
    ref = runs[0]
    for i, r in enumerate(runs[1:], 1):
        out.extend(f"{k}: flags #{i} {r.get(k)} != flags #0 {ref[k]}" for k in ref if r.get(k) != ref[k])
    if baseline is not None:
        out.extend(f"{k}: {ref.get(k)} != D2 baseline {v}" for k, v in baseline.items() if ref.get(k) != v)
    return out


# relay_batch 用例检查中的运行性检查（严格门槛）；其余（间隙、交接等场景质量检查）按 relay_quality_diff 与 D2 基线差分判定
RELAY_RUN_CHECKS = ("summary.exit_code", "summary.config_problems")


def relay_verdict(report: dict, case: str, hours: float) -> tuple[str, list[str]]:
    """从 relay_batch 的 report.json 取 r1 用例目录名与不通过项（运行性检查与 determinism 逐字节一致）。"""
    name = f"relay-{case}-{hours:g}h"
    cases = {c["case"]: c for c in report.get("cases", [])}
    problems = []
    for r in ("-r1", "-r2"):
        c = cases.get(name + r)
        if c is None:
            problems.append(f"{name}{r}: missing")
            continue
        problems += [f"{name}{r}: {k['check']}" for k in c.get("checks", []) if k["check"] in RELAY_RUN_CHECKS and not k["ok"]]
    det = cases.get(name + "-determinism")
    if det is None or not det.get("ok"):
        problems.append(f"{name}-determinism: {'missing' if det is None else 'not byte identical'}")
    return name + "-r1", problems


def _norm(v: object) -> str:
    """检查值的规范化文本（比较用：键排序、浮点按 JSON 文本）。"""
    return json.dumps(v, sort_keys=True, separators=(",", ":"), default=str)


def relay_checks_of(report: dict, case_name: str) -> dict[str, dict]:
    """relay_batch report.json 中某用例的场景质量检查（运行性检查之外）：{检查名: {ok, value, expect}}。"""
    c = {x["case"]: x for x in report.get("cases", [])}.get(case_name) or {}
    return {k["check"]: {"ok": bool(k["ok"]), "value": k.get("value"), "expect": k.get("expect")}
            for k in c.get("checks", []) if k["check"] not in RELAY_RUN_CHECKS}


def relay_quality_diff(report: dict, case: str, hours: float, base: dict | None) -> tuple[list[str], list[dict]]:
    """场景质量检查（间隙、交接、派遣延迟等）的差分口径（G04）："与 D2 基线相同即通过"。r1、r2 的每项检查：本身通过即通过；
    不通过时须与 baseline.json `relay_checks` 节中 D2 的同名检查同样不通过且取值逐项相同（D2 既有的场景质量缺陷不阻塞 D3，
    任何新增或变化的不通过即回退）。返回（不通过项，逐项对照表）。"""
    name = f"relay-{case}-{hours:g}h"
    bchecks = (base or {}).get("checks") or {}
    problems: list[str] = []
    rows: list[dict] = []
    for r in ("-r1", "-r2"):
        for chk, c in relay_checks_of(report, name + r).items():
            b = bchecks.get(chk)
            same = b is not None and b["ok"] == c["ok"] and _norm(b.get("value")) == _norm(c["value"])
            verdict = "pass" if c["ok"] else ("same-as-d2" if same else "regressed")
            rows.append({"case": name + r, "check": chk, "ok": c["ok"], "verdict": verdict, "value": c["value"],
                         "d2_ok": None if b is None else b["ok"], "d2_value": None if b is None else b.get("value")})
            if verdict == "regressed":
                problems.append(f"{name}{r}: {chk} fails" + (" (no D2 baseline)" if b is None else
                                                              f" ({_norm(c['value'])[:160]} != D2 {_norm(b.get('value'))[:160]})"))
    return problems, rows


def _run(argv: list[str], cwd: Path, timeout_s: float, env: dict | None = None, log: Path | None = None) -> tuple[int, float]:
    t0 = time.monotonic()
    e = {**os.environ, **(env or {})}
    with open(log, "w", encoding="utf-8") if log else open(os.devnull, "w") as f:
        try:
            r = subprocess.run(argv, cwd=cwd, env=e, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s, check=False)
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = 124
    return code, round(time.monotonic() - t0, 1)


def step_cmd(s: dict, out: Path) -> dict:
    argv = [PY if a == "{py}" else a for a in s["argv"]]
    code, dt = _run(argv, ROOT / s.get("cwd", "."), float(s.get("timeout_s", 3600)), log=out / f"{s['id']}.log")
    return {"pass": code == 0, "exit": code, "seconds": dt, "argv": argv}


def relay_root() -> Path:
    """relay_batch 所在的树：缺省为本检出；`--stack-root` 给出 d2 检出时取该检出。relay_batch 把自身所在树的 `python/` 置于
    worker 的 PYTHONPATH 之首（tools/bench/sandbox/relay_batch.py），只设 PYTHONPATH 不能让 worker 运行 d2 代码，记录 D2 基线
    （`digests`、`relay_checks`）时须以 d2 检出的 relay_batch 运行。"""
    return Path(PUBLIC_STACK_ROOT[0]) if PUBLIC_STACK_ROOT[0] else ROOT


def step_zero_impact(g: dict, out: Path, record: bool) -> dict:
    z = g["zero_impact"]
    rb = z["relay_batch"]
    root = relay_root()
    runs, detail, reps = [], [], []
    for k, flags in enumerate(z["flags"]):
        d = out / f"zero-{k}"
        argv = [PY, str(root / "tools/bench/sandbox/relay_batch.py"), "--case", rb["case"], "--hours", str(rb["hours"]),
                "--seed", str(rb["seed"]), "--no-lock", "--no-load-wait", "--determinism", "--out", str(d)]
        code, dt = _run(argv, root, 7200, env=flags, log=out / f"zero-{k}.log")
        rep = json.loads((d / "report.json").read_text(encoding="utf-8")) if (d / "report.json").exists() else {}
        reps.append(rep)
        case_name, problems = relay_verdict(rep, rb["case"], float(rb["hours"]))
        case_dir = d / case_name
        digests = {f: sha256(case_dir / f) for f in z["files"] if (case_dir / f).exists()}
        pcode, pdt = _run([PY, "-m", "pytest", "-q", *z["pytest"]], ROOT, 1800, env=flags, log=out / f"zero-{k}-pytest.log")
        runs.append(digests)
        detail.append({"flags": flags, "relay_batch_exit": code, "relay_problems": problems, "seconds": dt, "digests": digests,
                       "pytest_exit": pcode,
                       "pytest_seconds": pdt})
    missing = [f for f in z["files"] for r in runs if f not in r]
    name = f"relay-{rb['case']}-{float(rb['hours']):g}h"
    if record:
        write_baseline("relay_batch", rb)
        write_baseline("digests", runs[0])
        if relay_checks_of(reps[0], name + "-r1"):
            write_baseline("relay_checks", {"case": name + "-r1", "checks": relay_checks_of(reps[0], name + "-r1"),
                                            "source": {"tree": SOURCE_LABEL[0] or _rel_root(str(root)),
                                                       "flags": z["flags"][0], "host": host_fingerprint()["cpu_model"],
                                                       "digests": runs[0],
                                                       "recorded_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}})
    bl = load_baseline()
    base = bl.get("digests")
    for dd, rep in zip(detail, reps, strict=True):
        dd["quality_problems"], dd["quality"] = relay_quality_diff(rep, rb["case"], float(rb["hours"]), bl.get("relay_checks"))
    diffs = compare_digests(runs, base)
    ok = not missing and not diffs and all(not d["relay_problems"] and not d["quality_problems"] and d["pytest_exit"] == 0 for d in detail)
    return {"pass": ok, "runs": detail, "diffs": diffs, "missing": missing, "relay_root": str(root),
            "baseline": _rel_root(str(BASELINE)) if base else None}


# ---------------------------------------------------------------- resim 帧摘要（AWR-05 §4.8 第 2 条）
# 录制与重仿真只用 D1 起稳定的 awr 公共接口（SimCore、InputLog、resim），以便同一段代码在 d2 检出上记录基线
RESIM_KEYS = ("frames", "frames_sha256", "inputs", "inputs_sha256")
RESIM_CHUNK = 500


def _resim_principal(pid: str = "p-gate") -> dict:
    """录制用操作员主体（sim-core 无会话密钥时不验签；签名字段本来也不写入输入日志）。"""
    from awr.runtime.principal import Principal

    return Principal(pid, "operator", "api", "c-gate", True).fields()


def resim_record(run_dir: Path, spec: dict) -> int:
    """按 spec 的固定剧本录制 `run_dir/inputs.msgpack`（假墙钟单步驱动：命令在固定仿真时刻提交，apply_tick 确定）；
    返回录制结束时的 tick。命令未被受理即抛出 RuntimeError（剧本与当前代码不匹配，不能作为对拍输入）。"""
    import secrets

    from awr.contracts import LAYOUT_ID
    from awr.runtime.bus import LocalBus
    from awr.runtime.statering import LocalRing
    from awr.sim.fleet.fleet import FleetConfig
    from awr.sim.runtime.config import SimConfig
    from awr.sim.runtime.inputlog import InputLog, inputlog_header
    from awr.sim.runtime.main import SimCore

    tag = secrets.token_hex(3)
    fc = FleetConfig(kernel=spec.get("kernel", "numba"), path_capacity=int(spec.get("path_capacity", 65536)))
    cfg = SimConfig(run_id=str(spec.get("run_id", "gate-resim")), n_vehicles=int(spec["n_vehicles"]),
                    load_world=bool(spec.get("load_world", False)), world_id=str(spec.get("world_id", "shenzhen")),
                    autoplay=True, profile_id=str(spec.get("profile_id", "p600_mid360")),
                    spawn_xy=tuple(spec.get("spawn_xy", (0.0, 0.0))), spawn_spacing_m=float(spec.get("spawn_spacing_m", 15.0)),
                    fleet=fc, world_seed=int(spec.get("world_seed", 0)))
    path = f"/d3gate/{tag}/state.sim-core"
    ring, _ = LocalRing.open_or_create(path, capacity=fc.capacity, slots=32, layout_id=LAYOUT_ID, id_base=0, id_count=1024)
    bus = LocalBus.open("sim-core", namespace=f"awr/d3gate/{tag}")
    W = [1_000_000_000]
    core = SimCore(cfg, bus, ring, secret=None, wall_ns=lambda: W[0])
    il = None

    def advance_to(t_s: float) -> None:
        end = round(t_s * 1e9)
        step = 8 * core.clock.tick_ns  # 每次迭代墙钟前进 8 tick（同 M08 单步驱动夹具）
        guard = 0
        while core.clock.t_ns < end and guard < 10_000_000:
            W[0] += step
            core.iterate()
            guard += 1

    try:
        core.start()
        il = InputLog(Path(run_dir) / "inputs.msgpack", inputlog_header(core, cfg.run_id))
        core.inputlog = il
        core.engine.inputlog = il
        r = core._lease_op({"v": 1, "cid": "seat-gate", "op": "seat_claim", "principal": _resim_principal()})
        if r.get("code"):
            raise RuntimeError(f"seat_claim rejected: {r}")
        ids = [core.roster.by_slot[s].id for s in core.roster.slots_in_order()]
        for k, c in enumerate(sorted(spec["script"], key=lambda c: float(c["t_s"]))):
            advance_to(float(c["t_s"]))
            cid = f"gate-{k:03d}-{c['op']}"
            uav = c.get("uav", "*")
            msg = {"v": 1, "cid": cid, "op": c["op"], "uav": ids[uav] if isinstance(uav, int) else uav, "args": c.get("args") or {},
                   "principal": _resim_principal(), "lease": None, "t_wall_ns": 0, "epoch_seen": 1, "batch_id": None}
            adm = core.engine.handle(msg)
            if adm.get("status") != "accepted":
                raise RuntimeError(f"{cid}: {adm}")
        advance_to(float(spec["end_s"]))
        return int(core.clock.tick)
    finally:
        if il is not None:
            il.close()
        core.stop()
        bus.close()
        LocalRing.remove(path)


def resim_summary(digests: list[str], entries: list[dict], end_tick: int) -> dict:
    """帧摘要序列与输入条目序列的摘要：`frames_sha256` = sha256(各帧 sha256 十六进制串顺序拼接)（同 `resim_main`）；
    `inputs_sha256` 覆盖每条输入的 (tick, kind, cid, payload_sha256)，不含日志头（头中有版本与 FleetConfig 全集）；
    `chunks` 为每 500 帧一段的摘要，用于定位首个分歧段；`distinct` 为不同帧数（机体在运动的旁证）。"""
    inputs = "\n".join(f"{int(e['tick'])} {e['kind']} {e.get('cid')} {e['payload_sha256']}" for e in entries)
    return {"frames": len(digests), "distinct": len(set(digests)),
            "frames_sha256": hashlib.sha256("".join(digests).encode()).hexdigest(),
            "inputs": len(entries), "inputs_sha256": hashlib.sha256(inputs.encode()).hexdigest(), "end_tick": end_tick,
            "chunks": [hashlib.sha256("".join(digests[i:i + RESIM_CHUNK]).encode()).hexdigest()[:16]
                       for i in range(0, len(digests), RESIM_CHUNK)]}


def resim_worker(run_dir: Path, spec: dict) -> dict:
    """子进程：录制输入日志并重仿真，返回 `resim_summary`（附所用内核与 awr 版本）。"""
    from awr.sim.fleet.fleet import resolve_kernel
    from awr.sim.runtime.inputlog import _version, read_inputlog
    from awr.sim.runtime.resim import resim

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "inputs.msgpack").unlink(missing_ok=True)  # InputLog 以追加方式打开，残留的旧日志会混入第二个头
    end_tick = resim_record(run_dir, spec)
    _, entries = read_inputlog(run_dir / "inputs.msgpack")
    last = max((int(e["tick"]) for e in entries), default=0)
    digests = resim(run_dir, extra_ticks=max(0, end_tick - last))
    s = resim_summary(digests, entries, end_tick)
    s["kernel"] = resolve_kernel(spec.get("kernel", "numba"))[0]
    s["awr_version"] = _version()
    import awr

    s["awr_path"] = str(Path(awr.__file__).resolve().parent)
    return s


SOURCE_LABEL: list[str | None] = [None]  # --source-label：记录基线时写入 source.tree（例如 "d2@928c57b"）


def _rel_root(p: str | None) -> str | None:
    """awr 包路径：位于本检出内时写相对路径，否则只写 "external"（不把本机临时目录写进入库文件）。"""
    if not p:
        return None
    with contextlib.suppress(ValueError):
        return Path(p).resolve().relative_to(ROOT).as_posix()
    return "external"


def resim_digests(s: dict) -> dict[str, str]:
    return {k: str(s[k]) for k in RESIM_KEYS if k in s}


def step_resim(g: dict, out: Path, record: bool) -> dict:
    spec = g["resim"]
    runs, detail = [], []
    for k, flags in enumerate(g["zero_impact"]["flags"]):
        d = out / f"resim-{k}"
        argv = [PY, str(Path(__file__).resolve()), "--resim-worker", str(d)]
        code, dt = _run(argv, ROOT, float(spec.get("timeout_s", 1800)), env=flags, log=out / f"resim-{k}.log")
        s = json.loads((d / "resim.json").read_text(encoding="utf-8")) if code == 0 and (d / "resim.json").exists() else {}
        runs.append(resim_digests(s))
        detail.append({"flags": flags, "exit": code, "seconds": dt, "summary": s})
    if record and runs[0]:
        s0 = detail[0]["summary"]
        write_baseline("resim", {"spec_sha256": spec_sha256(spec), **runs[0], "chunks": s0.get("chunks", []),
                                 "source": {"tree": SOURCE_LABEL[0] or _rel_root(s0.get("awr_path")), "kernel": s0.get("kernel"),
                                            "awr_version": s0.get("awr_version"), "flags": detail[0]["flags"],
                                            "recorded_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}})
    base = load_baseline().get("resim")
    problems = [f"flags #{k}: exit {d['exit']}" for k, d in enumerate(detail) if d["exit"] != 0]
    if base is not None and base.get("spec_sha256") != spec_sha256(spec):
        problems.append("gate.json resim spec changed since the D2 baseline was recorded; re-record on d2")
    diffs = compare_digests(runs, {k: base[k] for k in RESIM_KEYS} if base else None) if not problems else []
    if base is None:
        problems.append(f"no resim baseline in {BASELINE.relative_to(ROOT)}")
    return {"pass": not problems and not diffs, "runs": detail, "diffs": diffs, "problems": problems,
            "first_diverging_chunk": first_diverging_chunk(detail, base)}


def first_diverging_chunk(detail: list[dict], base: dict | None) -> int | None:
    """首个与基线（无基线时与条件 #0）不同的 500 帧段序号；全部一致为 None。"""
    ref = (base or {}).get("chunks") or (detail[0]["summary"].get("chunks") if detail else None) or []
    for d in detail:
        got = d["summary"].get("chunks") or []
        for i in range(max(len(ref), len(got))):
            if i >= len(ref) or i >= len(got) or ref[i] != got[i]:
                return i
    return None


def spec_sha256(spec: dict) -> str:
    """resim 剧本的规范化摘要（剧本改动后基线失效，须在 d2 上重录）。"""
    keep = {k: v for k, v in spec.items() if k not in ("doc", "timeout_s")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------- K9 的 D3 扩展扫描（AWR-05 §4.8 第 6 条）
def scan_k9_d3(dist: Path, forbidden: list[str], text_ext: list[str]) -> dict:
    """D2 公开演示构建产物中 D3 入口字符串的出现次数：`counts` 逐字符串计数，`hits` 为 [文件, 字符串, 上下文]（每文件每串
    至多 3 条）；产物缺失（无 index.html）时 `missing` 为真。"""
    if not (dist / "index.html").is_file():
        return {"missing": True, "files": 0, "counts": dict.fromkeys(forbidden, 0), "hits": []}
    exts = tuple(e.lower() for e in text_ext)
    counts = dict.fromkeys(forbidden, 0)
    hits: list[list[str]] = []
    n = 0
    for p in sorted(dist.rglob("*")):
        if not p.is_file() or not p.name.lower().endswith(exts):
            continue
        n += 1
        text = p.read_text(encoding="utf-8", errors="replace")
        rel = p.relative_to(dist).as_posix()
        for s in forbidden:
            c = text.count(s)
            if not c:
                continue
            counts[s] += c
            i = -1
            for _ in range(min(c, 3)):
                i = text.find(s, i + 1)
                hits.append([rel, s, " ".join(text[max(0, i - 50):i + len(s) + 30].split())])
    return {"missing": False, "files": n, "counts": counts, "hits": hits}


def step_k9_d3(g: dict, out: Path) -> dict:
    k = g["k9_d3"]
    dist = (ROOT / k["dist"]).resolve()
    r = scan_k9_d3(dist, list(k["forbidden"]), list(k["text_ext"]))
    with open(out / "k9-d3.log", "w", encoding="utf-8") as f:
        for h in r["hits"]:
            f.write(f"{h[0]} {h[1]} ...{h[2]}...\n")
        f.write(f"k9-d3: {dist} files {r['files']} counts {json.dumps(r['counts'])}\n")
    bi = dist / "build-info.json"
    build = json.loads(bi.read_text(encoding="utf-8")) if bi.is_file() else None  # 产物对应的提交与构建时刻（判断是否陈旧）
    return {"pass": not r["missing"] and not any(r["counts"].values()), "dist": str(dist), "build_info": build, **r}


# ---------------------------------------------------------------- 公开站 e2e 与 Tier S 首屏（AWR-05 §4.8 第 6 条，G03）
# 本机 cgroup 模拟 emax 口径（AWR-04 §13）：后端栈为 tools/bench/sandbox/_sbbench.Stack（systemd-run --user --scope：单元 CPUQuota
# 250%、MemoryHigh 1.8G、MemoryMax 2G，taskset 4 核，单元内 cgroup-exec.sh 建 core/、sandbox/{ctl,plan,sb} 布局；supervisor 为
# emax 的资源配置：共享展示 S7、sandbox-pool、sandbox-api，无热备用、不钉核），Playwright 与浏览器在其余核上充当访客机器。
# 栈在独立子进程（--public-stack-worker）中运行：PYTHONPATH 与 _sbbench 都取自 stack_root，以便同一段代码以 d2 检出记录基线。
TIER_S_KEYS = ("ttfp", "ttfpFirstPixel", "warmupWaitMs", "tti", "firstScreenBytes", "programsAtReveal", "compiledAfterReveal",
               "glErrors", "drawn", "failed", "budgetViolations")


def _median(xs: list[float]) -> float | None:
    v = sorted(float(x) for x in xs if x is not None)
    if not v:
        return None
    m = len(v) // 2
    return round(v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2.0, 3)


def tier_s_summary(samples: list[dict]) -> dict[str, dict]:
    """每个世界的 Tier S 首屏汇总：时间量取中位（ttfp、ttfp_first_pixel、warmup_wait、tti），确定量取集合或极值（首屏字节、
    reveal 时的程序数、reveal 后新编译程序、GL 错误、失败节点、预算违例）；`tiers` 为出现过的 tier/deviceClass。"""
    out: dict[str, dict] = {}
    for w in dict.fromkeys(s["world"] for s in samples):
        ss = [s for s in samples if s["world"] == w]
        out[w] = {"n": len(ss), "tiers": sorted({f"{s.get('tier')}/{s.get('deviceClass')}" for s in ss}),
                  "ttfp_ms": _median([s.get("ttfp") for s in ss]), "ttfp_first_pixel_ms": _median([s.get("ttfpFirstPixel") for s in ss]),
                  "warmup_wait_ms": _median([s.get("warmupWaitMs") for s in ss]), "tti_ms": _median([s.get("tti") for s in ss]),
                  "first_screen_bytes": sorted({int(s.get("firstScreenBytes") or 0) for s in ss}),
                  "programs_at_reveal": max(int(s.get("programsAtReveal") or 0) for s in ss),
                  "compiled_after_reveal": max(int(s.get("compiledAfterReveal") or 0) for s in ss),
                  "drawn": _median([s.get("drawn") for s in ss]),
                  "gl_errors": sum(int(s.get("glErrors") or 0) for s in ss), "failed": sum(int(s.get("failed") or 0) for s in ss),
                  "budget_violations": sum(int(s.get("budgetViolations") or 0) for s in ss)}
    return out


def judge_tier_s(summary: dict[str, dict], base: dict | None, cfg: dict) -> list[str]:
    """Tier S 首屏"不回退"判据（D3-AC-01；阈值登记在 gate.json `public_e2e.tier_s`）：
    1. 每个采样 Tier S、software（C1 旗标的 SwiftShader）；GL 错误、失败节点、预算违例、reveal 后新编译程序均为 0；
    2. TTFP 中位 ≤ `ttfp_ms_max`（D1-AC-02 冻结值 1000 ms；不含着色器预热等待，同 D1-AC-02 口径）；
    3. 与 D2 基线：首屏字节逐字节相同（同一世界包与同一构建是确定量）；reveal 时程序数 ≤ 基线；绘制点数中位 ≥ 基线 ×
       (1 − `regress_max`)；TTFP 中位 ≤ max(基线 × (1 + `regress_max`), 基线 + `regress_floor_ms`)（本机共享主机的计时分辨率
       下限，数值见 gate.json）；
    tti 与预热等待只记录（SwiftShader 上由着色器编译主导、随主机负载变化，thresholds.json 的 tti_ms 只对 scene=pc 冻结）。"""
    out = []
    worlds = list(cfg.get("worlds") or summary)
    rmax = float(cfg.get("regress_max", 0.05))
    floor = float(cfg.get("regress_floor_ms", 50.0))
    for w in worlds:
        s = summary.get(w)
        if not s:
            out.append(f"{w}: no Tier S sample")
            continue
        if s["tiers"] != ["S/software"]:
            out.append(f"{w}: tier {s['tiers']} (expected S/software)")
        out.extend(f"{w}: {k} {s[k]}" for k in ("gl_errors", "failed", "budget_violations", "compiled_after_reveal") if s[k])
        if len(s["first_screen_bytes"]) != 1 or not s["first_screen_bytes"][0]:
            out.append(f"{w}: first screen bytes {s['first_screen_bytes']}")
        if s["ttfp_ms"] is None or s["ttfp_ms"] > float(cfg.get("ttfp_ms_max", 1000)):
            out.append(f"{w}: TTFP median {s['ttfp_ms']} ms > {cfg.get('ttfp_ms_max', 1000)} ms")
        b = ((base or {}).get("tier_s") or {}).get(w)
        if b is None:
            out.append(f"{w}: no D2 baseline")
            continue
        if s["first_screen_bytes"] != [b["first_screen_bytes"]]:
            out.append(f"{w}: first screen bytes {s['first_screen_bytes']} != D2 baseline {b['first_screen_bytes']}")
        if s["programs_at_reveal"] > b["programs_at_reveal"]:
            out.append(f"{w}: programs at reveal {s['programs_at_reveal']} > D2 baseline {b['programs_at_reveal']}")
        if s["drawn"] is not None and s["drawn"] < b["drawn"] * (1.0 - rmax):
            out.append(f"{w}: drawn points median {s['drawn']} < D2 baseline {b['drawn']} x {1.0 - rmax:g}")
        lim = max(b["ttfp_ms"] * (1.0 + rmax), b["ttfp_ms"] + floor)
        if s["ttfp_ms"] is not None and s["ttfp_ms"] > lim:
            out.append(f"{w}: TTFP median {s['ttfp_ms']} ms > {lim:.1f} ms (D2 baseline {b['ttfp_ms']} ms)")
    return out


def e2e_checks(metrics: dict, cfg: dict) -> list[str]:
    """spec 记录的功能检查：`required` 中每一项都出现且 ok（缺项即不通过，防止用例被跳过时误判通过）；D3 入口请求为 0。"""
    got = {c["test"]: c for c in metrics.get("checks") or []}
    out = [f"{t}: missing" if t not in got else f"{t}: {got[t].get('detail')}" for t in cfg.get("required", []) if not got.get(t, {}).get("ok")]
    if metrics.get("lab_dist") and not got.get("lab-static", {}).get("ok"):
        out.append(f"lab-static: {got.get('lab-static', {}).get('detail', 'missing')}")
    if metrics.get("d3_requests"):
        out.append(f"D3 entry requests from D2 pages: {metrics['d3_requests'][:5]}")
    return out


def public_stack_worker(cfg_path: Path) -> int:
    """子进程：以 stack_root 的 `_sbbench.Stack` 起 cgroup 模拟栈，打印一行 JSON（base、show_base、cgroup_mode…），stdin 关闭后停栈。"""
    import importlib.util

    c = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    root = Path(c["stack_root"]).resolve()
    spec = importlib.util.spec_from_file_location("_sbbench", root / "tools" / "bench" / "sandbox" / "_sbbench.py")
    sb = importlib.util.module_from_spec(spec)
    sys.modules["_sbbench"] = sb
    spec.loader.exec_module(sb)
    cg = c.get("cgroup") or {}
    st = c.get("stack") or {}
    stack = sb.Stack(world=st.get("world", "synthcity"), max_sessions=st.get("max_sessions", "auto"), cgroup=bool(cg.get("enabled", True)),
                     cpu_quota_pct=int(cg.get("cpu_quota_pct", 250)), mem_high=str(cg.get("mem_high", "1.8G")),
                     mem_max=str(cg.get("mem_max", "2G")), cpus=str(cg.get("cpus", "0-3")), show_scenario=st.get("show_scenario"),
                     quota_profile=str(st.get("quota_profile", "local")), ws_max_show=int(st.get("ws_max_show", 24)),
                     extra_set=list(st.get("set") or []))
    try:
        stack.start(ready_timeout_s=float(st.get("ready_timeout_s", 240)))
    except Exception as e:  # 启动失败：一行 JSON 报告原因（父进程据此判不通过）
        print(json.dumps({"error": str(e)[-2000:]}), flush=True)
        return 1
    try:
        import awr

        print(json.dumps({"base": stack.base, "show_base": stack.show_base, "run_id": stack.run_id, "cgroup_mode": stack.cgroup_mode,
                          "supervisor_pid": stack.supervisor_pid(), "awr_path": str(Path(awr.__file__).resolve().parent)}), flush=True)
        sys.stdin.read()  # 父进程关闭 stdin 即停栈
    finally:
        stack.stop()
    return 0


def _start_public_stack(c: dict, out: Path) -> tuple[subprocess.Popen, dict]:
    import threading

    cfg_path = out / "public-stack.json"
    cfg_path.write_text(json.dumps(c, indent=1) + "\n", encoding="utf-8")
    root = Path(c["stack_root"]).resolve()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": os.pathsep.join([str(root / "python"), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    with open(out / "public-stack.log", "w", encoding="utf-8") as log:  # 子进程持有自己的描述符，父进程随即关闭
        p = subprocess.Popen([PY, str(Path(__file__).resolve()), "--public-stack-worker", str(cfg_path)], cwd=root, env=env,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True)
    box: list[dict] = []

    def first_line() -> None:
        assert p.stdout is not None
        for line in p.stdout:
            if line.startswith("{"):
                box.append(json.loads(line))
                return

    t = threading.Thread(target=first_line, daemon=True)
    t.start()
    t.join(float((c.get("stack") or {}).get("ready_timeout_s", 240)) + 120)
    return p, (box[0] if box else {"error": "no stack line (see public-stack.log)"})


def _stop_public_stack(p: subprocess.Popen) -> None:
    with contextlib.suppress(OSError, ValueError):
        if p.stdin:
            p.stdin.close()
    try:
        p.wait(120)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(10)


def cpus_available(spec: str) -> str | None:
    """taskset 核列表（"4-7"、"4,5"）在本机全部可用时原样返回，否则 None（核数不足的主机上不钉核，报告注明）。"""
    if not spec:
        return None
    want: set[int] = set()
    try:
        for part in spec.split(","):
            lo, _, hi = part.partition("-")
            want.update(range(int(lo), int(hi or lo) + 1))
    except ValueError:
        return None
    have = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    return spec if want and want <= have else None


def _loadavg() -> list[float] | None:
    with contextlib.suppress(OSError):
        return [round(x, 2) for x in os.getloadavg()]
    return None


def step_public_e2e(g: dict, out: Path, record: bool) -> dict:
    """第 6 条：cgroup 模拟 emax 口径下 D2 公开站 e2e（tests/e2e/d2-public-gate.spec.ts）与 Tier S 首屏；排在 public-build 之后。"""
    c = dict(g["public_e2e"])
    c["stack_root"] = str((ROOT / c.get("stack_root", ".")).resolve()) if not PUBLIC_STACK_ROOT[0] else PUBLIC_STACK_ROOT[0]
    dist = (ROOT / (PUBLIC_DIST[0] or c["dist"])).resolve()
    lab = (ROOT / c["lab_dist"]).resolve() if c.get("lab_dist") else None
    lab = lab if lab is not None and (lab / "index.html").is_file() else None
    d = out / "public-e2e"
    d.mkdir(parents=True, exist_ok=True)
    bi = dist / "build-info.json"
    build = json.loads(bi.read_text(encoding="utf-8")) if bi.is_file() else None
    res: dict = {"dist": str(dist), "build_info": build, "lab_dist": str(lab) if lab else None, "stack_root": c["stack_root"]}
    if not (dist / "index.html").is_file() or (build or {}).get("channel") != "demo":
        return {**res, "pass": False, "problems": [f"{dist}: not a public demo build (run public-build first)"]}
    tier = c.get("tier_s") or {}
    res["load_before"] = _loadavg()
    proc, info = _start_public_stack(c, d)
    res["stack"] = {k: v for k, v in info.items() if k != "awr_path"} | {"tree": _rel_root(info.get("awr_path"))}
    if "error" in info:
        _stop_public_stack(proc)
        return {**res, "pass": False, "problems": [f"stack: {info['error'][-500:]}"]}
    metrics_path = d / "metrics.json"
    metrics_path.unlink(missing_ok=True)
    env = {"D2PG_DIST": str(dist), "D2PG_SANDBOX": info["base"], "D2PG_SHOW": info["show_base"], "D2PG_OUT": str(metrics_path),
           "D2PG_WORLDS": ",".join(tier.get("worlds", ["synthcity", "shenzhen"])), "D2PG_REPEATS": str(tier.get("repeats", 3)),
           "AWR_PERF_BASE": "http://127.0.0.1:1",  # 不起 playwright.config 的 vite preview（spec 自起仿 nginx 站点）
           "PLAYWRIGHT_JSON_OUTPUT_NAME": str(d / "playwright.json")}
    if lab:
        env["D2PG_LAB_DIST"] = str(lab)
    argv = ["npx", "playwright", "test", Path(c["spec"]).stem, "--project", "e2e", "--reporter=line,json"]
    client = cpus_available(str(c.get("client_cpus") or ""))
    res["client_cpus"] = client
    if client:
        argv = ["taskset", "-c", client, *argv]
    try:
        code, dt = _run(argv, ROOT / "apps" / "web", float(c.get("timeout_s", 2400)), env=env, log=d / "playwright.log")
    finally:
        _stop_public_stack(proc)
    res["load_after"] = _loadavg()
    m = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    summ = tier_s_summary(m.get("tier_s") or [])
    res.update({"playwright_exit": code, "seconds": dt, "checks": m.get("checks") or [], "tier_s": summ,
                "d3_requests": m.get("d3_requests") or []})
    cond = {"worlds": list(tier.get("worlds", [])), "repeats": int(tier.get("repeats", 3)), "viewport": m.get("viewport"),
            "cgroup": c.get("cgroup"), "client_cpus": c.get("client_cpus")}
    if record and code == 0 and summ:
        write_baseline("public_e2e", {
            "conditions": cond,
            "tier_s": {w: {"first_screen_bytes": s["first_screen_bytes"][0], "ttfp_ms": s["ttfp_ms"], "programs_at_reveal": s["programs_at_reveal"],
                           "drawn": s["drawn"], "ttfp_first_pixel_ms": s["ttfp_first_pixel_ms"], "tti_ms": s["tti_ms"], "n": s["n"]}
                       for w, s in summ.items()},
            "source": {"tree": SOURCE_LABEL[0] or res["stack"]["tree"], "build_info": build, "cgroup_mode": info.get("cgroup_mode"),
                       "load_before": res["load_before"], "recorded_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}})
    base = load_baseline().get("public_e2e")
    problems = [f"playwright exit {code} (see public-e2e/playwright.log)"] if code != 0 else []
    if info.get("cgroup_mode") != "systemd-scope" and c.get("require_cgroup", True):
        problems.append(f"cgroup emulation {info.get('cgroup_mode')} (systemd user scope unavailable): not an emax-like run")
    problems += e2e_checks(m, c)
    if base is not None and base.get("conditions", {}).get("worlds") != cond["worlds"]:
        problems.append("gate.json public_e2e.tier_s worlds changed since the D2 baseline was recorded; re-record")
    problems += judge_tier_s(summ, base, tier)
    return {**res, "pass": not problems, "problems": problems}


PUBLIC_STACK_ROOT: list[str | None] = [None]  # --stack-root：记录 D2 基线时栈取自 d2 检出（git archive 928c57b + worlds 链接）
PUBLIC_DIST: list[str | None] = [None]        # --public-dist：替代 gate.json 的 public_e2e.dist（例如 D2 标签构建）


# ---------------------------------------------------------------- 基线文件
def load_baseline() -> dict:
    return json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}


def write_baseline(section: str, value: dict) -> None:
    """只改写 baseline.json 的一节（`digests` 为零影响对拍，`resim` 为 resim 帧摘要），其余节与来源说明保留。"""
    b = load_baseline() or {"schema": "awr.d3.regress_gate.baseline.v1"}
    b[section] = value
    BASELINE.write_text(json.dumps(b, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- D2 性能子集（AWR-05 §4.8 第 3 条，G04）
# 每次重复在独立子进程（--perf-worker）中以 pytest 运行 gate.json `perf.pytest` 的 perf 用例；_PerfRecorder 记录每个用例 call
# 阶段的本线程 CPU 时间（`call_cpu_ms`；主机负载下的抢占与排队不计入）、墙钟与结果，并收集用例经 record_property 写入的数值
# 指标（例如 tests/targets/test_budget.py 的 stage_cpu_ms_per_sim_s）。前 `warmup` 次只预热（numba 缓存、页缓存），其后
# `repeats` 次逐指标取中位（ADR-033；记录基线时取 `baseline_repeats` 次），与 baseline.json 的 `perf` 节（D2 基线：同一套用例以 d2 检出的 awr 包在 dmax CPU 池
# 无负载窗口录制）比较，各指标 ≤ 基线 × (1 + max_regression)。D2 预算阈值本身在 D2 未达成的用例（M18-AC-022、M19-AC-026）
# 只要求结果与 D2 基线相同；D2 通过的用例须仍通过。主机指纹（CPU 型号、Python、numpy、numba）与基线不同即不可比，判不通过。
def host_fingerprint() -> dict[str, str]:
    """性能基线的主机指纹：CPU 型号与解释器、数值库版本（CPU 时间依赖这些；不同即须在本机重录 D2 基线）。"""
    import platform
    from importlib.metadata import PackageNotFoundError, version

    cpu = platform.processor() or "unknown"
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break

    def v(p: str) -> str:
        try:
            return version(p)
        except PackageNotFoundError:
            return ""

    return {"cpu_model": cpu, "python": ".".join(platform.python_version_tuple()[:2]), "numpy": v("numpy"), "numba": v("numba")}


def perf_recorder():  # 返回 pytest 插件对象（pytest 只在 worker 中导入）
    """pytest 插件：每个用例 call 阶段的本线程 CPU 时间、墙钟、结果与 record_property 的数值指标（`.tests`）。"""
    import pytest

    class PerfRecorder:
        def __init__(self) -> None:
            self.tests: dict[str, dict] = {}

        def _t(self, nodeid: str) -> dict:
            return self.tests.setdefault(nodeid, {"outcome": None, "metrics": {}})

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_call(self, item):  # pytest 钩子
            c0, w0 = time.thread_time_ns(), time.perf_counter_ns()
            try:
                yield
            finally:
                m = self._t(item.nodeid)["metrics"]
                m["call_cpu_ms"] = round((time.thread_time_ns() - c0) / 1e6, 3)
                m["call_wall_ms"] = round((time.perf_counter_ns() - w0) / 1e6, 3)

        def pytest_runtest_logreport(self, report) -> None:  # pytest 钩子
            t = self._t(report.nodeid)
            if report.when == "call" or (report.outcome != "passed" and t["outcome"] is None):
                t["outcome"] = report.outcome
            if report.when == "call":
                t["metrics"].update({str(k): float(v) for k, v in report.user_properties
                                     if isinstance(v, (int, float)) and not isinstance(v, bool)})

    return PerfRecorder()


def perf_worker(out_json: Path, spec: dict) -> int:
    """子进程：运行一次 perf 用例子集，写 {tests: {nodeid: {outcome, metrics}}, exit, host}。"""
    import pytest

    rec = perf_recorder()
    code = pytest.main(["-q", "-m", "perf", "-p", "no:cacheprovider", "-p", "no:warnings", *spec["pytest"]], plugins=[rec])
    import awr

    out_json.write_text(json.dumps({"exit": int(code), "tests": rec.tests, "host": host_fingerprint(),
                                    "awr_path": str(Path(awr.__file__).resolve().parent)}, indent=1) + "\n", encoding="utf-8")
    return 0


def perf_summary(runs: list[dict]) -> dict[str, dict]:
    """逐用例：结果取众数（不一致时记 "mixed"），各指标取中位，`spread` 为 (max − min) / 中位（重复间离散度，只记录）。"""
    import statistics

    out: dict[str, dict] = {}
    for nid in dict.fromkeys(n for r in runs for n in r.get("tests", {})):
        ts = [r["tests"][nid] for r in runs if nid in r.get("tests", {})]
        oc = {t.get("outcome") for t in ts}
        keys = dict.fromkeys(k for t in ts for k in t.get("metrics", {}))
        med, spread = {}, {}
        for k in keys:
            xs = [float(t["metrics"][k]) for t in ts if k in t.get("metrics", {})]
            med[k] = round(statistics.median(xs), 4)
            spread[k] = round((max(xs) - min(xs)) / med[k], 4) if med[k] else 0.0
        out[nid] = {"outcome": oc.pop() if len(oc) == 1 else "mixed", "n": len(ts), "metrics": med, "spread": spread}
    return out


def judge_perf(summary: dict[str, dict], base: dict | None, cfg: dict, host: dict) -> tuple[list[str], list[dict]]:
    """与 D2 基线比较：返回（不通过项，逐指标对照表）。判据见本节注释；`cfg["metrics"]` 为 {nodeid: [指标]}。"""
    problems: list[str] = []
    rows: list[dict] = []
    rmax = float(cfg.get("max_regression", 0.05))
    if base is None:
        return [f"no perf baseline in {BASELINE.relative_to(ROOT)} (record on d2: --steps perf --record-baseline)"], rows
    if base.get("host") != host:
        problems.append(f"host {host} != D2 baseline host {base.get('host')}: not comparable, re-record the D2 baseline on this host")
    if sorted(base.get("pytest") or []) != sorted(cfg.get("pytest") or []):
        problems.append("gate.json perf.pytest changed since the D2 baseline was recorded; re-record on d2")
    btests = base.get("tests") or {}
    for nid, keys in (cfg.get("metrics") or {}).items():
        s, b = summary.get(nid), btests.get(nid)
        if s is None:
            problems.append(f"{nid}: not run")
            continue
        if b is None:
            problems.append(f"{nid}: no D2 baseline")
            continue
        if s["outcome"] != "passed" and b["outcome"] == "passed":
            problems.append(f"{nid}: {s['outcome']} (D2 baseline passed)")
        elif s["outcome"] != b["outcome"] and s["outcome"] != "passed":
            problems.append(f"{nid}: outcome {s['outcome']} != D2 baseline {b['outcome']}")
        improved = s["outcome"] == "passed" and b["outcome"] != "passed"   # D2 未达成、现在达成：执行路径变长，只记录不比较
        for k in keys:
            cur, ref = s["metrics"].get(k), (b.get("metrics") or {}).get(k)
            row = {"test": nid, "metric": k, "value": cur, "d2": ref, "ratio": round(cur / ref, 4) if cur is not None and ref else None,
                   "spread": s["spread"].get(k)}
            rows.append(row)
            if cur is None or ref is None:
                problems.append(f"{nid}: metric {k} missing ({'run' if cur is None else 'D2 baseline'})")
            elif not improved and cur > ref * (1.0 + rmax):
                problems.append(f"{nid}: {k} {cur:g} > D2 baseline {ref:g} x {1.0 + rmax:g}")
    return problems, rows


def _host_load() -> tuple[float, int]:
    """1 分钟 loadavg 与主机逻辑 CPU 数（容器内 loadavg 为主机值，CPU 数亦取主机的，故以每 CPU 负载判定）。"""
    return os.getloadavg()[0], os.cpu_count() or 1


def wait_perf_load(cfg: dict) -> tuple[bool, list[float]]:
    """ADR-033 PR-2 的按核归一口径：1 分钟 loadavg / CPU 数 ≤ `load_max_per_cpu`，每 10 s 复查，最长 `load_wait_s`。"""
    lim = float(cfg.get("load_max_per_cpu", 0.5))
    deadline = time.monotonic() + float(cfg.get("load_wait_s", 600))
    seen = []
    while True:
        la, n = _host_load()
        seen.append(round(la / n, 3))
        if la / n <= lim:
            return True, seen
        if time.monotonic() > deadline:
            return False, seen
        print(f"[d3-gate] perf: waiting for host load {la:.1f} / {n} CPU > {lim}", flush=True)
        time.sleep(10.0)


def step_perf(g: dict, out: Path, record: bool) -> dict:
    """第 3 条：D2 性能子集与 D2 基线差 ≤ 5%（见本节注释）。`--record-baseline` 以当前 PYTHONPATH 的 awr 包写 `perf` 节。"""
    import fcntl

    p = g["perf"]
    d = out / "perf"
    d.mkdir(parents=True, exist_ok=True)
    lock = None
    # ADR-033 PR-1 / ADR-195 (3)：dmax 上为 $D/.perf.lock（flock -x）；调用方已持锁（make 的 AWR_PERF_LOCK_HELD，
    # mk/common.mk）时不再加排它锁——对父进程的共享锁加 LOCK_EX 会永远等下去
    if os.environ.get("AWR_PERF_LOCK") and not os.environ.get("AWR_PERF_LOCK_HELD"):
        lock = open(os.environ["AWR_PERF_LOCK"], "a+")  # noqa: SIM115 - 持有到本步骤结束
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    try:
        ready, load_wait = wait_perf_load(p)
        res: dict = {"load_wait_per_cpu": load_wait, "load_before": _loadavg(), "cpus": os.cpu_count(), "lock": bool(lock)}
        if not ready:
            return {**res, "pass": False, "problems": [f"host load per CPU {load_wait[-1]} > {p.get('load_max_per_cpu', 0.5)}: not judged"]}
        env = {str(k): str(v) for k, v in (p.get("env") or {}).items()}
        warm, reps = int(p.get("warmup", 1)), int(p.get("repeats", 3))
        if record:                                 # 基线取更多次重复的中位（共享 CPU 池上单次会话的中位约有 ±3% 的漂移）
            reps = int(p.get("baseline_repeats", reps))
        runs, exits = [], []
        t0 = time.monotonic()
        for k in range(warm + reps):
            j = d / f"run-{k}.json"
            j.unlink(missing_ok=True)
            code, _dt = _run([PY, str(Path(__file__).resolve()), "--perf-worker", str(j)], ROOT, float(p.get("timeout_s", 1800)), env=env,
                             log=d / f"run-{k}.log")
            r = json.loads(j.read_text(encoding="utf-8")) if code == 0 and j.is_file() else {"exit": code, "tests": {}}
            exits.append(r.get("exit", code))
            if k >= warm:
                runs.append(r)
        res.update({"seconds": round(time.monotonic() - t0, 1), "load_after": _loadavg(), "pytest_exits": exits})
    finally:
        if lock is not None:
            lock.close()
    host = host_fingerprint()
    summ = perf_summary(runs)
    tree = _rel_root(runs[0].get("awr_path")) if runs else None
    if record and runs and all(r.get("tests") for r in runs):
        write_baseline("perf", {"pytest": list(p["pytest"]), "host": host, "tests": summ, "warmup": warm, "repeats": reps,
                                "source": {"tree": SOURCE_LABEL[0] or tree, "cpus": os.cpu_count(), "load_before": res["load_before"],
                                           "load_after": res["load_after"],
                                           "recorded_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}})
    problems, rows = judge_perf(summ, load_baseline().get("perf"), p, host)
    problems += [f"run #{k}: pytest exit {c} (collection or internal error)" for k, c in enumerate(exits) if c not in (0, 1)]
    return {**res, "pass": not problems, "problems": problems, "host": host, "tree": tree, "tests": summ, "compare": rows}


BUILTIN = {"zero-impact": step_zero_impact, "resim": step_resim, "perf": step_perf,
           "k9-d3": lambda g, out, _rec: step_k9_d3(g, out), "public-e2e": step_public_e2e}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", default=None, help="comma separated step ids (default: all default steps)")
    ap.add_argument("--record-baseline", action="store_true", help="write the zero-impact digests as the D2 baseline")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gate", default=None, help="alternative gate.json (default tests/d3/regress_gate/gate.json)")
    ap.add_argument("--source-label", default=None, help="with --record-baseline: label of the recorded tree (e.g. d2@928c57b)")
    ap.add_argument("--stack-root", default=None, help="public-e2e and zero-impact: tree the backend stack and relay_batch run from "
                    "(default this checkout; a d2 checkout when recording the D2 baseline)")
    ap.add_argument("--public-dist", default=None, help="public-e2e: public demo build to serve (default gate.json public_e2e.dist)")
    ap.add_argument("--resim-worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--perf-worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--public-stack-worker", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    SOURCE_LABEL[0] = a.source_label
    PUBLIC_STACK_ROOT[0] = str(Path(a.stack_root).resolve()) if a.stack_root else None
    PUBLIC_DIST[0] = str(Path(a.public_dist).resolve()) if a.public_dist else None
    if a.public_stack_worker:
        return public_stack_worker(Path(a.public_stack_worker))
    try:
        g = load_gate(Path(a.gate) if a.gate else GATE)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    if a.perf_worker:
        return perf_worker(Path(a.perf_worker), g["perf"])
    if a.resim_worker:
        d = Path(a.resim_worker)
        s = resim_worker(d, g["resim"])
        (d / "resim.json").write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({k: s[k] for k in ("frames", "frames_sha256", "inputs", "inputs_sha256", "kernel", "awr_path")}))
        return 0
    try:
        steps = select_steps(g, a.steps.split(",") if a.steps else None)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(a.out) if a.out else ROOT / "runs" / "perf" / f"d3-gate-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for s in steps:
        print(f"[d3-gate] {s['id']} ...", flush=True)
        r = BUILTIN[s["id"]](g, out, a.record_baseline) if s["kind"] == "builtin" else step_cmd(s, out)
        r["item"] = s["item"]
        res[s["id"]] = r
        print(f"[d3-gate] {s['id']}: {'PASS' if r['pass'] else 'FAIL'}", flush=True)
    rep = {"schema": "awr.d3.regress_gate.report.v1", "utc": stamp, "steps": res, "pass": all(r["pass"] for r in res.values())}
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[d3-gate] report {out / 'report.json'}: {'PASS' if rep['pass'] else 'FAIL'}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
