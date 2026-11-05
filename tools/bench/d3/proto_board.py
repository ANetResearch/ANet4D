#!/usr/bin/env python3
"""tools/bench/d3/proto_board.py：P0a 原型榜（WP-D3-09、WP-D3-30）。按 run_spec 对参考基线 × 种子逐一运行 Env、评测并聚合成榜。

每个 (策略, 种子) 一个独立子进程（Realistic 档各自启动 gz 实例），产出写入 `<out>/runs/<policy>-s<seed>/`，评测写入同目录
`evaluation.json`；全部完成后写 `<out>/index.json`（评测行索引）、`config.json` 与 `rows.jsonl`（榜单版本目录，`awr.bench.boardstore`），并重算
`board.json`（`bench/board.v1`）与 `details.json`（BN-84 条目详情）；把 `<out>` 复制为 `<data_root>/boards/<board_version>` 即发布。已存在
`evaluation.json` 的运行跳过（断点续跑）；`--physics kinematic` 用于开发自检（评测标注 `physics_not_ranked`，配置标注
`track.dev`：开发榜，不得发布为当前版本）。

用法：
    python tools/bench/d3/proto_board.py --spec configs/bench.d3/crop-handoff-001.l1-3uav.run_spec.json \\
        --events-root data/d3/events --out runs/bench/proto-0 [--policies static_guard,track_share] [--seeds 1,2] [--jobs 3] \\
        [--physics kinematic] [--gz-launcher local] [--gz-export-root DIR] [--agents container --agent-image d3-agent-py:dev]
退出码 0 全部成功、1 有运行失败（失败项写入 `<out>/failures.json`，不进入榜单）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DISPLAY = {"static_guard": "Static guard", "coverage_scan": "Coverage scan", "track_share": "Track and share"}


def _one(job: dict) -> dict:
    sys.path[:0] = [str(ROOT / "python"), str(ROOT / "tools/sdk/anet4d/src")]
    from anet4d.baselines import BASELINES
    from anet4d.runner import run_episode

    from awr.agentsdk.env.core import BenchEnv, _open_world
    from awr.agentsdk.env.spec import EnvSpec
    from awr.agentsdk.env.tapecheck import is_formal
    from awr.bench.eval.evaluator import evaluate_run

    rs, pol, seed, out = job["spec"], job["policy"], job["seed"], Path(job["out"])
    rs = {**rs, "policy": {"kind": "reference", "name": pol}}
    if job.get("duration_s"):
        rs["duration_s"] = float(job["duration_s"])
    run_dir = out / "runs" / f"{pol}-s{seed}"
    ev_path = run_dir / "evaluation.json"
    if ev_path.exists():
        return {"policy": pol, "seed": seed, "run_dir": str(run_dir), "skipped": True}
    gz = {k: v for k, v in job["gz"].items() if v is not None}
    if gz:
        gz.setdefault("name", f"d3-gz-{job['tag']}-{pol}-s{seed}")
    if job["agents"] == "container":                     # 隔离容器 + 线协议（正式运行的执行方式）
        from awr.bench.runner import run_job

        r = run_job(rs, policy=rs["policy"], seed=seed, run_id=f"{job['tag']}-{pol}-s{seed}", run_dir=run_dir, worlds_root=Path(job["worlds_root"]),
                    events_root=Path(job["events_root"]), physics=job["physics"], gz=gz, agent_image=job["agent_image"])
        return {"policy": pol, "seed": seed, "run_dir": str(run_dir), "skipped": False, "run_wall_s": r["run_wall_s"], "eval_wall_s": r["eval_wall_s"],
                "pose_sha256": r["pose_sha256"], "components": r["components"]}
    spec = EnvSpec.from_run_spec(rs, run_id=f"{job['tag']}-{pol}-s{seed}", seed=seed, worlds_root=Path(job["worlds_root"]),
                                 events_root=Path(job["events_root"]), out_dir=run_dir, physics=job["physics"], gz=gz)
    t0 = time.time()
    m = run_episode(BenchEnv(spec), lambda _u: BASELINES[pol]())
    t1 = time.time()
    doc = evaluate_run(run_dir, spec.event_dir, _open_world(spec.world_dir, formal=is_formal(spec)))   # 正式运行不退回 DSM（K07）
    ev_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"policy": pol, "seed": seed, "run_dir": str(run_dir), "skipped": False, "run_wall_s": round(t1 - t0, 1),
            "eval_wall_s": round(time.time() - t1, 1), "pose_sha256": m.get("pose_sha256"), "components": doc["components"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="proto_board.py")
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--events-root", type=Path, required=True, help="directory holding <event_id>/v<N>/")
    ap.add_argument("--worlds-root", type=Path, default=ROOT / "worlds")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policies", default="static_guard,coverage_scan,track_share")
    ap.add_argument("--seeds", help="comma-separated; default: run_spec seeds")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--physics", choices=("realistic", "gz", "kinematic"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--agents", choices=("inproc", "container"), default="inproc",
                    help="container: each UAV agent in an isolated container over drone4d.bench.wire.v1 (formal runs)")
    ap.add_argument("--agent-image", default="d3-agent-py:dev")
    ap.add_argument("--gz-launcher", choices=("docker", "local"))
    ap.add_argument("--gz-image")
    ap.add_argument("--gz-export-root")
    ap.add_argument("--duration-s", type=float, help="override the run duration (smoke runs; not for published boards)")
    a = ap.parse_args(argv)
    rs = json.loads(a.spec.read_text(encoding="utf-8"))
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(rs["seeds"])
    physics = a.physics or rs.get("physics", "realistic")
    a.out.mkdir(parents=True, exist_ok=True)
    tag = rs["event"]["id"]
    gz = {"launcher": a.gz_launcher, "image": a.gz_image, "export_root": a.gz_export_root} if physics == "gz" else {}      # the gz instance options are only used by the gz tier (ADR-226)
    jobs = [{"spec": rs, "policy": p, "seed": s, "out": str(a.out), "tag": tag, "physics": physics, "gz": gz, "worlds_root": str(a.worlds_root),
             "events_root": str(a.events_root), "agents": a.agents, "agent_image": a.agent_image, "duration_s": a.duration_s} for p in a.policies.split(",") for s in seeds]
    done, failures = [], []
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = {ex.submit(_one, j): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                r = f.result()
            except Exception as e:                       # 单个运行失败不中断其余运行
                failures.append({"policy": j["policy"], "seed": j["seed"], "error": repr(e), "trace": traceback.format_exc()[-4000:]})
                print(f"FAIL {j['policy']} s{j['seed']}: {e!r}", flush=True)
                continue
            done.append(r)
            c = r.get("components")
            print(f"{r['policy']} s{r['seed']}: " + ("skipped (evaluation.json exists)" if r["skipped"] else
                  f"EC {c['ec']:.3f} EU {c['eu']:.3f} AO {c['ao']:.3f} CE {c['ce']}  run {r['run_wall_s']} s  eval {r['eval_wall_s']} s"), flush=True)
    if failures:
        (a.out / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    sys.path.insert(0, str(ROOT / "python"))
    from awr.bench.boardstore import append_rows, publish, slim_evaluation
    from awr.bench.eval.thresholds import SCORE_VERSION

    plan = json.loads((a.events_root / tag / f"v{rs['event'].get('version', 1)}" / "plan.json").read_text(encoding="utf-8"))
    index = [{"entry_id": f"baseline/{r['policy']}", "team": "D3 reference", "display_name": DISPLAY.get(r["policy"], r["policy"]), "baseline": True,
              "track": f"std{len(rs['fleet'])}-{rs['obs_level'].lower()}", "lane": plan["event_type"], "event_type": plan["event_type"], "event": tag,
              "seed": r["seed"],
              "evaluation": str(Path(r["run_dir"]) / "evaluation.json")} for r in sorted(done, key=lambda r: (r["policy"], r["seed"]))]
    (a.out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    vehicles = sorted({f.get("vehicle_ref", "x500") for f in rs["fleet"]})
    track = {"fleet_ref": f"{len(rs['fleet'])}x-{'+'.join(vehicles)}", "obs_level": rs["obs_level"], "dev": physics != "realistic"}
    gz_cfg = {k: v for k, v in gz.items() if v is not None}
    cfg = {"run_spec": {**rs, "seeds": seeds}, "seeds": seeds, "events_root": str(a.events_root.resolve()), "worlds_root": str(a.worlds_root.resolve()),
           "physics": physics,
           "gz": gz_cfg, "agent_image": a.agent_image, "score_version": SCORE_VERSION, "track": track, "track_id": index[0]["track"] if index else None,
           "lane": plan["event_type"], "event_type": plan["event_type"], "event": tag}
    (a.out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (a.out / "rows.jsonl").unlink(missing_ok=True)
    append_rows(a.out, [{**{k: v for k, v in r.items() if k != "evaluation"}, "evaluation": slim_evaluation(json.loads(Path(r["evaluation"]).read_text(encoding="utf-8")))}
                        for r in index])
    board = publish(a.out, n_boot=a.n_boot)
    print(f"{len(board['entries'])} entries, {len(index)} runs, {len(failures)} failures -> {a.out / 'board.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
