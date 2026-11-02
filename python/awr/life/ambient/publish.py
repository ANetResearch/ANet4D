"""Assemble, check and publish an exported ambient scene (awr.life.ambient.export) as a D3 civil event.

    python -m awr.life.ambient.publish WORK --out events/NAME/vN --world W --nav SUPPORT --semantics SEM --config CFG

The tape is assembled on the measured ground (awr.life.civil.assemble), checked by the final tape QA
(awr.life.civil.validate_tape) and the life audit with the gates of CFG (`life_gates`); only an event that passes both
is published. The sidecars of task binding (scenario overlay, activities, parking events, entities) are copied in.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

SIDECARS = (
    "scenario-overlay.json",
    "activity-decisions.jsonl",
    "activity-places.json",
    "activity-events.jsonl",
    "agendas.json",
)


def publish(work, out, *, world, nav, semantics, config, seed=7, warmup_s=120.0):
    from awr.life.civil.assemble import build as assemble
    from awr.life.civil.life_audit import audit as life_audit
    from awr.life.civil.life_audit import markdown as life_markdown
    from awr.life.civil.validate_tape import validate
    from awr.life.civil.world_reference import navigation_hashes
    from awr.life.envpack.execution import finish_output
    from awr.sim.events.compile import raycast_sha256

    work, out, world, nav = Path(work), Path(out), Path(world), Path(nav)
    cfg = json.loads(Path(config).read_text())
    sem = json.loads(Path(semantics).read_text())
    wdoc = json.loads((world / "world.json").read_text())
    ref = {
        "id": wdoc["id"],
        "content_version": wdoc["contentVersion"],
        "raycast_sha256": raycast_sha256(world),
        "navmesh_sha256": navigation_hashes(world, require_vehicle=True),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=out.name + ".pending-", dir=out.parent))
    assemble(
        work / "sim",
        sem,
        {"seed": seed, "warmup_s": warmup_s},
        {"persons": [], "vehicles": []},
        ref,
        out.parent.name,
        int(out.name.removeprefix("v")),
        pending,
        ground="grid",
        nav_dir=nav,
        world_dir=world,
        env=None,
        time_origin="2026-06-15T05:30:00Z",
        brief={
            "scope": "Ambient town centre: SUMO traffic and JuPedSim pedestrians on the measured ground, "
            "parked cars of the queries; outdoor activities only."
        },
    )
    result = validate(pending, nav)
    (pending / "final_qa.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["pass"]:
        raise ValueError("Final QA failed; pending event not published")
    for name in SIDECARS:
        shutil.copy(work / name, pending / name)
    actors = json.loads((work / "sim" / "actors.json").read_text())
    import numpy as np

    xy = np.load(work / "sim" / "tracks.npz")["xy"][0]
    (pending / "entities.json").write_text(
        json.dumps(
            {
                "schema": "awr.life.entities.v1",
                "entities": [{**a, "spawn_xy_m": [float(v) for v in xy[a["idx"]]], "road": None} for a in actors],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    shutil.copy(Path(config), pending / "testbed-config.json")
    for name in ("scene-report.json", "stage-network.json"):
        if (work / name).exists():
            shutil.copy(work / name, pending / name)
    report = life_audit(pending, cfg)
    report["event"] = str(out)
    (pending / "life-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float) + "\n")
    (pending / "life-audit.md").write_text(life_markdown(report))
    gates = report["gates"]["gates"]
    print(
        json.dumps(
            {
                "life_gates": {
                    k: dict(value=g["value"], threshold=g["threshold"], pass_=g["pass_"]) for k, g in gates.items()
                }
            },
            default=float,
        ),
        flush=True,
    )
    failed = [k for k, g in gates.items() if g["pass_"] is not True]
    if failed:
        rejected = out.parent / (out.name + ".rejected")
        pending.rename(rejected)
        raise ValueError(f"Life gates failed ({', '.join(failed)}); kept at {rejected}")
    target = finish_output(pending, out, dynamic_validation=False)
    print("published", target, flush=True)
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--nav", type=Path, required=True)
    ap.add_argument("--semantics", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--warmup", type=float, default=120.0)
    a = ap.parse_args()
    publish(
        a.work, a.out, world=a.world, nav=a.nav, semantics=a.semantics, config=a.config, seed=a.seed, warmup_s=a.warmup
    )


if __name__ == "__main__":
    main()
