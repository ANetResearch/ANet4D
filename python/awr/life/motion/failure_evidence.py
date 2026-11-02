"""Failed-step evidence and exact crowd-only replay, never a resume checkpoint.

Accepted recording prefixes exclude the rejected tick. The attempted Director
events may include side effects from that tick, explicitly identified as such.
No safety rule or movement decision is changed by saving this evidence.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from awr.life.common.schema import schema_id

from .crowd import Crowd

ARRAYS = ("ext", "kinds", "radius", "z_ok", "walk", "sdist", "free", "offsets", "speed_fracs", "times")
CONFIG = ("sx0", "sy0", "scell", "fx0", "fy0", "fstep", "grid", "horizon", "margin", "margin_growth", "weights")


def record_activity(out, engine):
    """Retain causal decisions on failure, without re-running a route or decision."""
    out = Path(out)
    directors = []
    for agenda in engine.agendas.values():
        director = agenda.director
        if all(director is not d for d in directors):
            directors.append(director)
    path = out.parent / (out.name + ".partial-activity-decisions.jsonl")
    with path.open("w") as stream:
        for index, director in enumerate(directors):
            for event in director.events:
                stream.write(json.dumps(dict(event, director_index=index)) + "\n")
    return path.name


def source_hashes():
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("crowd.py", "kernels.py", "edge_recovery.py", "engine.py", "failure_evidence.py")
    }


def save_crowd(path, crowd, inputs, history, proposed, driven=None):
    data = {"crowd_" + k: getattr(crowd, k) for k in ARRAYS}
    data.update({"crowd_" + k: v for k, v in history.items()})
    data.update({"input_" + k: v for k, v in inputs.items()})
    data.update({"proposed_" + k: v for k, v in proposed.items()})
    if driven is not None:
        data.update({"driven_" + str(i): v for i, v in enumerate(driven)})
    data["metadata_json"] = np.array(
        json.dumps(
            dict(
                schema="awr.life.crowd_rejected_step.v1",
                config={k: getattr(crowd, k) for k in CONFIG},
                stats=crowd.stats,
                input_names=list(inputs),
                driven=driven is not None,
                source_sha256=source_hashes(),
                diagnostic_only=True,
            )
        )
    )
    np.savez_compressed(path, **data)


def replay_crowd(path):
    """Restore every body/mask and driven prediction; no frozen-neighbour approximation."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata_json"]))
        if schema_id(meta["schema"]) != "awr.life.crowd_rejected_step.v1":
            raise ValueError("Unknown failed-step evidence schema")
        crowd = Crowd.__new__(Crowd)
        for k in (*ARRAYS, "waited", "aim"):
            setattr(crowd, k, z["crowd_" + k].copy())
        # Walking time since the last stand (older snapshots lack it; it changes only the next step's wait) and the
        # edging directions, which every step overwrites.
        crowd.walked = z["crowd_walked"].copy() if "crowd_walked" in z else np.zeros(len(crowd.ext))
        crowd.motion = np.full(len(crowd.ext), np.nan)
        for k, v in meta["config"].items():
            setattr(crowd, k, tuple(v) if k in ("grid", "weights") else v)
        crowd.stats, crowd.area = meta["stats"].copy(), None
        inputs = {k: z["input_" + k].copy() for k in meta["input_names"]}
        driven = tuple(z["driven_" + str(i)].copy() for i in range(7)) if meta["driven"] else None
        expected = tuple(z["proposed_" + k].copy() for k in ("xy", "yaw", "speed", "held"))
    actual = crowd.step(**inputs, driven=driven)
    return dict(
        exact=np.array([np.array_equal(a, b) for a, b in zip(actual, expected, strict=False)]).all().item(),
        implementation_matches_snapshot=source_hashes() == meta["source_sha256"],
        expected=expected,
        actual=actual,
        crowd=crowd,
        inputs=inputs,
    )


def record(out, engine, error, tick, time_s, ctx):
    out = Path(out)
    count = max(tick, 0)  # failed tick was never committed to recording arrays
    np.savez_compressed(
        out.parent / (out.name + ".partial.npz"),
        xy=ctx["tracks"][:count].astype(np.float32),
        v=ctx["vel"][:count].astype(np.float32),
        yaw=ctx["yaw_track"][:count].astype(np.float32),
        state=ctx["state"][:count],
        container=ctx["container_track"][:count],
        ids=np.array(engine.ids),
    )
    (out.parent / (out.name + ".partial-events.jsonl")).write_text("".join(json.dumps(e) + "\n" for e in engine.events))
    activity_events = record_activity(out, engine)
    from .traffic_evidence import record_traffic

    traffic_evidence = record_traffic(out, engine)
    replay = None
    if hasattr(error, "motion_inputs"):
        replay = out.name + ".rejected-crowd.npz"
        save_crowd(
            out.parent / replay,
            engine.crowd,
            error.motion_inputs,
            error.motion_history,
            error.motion_proposed,
            error.motion_driven,
        )
    rows = []
    for i, actor in enumerate(engine.ids):
        agenda = engine.agendas.get(i)
        intent = agenda.director.active.get(actor) if agenda is not None else None
        row = dict(
            actor=actor,
            index=i,
            kind=str(engine.kinds[i]),
            xy=engine.xy[i].tolist(),
            yaw_rad=float(engine.yaw[i]),
            speed_mps=float(engine.speed[i]),
            active=bool(engine.active[i]),
            container=int(engine.container[i]),
            phase=None if agenda is None else agenda.phase,
            target_xy=None,
            current_intent=None,
        )
        try:
            if agenda is not None:
                row["target_xy"] = list(engine.target(i))
            if intent is not None:
                row["current_intent"] = dict(
                    place=intent.place, purpose=intent.purpose, step_kind=intent.step.kind, step_xy=list(intent.step.xy)
                )
        except (AttributeError, IndexError, TypeError) as detail_error:
            row["agenda_detail_error"] = repr(detail_error)
        rows.append(row)
    diagnostic = dict(
        schema="awr.life.runtime_failure.v1",
        tick=tick,
        time_s=time_s,
        error=f"{type(error).__name__}: {error}",
        accepted_recorded_ticks=count,
        rejected_tick_recorded=False,
        actors=rows,
        crowd=engine.crowd.stats,
        motion_replay=replay,
        activity_events=activity_events,
        traffic_evidence=traffic_evidence,
        admitted=False,
        note="Engine transaction may have advanced agendas/transfers before rejection. Poses are failure-time inputs, not a committed frame. Events include attempted tick; this is not a resume checkpoint.",
    )
    (out.parent / (out.name + ".runtime-failure.json")).write_text(
        json.dumps(diagnostic, indent=2, default=float) + "\n"
    )
    return diagnostic
