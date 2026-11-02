"""Independent checks on the published representation, after Z and IDs exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from awr.life.civil.collision import collision_pairs
from awr.life.civil.surface import SurfaceError, SurfaceMap
from awr.sim.events.tape import open_tape

QA_CHECKS = (
    "nonfinite",
    "invalid_container",
    "unsupported_ground",
    "support_error",
    "excessive_vertical_step",
    "unexplained_position_step",
    "inconsistent_velocity",
    "body_collisions",
    "unsupported_footprint",
    "unsupported_vehicle_contacts",
    "illegal_visibility_transition",
    "excessive_acceleration",
    "excessive_turn_rate",
    "implausible_vehicle_tilt",
)


def validate(event: Path, nav: Path) -> dict:
    tape = open_tape(event)
    import jsonschema

    schema = Path(__file__).resolve().parents[4] / "packages/contracts/event/tape.v2.schema.json"
    if not schema.is_file():
        from awr.life.civil.assemble import TAPE_V2_MISSING

        raise RuntimeError(TAPE_V2_MISSING)
    jsonschema.Draft202012Validator(json.loads(schema.read_text())).validate(tape.header)
    tape.verify()
    surfaces = SurfaceMap(nav)
    cls = np.array([a["class"] for a in tape.actors])
    counts = dict.fromkeys(QA_CHECKS, 0)
    sizes = np.array([a["size_mm"] for a in tape.actors], float) * 0.001
    # People occupy the footprint of the pose both renderers draw at the recorded
    # speed of the tick (awr.life.civil.body); other actors their catalogue boxes.
    from awr.life.civil.body import actor_extents, boxes

    ext = actor_extents(tape.actors, sizes)
    examples = []
    prev = None
    dt = 1 / tape.header["tick_hz"]
    for bi in range(len(tape.blocks)):
        f = tape.block_frames(bi)
        pos = np.stack([f["x_mm"], f["y_mm"], f["z_mm"]], -1).astype(float) * 0.001
        visible = np.isin(f["state"], [1, 2])
        counts["nonfinite"] += int((~np.isfinite(pos)).sum())
        from awr.life.civil.contacts import vehicle_contacts

        for j in np.flatnonzero(cls == "vehicle"):
            ticks = np.flatnonzero(visible[:, j])
            if not len(ticks):
                continue
            offset = vehicle_contacts(tuple(sizes[j]))
            yaw = np.radians(f["yaw_cdeg"][ticks, j] * 0.01)
            c = np.cos(yaw)[:, None]
            s = np.sin(yaw)[:, None]
            points = pos[ticks, j, None, :2] + np.stack(
                [offset[:, 0] * c - offset[:, 1] * s, offset[:, 0] * s + offset[:, 1] * c], -1
            )
            ok = surfaces.supported(points, "vehicle").reshape(len(ticks), -1).all(1)
            counts["unsupported_vehicle_contacts"] += int((~ok).sum())
            if not ok.all() and len(examples) < 10:
                examples.append(dict(kind="unsupported_vehicle_contacts", actor=tape.actors[int(j)]["id"], block=bi))
        speed = np.hypot(f["vx_cmps"].astype(float), f["vy_cmps"].astype(float)) * 0.01
        for tick in range(len(f)):
            ids = np.flatnonzero(visible[tick])
            yaw = np.radians(f["yaw_cdeg"][tick, ids] * 0.01)
            centres, body = boxes(pos[tick, ids, :2], yaw, speed[tick, ids], ext[ids], sizes[ids, 2])
            pairs = collision_pairs(centres, yaw, body)
            counts["body_collisions"] += len(pairs)
            for i, j in pairs[: max(0, 10 - len(examples))]:
                examples.append(
                    {
                        "kind": "body_collision",
                        "actors": [tape.actors[int(ids[i])]["id"], tape.actors[int(ids[j])]["id"]],
                        "block": bi,
                        "tick": tick,
                    }
                )
            # Identical four-corner predicate, batched by profile. Long tapes
            # must not trade all-frame checks for sparse temporal sampling.
            offsets = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])[None] * body[:, None, :2] / 2
            c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
            corners = centres[:, None, :] + np.stack(
                [offsets[:, :, 0] * c - offsets[:, :, 1] * s, offsets[:, :, 0] * s + offsets[:, :, 1] * c], -1
            )
            for profile in ("person", "vehicle"):
                mask = (cls[ids] == "vehicle") if profile == "vehicle" else (cls[ids] != "vehicle")
                selected = ids[mask]
                ok = surfaces.supported(corners[mask], profile).reshape(-1, 4).all(1)
                counts["unsupported_footprint"] += int((~ok).sum())
                for j in selected[~ok][: max(0, 10 - len(examples))]:
                    examples.append(
                        {"kind": "unsupported_footprint", "actor": tape.actors[int(j)]["id"], "block": bi, "tick": tick}
                    )
        if "pitch_cdeg" in f.dtype.names:
            counts["implausible_vehicle_tilt"] += int(
                (
                    visible
                    & (cls == "vehicle")[None, :]
                    & ((np.abs(f["pitch_cdeg"].astype(float)) > 2000) | (np.abs(f["roll_cdeg"].astype(float)) > 1200))
                ).sum()
            )
        cont = f["container"].astype(np.int64)
        riding = f["state"] == 3
        valid_cont = (cont >= 0) & (cont < len(cls))
        bad = riding & ~valid_cont
        idx = riding & valid_cont
        bad[idx] |= cls[cont[idx]] != "vehicle"
        counts["invalid_container"] += int(bad.sum())
        for profile, cols in (("vehicle", cls == "vehicle"), ("person", cls != "vehicle")):
            mask = visible & cols[None, :]
            pp = pos[mask]
            # Per-position exceptions retain exact failures instead of hiding
            # every good sample in a failed batch.
            if len(pp):
                try:
                    zz = surfaces.sample(pp[:, :2], profile)
                    counts["support_error"] += int((np.abs(zz - pp[:, 2]) > 0.05).sum())
                except SurfaceError:
                    for p in pp:
                        try:
                            z = surfaces.sample(p[None, :2], profile)[0]
                            counts["support_error"] += int(abs(z - p[2]) > 0.05)
                        except SurfaceError:
                            counts["unsupported_ground"] += 1
        frames = np.concatenate([prev[None], f]) if prev is not None else f
        if len(frames) > 1:
            xyz = np.stack([frames["x_mm"], frames["y_mm"], frames["z_mm"]], -1).astype(float) * 0.001
            delta = np.diff(xyz, axis=0)
            # Visibility may change only via a declared physical transition.
            # Mid-scene absent/gone toggles are not occlusion: geometry handles occlusion.
            before, after = frames["state"][:-1], frames["state"][1:]
            toggled = np.isin(before, [1, 2]) != np.isin(after, [1, 2])
            counts["illegal_visibility_transition"] += int(
                (toggled & (np.isin(before, [0, 5]) | np.isin(after, [0, 5]))).sum()
            )
            both = np.isin(frames["state"][1:], [1, 2]) & np.isin(frames["state"][:-1], [1, 2])
            speed_limit = np.where(cls == "vehicle", 25.0, np.where(cls == "person", 4.5, 8.0))
            jump = (np.linalg.norm(delta, axis=-1) > speed_limit[None, :] * dt + 0.02) & both
            vertical = (np.abs(delta[..., 2]) > np.where(cls == "vehicle", 0.35, 0.25)) & both
            counts["unexplained_position_step"] += int(jump.sum())
            counts["excessive_vertical_step"] += int(vertical.sum())
            vel = np.stack([frames["vx_cmps"][1:], frames["vy_cmps"][1:]], -1).astype(float) * 0.01
            velocity = np.stack([frames["vx_cmps"], frames["vy_cmps"]], -1).astype(float) * 0.01
            acceleration = np.linalg.norm(np.diff(velocity, axis=0), axis=-1) / dt
            counts["excessive_acceleration"] += int(
                ((acceleration > np.where(cls == "vehicle", 6.0, 8.0)) & both).sum()
            )
            turn = ((np.diff(frames["yaw_cdeg"].astype(float), axis=0) + 18000) % 36000 - 18000) * np.pi / 18000 / dt
            counts["excessive_turn_rate"] += int(((np.abs(turn) > np.where(cls == "vehicle", 1.0, 3.0)) & both).sum())
            mismatch = (np.linalg.norm(delta[..., :2] / dt - vel, axis=-1) > 0.75) & both
            counts["inconsistent_velocity"] += int(mismatch.sum())
            for k, j in np.argwhere(jump | vertical)[: max(0, 10 - len(examples))]:
                examples.append(
                    {
                        "actor": tape.actors[int(j)]["id"],
                        "block": bi,
                        "local_interval": int(k),
                        "delta_m": delta[k, j].tolist(),
                    }
                )
        prev = f[-1]
    return {
        "schema": "awr.life.final_qa.v1",
        "pass": all(v == 0 for v in counts.values()),
        "counts": counts,
        "examples": examples,
        "ticks": tape.n_ticks,
        "tape_content_sha256": tape.hashes()["content_sha256"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", type=Path, required=True)
    ap.add_argument("--nav", type=Path, required=True)
    a = ap.parse_args()
    result = validate(a.event, a.nav)
    (a.event / "final_qa.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
