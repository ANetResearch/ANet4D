"""Assemble a simulation into a D3 event directory: ``tape.bin`` + ``tape.json`` (drone4d.tape.v2, all actors
scripted), ``plan.json`` (drone4d.plan.v1 subset), ``anchors.json`` (awr.anchors.v1), ``facts.jsonl`` and
``civil.json`` (provenance). Requires the upstream ``awr.sim.events.tape`` module on the path.

    python -m awr.life.civil.assemble --sim <sim dir> --semantics … --plans … --population … --world-ref <tape.json of a
        compiled event for the same world> --event-id crop-city-002 --out <event dir> [--ground grid|mesh --nav …]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

TICK_MS = 100
_ENV_KEYS = ("wind_mps", "wind_dir_deg", "gust_sigma_mps", "rain_mm_h", "visibility_m", "fog", "cloud")
DEFAULT_ENV = {
    "interpolation": "smooth",
    "keyframes": [
        {
            "t_ms": 0,
            "wind_mps": 2.0,
            "wind_dir_deg": 250.0,
            "gust_sigma_mps": 0.5,
            "rain_mm_h": 0.0,
            "visibility_m": 12000.0,
            "fog": 0.0,
            "cloud": 0.4,
        }
    ],
}


# The tape v2 record (u32 container, pitch and roll) lives in awr.sim.events.tape since 2026-11-04. The message below is
# only raised when an older ANet4D without that record is on the path.
SCENARIO_TAPE_PENDING = (
    "Scenario tape generation needs the tape v2 record (RECORD_V2, NO_CONTAINER_V2) of awr.sim.events.tape and "
    "packages/contracts/event/tape.v2.schema.json; this ANet4D checkout does not provide them (docs/life/README.md, section 2)."
)


def _patch(K: dict) -> dict:
    return {
        "wind": {
            "speed_ref_mps": K["wind_mps"],
            "dir_from_deg": K["wind_dir_deg"],
            "turb_sigma_u_ref_mps": K["gust_sigma_mps"],
        },
        "precip": {"rain_mmh": K["rain_mm_h"]},
        "atmosphere": {"mor_bg_m": K["visibility_m"], "fog_top_agl_m": K["fog"]},
        "cloud": {"cover": K["cloud"]},
    }


def compile_env_track(env: dict) -> list[dict]:
    """Same semantics as the upstream compiler (awr.sim.events.gen.compile_env_track)."""
    kf = env["keyframes"]
    ops = [{"t_ms": 0, "op": {"kind": "set", "patch": _patch(kf[0]), "mode": "step", "duration_s": 0}}]
    mode = env.get("interpolation", "smooth")
    for a, b in itertools.pairwise(kf):
        if mode == "step":
            ops.append({"t_ms": b["t_ms"], "op": {"kind": "set", "patch": _patch(b), "mode": "step", "duration_s": 0}})
            continue
        sub = 600_000 if mode == "smooth" else 1000
        t = a["t_ms"]
        while t < b["t_ms"]:
            t1 = min(t + sub, b["t_ms"])
            u = (t1 - a["t_ms"]) / (b["t_ms"] - a["t_ms"])
            K = {k: a[k] + (b[k] - a[k]) * u for k in _ENV_KEYS}
            dd = ((b["wind_dir_deg"] - a["wind_dir_deg"] + 180.0) % 360.0) - 180.0
            K["wind_dir_deg"] = (a["wind_dir_deg"] + dd * u) % 360.0
            ops.append(
                {
                    "t_ms": t,
                    "op": {"kind": "set", "patch": _patch(K), "mode": "smooth", "duration_s": (t1 - t) / 1000.0},
                }
            )
            t = t1
    return ops


def recorded_pose(x, y, yaw, vx, vy):
    """A body's pose and speed as the tape records them, for checks that must hold on the recording: the life
    runtime keeps float32 tracks (degrees for yaw), which _fix turns into mm, centidegrees and cm/s. Returns
    (x m, y m, yaw rad, speed m/s), the values the tape QA (awr.life.civil.validate_tape) reads back."""

    def q(v, k):
        return math.floor(float(np.float32(v)) * k + 0.5) / k

    return q(x, 1000.0), q(y, 1000.0), math.radians(q(math.degrees(yaw), 100.0)), math.hypot(q(vx, 100.0), q(vy, 100.0))


def _fix(v, k, dtype):
    info = np.iinfo(dtype)
    scaled = np.floor(np.asarray(v, np.float64) * k + 0.5)
    if not np.isfinite(scaled).all() or np.any((scaled < info.min) | (scaled > info.max)):
        raise ValueError(f"Value cannot be represented as {dtype}; refusing silent clipping")
    return scaled.astype(dtype)


def ground_z(
    xy: np.ndarray, mode: str, nav_dir: Path | None, world_dir: Path | None, profile: str = "person"
) -> np.ndarray:
    """Sample only a validated support surface; unknown terrain is an error."""
    from awr.life.civil.surface import SurfaceError, SurfaceMap

    if nav_dir is None:
        raise SurfaceError("A validated support map is required")
    hint = SurfaceMap(nav_dir).sample(xy, profile)
    if mode == "mesh":
        if world_dir is None:
            raise SurfaceError("mesh ground requires a world")
        from awr.sim.events.ground import build_ground_index
        from awr.world.mesh.query import open_mesh_world_query

        q = open_mesh_world_query(world_dir, world_dir.parent.parent / ".cache")
        lo, hi = xy.min(0), xy.max(0)
        gi = build_ground_index(q, (*lo, *hi))
        z, found = gi.ground_z_batch(xy, hint)
        if not found.all():
            raise SurfaceError(f"Missing mesh support at {int((~found).sum())} positions")
        return z
    return hint


def build(
    sim_dir: Path,
    sem: dict,
    plans: dict,
    pop: dict,
    world: dict,
    event_id: str,
    version: int,
    out: Path,
    *,
    ground: str,
    nav_dir: Path | None,
    world_dir: Path | None,
    env: dict | None,
    time_origin: str,
    brief: dict | None,
) -> dict:
    try:
        from awr.sim.events.tape import (
            NO_CONTAINER_V2 as NO_CONTAINER,
        )
        from awr.sim.events.tape import (
            RECORD_V2 as RECORD,
        )
        from awr.sim.events.tape import (
            manifest,
            record_spec,
            write_tape,
        )
    except ImportError as e:
        raise RuntimeError(SCENARIO_TAPE_PENDING) from e
    from awr.sim.events.jcs import jcs

    d = np.load(sim_dir / "tracks.npz")
    actors = json.loads((sim_dir / "actors.json").read_text(encoding="utf-8"))
    xy, v, yaw, state, cont = d["xy"], d["v"], d["yaw"], d["state"], d["container"]
    n_t, n = state.shape
    vis = (state == 1) | (state == 2)
    keep = np.flatnonzero(np.isin(state, [1, 2, 3, 4]).any(0))  # actors that are ever observable
    remap = {int(o): i for i, o in enumerate(keep)}
    table = []
    for i, o in enumerate(keep):
        a = actors[int(o)]
        table.append(
            {
                "idx": i,
                "id": a["id"],
                "class": a["class"],
                "scripted": True,
                "profile": a["profile"],
                "size_mm": a["size_mm"],
                "appearance": a["appearance"],
            }
        )
    xy, v, yaw, state, cont = xy[:, keep], v[:, keep], yaw[:, keep], state[:, keep], cont[:, keep]
    cont_new = np.full(cont.shape, NO_CONTAINER, np.int64)
    m = cont >= 0
    if m.any():
        lut = np.full(n, NO_CONTAINER, np.int64)
        for o, i in remap.items():
            lut[o] = i
        cont_new[m] = lut[cont[m]]
    # positions of absent/gone/inside actors: hold the last visible position (renderer ignores them)
    fr = np.zeros((n_t, len(keep)), RECORD)
    z = np.zeros(state.shape, np.float64)
    visible = np.isin(state, [1, 2])
    for profile in ("person", "vehicle"):
        columns = np.array([a["class"] == "vehicle" for a in table])
        if profile == "person":
            columns = ~columns
        selected = visible & columns[None, :]
        if selected.any():
            z[selected] = ground_z(xy[selected].astype(np.float64), ground, nav_dir, world_dir, profile)
    for k in range(n_t):
        hidden = ~visible[k]
        if k:
            z[k, hidden] = z[k - 1, hidden]
        riding = state[k] == 3
        if riding.any():
            parents = cont_new[k, riding]
            if (parents == NO_CONTAINER).any():
                raise ValueError("A retained rider references an omitted/unknown vehicle")
            z[k, riding] = z[k, parents]
    fr["x_mm"] = _fix(xy[..., 0], 1000.0, np.int32)
    fr["y_mm"] = _fix(xy[..., 1], 1000.0, np.int32)
    fr["z_mm"] = _fix(z, 1000.0, np.int32)
    fr["yaw_cdeg"] = ((_fix(yaw, 100.0, np.int64) + 18000) % 36000 - 18000).astype(np.int16)
    fr["vx_cmps"] = _fix(v[..., 0], 100.0, np.int16)
    fr["vy_cmps"] = _fix(v[..., 1], 100.0, np.int16)
    vz = np.zeros_like(z)
    if n_t > 1:
        vz[1:] = np.where(visible[1:] & visible[:-1], np.diff(z, axis=0) / (TICK_MS / 1000.0), 0.0)
        vz[0] = vz[1]
    fr["vz_cmps"] = _fix(vz, 100.0, np.int16)
    # Fit vehicle support orientation at wheel locations. Humans stay upright.
    from awr.life.civil.surface import SurfaceMap

    surface = SurfaceMap(nav_dir)
    for j, a in enumerate(table):
        if a["class"] != "vehicle":
            continue
        ticks = np.flatnonzero(visible[:, j])
        if not len(ticks):
            continue
        ang = np.radians(yaw[ticks, j])
        forward = np.column_stack([np.cos(ang), np.sin(ang)])
        left = np.column_stack([-np.sin(ang), np.cos(ang)])
        half_l = a["size_mm"][0] * 0.00032
        half_w = a["size_mm"][1] * 0.00037
        center = xy[ticks, j]
        front = surface.sample(center + forward * half_l, "vehicle")
        back = surface.sample(center - forward * half_l, "vehicle")
        zl = surface.sample(center + left * half_w, "vehicle")
        zr = surface.sample(center - left * half_w, "vehicle")
        fr["pitch_cdeg"][ticks, j] = _fix(-np.degrees(np.arctan2(front - back, 2 * half_l)), 100.0, np.int16)
        fr["roll_cdeg"][ticks, j] = _fix(np.degrees(np.arctan2(zl - zr, 2 * half_w)), 100.0, np.int16)
    fr["state"] = state
    fr["container"] = cont_new.astype(np.uint32)
    env = env or DEFAULT_ENV
    plan = {
        "schema": "drone4d.plan.v1",
        "event_id": event_id,
        "version": version,
        "event_type": "civil",
        "generator": "awr.life.civil",
        "world": {"id": world["id"], "content_version": world["content_version"]},
        "anchors_ref": "event:anchors.json",
        "seed": plans["seed"],
        "duration_ms": n_t * TICK_MS,
        "tick_hz": 10,
        "mode": "exogenous",
        "time_origin_utc": time_origin,
        "actors": [
            {
                "id": t["id"],
                "class": t["class"],
                "profile": t["profile"],
                "size_mm": t["size_mm"],
                "appearance": t["appearance"],
                "role": actors[int(keep[t["idx"]])].get("role"),
                "description": actors[int(keep[t["idx"]])].get("description"),
                "route": [],
            }
            for t in table
        ],
        "background": [],
        "environment": env,
        "comms": {"zones": []},
        "annotations": [],
        "relations": [],
        "brief": brief or {},
        "interactive": {},
        "labels": {},
    }
    anchors_doc = (
        json.loads((sim_dir.parent / "anchors.overlay.json").read_text(encoding="utf-8"))
        if (sim_dir.parent / "anchors.overlay.json").exists()
        else {"anchors": []}
    )
    if not anchors_doc["anchors"]:
        x0, y0, x1, y1 = sem["stage"]
        anchors_doc["anchors"] = [
            {
                "id": "ev_town_center",
                "kind": "zone",
                "category": "manual",
                "profile": "person",
                "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                "label_zh": "场景边界",
            }
        ]
    anchors = sorted(anchors_doc["anchors"], key=lambda a: a["id"])
    a_sha = hashlib.sha256(jcs(anchors)).hexdigest()
    world_hdr = {**world, "anchors_sha256": a_sha}
    event_sha = hashlib.sha256(jcs({"plan": plan, "anchors": a_sha})).hexdigest()
    header = {
        "schema": "drone4d.tape.v2",
        "event_id": event_id,
        "event_version": version,
        "event_sha256": event_sha,
        "mode": "exogenous",
        "seed": plans["seed"],
        "tick_hz": 10,
        "n_ticks": n_t,
        "duration_ms": n_t * TICK_MS,
        "record": record_spec(2),
        "codec": {"name": "col-delta-zstd@1", "block_ticks": 100},
        "actors": table,
        "world": world_hdr,
        "environment": env,
        "env_track": compile_env_track(env),
        "comms": {"zones": []},
        "anim": None,
        "realized": None,
    }
    out.mkdir(parents=True, exist_ok=True)
    h = write_tape(out / "tape.bin", header, fr, n_scripted=len(table))
    prov = {"generator": "awr.life.civil", "sim_dir": str(sim_dir), "ground": ground}
    (out / "tape.json").write_text(
        json.dumps(manifest(h, header, provenance=prov), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "anchors.json").write_text(
        json.dumps(
            {"schema": "awr.anchors.v1", "world_id": world["id"], "anchors": anchors}, ensure_ascii=False, indent=1
        ),
        encoding="utf-8",
    )
    if anchors_doc.get("definition") is not None:
        (out / "public-region-definition.json").write_text(
            json.dumps(anchors_doc["definition"], ensure_ascii=False, indent=2) + "\n"
        )
    (out / "event.json").write_text(
        json.dumps(
            {
                "schema": "awr.life.civil.event.v1",
                "event_id": event_id,
                "version": version,
                "seed": plans["seed"],
                "warmup_s": plans["warmup_s"],
                "duration_s": n_t * TICK_MS / 1000.0,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    from awr.life.civil.facts import write_facts

    nf = write_facts(sim_dir, sem, plans, pop, actors, keep, out / "facts.jsonl", n_t * TICK_MS / 1000.0)
    return {
        "actors": len(table),
        "dropped": int(n - len(keep)),
        "n_ticks": n_t,
        "bytes": h["bytes"],
        "content_sha256": h["content_sha256"],
        "facts": nf,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, type=Path)
    ap.add_argument("--semantics", required=True, type=Path)
    ap.add_argument("--plans", required=True, type=Path)
    ap.add_argument("--population", required=True, type=Path)
    ap.add_argument(
        "--world-ref",
        required=True,
        type=Path,
        help="tape.json of an event compiled for the same world (world header fields)",
    )
    ap.add_argument("--event-id", required=True)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ground", choices=["grid", "mesh"], default="grid")
    ap.add_argument("--nav", type=Path, default=None)
    ap.add_argument("--world", type=Path, default=None)
    ap.add_argument("--env", type=Path, default=None, help="JSON with environment keyframes")
    ap.add_argument("--time-origin", default="2026-06-15T08:30:00Z")
    a = ap.parse_args()
    sem = json.loads(a.semantics.read_text(encoding="utf-8"))
    plans = json.loads(a.plans.read_text(encoding="utf-8"))
    pop = json.loads(a.population.read_text(encoding="utf-8"))
    world = json.loads(a.world_ref.read_text(encoding="utf-8"))["header"]["world"]
    env = json.loads(a.env.read_text(encoding="utf-8")) if a.env else None
    r = build(
        a.sim,
        sem,
        plans,
        pop,
        world,
        a.event_id,
        a.version,
        a.out,
        ground=a.ground,
        nav_dir=a.nav,
        world_dir=a.world,
        env=env,
        time_origin=a.time_origin,
        brief=None,
    )
    print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
