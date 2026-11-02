"""QA checks on a simulation (tracks.npz + actors.json): separation, lane adherence, speeds, jumps, walkability.

python -m awr.life.civil.qa --sim <dir> --semantics … --nav … [--out qa.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString


def run(sim_dir: Path, sem: dict, nav_dir: Path) -> dict:
    d = np.load(sim_dir / "tracks.npz")
    A = json.loads((sim_dir / "actors.json").read_text(encoding="utf-8"))
    xy, v, st = d["xy"], d["v"], d["state"]
    cls = np.array([a["class"] for a in A])
    role = np.array([a.get("role") or "" for a in A])
    vtype = np.array([a.get("vtype") or "" for a in A])
    m = json.loads((nav_dir / "meta.json").read_text())
    walk = np.load(nav_dir / "walk_person.npy")
    walk_v = np.load(nav_dir / "walk_vehicle.npy")
    ny, nx = walk.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    pts = shapely.points((xs + m["x0"] + 0.5).ravel(), (ys + m["y0"] + 0.5).ravel())
    lane = np.zeros((ny, nx), bool)
    for r in sem["roads"]:
        lane.ravel()[shapely.contains(LineString(r["line"]).buffer(r["width_m"] / 2.0 + 0.6), pts)] = True
    slot_pts = np.array(
        [s["xy"] for s in sem["curb_slots"]] + [s["xy"] for l in sem["parking_lots"] for s in l["slots"]]
    )
    slot_tree = cKDTree(slot_pts.reshape(-1, 2))
    out = {}
    n_t = len(st)
    x0, y0, x1, y1 = sem["stage"]
    inside = (xy[..., 0] >= x0) & (xy[..., 0] <= x1) & (xy[..., 1] >= y0) & (xy[..., 1] <= y1)
    vis = (st == 1) | (st == 2)
    out["outside_stage_visible"] = int((vis & ~inside).sum())
    mov = st == 1
    P, V, D_ = cls == "person", cls == "vehicle", cls == "dog"

    def cells(Q):
        ix = np.clip(np.floor(Q[..., 0] - m["x0"]).astype(int), 0, nx - 1)
        iy = np.clip(np.floor(Q[..., 1] - m["y0"]).astype(int), 0, ny - 1)
        return iy, ix

    # speeds
    sp = np.linalg.norm(v, axis=2)
    ped_sp = sp[:, P][mov[:, P]]
    if not len(ped_sp):
        ped_sp = np.zeros(1)
    jog = role == "jogger"
    out["person_speed"] = {
        "p50": float(np.percentile(ped_sp, 50)),
        "p99": float(np.percentile(ped_sp, 99)),
        "max": float(ped_sp.max()),
        "over_2.0_nonjogger": int((sp[:, P & ~jog][mov[:, P & ~jog]] > 2.0).sum()),
        "over_4.0": int((ped_sp > 4.0).sum()),
    }
    veh_sp = sp[:, V][mov[:, V]]
    if not len(veh_sp):
        veh_sp = np.zeros(1)
    out["vehicle_speed"] = {
        "p50": float(np.percentile(veh_sp, 50)),
        "p99": float(np.percentile(veh_sp, 99)),
        "max": float(veh_sp.max()),
    }
    # jumps
    dx = np.linalg.norm(np.diff(xy, axis=0), axis=2)
    both = vis[1:] & vis[:-1]
    lim = sp[:-1] * 0.1 * 1.6 + 0.08
    jumps = (dx > np.maximum(lim, 0.4)) & both
    out["jumps"] = {
        "person": int(jumps[:, P].sum()),
        "vehicle": int(jumps[:, V].sum()),
        "dog": int(jumps[:, D_].sum()),
        "samples": int(both.sum()),
    }
    # walkability and lane adherence
    iy, ix = cells(xy)
    on_walk = walk[iy, ix]
    out["person_off_walkable_frac"] = float((~on_walk[:, P] & vis[:, P]).sum() / max(1, vis[:, P].sum()))
    onlane = lane[iy, ix]
    vm = vis[:, V]
    bikes = vtype == "bicycle"
    offlane = ~onlane[:, V] & vm
    if offlane.any():
        Q = xy[:, V][offlane]
        dsl, _ = slot_tree.query(Q)
        near_slot = dsl < 3.5
        out["vehicle_off_lane_frac"] = float(offlane.sum() / max(1, vm.sum()))
        out["vehicle_off_lane_not_in_slot_frac"] = float((~near_slot).sum() / max(1, vm.sum()))
        bad = np.zeros(offlane.shape, bool)
        bad[offlane] = ~near_slot
        bad_moving = bad & mov[:, V] & ~bikes[V][None, :]
        out["vehicle_moving_off_lane_samples"] = int(bad_moving.sum())
        ids = np.array([a["id"] for a in A])[V]
        out["vehicle_moving_off_lane_top"] = [
            (str(ids[i]), int(c)) for i, c in sorted(enumerate(bad_moving.sum(0)), key=lambda t: -t[1])[:5] if c
        ]
    # separation (every 5th tick)
    pv_min, pv_close, pp_close, vv_close, pv_pairs = 1e9, 0, 0, 0, 0
    pb_close = 0
    motor = V & ~bikes
    for k in range(0, n_t, 5):
        pm = vis[k] & P
        vmk = mov[k] & motor
        if pm.any() and vmk.any():
            tv = cKDTree(xy[k][vmk])
            dd, _ = tv.query(xy[k][pm])
            pv_min = min(pv_min, float(dd.min()))
            pv_close += int((dd < 1.2).sum())
        bmk = mov[k] & V & bikes
        if pm.any() and bmk.any():
            dd, _ = cKDTree(xy[k][bmk]).query(xy[k][pm])
            pb_close += int((dd < 0.5).sum())
        if pm.sum() > 1:
            pp_close += len(cKDTree(xy[k][pm]).query_pairs(0.35))
        if vmk.sum() > 1:
            vv_close += len(cKDTree(xy[k][vmk]).query_pairs(1.5))
    out["separation"] = {
        "person_vehicle_min_m": pv_min if pv_min < 1e8 else None,
        "person_vehicle_below_1.2m": pv_close,
        "person_bicycle_below_0.5m": pb_close,
        "person_person_below_0.35m": pp_close,
        "moving_vehicle_pairs_below_1.5m_centres": vv_close,
    }
    out["counts"] = {
        "actors": len(A),
        "ever_visible": {
            c: int(vis[:, cls == c].any(0).sum()) for c in ("person", "vehicle", "dog", "object", "machine")
        },
    }
    out["pass"] = bool(
        out["outside_stage_visible"] == 0
        and out["jumps"]["person"] == 0
        and out["jumps"]["vehicle"] == 0
        and out["person_speed"]["over_4.0"] == 0
        and out.get("vehicle_moving_off_lane_samples", 0) < 0.002 * max(1, int(mov[:, V].sum()))
        and out["person_off_walkable_frac"] == 0
        and out["separation"]["person_bicycle_below_0.5m"] == 0
        and out["separation"]["person_vehicle_below_1.2m"] == 0
        and out["separation"]["person_person_below_0.35m"] == 0
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, type=Path)
    ap.add_argument("--semantics", required=True, type=Path)
    ap.add_argument("--nav", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    r = run(a.sim, json.loads(a.semantics.read_text(encoding="utf-8")), a.nav)
    s = json.dumps(r, ensure_ascii=False, indent=1)
    (a.out or a.sim / "qa.json").write_text(s, encoding="utf-8")
    print(s)
    return 0 if r["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
