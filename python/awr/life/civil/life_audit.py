"""Activity-level audit of a recorded civil tape (life behaviour, not geometry).

The geometric/motion QA (`validate_tape`) proves that every recorded pose is supported, collision free and
kinematically consistent. It says nothing about *why* actors stand still. This audit reconstructs, for every visible
tick of every actor, the activity state it is in from the published activity record (agenda events, vehicle park /
depart / board / alight events) and the tape itself, and reports:

- per-minute state composition and moving concurrency per class;
- per-actor moving time and distance, trip lengths;
- explained versus unexplained stationary time (pre-departure standing, finished scripts, no progress in a trip);
- long no-progress intervals with their recorded reasons;
- activity density per public search zone and spatial coverage;
- local crowding: nearest-neighbour spacing, neighbours within 3 m and the largest group of people linked by gaps of
  at most 1.5 m (stacking at one spot), and vehicles stopped inside a trip with the nearest body ahead;
- the lifecycle after an itinerary completes.

Gates are configured per scenario (`life_gates` in the testbed config) because thresholds depend on what the scenario is
for; this module never invents a single "realism" score.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TICK_S = 0.1
MOVING_MPS = 0.2  # horizontal speed treated as actual motion (0.05 m per 0.25 s sample)
STALL_REPORT_S = 30.0  # no-progress intervals at least this long are listed

# person activity states
P_STATES = (
    "moving",
    "brief_wait",
    "dwelling",
    "queueing",
    "courtesy",
    "stalled_in_trip",
    "pre_departure",
    "finished_idle",
    "between_legs",
    "unscheduled",
    "in_vehicle",
    "inside",
    "absent",
)
# vehicle activity states
V_STATES = ("moving", "stopped_in_trip", "parked", "absent")
# stationary person states that have an explicit purpose in the activity record
EXPLAINED_PERSON_STATIC = ("brief_wait", "dwelling", "queueing", "courtesy", "between_legs")
BRIEF_WAIT_S = 3.0  # a pause inside a trip shorter than this is an encounter (letting someone pass), not a stall
UNEXPLAINED_PERSON_STATIC = ("pre_departure", "finished_idle", "stalled_in_trip", "unscheduled")


def _frames(event: Path):
    from awr.sim.events.tape import open_tape

    tape = open_tape(event, verify=False)
    frames = np.concatenate([tape.block_frames(i) for i in range(len(tape.blocks))])
    return tape, frames


def _tick(t_s: float, n: int) -> int:
    return int(min(max(round(float(t_s) / TICK_S), 0), n))


def _person_timeline(agenda: dict, n_ticks: int) -> np.ndarray:
    """Activity phase per tick from the agenda's event list (times relative to the recording start; negative = warm-up).

    Returns an array of phase names (object dtype).
    """
    phase = np.empty(n_ticks, object)
    events = sorted(agenda.get("events", []), key=lambda e: e["t_s"])
    # phase in force at t = 0: replay all events before the recording start
    current = "pre_departure" if float(agenda.get("departure_s", 0.0)) > 0 or not events else "travelling"
    if events and events[0]["t_s"] > 0 and events[0]["kind"] != "journey_started":
        current = "pre_departure"
    purpose = None
    aside_from = None
    marks = []  # (tick, phase, purpose)
    initial = current
    for e in events:
        k = e["kind"]
        new = None
        if k == "journey_started":
            new = "travelling"
        elif k == "arrived":
            new = "queueing"
        elif k == "visit_started":
            new = "dwelling"
            purpose = e.get("purpose", purpose)
        elif k == "visit_interrupted":
            new = "travelling"
        elif k == "visit_completed":
            new = "between_legs"
        elif k == "activity_decision_wait":
            new = "unscheduled"
        elif k == "itinerary_completed":
            new = "finished_idle"
        elif k == "stepped_aside":
            aside_from = current
            new = "courtesy"
        elif k == "returned_to_place":
            new = aside_from or current
            aside_from = None
        elif k in ("home_dwell_started",):
            new = "dwelling"
            purpose = e.get("purpose", "at_home")
        if new is None:
            continue
        current = new
        if e["t_s"] <= 0:
            initial = new
            continue
        marks.append((_tick(e["t_s"], n_ticks), new, purpose))
    phase[:] = None
    k0, cur = 0, initial
    for k, new, _ in marks:
        phase[k0:k] = cur
        k0, cur = k, new
    phase[k0:] = cur
    return phase


def link_groups(groups, step_ticks):
    """Link (tick, member set) groups sampled every step_ticks into episodes.

    A group continues an episode of the previous sample when they share at least
    half of the smaller group's members; each episode continues at most once per
    sample. Returns dicts with k0 and frames [(tick, members), ...].
    """
    episodes, open_eps = [], []
    for k in sorted({k for k, _ in groups}):
        nxt = []
        for m in (m for kk, m in groups if kk == k):
            hit = next(
                (
                    e
                    for e in open_eps
                    if all(e is not x for x in nxt)
                    and e["last_k"] == k - step_ticks
                    and len(e["last"] & m) >= 0.5 * min(len(e["last"]), len(m))
                ),
                None,
            )
            if hit is None:
                hit = dict(k0=k, frames=[])
                episodes.append(hit)
            hit["last"], hit["last_k"] = m, k
            hit["frames"].append((k, m))
            nxt.append(hit)
        open_eps = nxt
    return episodes


def audit(event: Path, config: dict | None = None) -> dict:
    event = Path(event)
    tape, fr = _frames(event)
    return audit_frames(
        event,
        tape.actors,
        fr,
        config,
        tape_content_sha256=tape.hashes()["content_sha256"] if hasattr(tape, "hashes") else None,
    )


def audit_frames(
    event: Path, actors: list, fr: np.ndarray, config: dict | None = None, *, tape_content_sha256: str | None = None
) -> dict:
    """Same life audit on recorded, tape-quantized frames before asset export.

    This is activity evidence only. It does not replace encoded-tape integrity,
    measured-ground QA, geometry checks or rendered camera witnesses. An absent
    tape hash remains absent; callers cannot treat this as a published tape.
    """
    event = Path(event)
    fields = {"x_mm", "y_mm", "vx_cmps", "vy_cmps", "state"}
    if fr.ndim != 2 or not len(fr) or fr.shape[1] != len(actors) or not fields <= set(fr.dtype.names or ()):
        raise ValueError("Life audit requires recorded tape-format frames and matching actors")
    n_ticks, n = fr.shape
    cls = np.array([a["class"] for a in actors])
    ids = [a["id"] for a in actors]
    index = {a: i for i, a in enumerate(ids)}
    state = fr["state"]
    visible = np.isin(state, (1, 2))
    xy = np.stack([fr["x_mm"], fr["y_mm"]], -1).astype(np.float64) * 0.001
    speed = np.hypot(fr["vx_cmps"], fr["vy_cmps"]).astype(np.float64) * 0.01
    # Tapes store no backward difference for the first sample (all zero); use
    # the next sample there instead of reporting a town standing still.
    if n_ticks > 1 and not speed[0].any():
        speed[0] = speed[1]
    moving = visible & (speed >= MOVING_MPS)
    step = np.r_[np.zeros((1, n)), np.linalg.norm(np.diff(xy, axis=0), axis=-1)]
    step[~(visible & np.r_[np.zeros((1, n), bool), visible[:-1]])] = 0.0
    cfg = config or {}
    duration_s = n_ticks * TICK_S
    gate_cfg = cfg.get("life_gates") or {}

    def gate_param(name, key, default):
        spec = gate_cfg.get(name)
        return float(spec.get(key, default)) if isinstance(spec, dict) else default

    trip_min_m = gate_param("vehicle_ineffective_trips_max", "min_distance_m", 50.0)
    trip_max_stop_s = gate_param("vehicle_ineffective_trips_max", "max_stop_s", 60.0)
    jam_size = int(gate_param("person_blockages_max", "min_size", 7))
    jam_min_s = gate_param("person_blockages_max", "min_duration_s", 10.0)
    jam_moving_share = gate_param("person_blockages_max", "min_moving_share", 0.5)

    agendas = json.loads((event / "agendas.json").read_text()) if (event / "agendas.json").exists() else []
    activity = (
        [json.loads(line) for line in (event / "activity-events.jsonl").read_text().splitlines()]
        if (event / "activity-events.jsonl").exists()
        else []
    )

    # ---------------- persons ----------------
    persons = np.flatnonzero(cls == "person")
    pstate = np.full((n_ticks, n), "", object)
    by_actor = {}
    for ag in agendas:
        if ag["actor"] in index:
            by_actor.setdefault(ag["actor"], []).append(ag)
    stall_rows = []
    longest_stall = 0.0
    for i in persons:
        aid = ids[i]
        ph = np.full(n_ticks, "unscheduled", object)
        ags = sorted(by_actor.get(aid, []), key=lambda a: min([e["t_s"] for e in a["events"]] or [1e9]))
        for ag in ags:
            tl = _person_timeline(ag, n_ticks)
            first = min([e["t_s"] for e in ag["events"]] or [1e9])
            # a later agenda (e.g. a courier's per-stop errand) overrides only from its first event on
            k0 = _tick(first, n_ticks) if ag is not ags[0] else 0
            ph[k0:] = tl[k0:]
        col = np.where(
            state[:, i] == 3, "in_vehicle", np.where(state[:, i] == 4, "inside", np.where(~visible[:, i], "absent", ""))
        )
        out = col.astype(object)
        vis = visible[:, i]
        mv = moving[:, i]
        travelling = ph == "travelling"
        out[vis & mv] = "moving"
        rest = vis & ~mv
        out[rest & travelling] = "stalled_in_trip"
        for name in (
            "dwelling",
            "queueing",
            "courtesy",
            "pre_departure",
            "finished_idle",
            "between_legs",
            "unscheduled",
        ):
            out[rest & (ph == name)] = name
        out[(out == "") & vis] = "unscheduled"
        # stalls inside a trip: contiguous stalled_in_trip runs; runs shorter than
        # BRIEF_WAIT_S are encounter pauses (brief_wait)
        st = (out == "stalled_in_trip").astype(np.int8)
        edges = np.flatnonzero(np.diff(np.r_[0, st, 0]))
        for a, b in zip(edges[::2], edges[1::2], strict=False):
            if (b - a) * TICK_S < BRIEF_WAIT_S:
                out[a:b] = "brief_wait"
        pstate[:, i] = out
        for a, b in zip(edges[::2], edges[1::2], strict=False):
            dur = (b - a) * TICK_S
            if dur < BRIEF_WAIT_S:
                continue
            longest_stall = max(longest_stall, dur)
            if dur >= STALL_REPORT_S:
                t0, t1 = a * TICK_S, b * TICK_S
                reasons = Counter(
                    e.get("reason", e["kind"])
                    for ag in ags
                    for e in ag["events"]
                    if t0 - 1 <= e["t_s"] <= t1 + 1 and e["kind"] in ("journey_delayed", "stepped_aside")
                )
                act = Counter(
                    e["kind"] for e in activity if e.get("actors", [None])[0] == aid and t0 - 1 <= e["t_s"] <= t1 + 1
                )
                stall_rows.append(
                    dict(
                        actor=aid,
                        t0_s=round(t0, 1),
                        t1_s=round(t1, 1),
                        duration_s=round(dur, 1),
                        reasons=dict(reasons),
                        activity=dict(act),
                        xy=[round(float(v), 1) for v in xy[a, i]],
                    )
                )

    # ---------------- vehicles ----------------
    vehicles = np.flatnonzero(cls == "vehicle")
    vstate = np.full((n_ticks, n), "", object)
    vpurpose = np.full((n_ticks, n), "", object)
    vev = defaultdict(list)
    for e in activity:
        a = (e.get("actors") or [None])[0]
        if a in index and cls[index[a]] == "vehicle" and e["kind"] in ("park", "depart"):
            vev[a].append(e)
    trips = []
    for i in vehicles:
        evs = sorted(vev.get(ids[i], []), key=lambda e: e["t_s"])
        in_trip = np.zeros(n_ticks, bool)
        purpose = np.full(n_ticks, "unrecorded", object)
        cur_trip, cur_purpose, k0 = False, "unrecorded", 0
        open_trip = None
        for e in evs:
            k = _tick(e["t_s"], n_ticks)
            in_trip[k0:k] = cur_trip
            purpose[k0:k] = cur_purpose
            if e["kind"] == "depart":
                cur_trip, cur_purpose = True, "trip:" + e.get("purpose", "trip")
                open_trip = dict(vehicle=ids[i], purpose=e.get("purpose"), t0_s=e["t_s"], k0=k)
            else:
                cur_trip, cur_purpose = False, "parked:" + e.get("purpose", "park")
                if open_trip is not None:
                    open_trip.update(t1_s=e["t_s"], end_purpose=e.get("purpose"), legal=e.get("legal"))
                    trips.append(open_trip)
                    open_trip = None
            k0 = k
        in_trip[k0:] = cur_trip
        purpose[k0:] = cur_purpose
        if open_trip is not None:
            open_trip.update(t1_s=None, end_purpose=None)
            trips.append(open_trip)
        vis = visible[:, i]
        out = np.where(
            ~vis, "absent", np.where(moving[:, i], "moving", np.where(in_trip, "stopped_in_trip", "parked"))
        ).astype(object)
        vstate[:, i] = out
        vpurpose[:, i] = purpose
    for tr in trips:
        i = index[tr["vehicle"]]
        k1 = _tick(tr["t1_s"], n_ticks) if tr["t1_s"] is not None else n_ticks
        k0 = max(tr["k0"], 0)
        tr["end_xy_m"] = xy[min(k1, n_ticks - 1), i].tolist() if tr["t1_s"] is not None else None
        tr["distance_m"] = round(float(step[k0 + 1 : k1 + 1, i].sum()), 1)
        tr["moving_s"] = round(float(moving[k0:k1, i].sum() * TICK_S), 1)
        st = (vstate[k0:k1, i] == "stopped_in_trip").astype(np.int8)
        edges = np.flatnonzero(np.diff(np.r_[0, st, 0]))
        tr["longest_stop_s"] = round(
            float(max((b - a for a, b in zip(edges[::2], edges[1::2], strict=False)), default=0) * TICK_S), 1
        )
        # An effective trip is a journey to a destination: it covers at least
        # trip_min_m and ends with a recorded parking purpose, and it is never
        # held up longer than trip_max_stop_s. A trip that began before the
        # recording or is still under way at its end is judged on what is recorded;
        # a delivery leg's length is the planned stop spacing and is not judged.
        tr["status"] = "completed" if tr["t1_s"] is not None else "under_way"
        tr["began_before_recording"] = tr["t0_s"] < 0
        reasons = []
        if tr["longest_stop_s"] > trip_max_stop_s:
            reasons.append("held_up")
        tr["delivery_leg"] = actors[i].get("role") == "delivery_van"
        if (
            tr["status"] == "completed"
            and not tr["began_before_recording"]
            and not tr["delivery_leg"]
            and tr["distance_m"] < trip_min_m
        ):
            reasons.append("short")
        if tr["status"] == "completed" and not tr.get("end_purpose"):
            reasons.append("no_recorded_end")
        tr["effective"] = not reasons
        tr["ineffective_reasons"] = reasons
        del tr["k0"]

    # ---------------- per-minute composition ----------------
    minutes = int(np.ceil(duration_s / 60.0))
    per_minute = []
    for m in range(minutes):
        a, b = int(m * 60 / TICK_S), min(int((m + 1) * 60 / TICK_S), n_ticks)
        row = dict(minute=m)
        for name, ids_c, table, states in (
            ("person", persons, pstate, P_STATES),
            ("vehicle", vehicles, vstate, V_STATES),
        ):
            block = table[a:b][:, ids_c]
            cnt = Counter(block.ravel().tolist())
            ticks = max(b - a, 1)
            row[name] = {s: round(cnt.get(s, 0) / ticks, 2) for s in states if cnt.get(s, 0)}
            mv = (block == "moving").sum(1)
            row[name + "_moving_min"] = int(mv.min()) if len(mv) else 0
            row[name + "_moving_max"] = int(mv.max()) if len(mv) else 0
        per_minute.append(row)

    def share(table, ids_c, states):
        block = table[:, ids_c]
        vis_ticks = max(
            int(np.isin(block, [s for s in states if s not in ("absent", "in_vehicle", "inside")]).sum()), 1
        )
        cnt = Counter(block.ravel().tolist())
        return {s: round(cnt.get(s, 0) / vis_ticks, 4) for s in states if cnt.get(s, 0)}, vis_ticks

    p_share, p_vis = share(pstate, persons, P_STATES)
    v_share, v_vis = share(vstate, vehicles, V_STATES)
    p_moving_conc = (pstate[:, persons] == "moving").sum(1)
    v_moving_conc = (vstate[:, vehicles] == "moving").sum(1)

    def per_actor(ids_c):
        mt = moving[:, ids_c].sum(0) * TICK_S
        dist = (step[:, ids_c] * moving[:, ids_c]).sum(0)
        return mt, dist

    p_mt, p_dist = per_actor(persons)
    v_mt, v_dist = per_actor(vehicles)

    # parked purposes (vehicle-ticks)
    vp = Counter()
    for i in vehicles:
        m = vstate[:, i] == "parked"
        vp.update(Counter(vpurpose[m, i].tolist()))
    total_parked = max(sum(vp.values()), 1)
    parked_purpose = {k: round(v / total_parked, 4) for k, v in vp.most_common()}

    # ---------------- zones and coverage ----------------
    zones = []
    anchors = event / "anchors.json"
    if anchors.exists():
        import shapely
        from shapely.geometry import Polygon

        for z in json.loads(anchors.read_text()).get("anchors", []):
            if z.get("kind") != "zone":
                continue
            poly = Polygon(z["polygon"])
            sample = slice(0, n_ticks, 10)  # 1 Hz
            pts = xy[sample][:, persons].reshape(-1, 2)
            inside = shapely.contains_xy(poly, pts[:, 0], pts[:, 1]).reshape(-1, len(persons))
            vis_s = visible[sample][:, persons]
            mov_s = moving[sample][:, persons]
            ins = inside & mov_s
            per_min = [
                int(ins[m * 60 : (m + 1) * 60].sum(1).max()) for m in range(minutes) if len(ins[m * 60 : (m + 1) * 60])
            ]
            zones.append(
                dict(
                    id=z["id"],
                    label=z.get("label_zh"),
                    person_present_mean=round(float((inside & vis_s).sum(1).mean()), 2),
                    person_moving_mean=round(float(ins.sum(1).mean()), 2),
                    minutes_with_moving_person=int(sum(1 for v in per_min if v > 0)),
                )
            )

    def q(arr):
        arr = np.asarray(arr, float)
        if not len(arr):
            return None
        return dict(
            n=int(len(arr)),
            p10=round(float(np.percentile(arr, 10)), 1),
            median=round(float(np.median(arr)), 1),
            p90=round(float(np.percentile(arr, 90)), 1),
            max=round(float(arr.max()), 1),
        )

    # ---------------- local crowding (1 Hz) ----------------
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    nn_moving, nn_static, within3, largest = [], [], [], []
    biggest = None
    big_groups = []  # (k, member actor indices) of groups of at least jam_size people
    for k in range(0, n_ticks, 10):
        vis_p = persons[visible[k, persons]]
        if len(vis_p) < 2:
            continue
        P = xy[k, vis_p]
        tree = cKDTree(P)
        nn = tree.query(P, 2)[0][:, 1]
        mv = moving[k, vis_p]
        nn_moving += nn[mv].tolist()
        nn_static += nn[~mv].tolist()
        within3 += [len(x) - 1 for x in tree.query_ball_point(P, 3.0)]
        pairs = tree.query_pairs(1.5, output_type="ndarray")
        size = 1
        if len(pairs):
            g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
            lab = connected_components(g, directed=False)[1]
            counts = np.bincount(lab)
            size = int(counts.max())
            for g_ in np.flatnonzero(counts >= jam_size):
                big_groups.append((k, frozenset(vis_p[lab == g_].tolist())))
            if biggest is None or size > biggest["size"]:
                members = vis_p[lab == counts.argmax()]
                biggest = dict(
                    size=size,
                    t_s=round(k * TICK_S, 1),
                    centre_xy=np.round(xy[k, members].mean(0), 1).tolist(),
                    members=[ids[j] for j in members],
                    states=dict(Counter(pstate[k, members].tolist())),
                )
        largest.append(size)
    largest = np.asarray(largest if largest else [0])
    # Group episodes: groups of at least jam_size people linked second to second
    # (half of the smaller group's members shared). An episode is a blockage when
    # it lasts at least jam_min_s and its travelling members (moving, pausing or
    # stalled inside a trip) are moving less than jam_moving_share of the time;
    # otherwise it is people crossing one another or passing a group that waits
    # for a purpose (queue, visit). Progress is the distance each travelling
    # member covers while it belongs to the group.
    TRAVELLING = ("moving", "stalled_in_trip", "brief_wait")
    episodes = link_groups(big_groups, 10)
    group_episodes = []
    for e in episodes:
        trav = mov = 0
        members, progress = set(), defaultdict(float)
        for k, m in e["frames"]:
            members |= m
            mm = np.fromiter(m, int)
            st_ = pstate[k, mm]
            tr_ = np.isin(st_, TRAVELLING)
            trav += int(tr_.sum())
            mov += int((st_ == "moving").sum())
            for j in mm[tr_]:
                progress[ids[j]] += float(step[k + 1 : min(k + 11, n_ticks), j].sum())
        dur = len(e["frames"]) * 1.0
        share_mv = mov / trav if trav else 1.0
        k_mid, mm = e["frames"][len(e["frames"]) // 2]
        group_episodes.append(
            dict(
                t0_s=round(e["k0"] * TICK_S, 1),
                duration_s=dur,
                max_size=max(len(m) for _, m in e["frames"]),
                centre_xy=np.round(xy[k_mid, list(mm)].mean(0), 1).tolist(),
                travelling_moving_share=round(share_mv, 2),
                travelling_members=len(progress),
                travelling_progress_m=dict(
                    median=round(float(np.median(list(progress.values()))), 1),
                    min=round(float(min(progress.values())), 1),
                )
                if progress
                else None,
                blockage=bool(dur >= jam_min_s and share_mv < jam_moving_share),
                members=sorted(ids[j] for j in members),
            )
        )
    group_episodes.sort(key=lambda r: (not r["blockage"], -r["duration_s"]))
    # vehicles stopped inside a trip: intervals of at least 5 s, with the nearest other body when the stop began
    v_waits = []
    for i in vehicles:
        st = vstate[:, i] == "stopped_in_trip"
        k = 0
        while k < n_ticks:
            if not st[k]:
                k += 1
                continue
            k1 = k
            while k1 < n_ticks and st[k1]:
                k1 += 1
            if (k1 - k) * TICK_S >= 5.0:
                others = np.flatnonzero(visible[k] & (np.arange(n) != i) & np.isin(cls, ("person", "vehicle")))
                d = np.linalg.norm(xy[k, others] - xy[k, i], axis=1) if len(others) else np.array([np.inf])
                j = others[int(np.argmin(d))] if len(others) else None
                v_waits.append(
                    dict(
                        vehicle=ids[i],
                        t0_s=round(k * TICK_S, 1),
                        duration_s=round((k1 - k) * TICK_S, 1),
                        nearest=ids[j] if j is not None else None,
                        nearest_m=round(float(d.min()), 2) if len(others) else None,
                        nearest_state=(pstate[k, j] if cls[j] == "person" else vstate[k, j]) if j is not None else None,
                    )
                )
            k = k1
    crowding = dict(
        person_nearest_neighbour_m=dict(moving=q(nn_moving), static=q(nn_static)),
        person_neighbours_within_3m=q(within3),
        person_largest_group_1_5m=dict(
            median=float(np.median(largest)),
            p95=float(np.percentile(largest, 95)),
            max=int(largest.max()),
            seconds_with_group_of_6_or_more=int((largest >= 6).sum()),
            seconds_with_group_of_7_or_more=int((largest >= 7).sum()),
            largest=biggest,
        ),
        group_episodes=dict(
            min_size=jam_size,
            blockage_min_duration_s=jam_min_s,
            blockage_moving_share_below=jam_moving_share,
            episodes=len(group_episodes),
            blockages=sum(1 for r in group_episodes if r["blockage"]),
            duration_s=q([r["duration_s"] for r in group_episodes]),
            list=group_episodes[:30],
        ),
        vehicle_stops_in_trip_over_5s=len(v_waits),
        vehicle_stops_in_trip=sorted(v_waits, key=lambda r: -r["duration_s"])[:40],
    )

    cell = 50.0
    pts = xy[::10][:, persons][moving[::10][:, persons]]
    coverage_cells = len({(int(np.floor(x / cell)), int(np.floor(y / cell))) for x, y in pts}) if len(pts) else 0

    # ---------------- itineraries ----------------
    first_depart = []
    finished = []  # residents whose daily chain ended inside the recording
    courier_stops = []  # completed delivery stops and the time until the courier boarded again
    boards = defaultdict(list)
    for e in activity:
        if e["kind"] == "board":
            boards[(e.get("actors") or [None])[0]].append(e["t_s"])
    for ag in agendas:
        ev = ag.get("events", [])
        done = [e["t_s"] for e in ev if e["kind"] == "itinerary_completed"]
        if ag.get("role") == "courier":
            if done and done[0] <= duration_s:
                nb = [t for t in boards.get(ag["actor"], []) if t >= done[0] - 1e-6]
                courier_stops.append(
                    dict(
                        actor=ag["actor"],
                        completed_s=done[0],
                        boarded_s=min(nb) if nb else None,
                        board_after_s=round((min(nb) if nb else duration_s) - done[0], 1),
                        boarded=bool(nb),
                    )
                )
            continue
        js = [e["t_s"] for e in ev if e["kind"] == "journey_started"]
        if js:
            first_depart.append(max(min(js), 0.0))
        if done and done[0] <= duration_s:
            finished.append(
                dict(actor=ag["actor"], completed_s=done[0], idle_after_s=round(duration_s - max(done[0], 0.0), 1))
            )
    legs = []
    for ag in agendas:
        if ag["actor"] not in index:
            continue
        i = index[ag["actor"]]
        ev = sorted(ag.get("events", []), key=lambda e: e["t_s"])
        start = None
        for e in ev:
            if e["kind"] == "journey_started":
                start = e
            elif e["kind"] == "arrived" and start is not None:
                k0, k1 = _tick(start["t_s"], n_ticks), _tick(e["t_s"], n_ticks)
                if k1 > k0:
                    legs.append(
                        dict(
                            distance_m=float(step[k0 + 1 : k1 + 1, i].sum()),
                            duration_s=(k1 - k0) * TICK_S,
                            purpose=start.get("purpose"),
                        )
                    )
                start = None
    dwell = defaultdict(list)
    for ag in agendas:
        st = {}
        for e in sorted(ag.get("events", []), key=lambda e: e["t_s"]):
            if e["kind"] == "visit_started":
                st[e["visit_index"]] = e
            elif e["kind"] == "visit_completed" and e["visit_index"] in st:
                s = st.pop(e["visit_index"])
                dwell[s.get("purpose", "visit")].append(e["t_s"] - s["t_s"])

    unexplained = sum(p_share.get(s, 0.0) for s in UNEXPLAINED_PERSON_STATIC)
    recorded_trips = [t for t in trips if not t["began_before_recording"]]
    from awr.life.civil.fleet_audit import recurring_fleet

    recurrence_gate = gate_cfg.get("vehicle_recurring_vehicles_min", {})
    recurrence_gate = recurrence_gate if isinstance(recurrence_gate, dict) else {}
    fleet_activity = recurring_fleet(
        trips,
        [ids[i] for i in vehicles],
        moving[:, vehicles],
        tick_s=TICK_S,
        min_trips=recurrence_gate.get("min_completed_trips", 2),
        min_destinations=recurrence_gate.get("min_distinct_destinations", 2),
        destination_separation_m=recurrence_gate.get("destination_separation_m", 20.0),
        require_activity_chain=recurrence_gate.get("require_activity_chain", False),
        activity_events=[json.loads(line) for line in (event / "activity-decisions.jsonl").read_text().splitlines()]
        if (event / "activity-decisions.jsonl").exists()
        else [],
        activity_places=json.loads((event / "activity-places.json").read_text())
        if (event / "activity-places.json").exists()
        else [],
    )
    report = dict(
        schema="awr.life.life_audit.v1",
        event=str(event),
        tape_content_sha256=tape_content_sha256,
        duration_s=duration_s,
        counts=dict(Counter(cls.tolist())),
        definitions=dict(
            moving=f"visible and horizontal tape speed >= {MOVING_MPS} m/s",
            brief_wait=f"a pause inside a trip shorter than {BRIEF_WAIT_S:g} s (encounter); longer pauses are stalled_in_trip",
            explained_person_static=list(EXPLAINED_PERSON_STATIC),
            unexplained_person_static=list(UNEXPLAINED_PERSON_STATIC),
            shares="fraction of visible actor-ticks of the class",
        ),
        person=dict(
            state_share=p_share,
            unexplained_static_share=round(unexplained, 4),
            moving_concurrency=dict(
                min=int(p_moving_conc.min()), median=float(np.median(p_moving_conc)), max=int(p_moving_conc.max())
            ),
            moving_time_s=q(p_mt),
            moving_distance_m=q(p_dist),
            first_departure_s=q(first_depart),
            finished_itineraries=len(finished),
            finished_idle_s=q([f["idle_after_s"] for f in finished]),
            leg_distance_m=q([l["distance_m"] for l in legs]),
            leg_duration_s=q([l["duration_s"] for l in legs]),
            dwell_s={k: q(v) for k, v in sorted(dwell.items())},
            stalls_over_30s=len(stall_rows),
            stalls_over_60s=sum(1 for r in stall_rows if r["duration_s"] >= 60),
            longest_stall_s=round(longest_stall, 1),
        ),
        vehicle=dict(
            state_share=v_share,
            recurring_fleet=fleet_activity,
            moving_concurrency=dict(
                min=int(v_moving_conc.min()), median=float(np.median(v_moving_conc)), max=int(v_moving_conc.max())
            ),
            moving_time_s=q(v_mt),
            moving_distance_m=q(v_dist),
            vehicles_moving_under_60s=int((v_mt < 60).sum()),
            vehicles_never_moving=int((v_mt == 0).sum()),
            trips=len(trips),
            trips_from_warm_up=sum(1 for t in trips if t["began_before_recording"]),
            trips_under_way_at_end=sum(1 for t in trips if t["status"] == "under_way"),
            # raw distributions over trips that start inside the recording
            trip_distance_m=q([t["distance_m"] for t in recorded_trips]),
            trip_moving_s=q([t["moving_s"] for t in recorded_trips]),
            trip_longest_stop_s=q([t["longest_stop_s"] for t in trips]),
            trip_starts_per_minute=[
                sum(1 for t in trips if m * 60 <= t["t0_s"] < (m + 1) * 60) for m in range(minutes)
            ],
            effective_trip_rule=dict(
                min_distance_m=trip_min_m,
                max_stop_s=trip_max_stop_s,
                note="trips begun before the recording are not judged on distance",
            ),
            trips_effective=sum(1 for t in trips if t["effective"]),
            trips_ineffective=[t for t in trips if not t["effective"]],
            trip_list=sorted(trips, key=lambda t: t["t0_s"])[:200],
            trip_purposes=dict(Counter(f"{t['purpose']}->{t['end_purpose']}" for t in trips)),
            parked_purpose_share=parked_purpose,
        ),
        zones=zones,
        person_moving_coverage_cells_50m=coverage_cells,
        crowding=crowding,
        per_minute=per_minute,
        stalls=sorted(stall_rows, key=lambda r: -r["duration_s"])[:60],
        finished=sorted(finished, key=lambda f: f["completed_s"])[:200],
        courier=dict(
            stops_completed=len(courier_stops),
            board_after_s=q([c["board_after_s"] for c in courier_stops]),
            stops=courier_stops[:50],
        ),
    )
    report["gates"] = evaluate_gates(report, cfg.get("life_gates"))
    return report


def evaluate_gates(report: dict, gates: dict | None) -> dict | None:
    """Scenario-specific thresholds; every gate carries its rationale from the config."""
    if not gates:
        return None
    out = {}
    P, V = report["person"], report["vehicle"]
    pm = report["per_minute"]
    checks = {
        "duration_s_min": lambda th: (report.get("duration_s", 0.0), report.get("duration_s", 0.0) >= th),
        "vehicle_recurring_vehicles_min": lambda th: (
            V.get("recurring_fleet", {}).get("qualified_vehicles", 0),
            V.get("recurring_fleet", {}).get("qualified_vehicles", 0) >= th,
        ),
        "vehicle_longest_no_driving_s_max": lambda th: (
            V.get("recurring_fleet", {}).get("longest_no_driving_s"),
            V.get("recurring_fleet", {}).get("longest_no_driving_s", float("inf")) <= th,
        ),
        "person_unexplained_static_share_max": lambda th: (
            P["unexplained_static_share"],
            P["unexplained_static_share"] <= th,
        ),
        "person_pre_departure_share_max": lambda th: (
            P["state_share"].get("pre_departure", 0.0),
            P["state_share"].get("pre_departure", 0.0) <= th,
        ),
        "person_finished_idle_share_max": lambda th: (
            P["state_share"].get("finished_idle", 0.0),
            P["state_share"].get("finished_idle", 0.0) <= th,
        ),
        "person_longest_stall_s_max": lambda th: (P["longest_stall_s"], P["longest_stall_s"] <= th),
        "person_moving_min_per_minute": lambda th: (
            min(r["person_moving_min"] for r in pm),
            min(r["person_moving_min"] for r in pm) >= th,
        ),
        "person_finished_itineraries_max": lambda th: (P["finished_itineraries"], P["finished_itineraries"] <= th),
        "vehicle_moving_minutes_min": lambda th: (
            sum(1 for r in pm if r["vehicle_moving_max"] > 0),
            sum(1 for r in pm if r["vehicle_moving_max"] > 0) >= th,
        ),
        "vehicle_unrecorded_parking_share_max": lambda th: (
            V["parked_purpose_share"].get("unrecorded", 0.0),
            V["parked_purpose_share"].get("unrecorded", 0.0) <= th,
        ),
        "vehicle_trip_moving_s_median_min": lambda th: (
            (V["trip_moving_s"] or {}).get("median", 0.0),
            (V["trip_moving_s"] or {}).get("median", 0.0) >= th,
        ),
        "person_largest_group_max": lambda th: (
            report["crowding"]["person_largest_group_1_5m"]["max"],
            report["crowding"]["person_largest_group_1_5m"]["max"] <= th,
        ),
        "person_blockages_max": lambda th: (
            report["crowding"]["group_episodes"]["blockages"],
            report["crowding"]["group_episodes"]["blockages"] <= th,
        ),
        "vehicle_ineffective_trips_max": lambda th: (len(V["trips_ineffective"]), len(V["trips_ineffective"]) <= th),
        "courier_board_after_stop_s_max": lambda th: (
            (report["courier"]["board_after_s"] or {}).get("max", 0.0),
            (report["courier"]["board_after_s"] or {}).get("max", 0.0) <= th,
        ),
        "vehicle_longest_stop_in_trip_s_max": lambda th: (
            max([r["duration_s"] for r in report["crowding"]["vehicle_stops_in_trip"]] or [0.0]),
            max([r["duration_s"] for r in report["crowding"]["vehicle_stops_in_trip"]] or [0.0]) <= th,
        ),
        "zones_with_moving_person_every_minute_min": lambda th: (
            sum(1 for z in report["zones"] if z["minutes_with_moving_person"] >= len(pm)),
            sum(1 for z in report["zones"] if z["minutes_with_moving_person"] >= len(pm)) >= th,
        ),
    }
    for name, spec in gates.items():
        if name.startswith("_"):
            continue
        th = spec["threshold"] if isinstance(spec, dict) else spec
        if name not in checks:
            out[name] = dict(threshold=th, value=None, pass_=None, note="unknown gate")
            continue
        value, ok = checks[name](th)
        out[name] = dict(
            threshold=th,
            value=value,
            pass_=bool(ok),
            rationale=spec.get("rationale") if isinstance(spec, dict) else None,
        )
    return dict(all_pass=all(g["pass_"] is True for g in out.values()), gates=out)


def markdown(report: dict) -> str:
    P, V = report["person"], report["vehicle"]

    def median_with_unit(summary, unit):
        # An absent actor family has no per-actor distribution, not a zero median.
        return f"{summary['median']} {unit}" if summary is not None else "—"

    lines = [
        f"# 生活行为审计 · {report['event']}",
        "",
        f"磁带 `{(report.get('tape_content_sha256') or '')[:16]}…`，时长 {report['duration_s']:.0f} s，"
        f"人 {report['counts'].get('person', 0)}、车 {report['counts'].get('vehicle', 0)}、物件 {report['counts'].get('object', 0)}。",
        "",
        "| 指标 | 行人 | 车辆 |",
        "|---|---:|---:|",
        f"| 同时移动 最小/中位/最大 | {P['moving_concurrency']['min']} / {P['moving_concurrency']['median']:g} / {P['moving_concurrency']['max']} | "
        f"{V['moving_concurrency']['min']} / {V['moving_concurrency']['median']:g} / {V['moving_concurrency']['max']} |",
        f"| 单体累计移动时长中位数 | {median_with_unit(P['moving_time_s'], 's')} | {median_with_unit(V['moving_time_s'], 's')} |",
        f"| 单体累计移动距离中位数 | {median_with_unit(P['moving_distance_m'], 'm')} | {median_with_unit(V['moving_distance_m'], 'm')} |",
        "",
        "行人可见时间构成：" + "，".join(f"{k} {v * 100:.1f}%" for k, v in P["state_share"].items()) + "。",
        f"无明确目的的静止（出发前站立、行程结束后站立、行程中 3 s 及以上无进展、无日程）合计 {P['unexplained_static_share'] * 100:.1f}%；"
        f"行程中不足 3 s 的短暂让行（相遇时减速停顿）{P['state_share'].get('brief_wait', 0.0) * 100:.1f}%，计为有目的的静止。",
        "",
        "车辆停放目的构成：" + "，".join(f"{k} {v * 100:.1f}%" for k, v in V["parked_purpose_share"].items()) + "。",
        "",
    ]
    C = report.get("crowding")
    if C:
        nn, g = C["person_nearest_neighbour_m"], C["person_largest_group_1_5m"]

        def fmt(d):
            return f"{d['p10']} / {d['median']} / {d['p90']} m" if d else "—"

        lines += [
            "## 局部聚集",
            "",
            f"行人最近邻间距（第 10 百分位 / 中位 / 第 90 百分位）：行走中 {fmt(nn['moving'])}，静止 {fmt(nn['static'])}。"
            f"3 m 内其他行人数中位 {C['person_neighbours_within_3m']['median'] if C['person_neighbours_within_3m'] else '—'}、"
            f"最大 {C['person_neighbours_within_3m']['max'] if C['person_neighbours_within_3m'] else '—'}。",
            f"间距不超过 1.5 m 相连的最大人群：中位 {g['median']:g}、第 95 百分位 {g['p95']:g}、最大 {g['max']}；"
            f"出现 6 人及以上人群的秒数 {g['seconds_with_group_of_6_or_more']}。"
            + (
                f"最大人群出现在 {g['largest']['t_s']} s、{g['largest']['centre_xy']}，状态 {g['largest']['states']}。"
                if g.get("largest")
                else ""
            ),
            f"车辆行程中停车 5 s 以上 {C['vehicle_stops_in_trip_over_5s']} 次"
            + (
                f"，最长 {C['vehicle_stops_in_trip'][0]['duration_s']} s（{C['vehicle_stops_in_trip'][0]['vehicle']}，最近目标 "
                f"{C['vehicle_stops_in_trip'][0]['nearest']} {C['vehicle_stops_in_trip'][0]['nearest_m']} m）"
                if C["vehicle_stops_in_trip"]
                else ""
            )
            + "。",
            "",
        ]
        ge = C.get("group_episodes")
        if ge:
            lines += [
                f"{ge['min_size']} 人及以上人群按秒连成的聚集过程 {ge['episodes']} 次，其中判为堵塞 {ge['blockages']} 次"
                f"（持续不少于 {ge['blockage_min_duration_s']:g} s，且行进中成员的移动时间占比低于 {ge['blockage_moving_share_below']:g}）。",
                "",
            ]
            if ge["list"]:
                lines += [
                    "| 开始 | 持续 | 最大人数 | 位置 | 行进中成员 | 移动占比 | 行进距离中位/最小 | 判定 |",
                    "|---:|---:|---:|---|---:|---:|---:|---|",
                ]
                for r in ge["list"][:12]:
                    pr = r["travelling_progress_m"]
                    lines.append(
                        f"| {r['t0_s']} s | {r['duration_s']:g} s | {r['max_size']} | {r['centre_xy']} | {r['travelling_members']} | "
                        f"{r['travelling_moving_share']} | {pr['median'] if pr else '—'} / {pr['min'] if pr else '—'} m | {'堵塞' if r['blockage'] else '交错通过'} |"
                    )
                lines.append("")
    rule = V.get("effective_trip_rule")
    if rule:

        def fq(d, u):
            return f"{d['p10']} / {d['median']} / {d['p90']} / {d['max']} {u}" if d else "—"

        lines += [
            "## 车辆行程",
            "",
            f"行程 {V['trips']} 段，有效 {V['trips_effective']} 段（距离不少于 {rule['min_distance_m']:g} m、以记录的停车目的结束、"
            f"途中单次停车不超过 {rule['max_stop_s']:g} s；录制开始前已出发的行程不按距离判定）。",
            f"录制开始前已出发 {V['trips_from_warm_up']} 段，录制结束时仍在途 {V['trips_under_way_at_end']} 段。"
            f"录制内出发的行程距离（第 10 百分位 / 中位 / 第 90 百分位 / 最大）{fq(V['trip_distance_m'], 'm')}，"
            f"行驶时长 {fq(V['trip_moving_s'], 's')}；全部行程途中最长停车 {fq(V['trip_longest_stop_s'], 's')}。",
            "每分钟新出发行程数：" + "、".join(str(x) for x in V["trip_starts_per_minute"]) + "。",
        ]
        for t in V["trips_ineffective"][:12]:
            lines.append(
                f"- 无效行程 {t['vehicle']}，{t['t0_s']} s 出发，{t['distance_m']} m，途中最长停车 {t['longest_stop_s']} s，原因 {', '.join(t['ineffective_reasons'])}"
            )
        lines.append("")
    observed = V.get("recurring_fleet", {}).get("driver_activities")
    if observed:
        summary = (
            f"其中 {observed['vehicles_with_completed_activity']} 辆车关联 "
            f"{observed['completed_driver_activities']} 条完整驾驶者活动链。"
            if observed["status"] == "checked"
            else "缺少活动日志或场所定义，完整驾驶者活动链数量未知。"
        )
        lines += [
            "## 驾驶者活动实测",
            "",
            f"完整录制的合法停车行程 {observed['completed_legal_moving_trips']} 段。" + summary,
            "完整链要求步行、上车、实际行驶、下车、步行到目的地及完成活动相互对应；"
            "不依赖车辆数量或最短距离门槛，未完成的服务与预热阶段跨界链不计入。"
            "这是活动证据，运动安全、相机可见性与发布准入仍须分别验证。",
            "",
        ]
    cr = report.get("courier")
    if cr is not None:
        b = cr["board_after_s"]
        lines += [
            "## 日程（按角色）",
            "",
            f"居民日程在录制时段内结束 {P['finished_itineraries']} 人。快递员完成派送 {cr['stops_completed']} 站，"
            + (f"每站返回车门后上车用时最大 {b['max']} s。" if b else "尚无完成的派送。"),
            "",
        ]
    if report.get("gates"):
        lines += ["## 场景门槛", "", "| 门槛 | 阈值 | 实测 | 结果 |", "|---|---:|---:|---|"]
        for k, g in report["gates"]["gates"].items():
            lines.append(f"| {k} | {g['threshold']} | {g['value']} | {'通过' if g['pass_'] else '未通过'} |")
        lines.append("")
    lines += [
        "## 每分钟构成（平均人数）",
        "",
        "| 分钟 | 行人移动 | 短暂让行 | 行人停留 | 出发前 | 行程结束 | 行程中无进展 | 车辆移动 | 车辆停放 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["per_minute"]:
        p, v = r["person"], r["vehicle"]
        lines.append(
            f"| {r['minute']} | {p.get('moving', 0)} | {p.get('brief_wait', 0)} | {p.get('dwelling', 0)} | {p.get('pre_departure', 0)} | {p.get('finished_idle', 0)} | "
            f"{p.get('stalled_in_trip', 0)} | {v.get('moving', 0)} | {v.get('parked', 0)} |"
        )
    return "\n".join(lines) + "\n"
