"""Reconcile resident-car journeys with intentions, legal berths and tape frames.

Distance is descriptive, never a required tour length. This verifies the drive
leg's causal progress, not completion of the subsequent facility activity,
physical collision safety, task solvability, or permission to publish a tape.
"""

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from awr.life.common.schema import schema_id


def intention_lifecycle(decisions, events, *, horizon_s):
    """Account for EVERY selected car intent, including ones that never depart.

    Descriptive event ledger, not another success gate. Boarding-to-departure is
    predeparture time, not necessarily traffic delay. Selection-to-boarding also
    includes access walking and boarding service. Intervals are clipped to the
    recording; pending durations are censored lower bounds, not completed waits.
    Unselected proposals and unlogged cancellation cannot be inferred here.
    """
    if not math.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError("Positive finite recording horizon required")
    chosen, director, physical = defaultdict(list), defaultdict(list), defaultdict(list)
    for e in decisions:
        key = (e.get("actor"), e.get("intent_id"))
        director[key].append(e)
        if e.get("kind") == "activity_selected" and e.get("mode") == "car":
            chosen[key].append(e)
    for e in events:
        if e.get("source") == "observed_drive_execution":
            physical[(e.get("driver"), e.get("intent_id"))].append(e)
    rows = []
    for key, selections in sorted(chosen.items(), key=lambda item: str(item[0])):
        row = dict(driver=key[0], intent_id=key[1], status="inconsistent", issues=[])
        rows.append(row)
        if len(selections) != 1:
            row["issues"].append("duplicate_selection")
            continue
        choice = selections[0]
        row.update(vehicle=choice.get("vehicle"), purpose=choice.get("purpose"), place=choice.get("place"))
        milestones = {
            "selected": [choice],
            "boarded": [e for e in director[key] if e.get("kind") == "step_completed" and e.get("step") == "board"],
            "departed": [e for e in physical[key] if e.get("kind") == "depart"],
            "parked": [e for e in physical[key] if e.get("kind") == "park"],
        }
        times = {}
        for name, group in milestones.items():
            if len(group) > 1:
                row["issues"].append("duplicate_" + name)
            if group:
                try:
                    value = float(group[0]["t_s"])
                    if not math.isfinite(value):
                        raise ValueError("nonfinite")
                    times[name] = value
                except (ValueError, TypeError, KeyError):
                    row["issues"].append("invalid_" + name + "_time")
        row["times_s"] = times
        ordered = [times[n] for n in milestones if n in times]
        if any(a > b for a, b in zip(ordered, ordered[1:], strict=False)):
            row["issues"].append("out_of_order")
        if "departed" in times and "boarded" not in times:
            row["issues"].append("missing_boarding")
        if "parked" in times and "departed" not in times:
            row["issues"].append("missing_departure")
        if row["issues"]:
            continue
        if times["selected"] >= horizon_s or times.get("parked", horizon_s) <= 0:
            row["status"] = "outside_recording"
            continue
        seen = {n: t for n, t in times.items() if t < horizon_s}
        row["status"] = (
            "parked"
            if "parked" in seen
            else "driving_pending"
            if "departed" in seen
            else "predeparture_pending"
            if "boarded" in seen
            else "access_or_boarding_pending"
        )
        row["left_censored"] = times["selected"] < 0
        phases = {}
        for label, start, end in [
            ("access_and_boarding", "selected", "boarded"),
            ("predeparture", "boarded", "departed"),
            ("drive", "departed", "parked"),
        ]:
            if start not in seen:
                continue
            t0, t1 = max(0.0, seen[start]), min(horizon_s, times.get(end, horizon_s))
            phases[label] = dict(
                observed_s=max(0.0, t1 - t0), right_censored=end not in seen, left_censored=seen[start] < 0
            )
        row["phases"] = phases
        row["selection_to_park_observed_s"] = max(
            0.0, min(horizon_s, times.get("parked", horizon_s)) - max(0.0, times["selected"])
        )
    counts = {
        status: sum(r["status"] == status for r in rows)
        for status in (
            "parked",
            "driving_pending",
            "predeparture_pending",
            "access_or_boarding_pending",
            "outside_recording",
            "inconsistent",
        )
    }
    totals = {
        phase: sum(r.get("phases", {}).get(phase, {}).get("observed_s", 0.0) for r in rows)
        for phase in ("access_and_boarding", "predeparture", "drive")
    }
    return dict(
        schema="awr.life.car_intention_lifecycle.v1",
        horizon_s=horizon_s,
        selected_intents=len(rows),
        counts=counts,
        observed_actor_seconds=totals,
        rows=rows,
        scope="Selected car intents only; phases are descriptive, not physical or task certification.",
        limitations=[
            "Predeparture duration does not attribute a cause or prove traffic delay.",
            "Pending intervals are censored; no cancellation or completion is inferred.",
            "Unselected demands and subsequent facility activity are outside this ledger.",
        ],
    )


def audit_journeys(event, actors, frames, config):
    """Load the bound scene's evidence; missing inputs are unavailable, not zero failures."""
    try:
        event = Path(event)
        name = (config.get("life", {}).get("driving") or {}).get("map")
        if not name:
            raise ValueError("No driving-map binding in scenario")
        path = Path(name)
        if not path.is_absolute():
            path = Path(config.get("environment_root", ".")) / path
        files = [path, event / "activity-decisions.jsonl", event / "activity-events.jsonl"]
        data = [p.read_bytes() for p in files]
        doc = json.loads(data[0])
        if schema_id(doc.get("schema")) != "awr.life.driving_map_candidate.v1":
            raise ValueError("Unknown driving-map schema")
        decisions, events = ([json.loads(line) for line in raw.splitlines() if line.strip()] for raw in data[1:])
        result = inspect_journeys(actors, frames, decisions, events, doc["nodes"])
        result["input_sha256"] = {str(p): hashlib.sha256(raw).hexdigest() for p, raw in zip(files, data, strict=False)}
        return result
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(
            schema="awr.life.journey_evidence.v1", available=False, admitted=False, failures=None, reason=str(error)
        )


def inspect_journeys(actors, frames, decisions, events, berths, *, tick_s=0.1):
    required = {"x_mm", "y_mm", "vx_cmps", "vy_cmps", "state", "container"}
    if (
        frames.ndim != 2
        or frames.shape[1] != len(actors)
        or not len(frames)
        or not required <= set(frames.dtype.names or ())
        or not math.isfinite(tick_s)
        or tick_s <= 0
    ):
        raise ValueError("Matching nonempty fixed-roster tape frames and a positive tick required")
    index = {a["id"]: i for i, a in enumerate(actors)}
    if len(index) != len(actors):
        raise ValueError("Duplicate actor identities")
    horizon = len(frames) * tick_s
    selected = defaultdict(list)
    director = defaultdict(list)
    physical = defaultdict(list)
    for e in decisions:
        key = (e.get("actor"), e.get("intent_id"))
        director[key].append(e)
        if e["kind"] == "activity_selected" and e.get("mode") == "car":
            selected[key].append(e)
    for e in events:
        if e.get("source") in ("observed_drive_execution", "traffic_revision"):
            physical[(e.get("driver"), e.get("intent_id"))].append(e)
    xy = np.stack([frames["x_mm"], frames["y_mm"]], axis=-1).astype(float) * 0.001
    velocity = np.stack([frames["vx_cmps"], frames["vy_cmps"]], axis=-1).astype(float) * 0.01
    covered = np.zeros(frames.shape, bool)
    rows = []

    def frame_at(t):
        return int(round(t / tick_s))

    def near(a, b):
        # Recording error only: mm quantization plus float32 intermediate ULP.
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != (2,) or b.shape != (2,) or not np.isfinite(a).all() or not np.isfinite(b).all():
            return False
        tolerance = 0.0015 + float(np.spacing(np.float32(np.max(np.abs(b)))))
        return bool(np.max(np.abs(a - b)) <= tolerance)

    for key, group in sorted(physical.items(), key=lambda item: str(item[0])):
        group = sorted(group, key=lambda e: e["t_s"])
        starts = [e for e in group if e["kind"] == "depart"]
        ends = [e for e in group if e["kind"] == "park"]
        if not starts and not ends:
            continue  # a route revision while waiting is not a departure
        driver, intent = key
        row = dict(driver=driver, intent_id=intent, checks={}, status="inconsistent")
        rows.append(row)
        checks = row["checks"]
        checks["single_intention_and_departure"] = len(selected[key]) == 1 and len(starts) == 1 and len(ends) <= 1
        if not checks["single_intention_and_departure"]:
            continue
        chosen, depart = selected[key][0], starts[0]
        end = ends[0] if ends else None
        vehicle = chosen.get("vehicle")
        row["vehicle"] = vehicle
        row["purpose"], row["place"] = chosen.get("purpose"), chosen.get("place")
        t0, t1 = float(depart["t_s"]), float(end["t_s"]) if end else horizon
        checks["ordered_finite_times"] = bool(
            math.isfinite(t0)
            and math.isfinite(t1)
            and float(chosen["t_s"]) <= t0 < t1
            and all(float(chosen["t_s"]) <= float(e["t_s"]) <= t1 for e in group)
        )
        checks["known_actor_types"] = (
            driver in index
            and vehicle in index
            and actors[index[driver]]["class"] == "person"
            and actors[index[vehicle]]["class"] == "vehicle"
        )
        checks["same_vehicle_intent_and_purpose"] = bool(
            chosen.get("purpose")
            and chosen.get("place")
            and all(
                e.get("actors") == [vehicle]
                and e.get("purpose") == chosen["purpose"]
                and e.get("place") == chosen["place"]
                for e in group
            )
        )
        steps = chosen.get("steps", [])
        drive_steps = [s for s in steps if s["kind"] == "drive"]
        checks["drive_declared_for_same_vehicle"] = len(drive_steps) == 1 and drive_steps[0].get("entity") == vehicle
        admissions = [e for e in director[key] if e["kind"] == "drive_departure_admitted"]
        checks["observed_departure_matches_admission"] = bool(
            len(admissions) == 1
            and admissions[0].get("vehicle") == vehicle
            and admissions[0].get("evidence") == depart.get("evidence")
            and abs(float(admissions[0]["t_s"]) - t0) < tick_s / 2
        )
        boards = [e for e in director[key] if e["kind"] == "step_completed" and e.get("step") == "board"]
        checks["boarded_before_driving"] = bool(
            len(boards) == 1 and boards[0].get("entity") == driver and float(boards[0]["t_s"]) <= t0
        )
        proof = group[0].get("replaces") if group[0]["kind"] == "reroute" else depart.get("evidence")
        target = chosen.get("destination_berth")
        valid_proof = bool(proof and proof in chosen.get("routes", []))
        for e in group:
            if e["kind"] == "reroute":
                valid_proof &= e.get("replaces") == proof and bool(e.get("evidence"))
                proof, target = e.get("evidence"), e.get("berth")
            else:
                valid_proof &= e.get("evidence") == proof
        checks["route_revision_chain"] = bool(valid_proof)
        if not checks["ordered_finite_times"] or not checks["known_actor_types"]:
            continue
        if t1 < 0 or t0 >= horizon:
            row["status"] = "outside_recording"
            continue
        i, j = index[driver], index[vehicle]
        k0, k1 = max(0, frame_at(t0)), min(len(frames), frame_at(t1))
        checks["observed_interval_present"] = k0 < k1
        if k0 >= k1:
            continue
        covered[k0 : k1 + 1, j] = True
        row.update(
            depart_s=t0,
            park_s=t1 if end else None,
            boundary_truncated=t0 < 0 or end is None or t1 >= horizon,
            distance_m=float(np.linalg.norm(np.diff(xy[k0 : k1 + 1, j], axis=0), axis=1).sum()),
        )
        checks["same_car_containment"] = bool(
            (frames["state"][k0:k1, i] == 3).all()
            and (frames["container"][k0:k1, i] == j).all()
            and np.max(np.abs(xy[k0:k1, i] - xy[k0:k1, j])) <= 0.001
        )
        checks["vehicle_remains_visible"] = bool(np.isin(frames["state"][k0:k1, j], (1, 2)).all())
        if 0 <= t0 < horizon:
            checks["departure_position_matches_frame"] = near(xy[k0, j], depart.get("xy"))
        if end and 0 <= t1 < horizon:
            node = berths.get(end.get("berth"), {})
            checks["legal_declared_destination"] = bool(
                end.get("berth") == target
                and end.get("legal") is True
                and node.get("kind") == "berth"
                and node.get("parking_legality", {}).get("legal") is True
            )
            checks["observed_park_at_destination"] = bool(
                node.get("pose") is not None
                and near(xy[k1, j], node["pose"][:2])
                and near(xy[k1, j], end.get("xy"))
                and np.linalg.norm(velocity[k1, j]) <= 0.02
                and frames["state"][k1, j] in (1, 2)
            )
            completions = [e for e in director[key] if e["kind"] == "step_completed" and e.get("step") == "drive"]
            checks["drive_step_completed_at_parking"] = bool(
                len(completions) == 1
                and completions[0].get("entity") == vehicle
                and abs(float(completions[0]["t_s"]) - t1) < tick_s / 2
            )
            if t0 >= 0:
                # A positive displacement and recorded motion distinguish a
                # journey from metadata-only board/park. No 50m/30s tour quota.
                checks["physical_progress_observed"] = bool(
                    row["distance_m"] > 0.002 and np.max(np.linalg.norm(velocity[k0:k1, j], axis=1)) > 0.02
                )
        row["status"] = (
            ("partial_verified" if row["boundary_truncated"] else "verified")
            if all(checks.values())
            else "inconsistent"
        )

    # An admitted journey that should have moved cannot vanish from the physical
    # event ledger. A selected activity still walking toward its car is not a trip.
    accounted = {(r["driver"], r["intent_id"]) for r in rows}
    for key, group in director.items():
        if key not in accounted and any(e["kind"] == "drive_departure_admitted" for e in group):
            rows.append(
                dict(
                    driver=key[0],
                    intent_id=key[1],
                    status="inconsistent",
                    checks=dict(physical_departure_present=False),
                )
            )
    # Missing both ledgers must not make moving cars disappear from the audit.
    # This contract is for the resident-car backend; another backend must supply
    # equivalent observed intention/departure evidence before using this gate.
    for j, actor in enumerate(actors):
        if actor["class"] != "vehicle":
            continue
        unknown = np.flatnonzero(
            ~covered[:, j] & np.isin(frames["state"][:, j], (1, 2)) & (np.linalg.norm(velocity[:, j], axis=1) > 0.02)
        )
        if len(unknown):
            rows.append(
                dict(
                    vehicle=actor["id"],
                    driver=None,
                    intent_id=None,
                    status="inconsistent",
                    checks=dict(all_observed_driving_attributed=False),
                    frames=len(unknown),
                    first_s=float(unknown[0] * tick_s),
                    last_s=float(unknown[-1] * tick_s),
                )
            )
    return dict(
        schema="awr.life.journey_evidence.v1",
        available=True,
        admitted=False,
        verified_complete=sum(r["status"] == "verified" for r in rows),
        verified_partial=sum(r["status"] == "partial_verified" for r in rows),
        failures=sum(r["status"] == "inconsistent" for r in rows),
        rows=rows,
        intention_lifecycle=intention_lifecycle(decisions, events, horizon_s=horizon),
        rule="Intent and observed driving/containment/arrival at a legal mapped berth; no minimum trip distance.",
        limitations=[
            "Drive-leg evidence only; no claim of subsequent facility activity completion.",
            "Full motion QA, scene binding and task witnesses remain separate requirements.",
            "Warm-up or unfinished journeys verify only their recorded interval.",
        ],
    )
