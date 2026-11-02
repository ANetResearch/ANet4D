"""Explicit migration from finite curb-chain quotas to observed autonomous journeys."""

import hashlib
import json
import math
from copy import deepcopy


def validate_acceptance_capabilities(config):
    """Reject unavailable evidence requirements before loading/building a scene.

    Missing capability is not a successful zero-violation observation. Do not
    silently remove gates: a walking-only recipe must explicitly declare its
    own acceptance scope. This does not predict the number of admitted drivers.
    """
    gates = config.get("life_gates") or {}
    if "vehicle_journey_evidence_failures_max" not in gates:
        return
    life = config.get("life") or {}
    modes = (life.get("activity_engine") or {}).get("modes", ["walk"])
    if (
        config.get("runtime") != "life_v1"
        or life.get("behavior") != "activities_v1"
        or not (life.get("driving") or {}).get("map")
        or not any(mode in ("car", "drive") for mode in modes)
    ):
        raise ValueError(
            "vehicle_journey_evidence_failures_max requires life_v1 "
            "activities_v1 with a driving map and car/drive capability; "
            "declare a walking-only acceptance scope explicitly"
        )


def purposeful_journeys(config):
    """Return a new configuration; never mutate a running/frozen scenario.

    Retain all unrelated gates and the stricter old stop limit. This migrates
    an acceptance policy, not a trajectory, map, demand or physical predicate.
    """
    life = config.get("life") or {}
    if (
        config.get("runtime") != "life_v1"
        or life.get("behavior") != "activities_v1"
        or not (life.get("driving") or {}).get("map")
    ):
        raise ValueError("Purposeful journey acceptance requires the autonomous resident-car backend and mapped berths")
    if config.get("acceptance_contract"):
        raise ValueError("Configuration already declares an acceptance contract; review rather than overwrite it")
    out = deepcopy(config)
    gates = out.get("life_gates") or {}
    previous = gates.pop("vehicle_ineffective_trips_max", None)
    if previous is None:
        raise ValueError("No legacy trip gate to migrate")
    old = previous if isinstance(previous, dict) else {"threshold": previous}
    if old.get("threshold") != 0:
        raise ValueError("Migration requires the original zero-invalid-trip rule")
    stop = float(old.get("max_stop_s", 60.0))
    if not math.isfinite(stop) or stop <= 0:
        raise ValueError("Invalid legacy stop limit")
    if "vehicle_journey_evidence_failures_max" in gates:
        raise ValueError("Journey evidence gate already present")
    gates["vehicle_journey_evidence_failures_max"] = dict(
        threshold=0,
        rationale="Every resident-car journey must reconcile intention, observed driving and driver containment; "
        "completed legs must end stationary at their legal mapped destination. "
        "Short trips are valid; no tour-length quota. Partial recordings are reported separately.",
    )
    name = "vehicle_longest_stop_in_trip_s_max"
    if name in gates:
        value = gates[name]["threshold"] if isinstance(gates[name], dict) else gates[name]
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError("Invalid existing stop gate")
        stop = min(stop, float(value))
    gates[name] = dict(
        threshold=stop,
        rationale="Retains the stricter existing/legacy in-trip stop limit during acceptance-policy migration.",
    )
    out["life_gates"] = gates
    out["acceptance_contract"] = dict(
        schema="awr.life.activity_acceptance.v1",
        id="purposeful-journeys-v1",
        source_config_sha256=hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        replaced_gate={"vehicle_ineffective_trips_max": deepcopy(previous)},
        reason="User goal: capacity- and purpose-driven activity, not a mandatory per-trip distance.",
        invariants=[
            "Original motion, support and collision QA unchanged.",
            "No population, map, activity choice or trajectory changes.",
            "All unrelated life gates retained; maximum in-trip stop limit not weakened.",
            "Original frozen config/report retained; new contract must be declared in a new version.",
            "No task, rendering or release admission implied.",
        ],
    )
    return out
