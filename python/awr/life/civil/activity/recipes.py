"""Declarative activity demand; never supplies geometry or actor trajectories."""

import math

from .director import Activity


def load_activities(rows, available_modes, existing=()):
    """Load the grounded adapter's optional activity_engine.activities list.

    These replace the legacy facility-purpose defaults. Capabilities come from
    Place evidence, not this recipe. Route, parking, access and reservations are
    still checked by the existing Director and physical adapters.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("activities must be a nonempty list")
    required = {"purpose", "kinds", "dwell_s", "recurrence_s", "requires"}
    optional = {"weight", "modes", "readiness", "tour", "roles"}
    result, purposes = [], set(existing)

    def names(value, field):
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(v, str) or not v or v.strip() != v for v in value)
            or len(set(value)) != len(value)
        ):
            raise ValueError(f"Activity {field} must contain unique nonempty names")
        return tuple(value)

    def number(value, field):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Activity {field} must be finite numeric data")
        return float(value)

    for row in rows:
        if not isinstance(row, dict) or not required <= row.keys() or row.keys() - required - optional:
            raise ValueError("Activity recipe has missing or unsupported fields")
        purpose = names([row["purpose"]], "purpose")[0]
        if purpose in purposes or purpose == "return_home":
            raise ValueError("Activity purpose must be unique and not return_home")
        purposes.add(purpose)
        kinds = names(row["kinds"], "kinds")
        if set(kinds) & {"home", "berth"}:
            raise ValueError("Home and berth are engine resources, not recipe destinations")
        requires = names(row["requires"], "requires")
        modes = names(row.get("modes", ["walk"]), "modes")
        roles = names(row.get("roles", ["resident"]), "roles")
        if not set(modes) <= set(available_modes):
            raise ValueError("Activity modes require an available scene backend")
        dwell = row["dwell_s"]
        if not isinstance(dwell, list) or len(dwell) != 2:
            raise ValueError("Activity dwell_s must contain two bounds")
        dwell = tuple(number(v, "dwell_s") for v in dwell)
        recurrence = number(row["recurrence_s"], "recurrence_s")
        weight = number(row.get("weight", 1.0), "weight")
        readiness = number(row.get("readiness", 0.5), "readiness")
        tour = row.get("tour", False)
        if (
            not 0 < dwell[0] <= dwell[1]
            or recurrence <= 0
            or weight <= 0
            or not 0 < readiness <= 1
            or type(tour) is not bool
            or (tour and modes != ("car",))
        ):
            raise ValueError("Invalid activity timing, weight or tour mode")
        result.append(
            Activity(
                purpose,
                kinds,
                dwell,
                recurrence,
                weight=weight,
                roles=roles,
                modes=modes,
                readiness=readiness,
                requires=requires,
                tour=tour,
            )
        )
    return result
