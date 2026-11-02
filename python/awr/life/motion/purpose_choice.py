"""Choose a nominated cycle only after bound direct-walking comparison.

The result is still a geometry-compilation candidate. Optimistic drive time is
never treated as actual travel time, and no resident or vehicle is assigned.
"""

import json
import math
from collections import Counter

from awr.life.civil.activity.access import _walk_path
from awr.life.common.schema import schema_id


def choose(selection, walks, places):
    if (
        not selection.get("purpose_screened")
        or schema_id(walks.get("schema")) != "awr.life.purpose_direct_walk_screen.v1"
    ):
        raise ValueError("Purpose nomination and direct-walk screen required")
    roster = {p["id"]: p for p in places}
    if len(roster) != len(places):
        raise ValueError("Duplicate place")
    pairs = {tuple(r["pair"]): r for r in selection["rows"]}
    seen = set()
    counts = Counter()
    candidates = []

    def canonical(r):
        return json.dumps(r, sort_keys=True, separators=(",", ":"))

    for row in walks["rows"]:
        pair = tuple(row["pair"])
        if pair in seen or pair not in pairs:
            raise ValueError("Unknown or duplicate screened pair")
        seen.add(pair)
        source = pairs[pair]
        options = source["purpose_screen"]["options"]
        if [canonical(o["nomination"]) for o in row["options"]] != [canonical(o) for o in options]:
            raise ValueError("Walking screen does not cover the exact nominated options")
        for item in row["options"]:
            option = item["nomination"]
            home = roster[option["home_place"]]
            costs = []
            targets = set()
            for attempt in item["walk_attempts"]:
                ident = attempt["place"]
                target = roster.get(ident)
                if (
                    ident in targets
                    or target is None
                    or target["kind"] != option["service_kind"]
                    or (target.get("group") or ident) != option["service_group"]
                ):
                    raise ValueError("Walk comparison changes the destination facility")
                targets.add(ident)
                outward = attempt["outward"]
                back = attempt["returning"]
                if outward is not None:
                    _walk_path(outward, home["xy"], target["xy"])
                if back is not None:
                    _walk_path(back, target["xy"], home["xy"])
                if outward is not None and back is not None:
                    costs.append(
                        (float(outward["arc"][-1] + back["arc"][-1]) / selection["purpose_policy"]["walk_mps"], ident)
                    )
            best = min(costs) if costs else None
            car = option["car_outward_lower_s"] + option["car_return_lower_s"]
            if not math.isfinite(car) or car <= 0:
                raise ValueError("Invalid car estimate")
            status = (
                "walking_not_certified"
                if best is None
                else "walk_dominates_optimistic_drive"
                if best[0] <= car + 1e-8
                else "requires_actual_drive_cost"
            )
            observed = item["best_walk_roundtrip_s"]
            if (observed is not None and not math.isfinite(observed)) or (
                status != item["status"]
                or item["car_roundtrip_lower_s"] != car
                or (observed is None) != (best is None)
                or (best and abs(observed - best[0]) > 1e-8)
            ):
                raise ValueError("Walk status/cost differs from its complete paths")
            counts[status] += 1
            if status == "requires_actual_drive_cost":
                candidates.append(
                    dict(
                        pair=list(pair),
                        option=option,
                        walk_roundtrip_s=best[0],
                        walk_service_place=best[1],
                        optimistic_drive_roundtrip_s=car,
                        optimistic_time_margin_s=best[0] - car,
                        required_turns=source["required_turns"],
                        support_rejected_edges=source["support_rejected_edges"],
                    )
                )
    candidates.sort(
        key=lambda c: (
            -c["optimistic_time_margin_s"],
            len(c["support_rejected_edges"]),
            len(c["required_turns"]),
            c["pair"],
            c["option"]["home_place"],
        )
    )
    return dict(
        schema="awr.life.purpose_cycle_choice.v1",
        candidate_only=True,
        admitted=False,
        executable=False,
        selected=candidates[0] if candidates else None,
        candidates=candidates,
        counts=dict(counts),
        ranking=[
            "Largest direct-walk time minus optimistic drive time",
            "Fewer support-rejected edges",
            "Fewer missing turns",
            "Stable identifiers",
        ],
        limitations=[
            "A positive margin is only a reason to attempt geometry compilation; measured driving may eliminate it.",
            "No assignment, physical transfer, second destination, capacity or recurring activity is certified.",
        ],
    )
