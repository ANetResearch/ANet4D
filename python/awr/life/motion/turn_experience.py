"""Version-bound failed proposals guide exploration; they never close a road."""

import hashlib

from awr.life.common.schema import schema_id


def deferred_proposals(reports, expected_bindings):
    """Caller supplies all current geometry/planner bindings, excluding workflow.

    A failed finite search is not an infeasibility proof. Its source/target
    pair remains in topology and can be revisited if alternatives are exhausted.
    """
    if not expected_bindings or not 1 <= len(reports) <= 32:
        raise ValueError("Bounded reports with explicit current bindings required")
    observations = []
    pairs = set()
    successful = set()
    for report in reports:
        if schema_id(report.get("schema")) != "awr.life.missing_source_turns.v1" or not report.get("topology", {}).get(
            "approach_port_search"
        ):
            raise ValueError("Bounded approach-search evidence required")
        for key, digest in expected_bindings.items():
            if report.get("input_sha256", {}).get(key) != digest:
                raise ValueError("Stale proposal geometry or planner: " + key)
        seen = set()
        for turn in report["required_turns"]:
            pair = (turn["source"], turn["target"])
            if pair in seen:
                raise ValueError("Duplicate turn evidence")
            seen.add(pair)
            identity = "source-turn-intent:" + hashlib.sha256("|".join(pair).encode()).hexdigest()[:24]
            attempts = [r for r in report["rows"] if r["chain"][1] == identity]
            if any(r.get("candidate_found") is True for r in attempts):
                successful.add(pair)
                continue
            searched = [r for r in attempts if "stats" in r]
            # Budget-skipped ports alone are not a failed proposal experience.
            if not searched:
                continue
            pairs.add(pair)
            observations.append(
                dict(
                    source=pair[0],
                    target=pair[1],
                    source_junction=turn["source_junction"],
                    searched_ports=len(searched),
                    attempted_ports=len(attempts),
                    max_expansions=sorted({r["max_expansions"] for r in searched}),
                    steering_fractions=sorted({r["steering_fraction"] for r in searched}),
                )
            )
    # A later or broader search may succeed. Positive evidence from any bound
    # report supersedes a finite failure, independent of input report order.
    pairs -= successful
    observations = [o for o in observations if (o["source"], o["target"]) in pairs]
    return dict(
        pairs=[list(p) for p in sorted(pairs)],
        observations=observations,
        excludes_topology=False,
        proves_infeasibility=False,
        scope="Prefer other proposals under unchanged geometry and planner; allow later revisits.",
    )
