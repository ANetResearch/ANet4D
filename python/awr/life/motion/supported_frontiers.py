"""Spend bounded junction repair work on supported source ports first.

Nearest-only nomination can repeatedly spend every family slot on unsupported
endpoints while ignoring usable mapped alternatives. This is a priority change,
not a physical admission or deletion of unsupported-but-repairable approaches.
"""

import numpy as np

from .missing_turns import source_turn_pairs


def select(graph, roads, nomination, support, *, max_families=4):
    if type(max_families) is not int or not 1 <= max_families <= 8:
        raise ValueError("Bounded family budget required")
    authority = {(p["source"], p["target"]): p for p in source_turn_pairs(graph, roads)}
    nodes = sorted({n for p in authority.values() for n in (p["source"], p["target"])})
    values = (
        np.asarray(support.supported(np.array([graph.nodes[n]["pose"][:2] for n in nodes]), "vehicle"))
        if nodes
        else np.empty(0, dtype=bool)
    )
    if values.shape != (len(nodes),) or values.dtype != np.dtype(bool):
        raise ValueError("Boolean measured support for every source port required")
    supported = dict(zip(nodes, values.tolist(), strict=False))

    def key(p):
        return p["source"], p["target"]

    def ok(p):
        return supported[p["source"]] and supported[p["target"]]

    def distance(p):
        road = roads[p["roads"][0]]
        j = road["line"][0 if road["source_from"] == p["source_junction"] else -1]
        return sum(float(np.linalg.norm(graph.nodes[p[k]]["pose"][:2] - j)) for k in ("source", "target"))

    ranked = []
    for row in nomination["ranked_families"]:
        if not row["pairs"] or any(authority.get(key(p)) != p for p in row["pairs"]):
            raise ValueError("Nomination must retain exact source-authorized ports")
        family = tuple(row["family"])
        if any((p["source_junction"], *p["roads"]) != family for p in row["pairs"]):
            raise ValueError("Nomination family differs from its source ports")
        pairs = sorted(row["pairs"], key=lambda p: (not ok(p), distance(p), *key(p)))
        ranked.append(dict(row, pairs=pairs, supported_pairs=sum(ok(p) for p in pairs)))
    ranked.sort(key=lambda r: (not bool(r["supported_pairs"]), -r["actor_count"], r["family"]))
    selected = []
    families = set()
    for row in ranked:
        junction, a, b = row["family"]
        family = (junction, tuple(sorted((a, b))))
        if family in families:
            continue
        if len(families) >= max_families:
            break
        selected.append(row["pairs"][0])
        families.add(family)
        reverse = [p for p in authority.values() if p["source_junction"] == junction and p["roads"] == [b, a]]
        if reverse:
            selected.append(min(reverse, key=lambda p: (not ok(p), distance(p), *key(p))))
    return dict(
        nomination,
        selected_pairs=selected,
        ranked_families=ranked,
        max_families=max_families,
        endpoint_precheck=dict(
            nodes_checked=len(nodes),
            selected_pairs_supported=sum(ok(p) for p in selected),
            families_with_supported_alternatives=sum(bool(r["supported_pairs"]) for r in ranked),
        ),
        candidate_only=True,
        release_evidence=False,
        limitations=nomination.get("limitations", [])
        + [
            "Measured centre support changes offline priorities only; body, full-route support and native checks remain required.",
            "Unsupported families remain fallback opportunities; no new direction, road width or motion permission.",
        ],
    )
