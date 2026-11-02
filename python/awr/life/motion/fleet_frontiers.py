"""Bounded, source-authorized junction repair priorities for a whole fleet.

Ranking uses topology to find where many original sources are stranded. It is
not a motion permission or capacity proof. Physical search remains mandatory.
"""

import hashlib
from collections import defaultdict

import numpy as np

from .berth_reachability import road_tree
from .missing_turns import source_turn_pairs
from .support_chains import corridor, validate_chain


def nominate(graph, roads, origins, *, max_families=4):
    if type(max_families) is not int or not 1 <= max_families <= 8:
        raise ValueError("Bounded source-junction family budget required")
    if not 1 <= len(origins) <= 120 or len({r["actor"] for r in origins}) != len(origins):
        raise ValueError("Bounded unique original fleet required")
    reached_by = defaultdict(set)
    for row in origins:
        reached, _ = road_tree(graph, row["berth"], ())
        for node in reached:
            reached_by[node].add(row["actor"])
    groups = defaultdict(list)
    for pair in source_turn_pairs(graph, roads):
        groups[(pair["source_junction"], *pair["roads"])].append(pair)

    def distance(pair):
        road = roads[pair["roads"][0]]
        junction = road["line"][0 if road["source_from"] == pair["source_junction"] else -1]
        return sum(float(np.linalg.norm(graph.nodes[pair[end]]["pose"][:2] - junction)) for end in ("source", "target"))

    ranked = []
    for key, pairs in groups.items():
        actors = sorted(set().union(*(reached_by[p["source"]] for p in pairs)))
        if actors:
            ranked.append(
                dict(
                    family=key,
                    actors=actors,
                    actor_count=len(actors),
                    pairs=sorted(pairs, key=lambda p: (distance(p), p["source"], p["target"])),
                )
            )
    ranked.sort(key=lambda r: (-r["actor_count"], r["family"]))
    selected = []
    families = set()
    for row in ranked:
        key = row["family"]
        family = (key[0], tuple(sorted(key[1:])))
        if family in families:
            continue
        if len(families) >= max_families:
            break
        # Preserve one explicitly authorized reverse opportunity when present.
        # A one-way road cannot acquire a reverse direction from this pairing.
        selected.append(row["pairs"][0])
        families.add(family)
        reverse = (key[0], key[2], key[1])
        if reverse in groups:
            selected.append(min(groups[reverse], key=lambda p: (distance(p), p["source"], p["target"])))
    return dict(
        selected_pairs=selected,
        ranked_families=ranked,
        candidate_only=True,
        release_evidence=False,
        max_families=max_families,
        limitations=["Topology-only priorities; occupancy, support and native capacity are not inferred."],
    )


def approach_chains(graph, roads, pairs, *, distance_m=20.0, max_edges=8):
    if not 1 <= len(pairs) <= 16 or not 0 < distance_m <= 20 or type(max_edges) is not int or not 1 <= max_edges <= 8:
        raise ValueError("Bounded source approaches required")
    authority = {(p["source"], p["target"]): p for p in source_turn_pairs(graph, roads)}
    if len({(p["source"], p["target"]) for p in pairs}) != len(pairs):
        raise ValueError("Duplicate source intent")
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for key, edge in graph.edges.items():
        if edge["kind"] == "travel" and np.all(edge["route"]["gear"] == 1):
            incoming[edge["target"]].append(key)
            outgoing[edge["source"]].append(key)

    def chain_at(node, backwards):
        links = incoming if backwards else outgoing
        chain = []
        visited = {node}
        identity = None
        length = 0.0
        while len(chain) < max_edges and length < distance_m:
            choices = [
                k
                for k in links[node]
                if corridor(graph, k) is not None
                and (identity is None or corridor(graph, k) == identity)
                and graph.edges[k]["source" if backwards else "target"] not in visited
            ]
            if not choices:
                break
            key = min(choices, key=lambda k: (-float(graph.edges[k]["route"]["arc"][-1]), k))
            chain.append(key)
            identity = corridor(graph, key)
            length += float(graph.edges[key]["route"]["arc"][-1])
            node = graph.edges[key]["source" if backwards else "target"]
            visited.add(node)
        if backwards:
            chain.reverse()
        if not chain:
            raise ValueError("Source endpoint has no directed approach")
        validate_chain(graph, chain)
        return chain

    chains = []
    intents = {}
    for row in pairs:
        pair = (row["source"], row["target"])
        if pair not in authority:
            raise ValueError("Missing explicit source turn authority")
        key = "fleet-turn-intent:" + hashlib.sha256("|".join(pair).encode()).hexdigest()[:24]
        intents[key] = authority[pair]
        chains.append([*chain_at(pair[0], True), key, *chain_at(pair[1], False)])
    return chains, intents
