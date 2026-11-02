"""Offline support repair windows over existing directed travel edges.

Expand across an unsupported subdivision point, never across a junction,
parking access, direction change or lateral guide. No source node is moved.
"""

import numpy as np


def corridor(graph, key):
    edge = graph.edges[key]
    if edge["kind"] != "travel":
        return None
    nodes = [graph.nodes[edge[k]] for k in ("source", "target")]
    identities = [tuple(n.get(k) for k in ("road", "direction", "guide_offset_m")) for n in nodes]
    if identities[0] != identities[1] or identities[0][0] is None or identities[0][1] not in ("forward", "reverse"):
        return None
    return identities[0]


def validate_chain(graph, chain):
    if not 1 <= len(chain) <= 8 or len(set(chain)) != len(chain) or set(chain) - set(graph.edges):
        raise ValueError("One to eight distinct known edges required")
    if len(chain) == 1:
        return
    identity = corridor(graph, chain[0])
    if identity is None:
        raise ValueError("Travel corridor required")
    for i, key in enumerate(chain):
        if corridor(graph, key) != identity:
            raise ValueError("Repair cannot cross source corridor authority")
        if i and graph.edges[chain[i - 1]]["target"] != graph.edges[key]["source"]:
            raise ValueError("Connected directed repair chain required")
        if not np.all(graph.edges[key]["route"]["gear"] == 1):
            raise ValueError("Forward travel required")


def windows(graph, paths, blocked, support, *, max_edges=8, padding_edges=(0, 1, 2)):
    if type(max_edges) is not int or not 1 <= max_edges <= 8 or not 1 <= len(paths) <= 32:
        raise ValueError("Bounded support repair paths required")
    if (
        not 1 <= len(padding_edges) <= 4
        or len(set(padding_edges)) != len(padding_edges)
        or any(type(n) is not int or not 0 <= n <= 4 for n in padding_edges)
        or 0 not in padding_edges
    ):
        raise ValueError("Bounded approach padding required")
    blocked = set(blocked)
    if blocked - set(graph.edges):
        raise ValueError("Unknown support rejection")
    rows = []
    seen = set()
    for path in paths:
        if not path or len(path) > 10000 or set(path) - set(graph.edges):
            raise ValueError("Known bounded path required")
        for a, b in zip(path, path[1:], strict=False):
            if graph.edges[a]["target"] != graph.edges[b]["source"]:
                raise ValueError("Disconnected path")
        for i, key in enumerate(path):
            if key not in blocked:
                continue
            identity = corridor(graph, key)
            lo = hi = i
            if identity is None:
                chain = (key,)
            else:
                lo = hi = i

                def good(index, end):
                    node = graph.nodes[graph.edges[path[index]][end]]
                    return bool(support.supported(np.array([node["pose"][:2]]), "vehicle")[0])

                # Only expand when the actual connection point is unsupported.
                # A shared failed endpoint therefore gets bypassed as one curve.
                while (
                    hi - lo + 1 < max_edges
                    and lo > 0
                    and not good(lo, "source")
                    and corridor(graph, path[lo - 1]) == identity
                ):
                    lo -= 1
                while (
                    hi - lo + 1 < max_edges
                    and hi + 1 < len(path)
                    and not good(hi, "target")
                    and corridor(graph, path[hi + 1]) == identity
                ):
                    hi += 1
                chain = tuple(path[lo : hi + 1])
            variants = [chain]
            if identity is not None:
                for pad in sorted(padding_edges):
                    left, right = lo, hi
                    for _ in range(pad):
                        if left > 0 and right - left + 1 < max_edges and corridor(graph, path[left - 1]) == identity:
                            left -= 1
                        if (
                            right + 1 < len(path)
                            and right - left + 1 < max_edges
                            and corridor(graph, path[right + 1]) == identity
                        ):
                            right += 1
                    if good(left, "source") and good(right, "target"):
                        variants.append(tuple(path[left : right + 1]))
            for chain in variants:
                validate_chain(graph, chain)
                if chain not in seen:
                    seen.add(chain)
                    rows.append(list(chain))
    if len(rows) > 32:
        raise ValueError("Repair window budget exceeded")
    return rows


def repair_paths(graph, pair, blocked):
    """Choose actual graph paths with fewest rejected edges, then length.

    Selection is diagnostic only. A rejected path never becomes executable.
    Do not route through somebody else's parking berth as a shortcut.
    """
    import heapq

    blocked = set(blocked)
    if (
        len(pair) != 2
        or len(set(pair)) != 2
        or set(pair) - set(graph.nodes)
        or any(graph.nodes[n]["kind"] != "berth" for n in pair)
        or blocked - set(graph.edges)
    ):
        raise ValueError("Two known berths and bound support rejections required")
    results = []
    for source, target in (pair, pair[::-1]):
        best = {source: (0, 0.0)}
        previous = {}
        queue = [(0, 0.0, source)]
        while queue:
            count, distance, node = heapq.heappop(queue)
            if (count, distance) != best[node]:
                continue
            if node == target:
                break
            if node != source and graph.nodes[node]["kind"] == "berth":
                continue
            for key in sorted(graph.links[node]):
                edge = graph.edges[key]
                end = edge["target"]
                cost = (count + int(key in blocked), distance + float(edge["route"]["arc"][-1]))
                if cost < best.get(end, (float("inf"), float("inf"))):
                    best[end] = cost
                    previous[end] = key
                    heapq.heappush(queue, (*cost, end))
        path = []
        cursor = target
        if target in best:
            while cursor != source:
                key = previous[cursor]
                path.append(key)
                cursor = graph.edges[key]["source"]
            path.reverse()
        results.append(
            dict(
                source=source,
                target=target,
                path=path,
                cost=best.get(target),
                rejected_edges=[k for k in path if k in blocked],
            )
        )
    return results
