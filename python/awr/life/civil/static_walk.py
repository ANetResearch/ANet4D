"""Compiled source trees for repeated activity queries on one static snapshot.

The activity director often checks many facilities from the same position.
One SciPy Dijkstra tree supplies their grid paths, avoiding a separate Python
A* and component labelling for each destination. Dynamic local avoidance still
uses WalkNetwork.plan. Every result retains the exact connector/body/turning
checks in WalkNetwork.finish_grid_path; this is not a motion controller.
"""

import time
from collections import OrderedDict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


class StaticWalkRouter:
    def __init__(self, network, free, *, cache_size=8, stats=None):
        if not isinstance(cache_size, int) or cache_size < 1:
            raise ValueError("Positive static source-tree cache size required")
        if np.shape(free) != network.free.shape or np.any(np.asarray(free, bool) & ~network.free):
            raise ValueError("Static planning snapshot must be a subset of the walking network")
        self.network = network
        # Preserve exact body metadata. np.ndarray subclass copying does not
        # automatically retain attributes on the existing ObstacleGrid class.
        self.free = free.copy()
        for name in ("positions", "sizes", "yaws", "radius", "circle_radii", "static_free", "turn_only"):
            if hasattr(free, name):
                value = getattr(free, name)
                setattr(self.free, name, value.copy() if isinstance(value, np.ndarray) else value)
        self.free.flags.writeable = False
        self.cache_size = cache_size
        self.stats = stats if stats is not None else {}
        self.trees = OrderedDict()
        began = time.perf_counter()
        mask = np.asarray(self.free, bool)
        self.cells = np.argwhere(mask)
        self.index = np.full(mask.shape, -1, np.int32)
        self.index[mask] = np.arange(len(self.cells), dtype=np.int32)
        weights = np.ones(mask.shape) if network.weight is None else np.asarray(network.weight, np.float64)
        if not np.isfinite(weights[mask]).all() or np.any(weights[mask] < 1.0):
            raise ValueError("Finite positive walking costs required")
        rows, cols, values = [], [], []
        for dy, dx in ((0, 1), (1, 0)):
            h, w = mask.shape[0] - dy, mask.shape[1] - dx
            adjacent = mask[:h, :w] & mask[dy:, dx:]
            rows.append(self.index[:h, :w][adjacent])
            cols.append(self.index[dy:, dx:][adjacent])
            values.append((weights[:h, :w][adjacent] + weights[dy:, dx:][adjacent]) / 2)
        self.graph = csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(self.cells), len(self.cells)),
        )
        _, self.components = connected_components(self.graph, directed=False)
        self.stats["static_graph_builds"] = self.stats.get("static_graph_builds", 0) + 1
        self.stats["static_graph_wall_s"] = self.stats.get("static_graph_wall_s", 0.0) + time.perf_counter() - began

    def plan(self, start, goal):
        network, free = self.network, self.free
        target = network.goal_cell(goal, free)
        if target is None:
            return None
        distances, indices = network.tree.query(start, k=min(24, len(network.xy)))
        target_index = int(self.index[target])
        candidates = [
            (d, int(i))
            for d, i in zip(np.atleast_1d(distances), np.atleast_1d(indices), strict=False)
            if d <= 1.5
            and free[tuple(network.nodes[i])]
            and self.components[self.index[tuple(network.nodes[i])]] == self.components[target_index]
            and network.clear_segment(start, network.xy[i], free)
        ]
        if not candidates:
            return None
        source = tuple(network.nodes[candidates[0][1]])
        if network.labels[source] != network.labels[target]:
            return None
        key = int(self.index[source])
        # Costs are symmetric on this undirected snapshot. An outward proof
        # already cached at the requested goal also proves grid reachability
        # for the return leg. Reuse that tree, then independently run every
        # exact connector/body/turn check in the requested travel direction.
        reverse = key not in self.trees and target_index in self.trees
        tree_key = target_index if reverse else key
        if tree_key not in self.trees:
            began = time.perf_counter()
            self.trees[tree_key] = dijkstra(self.graph, directed=False, indices=tree_key, return_predecessors=True)
            self.stats["static_source_searches"] = self.stats.get("static_source_searches", 0) + 1
            self.stats["static_source_wall_s"] = (
                self.stats.get("static_source_wall_s", 0.0) + time.perf_counter() - began
            )
            if len(self.trees) > self.cache_size:
                self.trees.popitem(last=False)
        else:
            self.stats["static_source_cache_hits"] = self.stats.get("static_source_cache_hits", 0) + 1
            self.trees.move_to_end(tree_key)
            if reverse:
                self.stats["static_reverse_tree_hits"] = self.stats.get("static_reverse_tree_hits", 0) + 1
        distances, previous = self.trees[tree_key]
        other = key if reverse else target_index
        if not np.isfinite(distances[other]):
            return None
        cells = [tuple(self.cells[other])]
        cursor = other
        while cursor != tree_key:
            cursor = int(previous[cursor])
            if cursor < 0:
                raise RuntimeError("Compiled static path lost its source")
            cells.append(tuple(self.cells[cursor]))
        if reverse:
            cells.reverse()
        return network.finish_grid_path(start, goal, cells, free)
