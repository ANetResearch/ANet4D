"""Deterministic outdoor walking routes on supported roadside corridors.

Roadside strips are a conservative *derived* pedestrian map, not surveyed
sidewalks. In the default model road interiors are forbidden except for
supplied crossing polygons. The shared-street model (`shared_streets`) also
admits the carriageway, as on residential streets without a continuous
sidewalk, at a higher path cost that grows towards the road centre, so that
routes keep to sidewalks and road edges and cross roads directly. Either way
the grid admits a rotating body disk, not just an actor centre, on measured
support with the same clearance tests. Missing support and disconnected
components remain blocked; no nearest-point teleport is used.
"""

import math
from collections import OrderedDict

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

# Planning keeps strictly more distance from bodies than the per-tick lookahead
# (`prediction.blocked_disks`, threshold radius+.08 at .025 m samples). Connector
# samples are .1 m apart and distance-to-body is 1-Lipschitz, so a .06 m margin
# guarantees that every point of an accepted connector stays beyond the lookahead
# threshold. Without it a route can graze a standing body exactly at the
# threshold, pass planning and then be refused in execution indefinitely.
PLANNING_MARGIN_M = 0.06


class ObstacleGrid(np.ndarray):
    """Raster candidates plus exact oriented bodies for connector checks."""


def merge_obstacles(target, addition):
    """Intersect candidate masks AND retain every exact physical obstacle.

    Updating only the Boolean mask lets string-pulling cross the newly parked
    body: clear_segment deliberately uses exact bodies rather than interpreting
    dynamic node masks as solid cells. Both representations must change together.
    """
    if not isinstance(target, ObstacleGrid) or not isinstance(addition, ObstacleGrid) or target.shape != addition.shape:
        raise ValueError("Compatible exact-obstacle grids are required")
    target[...] = np.asarray(target) & np.asarray(addition)
    for name in ("positions", "sizes", "yaws", "circle_radii"):
        setattr(target, name, np.concatenate([getattr(target, name), getattr(addition, name)]))
    target.radius = max(target.radius, addition.radius)
    target.static_free = np.asarray(target.static_free) & np.asarray(addition.static_free)
    return target


def restrict(free, mask):
    """`free & mask`, keeping the exact bodies of an ObstacleGrid; marked as turning cells only."""
    out = (np.asarray(free, bool) & mask).view(ObstacleGrid)
    empty = dict(
        positions=np.zeros((0, 2)), sizes=np.zeros((0, 3)), yaws=np.zeros(0), radius=0.0, circle_radii=np.zeros(0)
    )
    for name in ("positions", "sizes", "yaws", "radius", "circle_radii"):
        setattr(out, name, getattr(free, name, empty[name]))
    out.static_free = np.asarray(getattr(free, "static_free", free), bool) & mask
    out.turn_only = True
    return out


def polyline(points, name):
    p = np.asarray(points, float)
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-7]
    p = p[keep]
    if len(p) < 2:
        return None
    return {"road": name, "points": p, "arc": np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]}


class WalkNetwork:
    field_cache = 96
    # A shortcut may exceed the highest cost of the grid path it replaces by
    # this much (about 1 m deeper into a carriageway with the default costs).
    shortcut_tolerance = 0.0

    def __init__(self, free, origin, step=0.5, weight=None, search_backend="auto"):
        from awr.life.civil.grid_search import implementation

        self.search_backend = implementation(search_backend)[1]
        self.free = np.asarray(free, bool)
        # Optional per-cell path cost (>= 1). None keeps unit costs.
        self.weight = None if weight is None else np.asarray(weight, np.float32)
        self._fields = OrderedDict()
        self._graphs = {}
        self.origin = np.asarray(origin, float)
        self.step = step
        self.labels, _ = label(self.free)  # Four-neighbour edges, no corner cutting.
        from scipy.ndimage import find_objects

        self.component_slices = find_objects(self.labels)
        self.nodes = np.argwhere(self.free)
        self.xy = self.origin + self.nodes[:, ::-1] * step
        self.tree = cKDTree(self.xy)
        counts = np.bincount(self.labels.ravel())
        self.eligible = np.flatnonzero(counts[self.labels[self.free]] >= 80)

    @classmethod
    def from_scene(cls, surface, semantics, radius, step=0.5, lane_network=None, shared_streets=None, turn_radius=None):
        """Walking cells for a body of planning `radius` (body disk or, for walkers that pass each
        other by body boxes, the half width). With `turn_radius` (the turning disk) the network also
        records `turn_free`: cells where the body can rotate in place; routes only turn sharply there."""
        import shapely
        from shapely.geometry import LineString, Polygon
        from shapely.ops import unary_union

        roads = [LineString(r["line"]).buffer(r["width_m"] / 2) for r in semantics["roads"]]
        outer = [LineString(r["line"]).buffer(r["width_m"] / 2 + 4) for r in semantics["roads"]]
        crossings = [LineString(c["ends"]).buffer(c["width_m"] / 2) for c in semantics["crossings"]]
        # No inferred crossing at a junction. Only the supplied crossing geometry
        # can open the road-interior exclusion.
        public_areas = [
            Polygon(a["polygon"])
            for a in semantics.get("open_areas", [])
            if a.get("kind") in ("park", "square") and a.get("polygon")
        ]
        # Resident drivers need a supported walking route from a real door to
        # their parked car. The former adapter omitted verified paved parking
        # interiors even when the measured person surface admitted them.
        parking_areas, parking_aisles, parking_ids = [], [], []
        if shared_streets:
            for lot in semantics.get("verified_parking_regions", []):
                ident = lot.get("id")
                if (
                    not isinstance(ident, str)
                    or not ident
                    or ident in parking_ids
                    or lot.get("verified") is not True
                    or not isinstance(lot.get("evidence"), str)
                    or not lot["evidence"].strip()
                ):
                    raise ValueError("Shared parking access requires unique surveyed regions")
                boundary = Polygon(lot["polygon"])
                if boundary.is_empty or not boundary.is_valid or not lot.get("aisles") or not lot.get("entrances"):
                    raise ValueError("Parking walking access needs a valid boundary and mapped aisles/entrances")
                passages = [Polygon(q) for q in [*lot["aisles"], *lot["entrances"]]]
                if any(p.is_empty or not p.is_valid or not boundary.buffer(1e-7).covers(p) for p in passages):
                    raise ValueError("Parking walking passages leave their surveyed boundary")
                parking_ids.append(ident)
                parking_areas.append(boundary)
                parking_aisles.extend(passages)
            corridor = unary_union(outer + public_areas + parking_areas).union(unary_union(crossings))
        else:
            corridor = unary_union(outer + public_areas).difference(unary_union(roads)).union(unary_union(crossings))
        if lane_network is not None:
            corridor = corridor.union(lane_network.walking)
        clearance = radius + step / math.sqrt(2) + 0.08
        area = corridor
        corridor = corridor.buffer(-clearance)
        shapely.prepare(corridor)
        y, x = np.mgrid[: int(surface.height_m / step), : int(surface.width_m / step)]
        origin = np.array([surface.x0, surface.y0]) + step / 2
        xy = origin + np.stack([x, y], -1).reshape(-1, 2) * step
        legal = shapely.intersects(corridor, shapely.points(xy))
        ids = np.flatnonzero(legal)
        supported = surface.supported(xy[ids], "person")
        for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            supported &= surface.supported(xy[ids] + clearance * np.array([np.cos(angle), np.sin(angle)]), "person")
        legal[ids] &= supported
        turn = None
        if turn_radius is not None and turn_radius > radius:
            turn_clear = turn_radius + step / math.sqrt(2) + 0.08
            inner = corridor.buffer(-(turn_clear - clearance))
            shapely.prepare(inner)
            ids = np.flatnonzero(legal)
            ok = shapely.intersects(inner, shapely.points(xy[ids]))
            for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                ok &= surface.supported(xy[ids] + turn_clear * np.array([np.cos(angle), np.sin(angle)]), "person")
            turn = np.zeros(len(xy), bool)
            turn[ids] = ok
            turn = turn.reshape(x.shape)
            # Cells where the body only passes (cannot rotate) are kept within
            # `pass_only_reach_m` of a turning cell: squeezing past a parked car or
            # a narrowing is fine, a long single-file passage would trap two
            # walkers who meet in it (neither can turn back).
            from scipy.ndimage import distance_transform_edt as _edt

            reach = float((shared_streets or {}).get("pass_only_reach_m", 4.0))
            legal &= (_edt(~turn) * step <= reach).ravel()
        weight = None
        carriageway = None
        if shared_streets:
            from scipy.ndimage import distance_transform_edt

            # Carriageway: mapped road bodies and SUMO driving lanes, minus
            # crossings and SUMO pedestrian areas. Cost grows linearly with the
            # depth into the carriageway up to `centre_depth_m`.
            road_body = unary_union(roads + parking_aisles)
            if lane_network is not None:
                road_body = road_body.union(lane_network.driving).difference(lane_network.walking)
            road_body = road_body.difference(unary_union(crossings))
            shapely.prepare(road_body)
            carriageway = np.zeros(len(xy), bool)
            for lo in range(0, len(xy), 400000):
                carriageway[lo : lo + 400000] = shapely.intersects(road_body, shapely.points(xy[lo : lo + 400000]))
            carriageway = carriageway.reshape(x.shape)
            depth = distance_transform_edt(carriageway) * step
            base = float(shared_streets.get("carriageway_cost", 2.0))
            centre = float(shared_streets.get("carriageway_centre_cost", 1.0))
            reach = float(shared_streets.get("centre_depth_m", 3.0))
            weight = np.where(carriageway, base + centre * np.minimum(depth, reach) / reach, 1.0).astype(np.float32)
        network = cls(legal.reshape(x.shape), origin, step, weight=weight)
        if shared_streets:
            network.shortcut_tolerance = float(shared_streets.get("shortcut_cost_tolerance", 0.35))
        network.radius = radius
        network.corridor = corridor
        network.area = area  # the walking area before the planning clearance (local motion checks body support itself)
        network.carriageway = carriageway
        # Travel cost is not permission to dwell. Crossings and pedestrian
        # overlays intentionally have cheap walking cost even where vehicles
        # pass; subtracting them above must not create homes/waiting slots.
        no_stay = unary_union(roads + crossings + parking_aisles)
        if lane_network is not None:
            no_stay = no_stay.union(lane_network.driving)
        stay_margin = max(radius, turn_radius or radius) + step / math.sqrt(2)
        no_stay = no_stay.buffer(stay_margin)
        shapely.prepare(no_stay)
        standing = network.free.copy()
        for lo in range(0, len(network.xy), 400000):
            nodes = network.nodes[lo : lo + 400000]
            standing[nodes[:, 0], nodes[:, 1]] &= ~shapely.intersects(
                no_stay, shapely.points(network.xy[lo : lo + 400000])
            )
        network.standing_free = standing
        network.standing_policy = dict(
            version="separate_passage_and_stay.v1",
            body_margin_m=stay_margin,
            excludes=["mapped_carriageways", "crossings", "parking_aisles_and_entrances", "motor_lanes"],
            note="Candidate long-stay area; live body, circulation, resource and semantic checks still apply.",
        )
        network.parking_walk_regions = tuple(parking_ids)
        network.turn_free = turn
        network.surface = surface
        network.turn_body = None  # (length, width) of the largest walking body; set by the builder
        network.turn_clear = None  # cells clear of street objects and parked vehicles by the turning disk
        return network

    def _component_graph(self, component):
        graph = self._graphs.get(component)
        if graph is None:
            from scipy.sparse import csr_matrix

            region = self.component_slices[component - 1]
            mask = self.labels[region] == component
            index = np.full(mask.shape, -1, np.int32)
            index[mask] = np.arange(int(mask.sum()), dtype=np.int32)
            w = self.weight[region]
            rows, cols, costs = [], [], []
            for dy, dx in ((0, 1), (1, 0)):
                h, v = mask.shape[0] - dy, mask.shape[1] - dx
                both = mask[:h, :v] & mask[dy:, dx:]
                rows.append(index[:h, :v][both])
                cols.append(index[dy:, dx:][both])
                costs.append((w[:h, :v][both] + w[dy:, dx:][both]) / 2)
            n = int(mask.sum())
            graph = (
                region,
                index,
                csr_matrix((np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)),
            )
            self._graphs[component] = graph
        return graph

    def distance_field(self, cell):
        """Exact static path cost to `cell` over its static component (Dijkstra).

        Every free mask used for planning is a subset of the static network, so
        the field is an admissible and consistent A* heuristic for any of them.
        """
        cell = (int(cell[0]), int(cell[1]))
        field = self._fields.get(cell)
        if field is not None:
            self._fields.move_to_end(cell)
            return field
        component = int(self.labels[cell])
        if component == 0:
            return None
        from scipy.sparse.csgraph import dijkstra

        region, index, graph = self._component_graph(component)
        local = index[cell[0] - region[0].start, cell[1] - region[1].start]
        field = (region, index, dijkstra(graph, directed=False, indices=int(local)).astype(np.float32))
        self._fields[cell] = field
        if len(self._fields) > self.field_cache:
            self._fields.popitem(last=False)
        return field

    def cell(self, p):
        return np.floor((np.asarray(p) - self.origin) / self.step + 0.5).astype(int)[::-1]

    def clear_segment(self, a, b, free, max_weight=None):
        # Exact grid-boundary traversal. Uniform samples can miss arbitrarily
        # short intersections at a forbidden cell's corner.
        q = (np.asarray(a) - self.origin) / self.step
        end = (np.asarray(b) - self.origin) / self.step
        delta = end - q
        geometry = getattr(free, "static_free", free)
        from awr.life.civil.grid_search import compiled_segment_check

        check = compiled_segment_check()
        if check is not None:
            # Same sample times, offsets, rounding and cost bound as below, compiled.
            weight = self.weight if max_weight is not None and self.weight is not None else np.empty((0, 0), np.float32)
            if not check(
                float(q[0]),
                float(q[1]),
                float(end[0]),
                float(end[1]),
                np.asarray(geometry, np.bool_),
                weight,
                math.inf if max_weight is None else float(max_weight),
            ):
                return False
        elif not self._grid_segment_clear(q, end, delta, free, geometry, max_weight):
            return False
        return self._bodies_clear(a, b, free)

    def _grid_segment_clear(self, q, end, delta, free, geometry, max_weight):
        times = [0.0, 1.0]
        for axis in (0, 1):
            if abs(delta[axis]) < 1e-12:
                continue
            lo, hi = sorted([q[axis], end[axis]])
            boundaries = np.arange(math.floor(lo - 0.5) + 1, math.ceil(hi - 0.5)) + 0.5
            times.extend(((boundaries - q[axis]) / delta[axis]).tolist())
        times = np.unique(times)
        times = np.r_[times, (times[:-1] + times[1:]) / 2]
        p = q + times[:, None] * delta
        p = (p[:, None, :] + np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * 1e-8).reshape(-1, 2)
        rc = np.floor(p + 0.5).astype(int)[:, ::-1]
        inside = (rc >= 0).all(1) & (rc < np.array(free.shape)).all(1)
        # Dynamic cells describe candidate node centres, not solid squares.
        # An exact, collision-free pose may lie in a cell whose centre is too
        # close to a person. Validate its connector against static cells and
        # exact bodies, instead of trapping it inside a raster artefact.
        if not (inside.all() and geometry[rc[:, 0], rc[:, 1]].all()):
            return False
        if (
            max_weight is not None
            and self.weight is not None
            and float(self.weight[rc[:, 0], rc[:, 1]].max()) > max_weight + 1e-6
        ):
            return False
        return True

    def _bodies_clear(self, a, b, free):
        if isinstance(free, ObstacleGrid) and len(free.positions):
            from awr.life.civil.grid_search import compiled_bodies_check

            check = compiled_bodies_check()
            if check is None:
                return self._bodies_clear_numpy(a, b, free)
            # Same culling, samples, distances and release rule as below, compiled. The obstacle arrays are only
            # ever replaced (obstacle_mask, merge_obstacles), so cos/sin are kept with the yaw array they belong to.
            a, b = np.asarray(a, float), np.asarray(b, float)
            trig = getattr(free, "yaw_trig", None)
            if trig is None or trig[0] is not free.yaws:
                trig = free.yaw_trig = (free.yaws, np.cos(free.yaws), np.sin(free.yaws))
            return check(
                float(a[0]),
                float(a[1]),
                float(b[0]),
                float(b[1]),
                float(np.linalg.norm(b - a)),
                np.ascontiguousarray(free.positions, np.float64),
                np.ascontiguousarray(free.sizes, np.float64),
                trig[1],
                trig[2],
                np.ascontiguousarray(free.circle_radii, np.float64),
                float(free.radius),
                float(free.radius + 0.08 + PLANNING_MARGIN_M),
            )
        return True

    def _bodies_clear_numpy(self, a, b, free):
        if isinstance(free, ObstacleGrid) and len(free.positions):
            a, b = np.asarray(a), np.asarray(b)
            length = np.linalg.norm(b - a)
            limit = free.radius + 0.08 + PLANNING_MARGIN_M
            # A segment capsule is much tighter than the old midpoint ball on
            # long errands. Cull only bodies whose enclosing disk cannot reach
            # any point of the segment, retaining the FULL planning margin.
            # Round bodies may exceed their nominal rectangular asset size.
            body_radius = np.where(
                np.isfinite(free.circle_radii), free.circle_radii, np.linalg.norm(free.sizes[:, :2], axis=1) / 2
            )
            along = np.clip((free.positions - a) @ (b - a) / max(float(length * length), 1e-24), 0.0, 1.0)
            closest = a + along[:, None] * (b - a)
            nearby = np.linalg.norm(free.positions - closest, axis=1) <= body_radius + limit + 1e-9
            if not nearby.any():
                return True
            positions, sizes, yaws = free.positions[nearby], free.sizes[nearby], free.yaws[nearby]
            # Distance-to-box is 1-Lipschitz. The planning margin exceeds half
            # the .1 m sample spacing, so gaps cannot hide a lookahead refusal.
            points = a + np.linspace(0, 1, max(2, math.ceil(length / 0.1) + 1))[:, None] * (b - a)
            delta = points[:, None, :] - positions
            c, s = np.cos(yaws), np.sin(yaws)
            local = np.abs(
                np.stack([delta[..., 0] * c + delta[..., 1] * s, -delta[..., 0] * s + delta[..., 1] * c], -1)
            )
            distances = np.linalg.norm(np.maximum(local - sizes[None, :, :2] / 2, 0), axis=-1)
            circles = free.circle_radii[nearby]
            round_ids = np.isfinite(circles)
            distances[:, round_ids] = np.linalg.norm(delta[:, round_ids], axis=-1) - circles[round_ids]
            # A connector that starts inside a body's planning margin (two
            # walkers who stopped face to face, legally, closer than planning
            # would bring them) may leave it: towards that body the distance
            # must never decrease along the segment. Without this nobody in the
            # margin can plan any route, including stepping aside or back.
            # A start in actual contact (within the planning radius itself) is
            # never released this way.
            inside = distances[0] <= limit
            if np.any(distances[:, ~inside] <= limit) or np.any(distances[0, inside] <= free.radius):
                return False
            if inside.any() and np.any(distances[:, inside] < distances[0, inside] - 1e-6):
                return False
        return True

    def obstacle_mask(self, positions, sizes, radius, yaws=None, min_separation=0.0, circle_radii=None):
        free = self.free.copy().view(ObstacleGrid)
        # Moving pedestrian is a turning disk; obstacles retain their oriented
        # full body boxes. A long bus must not block an entire nearby sidewalk
        # merely because its circumscribed circle extends several metres aside.
        if yaws is None:
            yaws = np.zeros(len(positions))
        free.positions = np.asarray(positions, float).reshape(-1, 2)
        free.sizes = np.asarray(sizes, float).reshape(-1, 3)
        free.yaws = np.asarray(yaws, float)
        free.radius = radius
        free.static_free = self.free
        free.circle_radii = np.full(len(positions), np.nan) if circle_radii is None else np.asarray(circle_radii, float)
        for p, size, yaw, circle in zip(positions, sizes, yaws, free.circle_radii, strict=False):
            r = max(min_separation, radius + np.linalg.norm(size[:2]) / 2 + 0.08 + PLANNING_MARGIN_M)
            rc = self.cell(p)
            cells = math.ceil(r / self.step)
            lo = np.maximum(0, rc - cells)
            hi = np.minimum(free.shape, rc + cells + 1)
            if (hi <= lo).any():
                continue
            y, x = np.mgrid[lo[0] : hi[0], lo[1] : hi[1]]
            points = self.origin + np.stack([x, y], -1) * self.step
            delta = points - p
            c, s = np.cos(yaw), np.sin(yaw)
            local = np.abs(
                np.stack([delta[..., 0] * c + delta[..., 1] * s, -delta[..., 0] * s + delta[..., 1] * c], -1)
            )
            distance = np.linalg.norm(np.maximum(local - np.asarray(size[:2]) / 2, 0), axis=-1)
            if np.isfinite(circle):
                distance = np.linalg.norm(delta, axis=-1) - circle
            free[lo[0] : hi[0], lo[1] : hi[1]] &= (distance > radius + 0.08 + PLANNING_MARGIN_M) & (
                np.linalg.norm(delta, axis=-1) >= min_separation
            )
        return free

    def goal_cell(self, goal, free, reach_m=1.5):
        """The grid cell a route to `goal` ends in, or None.

        The goal's own cell when `free` holds it. A goal whose cell only the clearance of a body closes (the cell is
        on the static geometry; the exact goal may still be clear of every body, e.g. the door side of a parked car,
        whose clearance closes cells at grid resolution) ends at the nearest free cell within reach_m joined to the
        goal by a clear segment, the same connector rule as at the start of a route. The route's last segment then
        passes the exact geometry and body checks (finish_grid_path)."""
        target = tuple(self.cell(goal))
        if not (0 <= target[0] < free.shape[0] and 0 <= target[1] < free.shape[1]):
            return None
        if free[target]:
            return target
        geometry = getattr(free, "static_free", None)
        if geometry is None or not geometry[target]:
            return None
        distances, indices = self.tree.query(np.asarray(goal, float), k=min(24, len(self.xy)))
        for d, i in zip(np.atleast_1d(distances), np.atleast_1d(indices), strict=False):
            node = tuple(self.nodes[int(i)])
            if d <= reach_m and free[node] and self.clear_segment(self.xy[int(i)], goal, free):
                return node
        return None

    def plan(self, start, goal, free=None):
        free = self.free if free is None else free
        target = self.goal_cell(goal, free)
        if target is None:
            return None
        # A connector must start at the exact current pose and be fully legal.
        distances, indices = self.tree.query(start, k=min(24, len(self.xy)))
        candidates = [
            (d, int(i))
            for d, i in zip(np.atleast_1d(distances), np.atleast_1d(indices), strict=False)
            if d <= 1.5 and free[tuple(self.nodes[i])] and self.clear_segment(start, self.xy[i], free)
        ]
        if not candidates:
            return None
        component = int(self.labels[target])
        candidates = [(d, i) for d, i in candidates if self.labels[tuple(self.nodes[i])] == component]
        if not candidates:
            return None
        # Reject a dynamically cut component in C before Python A* explores
        # every cell. This changes neither connectivity nor accepted paths.
        region = self.component_slices[component - 1]
        origin = np.array([region[0].start, region[1].start])
        dynamic, _ = label(free[region] & (self.labels[region] == component))
        target_component = dynamic[tuple(np.asarray(target) - origin)]
        candidates = [(d, i) for d, i in candidates if dynamic[tuple(self.nodes[i] - origin)] == target_component]
        if not candidates:
            return None
        source = tuple(self.nodes[candidates[0][1]])
        # The compiled backend retains the same costs, heuristic and heap tie
        # order. It changes no physical checks or connectivity rule.
        from awr.life.civil.grid_search import grid_path

        field = (
            self.distance_field(target)
            if self.weight is not None and abs(source[0] - target[0]) + abs(source[1] - target[1]) > 60
            else None
        )
        cells = grid_path(free, self.weight, source, target, region=region, field=field, backend=self.search_backend)
        if not cells:
            return None
        return self.finish_grid_path(start, goal, cells, free)

    def finish_grid_path(self, start, goal, cells, free, shortcut_rule=None):
        """Check/simplify a target-to-source grid path from either router.

        Sharing this stage preserves exact connector, body, path-cost and
        turning checks when a compiled static graph supplies the cell path.
        `shortcut_rule(raw)`, if given, returns a test f(k, j) that a shortcut
        from raw[k] to raw[j] must also pass (a query with extra cell costs
        keeps the detour that its search found).
        """
        target, source = cells[0], cells[-1]
        weight = self.weight
        raw = [np.asarray(start), *(self.origin + np.array(cells[::-1])[:, ::-1] * self.step)]
        if not self.clear_segment(raw[-1], goal, free):
            return None
        raw.append(np.asarray(goal))
        admit = None if shortcut_rule is None else shortcut_rule(raw)
        # String pulling removes grid zigzags but cannot cross any blocked cell.
        simplified = [raw[0]]
        k = 0
        # With path costs a shortcut may not pass costlier cells than the grid
        # path it replaces (no diagonal cuts across a carriageway that the
        # path itself kept to the edge of).
        raw_weight = None
        if weight is not None:
            raw_weight = np.array(
                [float(weight[source])] + [float(weight[c]) for c in cells[::-1]] + [float(weight[target])], np.float32
            )
        while k < len(raw) - 1:
            # Find a visible prefix with exponential probes. Scanning backwards
            # from a kilometre-away goal at every corner was quadratic and made
            # long errands prohibitively slow. Every accepted connector still
            # goes through the exact same conservative collision check.
            limit = None if raw_weight is None else np.maximum.accumulate(raw_weight[k:])

            def visible(j):
                if admit is not None and j > k + 1 and not admit(k, j):
                    return False
                return self.clear_segment(
                    raw[k],
                    raw[j],
                    free,
                    None if limit is None or j == k + 1 else float(limit[j - k]) + self.shortcut_tolerance,
                )

            end = k + 1
            if not visible(end):
                return None
            stride = 2
            while end < len(raw) - 1:
                probe = min(k + stride, len(raw) - 1)
                if not visible(probe):
                    lo, hi = end, probe
                    while hi - lo > 1:
                        mid = (lo + hi) // 2
                        if visible(mid):
                            lo = mid
                        else:
                            hi = mid
                    end = lo
                    break
                end = probe
                stride *= 2
            simplified.append(raw[end])
            k = end
        if weight is not None:
            simplified = self._round_corners(simplified, free)
        if getattr(self, "turn_body", None) is not None and not getattr(free, "turn_only", False):
            # A corner kept sharp (> .25 rad after rounding) makes the walker stop
            # and rotate in place there. If the body cannot rotate at that corner,
            # plan again through turning cells only; if that fails as well, keep
            # the route (the per-tick checks decide at the corner itself).
            for k in range(1, len(simplified) - 1):
                a = simplified[k] - simplified[k - 1]
                b = simplified[k + 1] - simplified[k]
                if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
                    continue
                if abs(math.atan2(a[0] * b[1] - a[1] * b[0], float(np.dot(a, b)))) > 0.25 and not self.can_turn(
                    simplified[k], math.atan2(a[1], a[0]), math.atan2(b[1], b[0])
                ):
                    other = self.plan(start, goal, restrict(free, self.turn_free))
                    if other is not None:
                        return other
                    break
        return polyline(simplified, f"walk-component-{self.labels[source]}")

    def can_turn(self, p, h0, h1, margin=0.05):
        """Whether the walking body (`turn_body`: length, width) can rotate in place at `p` from
        heading h0 to h1: every corner and the centre stay on supported ground at 0.1 rad steps
        (the per-tick test, with `margin`), and no street object or parked vehicle is within the
        turning disk (`turn_clear`)."""
        clear = getattr(self, "turn_clear", None)
        c = self.cell(p)
        if clear is not None and not clear[tuple(c)]:
            return False
        d = (h1 - h0 + math.pi) % (2 * math.pi) - math.pi
        angles = h0 + np.linspace(0.0, 1.0, max(2, math.ceil(abs(d) / 0.1) + 1)) * d
        half = np.asarray(self.turn_body[:2], float) / 2 + margin
        unit = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * half
        cs, sn = np.cos(angles)[:, None], np.sin(angles)[:, None]
        corners = np.asarray(p) + np.stack(
            [unit[:, 0] * cs - unit[:, 1] * sn, unit[:, 0] * sn + unit[:, 1] * cs], -1
        ).reshape(-1, 2)
        return bool(self.surface.supported(np.vstack([corners, [p]]), "person").all())

    def _round_corners(self, points, free, turn_limit=0.25, step_turn=0.2, tangents=(1.0, 0.7, 0.45, 0.3)):
        """Replace corners sharper than `turn_limit` by short circular arcs.

        A walker then turns while walking instead of stopping to rotate. Each arc
        is accepted only if all of its chords pass the same connector check as
        the route itself; otherwise a shorter arc is tried and, failing that,
        the corner is kept.
        """
        pts = [np.asarray(q, float) for q in points]
        if len(pts) < 3:
            return pts
        out = [pts[0]]
        for k in range(1, len(pts) - 1):
            prev, v, nxt = out[-1], pts[k], pts[k + 1]
            a, b = v - prev, nxt - v
            la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if la < 1e-6 or lb < 1e-6:
                out.append(v)
                continue
            a, b = a / la, b / lb
            turn = math.atan2(a[0] * b[1] - a[1] * b[0], float(np.dot(a, b)))
            if abs(turn) <= turn_limit:
                out.append(v)
                continue
            # The previous arc may already use part of the incoming segment.
            room = min(0.45 * la if k > 1 else 0.9 * la, 0.45 * lb)
            done = False
            for d in tangents:
                if d > room or abs(turn) > 2.6:
                    continue
                t1, t2 = v - a * d, v + b * d
                r = d / math.tan(abs(turn) / 2)
                side = 1.0 if turn > 0 else -1.0
                centre = t1 + np.array([-a[1], a[0]]) * side * r
                m = max(2, math.ceil(abs(turn) / step_turn))
                arc = []
                for j in range(1, m):
                    ang = turn * j / m
                    c, s_ = math.cos(ang), math.sin(ang)
                    q = t1 - centre
                    arc.append(centre + np.array([c * q[0] - s_ * q[1], s_ * q[0] + c * q[1]]))
                chain = [prev, t1, *arc, t2]
                if all(self.clear_segment(chain[j], chain[j + 1], free) for j in range(len(chain) - 1)):
                    out.extend([t1, *arc, t2])
                    done = True
                    break
            if not done:
                out.append(v)
        out.append(pts[-1])
        return out

    def trip(self, rng, start=None, positions=(), sizes=(), radius=0.72, yaws=None):
        if not len(self.eligible):
            return None
        free = self.obstacle_mask(positions, sizes, radius, yaws, min_separation=1.8 if start is None else 0.0)
        possible = self.eligible[free[self.nodes[self.eligible, 0], self.nodes[self.eligible, 1]]]
        if not len(possible):
            return None
        components = {}
        for _ in range(160 if start is None else 40):
            if start is None:
                node = int(rng.choice(possible))
                p = self.xy[node]
            else:
                p = np.asarray(start)
                _, node = self.tree.query(p)
            static_component = self.labels[tuple(self.nodes[node])]
            if static_component == 0:
                continue
            region = self.component_slices[static_component - 1]
            origin = np.array([region[0].start, region[1].start])
            if static_component not in components:
                components[static_component] = label(free[region] & (self.labels[region] == static_component))[0]
            dynamic_labels = components[static_component]
            component = dynamic_labels[tuple(self.nodes[node] - origin)]
            if component == 0:
                continue
            dist = np.linalg.norm(self.xy - p, axis=1)
            candidates = np.flatnonzero((self.labels[self.free] == static_component) & (dist >= 8) & (dist <= 600))
            local = self.nodes[candidates] - origin
            choices = candidates[dynamic_labels[local[:, 0], local[:, 1]] == component]
            if not len(choices):
                continue
            target = self.xy[int(rng.choice(choices))]
            route = self.plan(p, target, free)
            if route is not None and route["arc"][-1] >= 8:
                return route
        return None

    def summary(self):
        counts = np.bincount(self.labels.ravel())[1:]
        return {
            "schema": "awr.life.walk_network.v1",
            "step_m": self.step,
            "grid_search_backend": self.search_backend,
            "free_area_m2": float(self.free.sum() * self.step**2),
            "components": len(counts),
            "largest_component_m2": float(counts.max() * self.step**2),
            "standing_area_m2": float(self.standing_free.sum() * self.step**2)
            if hasattr(self, "standing_free")
            else None,
            "standing_policy": getattr(self, "standing_policy", None),
            "semantics": "Derived roadside strips, explicit crossings, measured support; not surveyed sidewalks.",
        }
