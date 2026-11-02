"""Street furniture of the ambient scene: the catalogue's street objects placed on the walkway by fixed rules.

- Benches with their backs to a wall or planting at the edges of the parks and pitches and at the bus stop; a litter
  bin at the end of every other one.
- A litter bin at each facility (post office, bank, shops, town hall, court, community centre, school, church); a
  bicycle rack with bicycles leaning beside it at the supermarket, school, community centre, town hall and café.
- Bicycles leaning at a few houses, hydrants along the streets, rows of bollards where an open square meets a road.

An object stands on the walkway with its back BACK_GAP_M from the walkway's edge (a hydrant and a bollard anywhere),
keeps LANE_CLEAR_M from every car lane (the streets are shared, awr.life.ambient.walkway: the cars pass it at their
lanes), keeps clear of the parked cars and of the people's standing slots, stands where the walkway is at least
WIDE_ROOM_M from edge to middle, and never cuts the way past it: the walkway around it as the people walk it (one
walker, and two abreast as at the standing slots, awr.life.ambient.crowd) does not fall apart. It is then an obstacle of
the walkway, like a parked car. No random choice: a stage gets the same furniture whatever the seed of its people
and cars.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from shapely import contains_xy
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union

from awr.life.ambient import layout
from awr.life.ambient.crowd import EDGE_M, RADIUS_M, SLOT_PASS_GAP_M, WALL_BUFFER_M
from awr.life.ambient.walkway import OBSTACLE_PAD_M

LAYOUT_SEED = 7  # the places the furniture is laid out by (awr.life.ambient.crowd.compile_places)
LABELS = {
    "bench": "长椅",
    "litter-bin": "垃圾桶",
    "bicycle-rack": "自行车停放架",
    "parked-bicycle": "自行车",
    "hydrant": "消防栓",
    "bollard": "隔离桩",
}
BACK_GAP_M = 0.15  # between an object's back and the walkway's edge (a wall, a hedge)
LANE_CLEAR_M = 2.5  # a footprint from a car lane's centre line: beyond the carriageway band of the walkway
# (LANE_BAND_M 2.2: a car's half width, a person's reach and room to pass)
PARKED_CLEAR_M = 1.2  # from a parked car's body (its doors, a person getting past)
SLOT_CLEAR_M = 1.0  # from a standing slot: a person standing there (a disc) and a little room
OBJECT_GAP_M = 0.5  # between the footprints of two objects (a bin at a bench end, a bicycle at a rack: .3)
WIDE_ROOM_M = 2.6  # the walkway around an object (within NEAR_M) reaches this far from its edge somewhere
NEAR_M = 3.0
WINDOW_M = 8.0  # the way past an object is judged within this distance of it
REACH_M = (2.0, 12.0)  # an object for a place or a facility stands this far from it
BENCHES, BENCH_SPACING_M = 16, 12.0
FACILITY_BINS, BIN_SPACING_M = 12, 8.0
RACKS = 6
HOUSE_BICYCLES, HOUSE_BICYCLE_SPACING_M = 6, 25.0
HYDRANTS, HYDRANT_SPACING_M = 10, 40.0
HYDRANT_STREET_M = 4.5  # a hydrant stands within this of a car lane (the street it serves)
BOLLARD_ROWS, BOLLARDS_PER_ROW = 3, 3
BOLLARD_SPACING_M = 3.6  # walkers pass between two: a disc, JuPedSim's wall buffer and the walkway's edge erosion clear
# of each padded hole
BOLLARD_LANE_M = 2.7  # a row of bollards this far from the lane it guards
BOLLARD_ROW_SPACING_M = 60.0
SQUARE_ROOM_M = 4.0  # ... on the side of an open square (the walkway this far from its edge beyond the row)
BIN_KINDS = (
    "post_office",
    "bank",
    "supermarket",
    "cafe",
    "townhall",
    "courthouse",
    "community_centre",
    "school",
    "place_of_worship",
    "bus_station",
)
RACK_KINDS = ("supermarket", "school", "community_centre", "townhall", "cafe")
BENCH_PLACE_KINDS = ("park", "pitch")


def place(walkway, places, semantics, lanes, parked, sizes, bbox, support=None):
    """The furniture of a stage: list of dict(id, asset, xy, yaw, label, context). `walkway`: awr.life.ambient.walkway
    (around the parked cars); `places`: awr.life.ambient.crowd places; `lanes`: car lane centre lines; `parked`: the parked
    cars (id, vtype, xy, yaw); `sizes`: asset id -> dimensions (m); `support`: awr.life.ambient.support, whose person
    cells every corner of a footprint stands on (the published tape's final QA tests the four corners; the walkway's
    polygon is simplified and may reach past them)."""
    return _Layout(walkway, places, semantics, lanes, parked, sizes, bbox, support).run()


def footprints(furniture, sizes):
    """The footprints of placed furniture (obstacles of the walkway)."""
    return [layout.body(f["xy"], f["yaw"], sizes[f["asset"]]) for f in furniture]


class _Layout:
    def __init__(self, walkway, places, semantics, lanes, parked, sizes, bbox, support=None):
        self.w, self.places, self.sizes, self.support = walkway, places, sizes, support
        self.walk = walkway.walk.copy()
        self.lanes = unary_union(lanes)
        self.parked = (
            unary_union([layout.body(p["xy"], p["yaw"], sizes[p["vtype"]]) for p in parked]) if parked else None
        )
        self.slots = MultiPoint([tuple(s) for p in places for s in p.slots])

        def inside(xy):
            return bbox[0] <= xy[0] <= bbox[2] and bbox[1] <= xy[1] <= bbox[3]

        self.pois = sorted((p for p in semantics.get("pois", []) if inside(p["xy"])), key=lambda p: p["id"])
        self.out, self.bodies = [], []
        self._room()
        smooth = ndimage.gaussian_filter(self.room0, 1.0)
        gy, gx = np.gradient(smooth)
        norm = np.hypot(gx, gy)
        self.normal = np.stack([gx, gy], -1) / np.maximum(norm, 1e-9)[..., None]

    def _room(self):
        self.room = ndimage.distance_transform_edt(np.pad(self.walk, 1))[1:-1, 1:-1] * self.w.cell_m
        if not hasattr(self, "room0"):
            self.room0 = self.room.copy()

    def run(self):
        self._benches()
        self._facilities()
        self._house_bicycles()
        self._hydrants()
        self._bollards()
        return self.out

    # ---- rules ----
    def _benches(self):
        stops = [p for p in self.pois if p["kind"] == "bus_station"]
        # the places of the parks and pitches taken in turns, one of each park per round
        parks = {}
        for p in sorted(self.places, key=lambda p: p.id):
            if p.kind in BENCH_PLACE_KINDS:
                parks.setdefault(p.area, []).append(p)
        rounds = [
            [parks[a][k] for a in sorted(parks) if k < len(parks[a])]
            for k in range(max(map(len, parks.values()), default=0))
        ]
        spots = [(np.asarray(s["xy"], float), s["id"]) for s in stops] + [(p.xy, p.id) for r in rounds for p in r]
        n = 0
        for xy, ctx in spots:
            if n >= BENCHES:
                break
            bench = self._at_edge("bench", xy, ctx, spacing=BENCH_SPACING_M)
            if bench is None:
                continue
            n += 1
            if n % 2 == 1:  # a litter bin at one end of every other bench
                self._beside(bench, "litter-bin", ctx)

    def _facilities(self):
        bins = racks = 0
        for poi in self.pois:
            geom = Polygon(poi["polygon"]) if poi.get("polygon") else Point(poi["xy"])
            xy = np.asarray(geom.centroid.coords[0])
            if poi["kind"] in BIN_KINDS and bins < FACILITY_BINS:
                if self._at_edge("litter-bin", xy, poi["id"], spacing=BIN_SPACING_M, geom=geom) is not None:
                    bins += 1
            if poi["kind"] in RACK_KINDS and racks < RACKS:
                rack = self._at_edge("bicycle-rack", xy, poi["id"], geom=geom)
                if rack is not None:
                    racks += 1
                    for side in (1, -1):  # bicycles leaning along the edge beyond either end
                        self._beside(rack, "parked-bicycle", poi["id"], side=side, lengthwise=True)

    def _house_bicycles(self):
        n = 0
        for p in sorted(self.places, key=lambda p: p.id):
            if n >= HOUSE_BICYCLES:
                break
            if p.kind != "house":
                continue
            if (
                self._at_edge("parked-bicycle", p.xy, p.id, spacing=HOUSE_BICYCLE_SPACING_M, lengthwise=True)
                is not None
            ):
                n += 1

    def _hydrants(self):
        """Along the streets: edge spots near a car lane, spread over the stage."""
        c = self.w.cell_m
        rows, cols = np.nonzero(self.walk & (self.room0 >= 0.5) & (self.room0 <= 1.0))
        xy = np.c_[self.w.x0 + (cols + 0.5) * c, self.w.y0 + (rows + 0.5) * c]
        from shapely import distance, points

        d = distance(self.lanes, points(xy))
        keep = (d >= LANE_CLEAR_M + 0.2) & (d <= HYDRANT_STREET_M)
        xy, rows, cols = xy[keep], rows[keep], cols[keep]

        def put(i):
            t = self._tangent(rows[i], cols[i])
            return self._try("hydrant", xy[i], np.arctan2(t[1], t[0]), "street")

        _spread(xy, put, HYDRANTS, HYDRANT_SPACING_M)

    def _bollards(self):
        """Rows parallel to a lane at an open square: BOLLARD_LANE_M from the lane, the walkway open beyond."""
        from shapely import distance, points

        c = self.w.cell_m
        rows, cols = np.nonzero(self.walk & (self.room0 >= SQUARE_ROOM_M))
        if not len(rows):
            return
        sq = np.c_[self.w.x0 + (cols + 0.5) * c, self.w.y0 + (rows + 0.5) * c]
        d = distance(self.lanes, points(sq))
        near = sq[(d >= BOLLARD_LANE_M + 1.0) & (d <= BOLLARD_LANE_M + 3.0)]
        from shapely.ops import nearest_points

        def put(i):
            a = np.asarray(nearest_points(self.lanes, Point(*near[i]))[0].coords[0])
            u = near[i] - a
            if np.hypot(*u) < 1e-6:
                return None
            u /= np.hypot(*u)
            t = np.array([-u[1], u[0]])
            mid = a + u * BOLLARD_LANE_M
            placed = []
            for k in np.arange(BOLLARDS_PER_ROW) - (BOLLARDS_PER_ROW - 1) / 2:
                f = self._try(
                    "bollard",
                    mid + t * BOLLARD_SPACING_M * k,
                    np.arctan2(t[1], t[0]),
                    "square",
                    lane_clear=BOLLARD_LANE_M - 0.3,
                )
                if f is None:
                    for g in placed:
                        self._remove(g)
                    return None
                placed.append(f)
            return placed

        _spread(near, put, BOLLARD_ROWS, BOLLARD_ROW_SPACING_M)

    # ---- geometry ----
    def _tangent(self, r, q):
        n = self.normal[r, q]
        return np.array([-n[1], n[0]])

    def _at_edge(self, asset, xy, ctx, *, spacing=0.0, geom=None, lengthwise=False):
        """The nearest edge spot within REACH_M of `xy` (or of `geom`) for `asset`, its back to the edge (benches
        face away from it: their backrest is on the mesh's -y side); its long side along the edge, or its length
        along it for a bicycle (`lengthwise`)."""
        size = self.sizes[asset]
        depth = size[0] if lengthwise else size[1]
        c = self.w.cell_m
        target = depth / 2 + BACK_GAP_M + c / 2
        # the cells within reach of the point, or of the whole building
        x0, y0, x1, y1 = geom.bounds if geom is not None else (xy[0], xy[1], xy[0], xy[1])
        (ra, qa), (rb, qb) = (
            self.w.cell((x0 - REACH_M[1], y0 - REACH_M[1])),
            self.w.cell((x1 + REACH_M[1], y1 + REACH_M[1])),
        )
        rs = slice(max(ra, 0), min(rb + 1, self.walk.shape[0]))
        qs = slice(max(qa, 0), min(qb + 1, self.walk.shape[1]))
        rr, qq = np.nonzero(self.walk[rs, qs] & (np.abs(self.room0[rs, qs] - target) <= c / 2 + 1e-6))
        rr, qq = rr + rs.start, qq + qs.start
        pts = np.c_[self.w.x0 + (qq + 0.5) * c, self.w.y0 + (rr + 0.5) * c]
        if geom is not None:
            from shapely import distance, points

            d = distance(geom, points(pts))
        else:
            d = np.hypot(*(pts - xy).T)
        ok = (d >= REACH_M[0]) & (d <= REACH_M[1])
        for i in np.flatnonzero(ok)[np.argsort(d[ok], kind="stable")]:
            p = pts[i]
            if spacing and any(
                f["asset"] == asset and np.hypot(*(np.asarray(f["xy"]) - p)) < spacing for f in self.out
            ):
                continue
            n = self.normal[rr[i], qq[i]]
            t = np.array([-n[1], n[0]])
            # yaw turns the mesh's +x along the edge with its +y (a bench's front) towards the walkway, or for a
            # bicycle its +y (its length) along the edge
            yaw = np.arctan2(-t[1], -t[0]) if not lengthwise else np.arctan2(n[1], n[0])
            f = self._try(asset, p, yaw, ctx)
            if f is not None:
                return f
        return None

    def _beside(self, base, asset, ctx, *, side=1, lengthwise=False):
        """`asset` at one end of `base` along the edge, its back in line with the base's back."""
        yaw = base["yaw"]
        t = np.array([np.cos(yaw), np.sin(yaw)])
        n = np.array([-t[1], t[0]])
        bs, size = self.sizes[base["asset"]], self.sizes[asset]
        long, depth = (size[1], size[0]) if lengthwise else (size[0], size[1])
        back = np.asarray(base["xy"]) - n * bs[1] / 2
        xy = back + n * depth / 2 + side * t * (bs[0] / 2 + 0.3 + long / 2)
        return self._try(asset, xy, yaw - np.pi / 2 if lengthwise else yaw, ctx, gap=0.25)

    def _try(self, asset, xy, yaw, ctx, *, gap=OBJECT_GAP_M, lane_clear=LANE_CLEAR_M):
        xy = np.asarray(xy, float)
        fp = layout.body(xy, yaw, self.sizes[asset])
        if not self.w.polygon.contains(fp):
            return None
        if self.support is not None:
            half = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * np.asarray(self.sizes[asset][:2]) / 2
            c, s = np.cos(yaw), np.sin(yaw)
            if not self.support.supported(
                xy + np.c_[half[:, 0] * c - half[:, 1] * s, half[:, 0] * s + half[:, 1] * c], "person"
            ).all():
                return None
        if self.lanes.distance(fp) < lane_clear:
            return None
        if self.parked is not None and self.parked.distance(fp) < PARKED_CLEAR_M:
            return None
        if not self.slots.is_empty and self.slots.distance(fp) < SLOT_CLEAR_M:
            return None
        if any(b.distance(fp) < gap for b in self.bodies):
            return None
        if self._narrow(xy) or self._cuts(fp):
            return None
        f = dict(
            id=f"{asset}-{len(self.out):03d}",
            asset=asset,
            xy=[round(float(v), 3) for v in xy],
            yaw=round(float(yaw), 4),
            label=LABELS[asset],
            context=ctx,
        )
        self.out.append(f)
        self.bodies.append(fp)
        self.walk &= ~self._hole(fp, slice(0, self.walk.shape[0]), slice(0, self.walk.shape[1]))
        self._room()
        return f

    def _remove(self, f):
        i = self.out.index(f)
        del self.out[i], self.bodies[i]
        self.walk = self.w.walk.copy()
        for fp in self.bodies:
            self.walk &= ~self._hole(fp, slice(0, self.walk.shape[0]), slice(0, self.walk.shape[1]))
        self._room()

    def _narrow(self, xy):
        c = self.w.cell_m
        k = int(np.ceil(NEAR_M / c))
        r0, q0 = self.w.cell(xy)
        win = self.room[max(r0 - k, 0) : r0 + k + 1, max(q0 - k, 0) : q0 + k + 1]
        return not win.size or win.max() < WIDE_ROOM_M

    def _hole(self, fp, rs, qs):
        """Walkway cells an obstacle's footprint takes (awr.life.ambient.walkway pads obstacles the same way)."""
        c = self.w.cell_m
        yy, xx = np.mgrid[rs, qs]
        X, Y = self.w.x0 + (xx + 0.5) * c, self.w.y0 + (yy + 0.5) * c
        return contains_xy(fp.buffer(OBSTACLE_PAD_M + c / 2), X, Y)

    def _cuts(self, fp):
        """Whether the footprint cuts the way past it: the cells a walker's centre may take (one walker, and the
        midpoint of two abreast SLOT_PASS_GAP_M apart) within WINDOW_M fall apart into more pieces (of at least a
        square metre) once its cells are taken."""
        c = self.w.cell_m
        x0, y0, x1, y1 = fp.bounds
        m = int(np.ceil((WINDOW_M + 3.0) / c))
        (ra, qa), (rb, qb) = self.w.cell((x0, y0)), self.w.cell((x1, y1))
        rs = slice(max(ra - m, 0), min(rb + m + 1, self.walk.shape[0]))
        qs = slice(max(qa - m, 0), min(qb + m + 1, self.walk.shape[1]))
        before = self.walk[rs, qs]
        after = before & ~self._hole(fp, rs, qs)
        room_b = ndimage.distance_transform_edt(before) * c
        room_a = ndimage.distance_transform_edt(after) * c
        inner = (slice(int(3.0 / c), -int(3.0 / c) or None),) * 2
        eight = np.ones((3, 3), bool)
        crumb = 1.0 / (c * c)
        for need in (EDGE_M + RADIUS_M + WALL_BUFFER_M, EDGE_M + WALL_BUFFER_M + 2 * RADIUS_M + SLOT_PASS_GAP_M / 2):
            ub, ua = room_b[inner] >= need, room_a[inner] >= need
            lb, _ = ndimage.label(ub, structure=eight)
            la, _ = ndimage.label(ua, structure=eight)
            for lab in set(np.unique(lb[ub & ~ua])) - {0}:
                _, n = np.unique(la[(lb == lab) & (la > 0)], return_counts=True)
                if (n >= crumb).sum() > 1:
                    return True
        return False


def _spread(pts, put, k, spacing):
    """Up to `k` of the candidate points `pts` taken by `put(i)` (None where it cannot) spread over the stage: each
    next one the candidate farthest from those taken (and at least `spacing` from them), the first the one nearest the
    candidates' middle."""
    if not len(pts):
        return
    left = np.ones(len(pts), bool)
    taken = []
    far = np.full(len(pts), np.inf)
    first = True
    while len(taken) < k and left.any():
        if first:
            i = int(np.argmin(np.where(left, np.hypot(*(pts - pts.mean(0)).T), np.inf)))
        else:
            i = int(np.argmax(np.where(left, far, -np.inf)))
            if far[i] < spacing:
                return
        left[i] = False
        if put(i) is None:
            continue
        first = False
        taken.append(i)
        far = np.minimum(far, np.hypot(*(pts - pts[i]).T))
