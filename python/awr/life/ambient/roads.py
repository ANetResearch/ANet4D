"""Compile the vehicle network of a stage from the town's SUMO net and the measured vehicle support.

The OSM-derived SUMO net places each lane at a mapped width from the road axis. On the photogrammetric town those lanes
leave the supported ground in many places (17 % of the lane area of the town centre, oryahovo-crop1). A road here
keeps the axis and the junctions of the SUMO net and takes its lanes from the supported corridor around the axis:
scanned along the normal every SAMPLE_M, the supported run nearest the previous centre gives the corridor centre and
width. A road whose corridor is at least TWO_WAY_M wide along its whole interior keeps a lane each way, LANE_OFFSET_M to
either side of the smoothed corridor centre; one at least ONE_WAY_M wide becomes one-way along the centre; a narrower
road carries no traffic. One-way directions are chosen for the largest strongly connected network, and only that
network is kept: an ambient car can always drive on. netconvert then builds the junctions (smooth internal curves,
no turnarounds, turn speeds limited to JUNCTION_LATERAL_MPS2 of lateral acceleration), and every lane, the internal
ones included, is checked with the car's footprint against the support; a junction connection that leaves it is
removed and the network rebuilt.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np
from shapely.geometry import LineString, Point

SAMPLE_M = 0.5
SCAN_M = 7.0  # half-width of the lateral scan around the road axis
SCAN_STEP_M = 0.125
LANE_MARGIN_M = 0.1  # lateral margin of a car's sides to unsupported ground
TWO_WAY_GAP_M = 2.8  # distance between the lane centres of a two-way road at least
SHIFT_COST, CENTRE_COST = 0.3, 0.02  # lane placement: per lateral step, per step from the corridor centre
END_M = 6.0  # this close to a junction the ground is the junction's
SPEED_MPS = 8.33  # 30 km/h in the town centre
LANE_WIDTH_M = 3.0
JUNCTION_LATERAL_MPS2 = 2.5
VEHICLE_WIDTH_M = 2.3  # widest car body (2.24 m) for SUMO's conflict points in junctions (its default is 1.8 m)
YAW_RATE_MPS = 0.8  # speed limit of a bend: at most this yaw rate (rad/s; the tape QA allows 1)
CAR_HALF_LENGTH_M, CAR_HALF_WIDTH_M = 2.42, 1.12  # envelope of the sedan and SUV assets (catalogue boxes)
# Tyre contacts the tape QA probes (awr.life.civil.contacts.vehicle_contacts) of the sedan, hatchback and SUV assets.
CONTACT_PROBES = np.array(
    [
        [1.549, 0.0],
        [-1.549, 0.0],
        [0.0, 0.807],
        [0.0, -0.807],
        [-1.387, -0.81],
        [-1.387, 0.81],
        [1.416, -0.81],
        [1.416, 0.81],
        [1.365, 0.0],
        [-1.365, 0.0],
        [0.0, 0.783],
        [0.0, -0.783],
        [-1.337, -0.775],
        [-1.337, 0.775],
        [1.246, -0.775],
        [1.246, 0.775],
        [1.523, 0.0],
        [-1.523, 0.0],
        [0.0, 0.827],
        [0.0, -0.827],
        [-1.496, -0.84],
        [-1.496, 0.84],
        [1.393, -0.84],
        [1.393, 0.84],
    ]
)
MIN_SUPPORTED = 1.0  # share of footprints of a lane on supported ground


@dataclass
class Road:
    base: str  # SUMO id without the direction suffix
    nodes: tuple  # (from, to) junctions of the forward direction
    axis: LineString
    length: float
    offset: np.ndarray = field(repr=False)  # per sample: left offset of the lane (one-way) or of the 'f' lane
    offset_b: np.ndarray = field(default=None, repr=False)  # two-way: left offset of the 'b' lane
    kind: str = "none"  # 'two_way' | 'one_way' | 'none'
    direction: str = "f"  # one-way: 'f' along the axis, 'b' against it
    two_way_room: np.ndarray = field(default=None, repr=False)  # per sample: two lanes fit
    samples: np.ndarray = field(default=None, repr=False)

    def lane(self, direction):
        """Lane centreline of `direction` ('f' or 'b') in its driving direction."""
        s = _samples(self.axis)
        P, n = _frame(self.axis, s)
        o = self.offset_b if self.kind == "two_way" and direction == "b" else self.offset
        pts = P + n * o[:, None]
        return pts if direction == "f" else pts[::-1]


AXIS_SMOOTH_M = 1.5  # Gaussian smoothing of a mapped road axis: lanes are offsets from it and inherit its corners


def smooth_axis(axis):
    """The road axis resampled every SAMPLE_M with its corners rounded (Gaussian, AXIS_SMOOTH_M), ends kept."""
    s = _samples(axis)
    if len(s) < 5:
        return axis
    P = np.array([axis.interpolate(v).coords[0][:2] for v in s])
    k = min(
        int(np.ceil(3 * AXIS_SMOOTH_M / SAMPLE_M)), len(P) - 2
    )  # a short axis: the kernel truncated to its reflection
    w = np.exp(-0.5 * (np.arange(-k, k + 1) * SAMPLE_M / AXIS_SMOOTH_M) ** 2)
    w /= w.sum()
    # reflect about the end points so the ends stay put and keep their direction
    pad = np.r_[2 * P[0] - P[k:0:-1], P, 2 * P[-1] - P[-2 : -k - 2 : -1]]
    Q = np.c_[np.convolve(pad[:, 0], w, "valid"), np.convolve(pad[:, 1], w, "valid")]
    Q[0], Q[-1] = P[0], P[-1]
    return LineString(Q)


def _samples(axis):
    s = np.arange(0.0, axis.length + 1e-9, SAMPLE_M)
    return np.r_[s, axis.length] if axis.length - s[-1] > 1e-6 else s


def _frame(axis, s):
    P = np.array([axis.interpolate(v).coords[0][:2] for v in s])
    a = np.array([axis.interpolate(max(v - 0.5, 0.0)).coords[0][:2] for v in s])
    b = np.array([axis.interpolate(min(v + 0.5, axis.length)).coords[0][:2] for v in s])
    t = b - a
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return P, np.c_[-t[:, 1], t[:, 0]]


def _running(values, window, how="mean"):
    k = max(int(round(window / SAMPLE_M)), 1)
    pad = np.pad(values, (k // 2, k - 1 - k // 2), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(pad, k)
    return {"mean": view.mean(1), "min": view.min(1), "median": np.median(view, 1)}[how]


OFFSETS = np.arange(-SCAN_M, SCAN_M + 1e-9, SCAN_STEP_M)


def scan(axis, support):
    """Samples along `axis` and, per sample and lateral offset (OFFSETS, left positive), whether the ground is
    supported for a car."""
    s = _samples(axis)
    P, n = _frame(axis, s)
    ok = support.supported((P[:, None, :] + n[:, None, :] * OFFSETS[None, :, None]).reshape(-1, 2), "vehicle")
    return s, ok.reshape(len(s), len(OFFSETS))


TURNS = (-1, 0, 1)  # lateral steps of SCAN_STEP_M per sample a lane may take


def lane_cells(axis, s, support):
    """Per lateral step (TURNS), sample and offset (OFFSETS): a car centred there and heading along the road turned by
    that step has the corners and the middles of the sides of its body (as the final tape QA checks them),
    LANE_MARGIN_M wider, on supported ground."""
    P, n = _frame(axis, s)
    t = np.c_[n[:, 1], -n[:, 0]]
    half = CAR_HALF_WIDTH_M + LANE_MARGIN_M
    out = {}
    for turn in TURNS:
        a = np.arctan2(turn * SCAN_STEP_M, SAMPLE_M)
        tt, nn = t * np.cos(a) + n * np.sin(a), n * np.cos(a) - t * np.sin(a)
        ok = np.ones((len(s), len(OFFSETS)), bool)
        for dl in (-CAR_HALF_LENGTH_M, 0.0, CAR_HALF_LENGTH_M):
            for dw in (-half, half):
                q = P[:, None, :] + n[:, None, :] * OFFSETS[None, :, None] + (tt * dl + nn * dw)[:, None, :]
                ok &= support.supported(q.reshape(-1, 2), "vehicle").reshape(ok.shape)
        out[turn] = ok
    return out


def place(cells, interior, prefer, allowed):
    """The least-cost path of offset indices through the interior samples (one lateral step of SCAN_STEP_M per
    sample at most, SHIFT_COST per step, CENTRE_COST per step from `prefer`) on which the car is supported at the
    heading of each step (`cells`, see lane_cells) and stays within `allowed`; None without one."""
    K, J = allowed.shape
    INF = 1e18
    idx = np.arange(J)
    free = ~interior[:, None]
    ok = {turn: (cells[turn] | free) & allowed for turn in TURNS}
    cost = np.where(ok[0][0], np.abs(idx - prefer[0]) * CENTRE_COST, INF)
    back = np.zeros((K, J), np.int64)
    for k in range(1, K):
        # arriving at j from j-1 (turn +1), j (0) or j+1 (-1), each checked at its own heading
        cand = np.stack(
            [
                np.where(ok[1][k], np.r_[INF, cost[:-1]] + SHIFT_COST, INF),
                np.where(ok[0][k], cost, INF),
                np.where(ok[-1][k], np.r_[cost[1:], INF] + SHIFT_COST, INF),
            ]
        )
        arg = cand.argmin(0)
        back[k] = idx + arg - 1
        cost = cand.min(0) + np.abs(idx - prefer[k]) * CENTRE_COST
        cost[cost >= INF] = INF
    if cost.min() >= INF:
        return None
    path = np.empty(K, np.int64)
    path[-1] = int(cost.argmin())
    for k in range(K - 1, 0, -1):
        path[k - 1] = back[k, path[k]]
    return path


RELAX_ITERATIONS = 400


def _relaxed(path, P, n, interior, support, valid, lo=-np.inf, hi=np.inf):
    """Lane offsets from the placed `path`, smoothed by repeated averaging of neighbouring samples (1-2-1) wherever
    no sample loses the support of the car body (LANE_MARGIN_M wider, heading along the lane). Returns the
    smoothest stage of the smoothing that holds the car along the whole lane (`valid`), None if not even the
    placed path does."""
    o = OFFSETS[path].astype(float)

    def ok(off):
        pts = P + n * off[:, None]
        d = np.gradient(pts, axis=0)
        return body_supported(pts, np.arctan2(d[:, 1], d[:, 0]), support)

    failing = ~ok(o) & interior
    snapshots = [o]
    for it in range(1, RELAX_ITERATIONS + 1):
        new = o.copy()
        new[1:-1] = (o[:-2] + 2 * o[1:-1] + o[2:]) / 4
        new = np.clip(new, lo, hi)
        if np.abs(new - o).max() < 1e-3:
            break
        mask = np.ones(len(o), bool)
        for _ in range(6):  # drop the moves near new failures until none is left
            cand = np.where(mask, new, o)
            worse = ~ok(cand) & interior & ~failing
            if not worse.any():
                break
            mask &= ~(worse | np.r_[worse[1:], False] | np.r_[False, worse[:-1]])
        else:
            break
        o = cand
        failing = ~ok(o) & interior
        if it in (10, 25, 50, 100, 200):
            snapshots.append(o)
    # the smoothest stage that holds the car along the whole lane (checked more finely than the samples)
    for o in [o] + snapshots[::-1]:
        if valid(o):
            return o
    return None


def measure_axis(base, nodes, axis, support):
    """The road `base` from junction nodes[0] to nodes[1] along `axis` with its lane placement and kind: two lanes
    held apart by TWO_WAY_GAP_M if they hold the car on all but END_M at either end, else one lane, else none."""
    gap = int(round(TWO_WAY_GAP_M / 2 / SCAN_STEP_M))
    s, ok = scan(axis, support)
    interior = (s >= END_M) & (s <= axis.length - END_M)
    if not interior.any():
        interior = np.ones(len(s), bool)  # a short connector inside a junction cluster is checked whole
    road = Road(base, nodes, axis, axis.length, np.zeros(len(s)))
    # Prefer the middle of the supported ground nearest the axis.
    mid = np.array([np.flatnonzero(r).mean() if r.any() else len(OFFSETS) // 2 for r in ok])
    prefer = _running(mid, 9.0)
    cells = lane_cells(axis, s, support)
    J = len(OFFSETS)
    side = np.arange(J)[None, :] - np.round(prefer)[:, None]
    P, n = _frame(axis, s)
    k = np.flatnonzero(interior)

    def valid(o):  # driven either way: the samples of both directions
        pts = (P + n * o[:, None])[k[0] : k[-1] + 1]
        return k[-1] == k[0] or (path_supported(pts, support) and path_supported(pts[::-1], support))

    right = place(cells, interior, prefer - gap, side <= -gap)
    left = place(cells, interior, prefer + gap, side >= gap)
    centre = OFFSETS[np.clip(np.round(prefer).astype(int), 0, J - 1)]
    half = TWO_WAY_GAP_M / 2
    right, left = (
        (None, None)
        if right is None or left is None
        else (
            _relaxed(right, P, n, interior, support, valid, hi=centre - half),
            _relaxed(left, P, n, interior, support, valid, lo=centre + half),
        )
    )
    if (
        right is not None
        and left is not None
        and not lanes_clear((P + n * right[:, None])[k[0] : k[-1] + 1], (P + n * left[:, None])[k[0] : k[-1] + 1][::-1])
    ):
        right = left = None
    if right is not None and left is not None:
        road.kind, road.offset, road.offset_b = "two_way", right, left
    else:
        path = place(cells, interior, prefer, np.ones((len(s), J), bool))
        o = None if path is None else _relaxed(path, P, n, interior, support, valid)
        if o is not None:
            road.kind, road.offset = "one_way", o
    road.two_way_room = _two_way_room(cells, prefer, gap) & interior | ~interior
    road.samples = s
    return road


ONCOMING_CLEARANCE_M = 0.15  # between the bodies of two cars passing on the lanes of a two-way road


def lanes_clear(a, b):
    """Whether cars on the opposite lanes `a` and `b` (each in its driving direction) never overlap: every pose
    along `a` against the nearest pose on `b`, bodies ONCOMING_CLEARANCE_M wider."""
    from awr.life.civil.collision import obb_overlap

    if len(a) < 2 or len(b) < 2:
        return True
    la, lb = LineString(a), LineString(b)
    if la.length < 0.5 or lb.length < 0.5:
        return True
    s = np.r_[np.arange(0.0, la.length, 0.25), la.length]
    Pa, na = _frame(la, s)
    sb = np.array([lb.project(Point(p)) for p in Pa])
    Pb, nb = _frame(lb, sb)
    ha, hb = np.arctan2(-na[:, 0], na[:, 1]), np.arctan2(-nb[:, 0], nb[:, 1])
    size = np.tile(
        [2 * CAR_HALF_LENGTH_M + ONCOMING_CLEARANCE_M, 2 * CAR_HALF_WIDTH_M + ONCOMING_CLEARANCE_M, 1.0], (len(s), 1)
    )
    return not obb_overlap(Pa, ha, size, Pb, hb, size).any()


def _two_way_room(cells, prefer, gap):
    """Per sample: a lane fits on either side of the preferred centre, TWO_WAY_GAP_M apart."""
    J = cells[0].shape[1]
    side = np.arange(J)[None, :] - np.round(prefer)[:, None]
    fits = cells[0] | cells[1] | cells[-1]
    return (fits & (side <= -gap)).any(1) & (fits & (side >= gap)).any(1)


PINCH_PAD_M = END_M + 3.0  # a single-lane stretch extends this far beyond the narrow ground (room for the lane shift)
PINCH_MERGE_M = 20.0  # narrow stretches closer than this form one single-lane stretch
MIN_PIECE_M = 15.0  # a two-way piece of a split road at least this long


def measure_pieces(edge, support):
    """The road of the SUMO `edge` (either direction), split where it is too narrow for two lanes: two-way pieces
    and single-lane pieces between them, joined at new nodes on the axis (ids '<road>~<metres along it>'). A road
    narrow along most of its length stays one single-lane road."""
    eid = edge.getID()
    base, direction = (eid[:-2], eid[-1]) if eid[-2:] in ("-f", "-b") else (eid, "f")
    shape = [p[:2] for p in edge.getShape()]
    nodes = (edge.getFromNode().getID(), edge.getToNode().getID())
    if direction == "b":
        shape, nodes = shape[::-1], nodes[::-1]
    axis = smooth_axis(LineString(shape))
    whole = measure_axis(base, nodes, axis, support)
    if whole.kind != "one_way":
        return [whole]
    s, narrow = whole.samples, ~whole.two_way_room
    if not narrow.any():
        return [whole]
    # narrow stretches, merged and padded
    idx = np.flatnonzero(narrow)
    groups = [[s[idx[0]], s[idx[0]]]]
    for v in s[idx[1:]]:
        if v - groups[-1][1] < PINCH_MERGE_M:
            groups[-1][1] = v
        else:
            groups.append([v, v])
    stretches = [(max(a - PINCH_PAD_M, 0.0), min(b + PINCH_PAD_M, axis.length)) for a, b in groups]
    cuts = [0.0]
    kinds = []
    for a, b in stretches:
        if a - cuts[-1] >= MIN_PIECE_M:
            cuts.append(a)
            kinds.append("wide")
        if kinds and kinds[-1] == "narrow":
            cuts[-1] = b
        else:
            cuts.append(b)
            kinds.append("narrow")
    if axis.length - cuts[-1] >= MIN_PIECE_M:
        cuts.append(axis.length)
        kinds.append("wide")
    else:
        cuts[-1] = axis.length
    if "wide" not in kinds:
        return [whole]
    from shapely.ops import substring

    pieces, ids = [], [nodes[0]] + [f"{base}~{c:.0f}" for c in cuts[1:-1]] + [nodes[1]]
    for i in range(len(cuts) - 1):
        sub = substring(axis, cuts[i], cuts[i + 1])
        pieces.append(measure_axis(f"{base}~{i}", (ids[i], ids[i + 1]), LineString(sub.coords), support))
    return pieces


def _base(eid):
    return eid[:-2] if eid[-2:] in ("-f", "-b") else eid


def measure(net, support, bbox):
    """The roads of `net` (a sumolib net) with a passenger lane, by base id; a road narrow only in places is split
    into pieces (measure_pieces). The OSM directions of a road are separate SUMO edges whose axes lie a few metres
    apart; each is measured and the one carrying more lane length (two-way counted twice) is kept."""
    edges = {}
    for e in net.getEdges():
        if any(l.allows("passenger") for l in e.getLanes()):
            edges.setdefault(_base(e.getID()), []).append(e)
    roads = {}
    for base, options in sorted(edges.items()):
        results = [measure_pieces(e, support) for e in sorted(options, key=lambda e: e.getID())]

        def score(pieces):
            return sum(r.length * {"two_way": 2, "one_way": 1}.get(r.kind, 0) for r in pieces)

        for r in max(results, key=score):
            roads[r.base] = r
    return roads


def _arcs(roads):
    arcs = []
    for r in roads.values():
        if r.kind == "two_way":
            arcs += [(r.nodes[0], r.nodes[1], r.base + "-f"), (r.nodes[1], r.nodes[0], r.base + "-b")]
        elif r.kind == "one_way":
            a, b = r.nodes if r.direction == "f" else r.nodes[::-1]
            arcs.append((a, b, r.base + "-" + r.direction))
    return arcs


def _largest_scc(arcs):
    from awr.life.civil.activity.driving import _components

    links, nodes = {}, set()
    for a, b, _ in arcs:
        links.setdefault(a, []).append(b)
        nodes |= {a, b}
    comps = _components(sorted(nodes), links)
    best = max(comps, key=len) if comps else []
    return set(best)


def orient(roads):
    """Directions of the one-way roads for a strongly connected network (Robbins: in a depth-first search of the
    road graph, tree edges point away from the root and the other edges back towards it, so every road that is not a
    bridge lies on a cycle); the arcs of the largest strongly connected part."""
    adjacent = {}
    for r in roads.values():
        if r.kind != "none":
            a, b = r.nodes
            adjacent.setdefault(a, []).append((b, r.base))
            adjacent.setdefault(b, []).append((a, r.base))
    seen, depth, done = set(), {}, set()
    for root in sorted(adjacent):
        if root in depth:
            continue
        depth[root] = 0
        stack = [(root, iter(sorted(adjacent[root])))]
        while stack:
            v, it = stack[-1]
            for w, base in it:
                if base in done:
                    continue
                done.add(base)
                r = roads[base]
                if w not in depth:  # tree edge v -> w
                    depth[w] = depth[v] + 1
                    stack.append((w, iter(sorted(adjacent[w]))))
                    forward = (v, w)
                else:  # back edge towards the shallower end
                    forward = (v, w) if depth[w] < depth[v] else (w, v)
                if r.kind == "one_way":
                    r.direction = "f" if forward == r.nodes else "b"
                break
            else:
                stack.pop()
    keep = _largest_scc(_arcs(roads))
    return [a for a in _arcs(roads) if a[0] in keep and a[1] in keep]


def speed_limit(pts, window_m=1.0):
    """Speed limit (m/s) of the polyline `pts`: SPEED_MPS, lowered on its sharpest bend (heading change over
    `window_m`) to JUNCTION_LATERAL_MPS2 of lateral acceleration and YAW_RATE_MPS of yaw rate."""
    line = LineString(pts)
    if line.length <= window_m:
        return SPEED_MPS
    # evenly spaced samples: a sliver of a last segment would have a heading of rounding noise
    n = int(np.ceil(line.length / 0.25))
    s = np.linspace(0.0, line.length, n + 1)
    P = np.array([line.interpolate(v).coords[0][:2] for v in s])
    h = np.unwrap(np.arctan2(*np.diff(P, axis=0).T[::-1]))
    k = max(int(round(window_m / (line.length / n))), 1)
    turn = np.abs(h[k:] - h[:-k]).max() if len(h) > k else 0.0
    if turn < 1e-6:
        return SPEED_MPS
    radius = window_m / turn
    return float(min(SPEED_MPS, YAW_RATE_MPS * radius, np.sqrt(JUNCTION_LATERAL_MPS2 * radius)))


CAR_ROOM_M = 7.5  # a waiting car and its gap to the car behind


def write_plain(roads, arcs, net, out, lanes=None):
    """Plain SUMO input (nodes, edges) for the kept `arcs`; `lanes` overrides the lane shape of an edge."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    used = sorted({a for a, _, _ in arcs} | {b for _, b, _ in arcs})
    with open(out / "stage.nod.xml", "w") as f:
        f.write("<nodes>\n")
        ends = {}
        for r in roads.values():
            ends[r.nodes[0]], ends[r.nodes[1]] = r.axis.coords[0][:2], r.axis.coords[-1][:2]
        for nid in used:
            if net.hasNode(nid):
                node = net.getNode(nid)
                (x, y), kind = node.getCoord()[:2], node.getType()
            else:
                (x, y), kind = ends[nid], "priority"  # a node splitting a road at a narrow stretch
            kind = kind if kind in ("priority", "right_before_left") else "priority"
            f.write(f'  <node id={quoteattr(nid)} x="{x:.3f}" y="{y:.3f}" type="{kind}"/>\n')
        f.write("</nodes>\n")
    with open(out / "stage.edg.xml", "w") as f:
        f.write("<edges>\n")
        ids = {eid for _, _, eid in arcs}
        for a, b, eid in sorted(arcs, key=lambda r: r[2]):
            pts = lanes[eid] if lanes and eid in lanes else roads[eid[:-2]].lane(eid[-1])
            shape = " ".join(f"{x:.3f},{y:.3f}" for x, y in pts)
            # A car waiting at the end of an edge shorter than CAR_ROOM_M stands in the junction behind it, one waiting
            # at the end of a single-lane road driven both ways keeps the road from the other direction: such edges
            # have the right of way at a priority junction, so the junctions and roads without room empty first.
            alternating = roads[eid[:-2]].kind == "one_way" and {eid[:-2] + "-f", eid[:-2] + "-b"} <= ids
            priority = 4 if LineString(pts).length < CAR_ROOM_M or alternating else 3
            # The simulated length is the planned lane's, not that of netconvert's lane drawn to the junction outline.
            f.write(
                f'  <edge id={quoteattr(eid)} from={quoteattr(a)} to={quoteattr(b)} priority="{priority}" numLanes="1" '
                f'speed="{speed_limit(pts):.2f}" width="{LANE_WIDTH_M}" spreadType="center" allow="passenger" '
                f'length="{max(LineString(pts).length, 0.1):.3f}" shape="{shape}"/>\n'
            )
        f.write("</edges>\n")


def build(netconvert, plain, net_out, deleted=(), shapes=None):
    """netconvert of the plain input; `deleted` (from, to) edge pairs are not connected; `shapes`, when given, is the
    complete set of connections, (from, to) -> internal lane shape."""
    con = Path(plain) / "stage.con.xml"
    with open(con, "w") as f:
        f.write("<connections>\n")
        for (a, b), pts in sorted((shapes or {}).items()):
            shape = " ".join(f"{x:.3f},{y:.3f}" for x, y in pts)
            f.write(
                f'  <connection from={quoteattr(a)} to={quoteattr(b)} fromLane="0" toLane="0" '
                f'speed="{speed_limit(pts):.2f}" length="{max(LineString(pts).length, 0.1):.3f}" shape="{shape}"/>\n'
            )
        for a, b in sorted(deleted):
            f.write(f"  <delete from={quoteattr(a)} to={quoteattr(b)}/>\n")
        f.write("</connections>\n")
    cmd = [
        str(netconvert),
        "--node-files",
        str(Path(plain) / "stage.nod.xml"),
        "--edge-files",
        str(Path(plain) / "stage.edg.xml"),
        "--connection-files",
        str(con),
        "--no-turnarounds",
        "true",
        "--offset.disable-normalization",
        "true",
        "--junctions.internal-link-detail",
        "12",
        "--junctions.limit-turn-speed",
        str(JUNCTION_LATERAL_MPS2),
        "--geometry.min-radius.fix",
        "true",
        "--junctions.endpoint-shape",
        "true",
        "--internal-junctions.vehicle-width",
        str(VEHICLE_WIDTH_M),
        "--no-warnings",
        "true",
        "-o",
        str(net_out),
    ]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError(f"netconvert failed: {run.stderr.strip()[-2000:]}")


def body_supported(P, heading, support, margin=LANE_MARGIN_M):
    """Per pose: the corners and side middles of the car body, and of the body `margin` wider, and the tyre contacts
    on supported vehicle ground. Both bodies are checked: the probes of the wider body can step over a hole in the
    support that a corner of the body itself (as the tape QA probes it) falls into."""
    t = np.c_[np.cos(heading), np.sin(heading)]
    n = np.c_[-t[:, 1], t[:, 0]]
    ok = np.ones(len(P), bool)
    for half in sorted({CAR_HALF_WIDTH_M, CAR_HALF_WIDTH_M + margin}):
        for dl in (-CAR_HALF_LENGTH_M, 0.0, CAR_HALF_LENGTH_M):
            for dw in (-half, half):
                ok &= support.supported(P + t * dl + n * dw, "vehicle")
    for dl, dw in CONTACT_PROBES:
        ok &= support.supported(P + t * dl + n * dw, "vehicle")
    return ok


R_MIN_M = 4.5  # smallest turning radius of a junction curve
CURVE_STEP_M = 0.25
CURVE_HANDLES = (0.15, 0.25, 0.35, 0.45, 0.55, 0.7, 0.85)  # Bezier handle lengths, share of the chord


MAX_TURN_RAD = np.radians(150.0)  # sharper turns between two roads are not driven


def _bezier(p0, h0, p1, h1, support):
    """The supported cubic Bezier from pose (p0, h0) to (p1, h1) of least peak curvature (at most 1/R_MIN_M);
    None without one."""
    chord = float(np.linalg.norm(p1 - p0))
    if chord < CURVE_STEP_M:
        return np.array([p0, p1])
    t0, t1 = np.array([np.cos(h0), np.sin(h0)]), np.array([np.cos(h1), np.sin(h1)])
    u = np.linspace(0.0, 1.0, max(int(chord / CURVE_STEP_M * 1.5), 8))[:, None]
    best, best_k = None, 1.0 / R_MIN_M
    for a in CURVE_HANDLES:
        for b in CURVE_HANDLES:
            c0, c1 = p0 + t0 * a * chord, p1 - t1 * b * chord
            P = (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * c0 + 3 * (1 - u) * u**2 * c1 + u**3 * p1
            d1 = 3 * (1 - u) ** 2 * (c0 - p0) + 6 * (1 - u) * u * (c1 - c0) + 3 * u**2 * (p1 - c1)
            d2 = 6 * (1 - u) * (c1 - 2 * c0 + p0) + 6 * u * (p1 - 2 * c1 + c0)
            speed = np.linalg.norm(d1, axis=1)
            if speed.min() < 1e-6:
                continue
            k = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / speed**3
            if k.max() > best_k:
                continue
            if body_supported(P, np.arctan2(d1[:, 1], d1[:, 0]), support).all() and path_supported(P, support):
                best, best_k = P, k.max()
    return best


SEARCH_STEP_M = 1.5  # forward search: arc primitives of this length
SEARCH_EXPANSIONS = 4000


def _search(p0, h0, p1, h1, support):
    """Bounded forward search (hybrid A*) from pose (p0, h0) to (p1, h1): arcs of curvature 0, +-1/(2 R_MIN_M) and
    +-1/R_MIN_M with the car body supported, finished by a supported Bezier (_bezier) within 12 m of the goal.
    The emitted path keeps the exact start and goal poses; the pose lattice only bounds the exploration."""
    import heapq

    chord = float(np.linalg.norm(p1 - p0))
    budget = 2.5 * chord + 20.0
    s = np.linspace(0.0, SEARCH_STEP_M, int(SEARCH_STEP_M / CURVE_STEP_M) + 1)[1:]

    def leg(x, y, h, k):
        yaw = h + k * s
        if abs(k) < 1e-12:
            pts = np.c_[x + s * np.cos(h), y + s * np.sin(h)]
        else:
            pts = np.c_[x + (np.sin(yaw) - np.sin(h)) / k, y + (np.cos(h) - np.cos(yaw)) / k]
        return pts, yaw

    def key(x, y, h):
        return int(round(x / 0.5)), int(round(y / 0.5)), int(round((h % (2 * np.pi)) / (2 * np.pi) * 72)) % 72

    nodes = [(p0[0], p0[1], h0, 0.0, -1, None)]
    queue, best, expanded = [(chord, 0)], {key(p0[0], p0[1], h0): 0.0}, 0
    while queue and expanded < SEARCH_EXPANSIONS:
        _, i = heapq.heappop(queue)
        x, y, h, cost, _, _ = nodes[i]
        expanded += 1
        d = float(np.hypot(p1[0] - x, p1[1] - y))
        if d <= 12.0:
            tail = _bezier(np.array([x, y]), h, p1, h1, support)
            if tail is not None:
                parts, j = [tail], i
                while j > 0:
                    parts.append(nodes[j][5])
                    j = nodes[j][4]
                return np.concatenate([np.array([p0])] + parts[::-1][:-1] + [tail[1:]])
        for k in (0.0, -0.5 / R_MIN_M, 0.5 / R_MIN_M, -1.0 / R_MIN_M, 1.0 / R_MIN_M):
            pts, yaw = leg(x, y, h, k)
            c = cost + SEARCH_STEP_M
            if c + np.hypot(p1[0] - pts[-1, 0], p1[1] - pts[-1, 1]) > budget:
                continue
            q = key(pts[-1, 0], pts[-1, 1], yaw[-1])
            if c >= best.get(q, np.inf):
                continue
            if not body_supported(pts, yaw, support).all():
                continue
            best[q] = c
            nodes.append((pts[-1, 0], pts[-1, 1], yaw[-1], c, i, pts))
            heapq.heappush(queue, (c + np.hypot(p1[0] - pts[-1, 0], p1[1] - pts[-1, 1]), len(nodes) - 1))
    return None


def junction_curve(p0, h0, p1, h1, support):
    """A supported junction curve from pose (p0, h0) to (p1, h1): the smoothest Bezier, else a bounded forward
    search; None without one or when the turn is sharper than MAX_TURN_RAD."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    if abs((h1 - h0 + np.pi) % (2 * np.pi) - np.pi) > MAX_TURN_RAD:
        return None
    curve = _bezier(p0, h0, p1, h1, support)
    if curve is None:
        curve = _search(p0, h0, p1, h1, support)
    # as the converter poses the car on it: heading along the polyline
    return curve if curve is not None and path_supported(curve, support) else None


def _heading(a, b):
    return float(np.arctan2(b[1] - a[1], b[0] - a[0]))


def connections(vnet):
    """(from, to) edge pairs of the built net with their internal lane shape (internal lanes in sequence joined)."""
    out = {}
    for e in vnet.getEdges(withInternal=False):
        for lane in e.getLanes():
            for conn in lane.getOutgoing():
                pts, via = [], conn.getViaLaneID()
                while via:
                    inner = vnet.getLane(via)
                    pts += [p[:2] for p in inner.getShape()]
                    nxt = [c for c in inner.getOutgoing() if c.getViaLaneID()]
                    via = nxt[0].getViaLaneID() if nxt else ""
                out[(e.getID(), conn.getTo().getID())] = pts
    return out


def footprint_support(shape, support):
    """Share of car footprints along the lane `shape` (every 0.5 m) on supported vehicle ground."""
    line = LineString(shape)
    if line.length < 0.5:
        return 1.0, []
    s = np.arange(0.0, line.length + 1e-9, 0.5)
    P, n = _frame(line, s)
    t = np.c_[n[:, 1], -n[:, 0]]
    corners = np.concatenate(
        [P + t * sx * CAR_HALF_LENGTH_M + n * sy * CAR_HALF_WIDTH_M for sx in (-1, 1) for sy in (-1, 1)]
    )
    ok = support.supported(corners, "vehicle").reshape(4, -1).all(0)
    return float(ok.mean()), P[~ok]


RETREAT_STEP_M = 2.0  # a stop line moves back from the junction in these steps
RETREAT_MAX_M = 24.0
MIN_EDGE_M = 2.0  # length of an edge between its stop lines at least


def _pose(line, at):
    d = min(max(at, 0.0), line.length)
    a, b = line.interpolate(max(d - 0.5, 0.0)), line.interpolate(min(d + 0.5, line.length))
    return np.array(line.interpolate(d).coords[0][:2]), float(np.arctan2(b.y - a.y, b.x - a.x))


def plan_junctions(arcs, roads, support):
    """Stop lines and junction curves. Each lane ends END_M before its junctions; for a turn (no U-turns) without a
    supported junction_curve between them, the stop lines of both lanes move back by RETREAT_STEP_M (up to
    RETREAT_MAX_M, leaving MIN_EDGE_M of edge) until one is found. Returns the cut lane shapes by edge, the curves by
    (from, to) and the turns left without a curve."""
    lines = {eid: LineString(roads[eid[:-2]].lane(eid[-1])) for _, _, eid in arcs}
    into, out_of = {}, {}
    for a, b, eid in arcs:
        out_of.setdefault(a, []).append(eid)
        into.setdefault(b, []).append(eid)
    turns = {(i, o) for j in into for i in into[j] for o in out_of.get(j, ()) if i[:-2] != o[:-2]}
    retreat = {
        (eid, side): max(min(END_M, (line.length - MIN_EDGE_M) / 2), 0.0)
        for eid, line in lines.items()
        for side in (0, 1)
    }

    def room(eid, side):
        return lines[eid].length - retreat[(eid, 1 - side)] - retreat[(eid, side)] - MIN_EDGE_M

    def curve(i, o):
        p0, h0 = _pose(lines[i], lines[i].length - retreat[(i, 1)])
        p1, h1 = _pose(lines[o], retreat[(o, 0)])
        return junction_curve(p0, h0, p1, h1, support)

    curves, pending = {}, set(turns)
    while pending:
        failed = set()
        for i, o in sorted(pending):
            c = curve(i, o)
            if c is None:
                failed.add((i, o))
                curves.pop((i, o), None)
            else:
                curves[(i, o)] = c
        moved = {
            key
            for i, o in failed
            for key in ((i, 1), (o, 0))
            if retreat[key] + RETREAT_STEP_M <= RETREAT_MAX_M and room(*key) >= RETREAT_STEP_M
        }
        if not moved:
            break
        for key in moved:
            retreat[key] += RETREAT_STEP_M
        pending = failed | {(i, o) for i, o in turns if (i, 1) in moved or (o, 0) in moved}
    cut = {}
    for eid, line in lines.items():
        s0, s1 = retreat[(eid, 0)], max(line.length - retreat[(eid, 1)], retreat[(eid, 0)] + 0.1)
        s = np.arange(s0, s1 + 1e-9, SAMPLE_M)
        s = np.r_[s, s1] if s1 - s[-1] > 1e-6 else s
        cut[eid] = [line.interpolate(v).coords[0][:2] for v in s]
    # A junction curve must not sweep over the lanes of other roads beyond their stop lines: SUMO models the
    # conflicts of junction curves among themselves only.
    for (i, o), c in list(curves.items()):
        near = LineString(c).buffer(2 * CAR_HALF_LENGTH_M + 2.0)
        for eid, pts in cut.items():
            if eid[:-2] in (i[:-2], o[:-2]) or len(pts) < 2 or not near.intersects(LineString(pts)):
                continue
            if not lanes_clear(c, pts) or not lanes_clear(c, pts[::-1]):
                del curves[(i, o)]
                break
    missing = sorted(turns - set(curves))
    return cut, {k: [tuple(map(float, p)) for p in v] for k, v in curves.items()}, missing


def _edge_scc(edges, links):
    from awr.life.civil.activity.driving import _components

    graph = {}
    for a, b in links:
        graph.setdefault(a, []).append(b)
    comps = _components(sorted(edges), graph)
    return set(max(comps, key=len)) if comps else set()


def path_supported(pts, support, step=0.05, margin=0.02):
    """Whether the car following the polyline `pts` (heading along it, every `step`) has the corners and side
    middles of its body, `margin` wider, on supported ground. The default margin covers any pose between the
    samples, where the tape QA checks the body (margin 0) at the recorded positions."""
    line = LineString(pts)
    if line.length < 1e-6:
        return bool(body_supported(np.asarray(pts[:1], float), np.zeros(1), support, margin)[0])
    s = np.r_[np.arange(0.0, line.length, step), line.length]
    P, n = _frame(line, s)
    return bool(body_supported(P, np.arctan2(-n[:, 0], n[:, 1]), support, margin).all())


def _scc_length(edges, links, length):
    keep = _edge_scc(edges, {(a, b) for a, b in links if a in edges and b in edges})
    return sum(length[e] for e in keep), keep


def one_way_where_possible(roads, keep, links, length):
    """Fix single-lane roads to one direction wherever the strongly connected network still reaches every road of
    `keep` (a two-way road may lose the lane that only led into the dropped direction); longest roads first, the
    Robbins direction tried first. The roads left with both directions are the ones the network cannot do without
    in either direction. Returns the kept edges."""
    keep = set(keep)
    covered = {e[:-2] for e in keep}
    for b in sorted(
        (b for b, r in roads.items() if r.kind == "one_way" and {b + "-f", b + "-b"} <= keep),
        key=lambda b: -roads[b].length,
    ):
        if not {b + "-f", b + "-b"} <= keep:
            continue
        for drop in ("b", "f") if roads[b].direction == "f" else ("f", "b"):
            trial = _scc_length(keep - {b + "-" + drop}, links, length)[1]
            if {e[:-2] for e in trial} == covered:
                keep = trial
                break
    return keep


def set_planned_shapes(net_path, cut, curves):
    """Replace the lane shapes netconvert drew (lanes extended to the junction outline, internal lanes trimmed to
    them) by the planned geometry, so that SUMO computes the conflict points of the junction curves on the curves the
    cars are recorded on. The simulated lengths are already the planned ones."""
    from xml.etree import ElementTree as ET

    import sumolib
    from shapely.ops import substring

    net = sumolib.net.readNet(str(net_path), withInternal=True)
    shapes = {e + "_0": np.asarray(pts, float) for e, pts in cut.items()}
    for e in net.getEdges(withInternal=False):
        for conn in e.getLanes()[0].getOutgoing():
            curve = LineString(curves[(e.getID(), conn.getTo().getID())])
            via, at = conn.getViaLaneID(), 0.0
            while via:
                inner = net.getLane(via)
                part = substring(curve, at, min(at + inner.getLength(), curve.length))
                shapes[via] = (
                    np.asarray(part.coords)[:, :2]
                    if part.geom_type == "LineString"
                    else np.asarray([part.coords[0]] * 2)
                )
                at += inner.getLength()
                nxt = [c for c in inner.getOutgoing() if c.getViaLaneID()]
                via = nxt[0].getViaLaneID() if nxt else ""
    tree = ET.parse(net_path)
    for lane in tree.getroot().iter("lane"):
        if lane.get("id") in shapes:
            lane.set("shape", " ".join(f"{x:.3f},{y:.3f}" for x, y in shapes[lane.get("id")]))
    tree.write(net_path, encoding="UTF-8", xml_declaration=True)


def compile_stage(net_path, support, bbox, out, netconvert):
    """Compile the stage vehicle network into `out` (stage.net.xml, the plain inputs, stage-geometry.json and
    stage-network.json); returns the report.

    Every measured road is offered in both directions: a two-way road on its two lanes, a road wide enough for one
    car only ('single_lane' in stage-network.json) on its one lane. A single-lane road is then made one-way wherever
    the network stays strongly connected (one_way_where_possible); the remaining ones ('alternating') are used by one
    direction at a time under the traffic controller. plan_junctions places the stop lines and the junction curves of all turns, every
    lane between its stop lines is checked with the car body as the tape QA checks it, and the largest strongly
    connected network over the supported curves is kept (no turnarounds, so a spur to a dead end is left out).
    netconvert builds it with the curves as its complete set of connections.

    netconvert extends a lane to the junction outline and may shift it slightly; the network's geometry for the tape
    is therefore the planned one (stage-geometry.json): an edge's lane between its stop lines and a connection's curve,
    to which the converter maps SUMO's lane positions in proportion to the lengths."""
    import sumolib

    net = sumolib.net.readNet(str(net_path))
    roads = measure(net, support, bbox)
    offered = []
    for r in roads.values():
        if r.kind != "none":
            offered += [(r.nodes[0], r.nodes[1], r.base + "-f"), (r.nodes[1], r.nodes[0], r.base + "-b")]
    cut, curves, missing = plan_junctions(offered, roads, support)
    bad = {eid for eid, pts in cut.items() if not path_supported(pts, support)}
    length = {e: LineString(p).length for e, p in cut.items()}
    keep = _scc_length({e for e in cut if e not in bad}, set(curves), length)[1]
    orient(roads)  # Robbins directions are tried first
    keep = one_way_where_possible(roads, keep, set(curves), length)
    arcs = [arc for arc in offered if arc[2] in keep]
    used = {pair: c for pair, c in curves.items() if pair[0] in keep and pair[1] in keep}
    built = Path(out) / "stage.net.xml"
    write_plain(roads, arcs, net, out, cut)
    build(netconvert, out, built, shapes=used)
    vnet = sumolib.net.readNet(str(built), withInternal=True)
    links = set(connections(vnet))
    if {e.getID() for e in vnet.getEdges(withInternal=False)} != keep or links != set(used):
        raise RuntimeError(
            f"netconvert changed the network: edges {len(keep)} -> {len(vnet.getEdges(withInternal=False))}, "
            f"connections {len(used)} -> {len(links)}"
        )
    set_planned_shapes(built, {e: cut[e] for e in keep}, used)
    for r in roads.values():
        if r.kind != "none" and not any(r.base + d in keep for d in ("-f", "-b")):
            r.kind = "none"
    report = dict(
        roads={kind: sum(r.kind == kind for r in roads.values()) for kind in ("two_way", "one_way", "none")},
        edges=len(keep),
        connections=len(used),
        lane_km=round(sum(LineString(cut[e]).length for e in keep) / 1000.0, 2),
        offered_edges=len(offered),
        unsupported_edges=sorted(bad),
        missing_turns=missing,
    )
    (Path(out) / "stage-geometry.json").write_text(
        json.dumps(
            dict(
                schema="awr.life.ambient.stage_geometry.v1",
                edges={e: np.round(cut[e], 3).tolist() for e in sorted(keep)},
                connections=[
                    dict(source=a, target=b, points=np.round(c, 3).tolist()) for (a, b), c in sorted(used.items())
                ],
            )
        )
    )
    (Path(out) / "stage-network.json").write_text(
        json.dumps(
            dict(
                schema="awr.life.ambient.stage_network.v2",
                bbox=list(bbox),
                report=report,
                roads={
                    b: dict(
                        kind={"one_way": "single_lane"}.get(r.kind, r.kind),
                        length_m=round(r.length, 1),
                        edges=[b + d for d in ("-f", "-b") if b + d in keep],
                    )
                    for b, r in sorted(roads.items())
                },
                alternating=sorted(b for b, r in roads.items() if r.kind == "one_way" and {b + "-f", b + "-b"} <= keep),
            ),
            indent=1,
        )
        + "\n"
    )
    return report
