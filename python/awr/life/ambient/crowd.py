"""People of the stage: the places they go to, what each does there, and their walking, moved by JuPedSim.

Places are standing spots in front of the mapped facilities (post office, banks, shops, town hall, ...), at the
buildings' fronts and inside the parks, on walkway cells clear of the car lanes. A person has a persona that picks the
next place when a stay ends: errands go between facilities and houses, strollers between the spots of a park, a group
(two or three) walks and stays together behind its leader, and a conversation is two or more people standing face to
face; people arriving where somebody stands may stop and talk.

JuPedSim (AnticipationVelocityModel, direct steering) moves everybody and keeps the bodies apart. Each tick the
crowd hands every agent a target on its route (a point LOOKAHEAD_M ahead) and a desired speed that changes by at most
ACCEL_MPS2 / DECEL_MPS2, so recorded speeds stay smooth; holds before cars (awr.life.ambient.coupling) set the desired
speed to zero in the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from shapely.geometry import Point, Polygon

from awr.life.ambient.coupling import box_gap, person_box

RADIUS_M = 0.6  # JuPedSim disc; walking bodies are about 1.0 x 0.74 m (awr.life.civil.body)
TIME_GAP_S = 0.35  # JuPedSim's own headway; the crowd's governor keeps HEADWAY_S over a wider corridor
HEADWAY_S = 1.0
CORRIDOR_M = 1.6  # half-width of the corridor ahead in which the governor looks for crossing people
NEAR_LANE_M = 0.5  # ... and beyond the discs' corridor, for anybody else
CUT_IN_MPS = 0.6  # ... where the speed stays within this of what a neighbour entering the corridor leaves
PASS_MPS = 0.5  # ... and no lower than this
CROSSING_DEG = 45.0  # walking directions further apart than this (and not oncoming) cross
NARROW_M = 1.8  # routes avoid walkway narrower than twice this (clearance from the edge)
NARROW_COST = 2.0
STALL_S = 2.5  # a walker blocked this long plans its way again
NO_PROGRESS_S = 8.0  # ... and a walker that has not got PROGRESS_M further along its route in this long
PROGRESS_M = 1.0
GIVE_WAY_M = 1.2  # a person in the way of a waiting car stands this far beside the car's path (+ half width)
GIVE_WAY_OTHER_M = 0.8  # ... and this far from the paths of the other cars
GIVE_WAY_REACH_M = 15.0
GIVE_WAY_AFTER_S = 1.0  # ... once the car has waited for it this long
GIVE_WAY_PASS_M = 4.0  # it stays at its spot until that car has driven this far (or has gone) ...
GIVE_WAY_HOLD_S = 25.0  # ... at most this long
GIVE_WAY_TRIES = 12  # nearest spots tried for a way there (spots beside a standing car's body are left out)
CAR_CHECK_S = 1.0  # how often a walker looks for standing cars on the next ROUTE_CHECK_M of its route
ROUTE_CHECK_M = 8.0
STANDING_M = 1.25  # routes keep this far from people standing (centre to centre)
STANDING_COST = 20.0
STANDING_CAR_M = 1.8  # routes keep this far from the body of a standing car where they can
STANDING_CAR_COST = 100.0
STANDING_CAR_BLOCK_M = 1.2  # ... and never come nearer than this (a walking body's reach and the coupling's TOUCH_M,
# awr.life.ambient.coupling, with a quarter cell to spare)
STANDING_CAR_MIN_M = 0.95  # ... or than this where nothing else leads (the reach and TOUCH_M with a little to spare)
WAITING_AHEAD_M = 8.0  # routes keep off this stretch of a standing car's way ahead of its front where they can
WAITING_PATH_M = 0.7  # ... widened by this beyond the car's half width (a body's half width and the coupling's
# YIELD_GAP_M, awr.life.ambient.coupling): cells there cost WAITING_PATH_COST more
WAITING_PATH_COST = 5.0
REPLANS = 2
NEAR_END_M = 2.5  # a walker blocked this close to its slot stands there instead
SETTLE_S = 1.0  # a standing person stands where it stopped after this
PAUSE_S = 1.0  # a walker that moved back and forth waits this long
BLOCK_M = 0.1  # a disc ahead this close to contact leaves a walker (next to) no speed
SLIDE_S = 1.5
SLIDE_MPS = 0.5
PINNED_S = 1.0  # a walker that wants to go and has not moved this long asks the idle people touching it to step aside
EDGE_M = 0.3  # JuPedSim's area is the walkway eroded by this
WALL_BUFFER_M = 0.1
REACTION_S = 0.6
LOOKAHEAD_M = 2.0
ACCEL_MPS2 = 1.0
DECEL_MPS2 = 2.5
ARRIVE_MPS2 = 0.5
ARRIVED_M = 0.6  # a walker this close to the end of its route, and nearly stopped, has arrived
TURN_LIMIT = np.radians(50.0)  # largest angle between a walker's motion and its target direction
STEER_MPS = 0.2  # ... for walkers at least this fast
TURN_MPS2 = 4.0  # ... and no larger than this change of velocity within a tick
WALL_SOFT_M = 2.0  # walkers this near the area's edge approach it at no more than WALL_TOWARD_MPS at WALL_HARD_M
WALL_HARD_M = 1.4  # (JuPedSim's wall influence, twice the disc and wall buffer)
WALL_TOWARD_MPS = 0.3
REACH_MARGIN_M = 0.3  # routes keep this much more from the area's edge than JuPedSim's wall buffer
LATERAL_MPS2 = 1.2  # sideways acceleration on bends of the route
ROUND_M = 0.6  # corners of a route are rounded off by up to this
SUBSTEPS = 10  # JuPedSim iterations per tick
YAW_RATE = 2.5  # rad/s, recorded heading
BODY_WALK = (-0.423, 0.576, -0.369, 0.369)  # walking body extents (back, front, right, left; awr.life.ambient.coupling)
BODY_LOOK_S = 0.3  # two walkers whose bodies (extents by speed, awr.life.ambient.coupling.person_box) would come
BODY_GAP_M = 0.02  # ... within this of each other in this time slow down together
GRAZE_M = 0.05  # a turning body keeps this clear of a car's body
FACE_RATE = 1.2  # rad/s, turning on the spot towards a partner
SLOT_ROOM_M = 1.1  # a standing slot keeps this from the edge of the walkway
SLOT_SPACING_M = 1.3
SLOT_PASS_GAP_M = 0.2  # a person standing at a slot leaves a way past it for two walkers abreast (passing each other),
# this far apart and from it ...
SLOT_WINDOW_M = 6.0  # ... where the walkway within this distance of the slot is passable at all
TALK_M = 1.3  # distance between two people talking
GROUP_SIDE_M = 1.3  # a follower's lateral offset from its leader
SINGLE_FILE_M = 1.4  # distance between people of a group walking in single file
REJOIN_M = 8.0  # a follower ahead of its leader within this distance walks back to its side
LOST_M = 5.0  # a follower further than this from its leader goes its own way back to it (until half as far)
CATCH_S = 3.0  # ... planned again this often, or when its goal has moved CATCH_MOVE_M from the way's end
CATCH_MOVE_M = 2.0
OFFSET_BLEND_M = 3.0

DWELL_S = {
    "post_office": (40, 160),
    "bank": (30, 120),
    "townhall": (40, 150),
    "courthouse": (40, 150),
    "community_centre": (40, 180),
    "bus_station": (60, 240),
    "cafe": (60, 240),
    "supermarket": (30, 120),
    "place_of_worship": (30, 150),
    "school": (30, 90),
    "pitch": (30, 120),
    "park": (15, 90),
    "house": (20, 90),
    "parking": (10, 40),
}
ERRAND_WEIGHT = {
    "post_office": 3.0,
    "bank": 2.0,
    "townhall": 1.5,
    "courthouse": 1.0,
    "community_centre": 1.5,
    "bus_station": 2.0,
    "cafe": 3.0,
    "supermarket": 4.0,
    "place_of_worship": 1.0,
    "school": 1.0,
    "pitch": 0.5,
    "park": 1.5,
    "house": 0.25,
    "parking": 0.5,
}
PERSONAS = (("errand", 0.45), ("stroller", 0.25), ("group", 0.2), ("talk", 0.1))
TALK_CHANCE = 0.35  # a person arriving next to somebody standing alone stops to talk
ERRAND_TRIP_M = 100.0  # errands go to places about this far (straight line), less often much nearer or further


@dataclass
class Place:
    id: str
    kind: str
    area: str  # the POI, park or building it belongs to
    xy: np.ndarray
    slots: np.ndarray  # (k, 2) standing positions
    taken: list = field(default_factory=list)

    def free_slot(self):
        for k in range(len(self.slots)):
            if k not in self.taken:
                return k
        return None

    def free_slots(self):
        return [k for k in range(len(self.slots)) if k not in self.taken]


def reachable(walkway):
    """The cells routes may take (awr.life.ambient.walkway.path, which does not cut corners): the walkway eroded by
    JuPedSim's edge, a disc, the wall buffer and REACH_MARGIN_M. A diagonal corridor rasterises there as cells that
    touch only at their corners; the cell beside each such joint is added where the walkway holds a walker
    (EDGE_M+RADIUS_M+WALL_BUFFER_M from its edge). Then only the largest part joined side by side: a place in another
    part is an island, and a person starting there never leaves it (nor does anybody else reach it)."""
    reach = ndimage.binary_erosion(
        walkway.walk, iterations=int(np.ceil((EDGE_M + RADIUS_M + WALL_BUFFER_M + REACH_MARGIN_M) / walkway.cell_m))
    )
    room = ndimage.distance_transform_edt(np.pad(walkway.walk, 1))[1:-1, 1:-1] * walkway.cell_m
    holds = room >= EDGE_M + RADIUS_M + WALL_BUFFER_M
    for _ in range(8):
        add = np.zeros_like(reach)
        a, b, c, d = reach[:-1, :-1], reach[1:, 1:], reach[:-1, 1:], reach[1:, :-1]
        joint = a & b & ~c & ~d
        add[:-1, 1:] |= joint & holds[:-1, 1:]
        add[1:, :-1] |= joint & ~holds[:-1, 1:] & holds[1:, :-1]
        joint = c & d & ~a & ~b
        add[:-1, :-1] |= joint & holds[:-1, :-1]
        add[1:, 1:] |= joint & ~holds[:-1, :-1] & holds[1:, 1:]
        if not (add & ~reach).any():
            break
        reach |= add
    lab, n = ndimage.label(reach)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        reach = lab == sizes.argmax()
    return reach


def compile_places(walkway, semantics, bbox, *, seed=7):
    """Standing places of the stage from the semantics' POIs, open areas and buildings; all their slots reach the
    route network (`reachable`, within the snap of awr.life.ambient.walkway)."""
    from awr.life.ambient.walkway import _finite

    rng = np.random.default_rng(seed)
    network = np.where(reachable(walkway), 1.0, np.inf)

    def reached(q):
        return bool(np.isfinite(network[_finite(network, walkway.cell(q))]))

    ok = walkway.walk & (walkway.cost <= 1.0)
    room = ndimage.distance_transform_edt(np.pad(walkway.walk, 1))[1:-1, 1:-1] * walkway.cell_m
    ok &= room >= SLOT_ROOM_M + 0.3
    rows, cols = np.nonzero(ok)
    xy = np.c_[walkway.x0 + (cols + 0.5) * walkway.cell_m, walkway.y0 + (rows + 0.5) * walkway.cell_m]

    def inside(p):
        return bbox[0] <= p[0] <= bbox[2] and bbox[1] <= p[1] <= bbox[3]

    import shapely

    pts = shapely.points(xy)
    places = []

    def add(area, kind, choices, spacing, limit):
        got = []
        for k in choices:
            p = xy[k]
            if any(np.hypot(*(p - q)) < spacing for q in got) or any(np.hypot(*(p - q.xy)) < 3.0 for q in places):
                continue
            slots = _slots(walkway, room, p)
            if len(slots) < 2 or not all(reached(q) for q in slots):
                continue
            got.append(p)
            places.append(Place(f"{kind}:{area}#{len(got)}", kind, area, p, slots))
            if len(got) >= limit:
                break

    areas = {a["id"]: a for a in semantics.get("open_areas", [])}
    for poi in sorted(semantics.get("pois", []), key=lambda p: p["id"]):
        if not inside(poi["xy"]) or poi["kind"] not in DWELL_S:
            continue
        geom = Polygon(poi["polygon"]) if poi.get("polygon") else Point(poi["xy"])
        if poi["kind"] in ("park", "pitch") and poi.get("polygon"):
            choices = rng.permutation(np.flatnonzero(shapely.contains(geom, pts)))
            add(poi["id"], poi["kind"], choices, 8.0, 30)
        else:
            d = shapely.distance(geom, pts)
            choices = np.flatnonzero((d >= 1.0) & (d <= 14.0))
            add(poi["id"], poi["kind"], choices[np.argsort(d[choices], kind="stable")], 4.0, 3)
    for a in areas.values():
        if a["id"] not in {p.area for p in places} and a.get("polygon"):
            choices = rng.permutation(np.flatnonzero(shapely.contains(Polygon(a["polygon"]), pts)))
            add(a["id"], a["kind"], choices, 8.0, 30)
    for b in sorted(semantics.get("buildings", []), key=lambda b: b["id"]):
        poly = Polygon(b["polygon"])
        if not inside(poly.centroid.coords[0]):
            continue
        d = shapely.distance(poly, pts)
        choices = np.flatnonzero((d >= 0.8) & (d <= 6.0))
        add(b["id"], "house", choices[np.argsort(d[choices], kind="stable")], 1e9, 1)
    return places


def _blocks(walkway, room, qs):
    """Whether people standing at the points `qs` (one or several) cut the way past them for two walkers abreast,
    SLOT_PASS_GAP_M apart (two walkers meeting there): the cells their midpoint may take (clear of the walkway's edge
    as JuPedSim's area, its wall buffer and a disc and a half with half the gap require) within SLOT_WINDOW_M of the
    points fall apart into more pieces (of at least a square metre) without those where the nearer one would come
    within SLOT_PASS_GAP_M of a person standing. On a walkway too narrow for that, a slot blocks it (a building front
    along a narrow lane: one walker squeezing past and one meeting it there stop each other for good). The slots of a
    place are tested together, all taken: a full place on a narrow street closes it as one person does."""
    qs = np.atleast_2d(np.asarray(qs, float))
    c = walkway.cell_m
    centre = qs.mean(0)
    r0, c0 = walkway.cell(centre)
    k = int(np.ceil((SLOT_WINDOW_M + np.hypot(*(qs - centre).T).max()) / c))
    rs = slice(max(r0 - k, 0), min(r0 + k + 1, room.shape[0]))
    cs = slice(max(c0 - k, 0), min(c0 + k + 1, room.shape[1]))
    usable = room[rs, cs] >= EDGE_M + WALL_BUFFER_M + 2 * RADIUS_M + SLOT_PASS_GAP_M / 2
    yy, xx = np.mgrid[rs, cs]
    near = np.zeros(yy.shape, bool)
    for q in qs:
        rq, cq = walkway.cell(q)
        near |= np.hypot(yy - rq, xx - cq) * c < 3 * RADIUS_M + 1.5 * SLOT_PASS_GAP_M
    eight = np.ones((3, 3), bool)  # routes step diagonally too
    before, _ = ndimage.label(usable, structure=eight)
    after, _ = ndimage.label(usable & ~near, structure=eight)
    crumb = 1.0 / (c * c)
    for lab in set(np.unique(before[near & usable])) - {0}:
        parts, sizes = np.unique(after[(before == lab) & (after > 0)], return_counts=True)
        if (sizes >= crumb).sum() > 1:
            return True
    return False


def _slots(walkway, room, p):
    """Standing positions around `p`: the spot and neighbours SLOT_SPACING_M away with room, all of them taken
    together not cutting the way past them (_blocks); none where the spot itself does."""
    if _blocks(walkway, room, p):
        return []
    out = [p]
    for a in np.radians(np.arange(0, 360, 45)):
        q = p + SLOT_SPACING_M * np.array([np.cos(a), np.sin(a)])
        r, c = walkway.cell(q)
        if (
            not (0 <= r < room.shape[0] and 0 <= c < room.shape[1])
            or room[r, c] < SLOT_ROOM_M
            or walkway.cost[r, c] > 1
        ):
            continue
        if not all(np.hypot(*(q - o)) >= SLOT_SPACING_M - 0.01 for o in out):
            continue
        if not _blocks(walkway, room, out + [q]):
            out.append(q)
        if len(out) >= 4:
            break
    return np.array(out)


@dataclass
class Person:
    i: int
    persona: str
    speed: float  # preferred walking speed (m/s)
    leader: int | None = None
    side: float = 0.0  # follower: +1 left, -1 right of the leader
    followers: list = field(default_factory=list)
    state: str = "stay"  # 'walk' | 'stay'
    place: Place | None = None
    slot: int | None = None
    stand: np.ndarray | None = None  # where the person stands during a stay
    face: np.ndarray | None = None  # point faced while standing
    until: float = 0.0
    route: np.ndarray | None = None
    arc: np.ndarray | None = None
    vmax: np.ndarray | None = None  # speed allowed along the route (bends, arrival)
    s: float = 0.0
    partner: int | None = None
    area: str | None = None  # a stroller's park
    allowed: set | None = None  # the only destinations (a query's person)
    offset: np.ndarray | None = None  # follower: lateral offset along the leader's route
    behind: float = SINGLE_FILE_M  # follower: distance behind the leader in single file
    recent: list = field(default_factory=list)
    hold: bool = False  # waiting for a car (awr.life.ambient.coupling)
    hold_standing: bool = False  # ... that stands (it does not pass: the walker plans around it)
    held_s: float = 0.0  # how long it has been waiting for cars (counting down while free)
    squeeze_until: float = -1.0  # until then it keeps only clear of cars' bodies and braking distances
    still: float = 0.0  # time standing still (s)
    pause_until: float = -1.0
    last: np.ndarray = field(default_factory=lambda: np.zeros(2))
    intent: np.ndarray | None = None  # intended velocity this tick, before holds
    stalled: float = 0.0
    replans: int = 0
    blocking: tuple | None = None  # (car id, car path polyline, half width): a waiting car waits for this person
    blocking_since: float | None = None
    give: np.ndarray | None = None  # where the person stands to let that car pass
    give_path: list | None = None  # [route, arcs, progress] of the way there (around the standing cars)
    gave_for: tuple | None = None  # (car id, where it stood, since): the car it stands aside for
    mark: tuple = (0.0, 0.0, None)  # (time, progress, route) when the walker last made headway on its route
    room: tuple | None = None  # (direction, until) of a step aside for a walker pinned against this person
    checked: float = -1e9  # last look for standing cars on the route
    avoided: set = field(default_factory=set)  # standing cars the route was planned around
    slide: tuple | None = None  # (direction, until): moving clear of people in contact
    catch: list | None = None  # follower away from its leader: [route, arcs, progress, planned at, speed profile]
    heading: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))  # JuPedSim's direction of motion
    v: float = 0.0
    yaw: float = 0.0


class Crowd:
    def __init__(self, walkway, places, people=150, *, seed=7, step_s=0.1, roles=None):
        """`roles`: key -> dict(places=[place ids]) for people whose destinations are fixed (the persons of
        queries); they are errands among those places only and the first people of the crowd."""
        import jupedsim as jps

        self.jps, self.walkway, self.places, self.step_s = jps, walkway, places, step_s
        self.rng = rng = np.random.default_rng(seed)
        area = walkway.polygon.buffer(-EDGE_M)
        if area.geom_type != "Polygon":
            area = max(area.geoms, key=lambda g: g.area)
        import shapely

        shapely.prepare(area)
        self.area = area
        # targets keep a disc and JuPedSim's wall buffer from the edge: a walker heading into the buffer has the
        # component towards the wall removed at once, a sudden turn
        self.inner = area.buffer(-(RADIUS_M + WALL_BUFFER_M))
        shapely.prepare(self.inner)
        # the wall segments as JuPedSim sees them, with each one's neighbours along its ring
        a, b, prev, nxt = [], [], [], []
        for ring in (area.exterior, *area.interiors):
            c = np.asarray(ring.coords)[:, :2]
            k, base = len(c) - 1, sum(map(len, a))
            a.append(c[:-1]), b.append(c[1:])
            prev.append(base + (np.arange(k) - 1) % k), nxt.append(base + (np.arange(k) + 1) % k)
        self.wall_a, self.wall_b = np.concatenate(a), np.concatenate(b)
        self.wall_prev, self.wall_next = np.concatenate(prev), np.concatenate(nxt)
        self.walls = shapely.STRtree(shapely.linestrings(np.stack([self.wall_a, self.wall_b], 1)))
        reach = reachable(walkway)
        room = ndimage.distance_transform_edt(np.pad(walkway.walk, 1))[1:-1, 1:-1] * walkway.cell_m
        self.room = room
        self.cost = np.where(reach, walkway.cost * np.where(room < NARROW_M, NARROW_COST, 1.0), np.inf)
        self.sim = jps.Simulation(
            model=jps.AnticipationVelocityModel(rng_seed=seed), geometry=area, dt=step_s / SUBSTEPS
        )
        stage = self.sim.add_direct_steering_stage()
        self.journey = self.sim.add_journey(jps.JourneyDescription([stage]))
        self.stage = stage
        self.parks = sorted({p.area for p in places if p.kind == "park"})
        self.people, self.agent = [], []
        self.roles = {}
        self._populate(people, roles or {})
        self.t = 0.0
        self.xy = np.array([self.sim.agent(a).position for a in self.agent])
        # where each person stands at the start: the place of the first stay (awr.life.ambient.export)
        self.events = [
            dict(t=0.0, person=per.i, kind="start", place=per.place.id) for per in self.people if per.place is not None
        ]
        self.cars = []  # standing cars (id, centre, yaw, half length, half width), set by awr.life.ambient.coupling
        self.corridors = []  # every car's (id, body and path ahead, half width), set by awr.life.ambient.coupling
        self.waiting = {}  # standing cars' ways ahead of their fronts (id -> (path, half width)), the same
        self._waiting_cost = (None, [])
        self.bodies = []  # every car's (id, centre, yaw, half length, half width), set by awr.life.ambient.coupling

    # ---- population ----
    def _populate(self, n, roles):
        rng, places = self.rng, self.places
        kinds, weights = zip(*PERSONAS, strict=False)
        counts = np.floor(np.array(weights) * n).astype(int)
        counts[0] += n - counts.sum()
        quota = dict(zip(kinds, counts, strict=False))
        order = rng.permutation(len(places))
        free = [places[k] for k in order]
        i = 0
        by_id = {p.id: p for p in places}
        for key, role in roles.items():
            allowed = [by_id[k] for k in role["places"]]
            starts = [by_id[k] for k in role.get("start", role["places"])]
            start = next(
                (
                    starts[k]
                    for k in rng.permutation(len(starts))
                    if starts[k].free_slot() is not None and self._ok(starts[k].slots[starts[k].free_slot()])
                ),
                None,
            )
            if start is None:
                raise RuntimeError(f"No free standing place for role {key}")
            per = Person(i, "errand", rng.uniform(1.15, 1.35))
            per.allowed = {p.id for p in allowed}
            self._stay(per, start, role.get("stay_s", 0.0) + rng.uniform(5.0, 40.0))
            self.people.append(per)
            self.roles[key] = i
            i += 1
            quota["errand"] -= 1

        def take(kind=None, n=1):
            for p in free:
                if (kind is None or p.kind in kind) and sum(self._ok(p.slots[k]) for k in p.free_slots()) >= n:
                    return p
            return None

        while quota["talk"] >= 2:
            k = int(min(rng.integers(2, 5), quota["talk"]))
            p = take(("park", "house", "cafe", "post_office", "bus_station", "supermarket"))
            ring = self._ring(p.xy, k) if p is not None else None
            if ring is None:
                free.remove(p) if p is not None else None
                if p is None:
                    break
                continue
            free.remove(p)
            members = []
            for q in ring:
                per = Person(i, "talk", rng.uniform(1.1, 1.4))
                per.stand, per.face, per.place = q, p.xy, p
                per.until = rng.uniform(100.0, 400.0)
                members.append(per)
                i += 1
            for per in members:
                per.partner = members[(members.index(per) + 1) % k].i
            self.people += members
            quota["talk"] -= k
        quota["errand"] += quota.pop("talk")
        while quota["group"] >= 2:
            k = int(min(rng.integers(2, 4), quota["group"]))
            p = take(n=k)
            if p is None:
                break
            lead = Person(i, str(rng.choice(["errand", "stroller"], p=[0.6, 0.4])), rng.uniform(1.0, 1.3))
            self._stay(lead, p, rng.uniform(0.0, 60.0))
            self.people.append(lead)
            i += 1
            for m in range(k - 1):
                f = Person(i, "follower", lead.speed + 0.3, leader=lead.i, side=(1.0, -1.0)[m])
                slot = next((s for s in p.free_slots() if self._ok(p.slots[s])), None)
                if slot is None:
                    break
                p.taken.append(slot)
                f.place, f.slot, f.stand = p, slot, p.slots[slot]
                lead.followers.append(f.i)
                self.people.append(f)
                i += 1
            if lead.persona == "stroller":
                lead.area = p.area
            quota["group"] -= 1 + len(lead.followers)
        quota["errand"] += quota.pop("group")
        for kind in ("stroller", "errand"):
            for _ in range(quota[kind]):
                p = take(("park",) if kind == "stroller" else None)
                if p is None:
                    raise RuntimeError("Not enough standing places for the crowd")
                per = Person(i, kind, rng.uniform(0.85, 1.1) if kind == "stroller" else rng.uniform(1.1, 1.45))
                per.area = p.area if kind == "stroller" else None
                self._stay(per, p, rng.uniform(0.0, 90.0))
                self.people.append(per)
                i += 1
        for per in self.people:
            if per.face is None:
                per.face = per.stand + np.array([1.0, 0.0])
            d = per.face - per.stand
            per.yaw = float(np.arctan2(d[1], d[0]))
            params = self.jps.AnticipationVelocityModelAgentParameters(
                position=tuple(per.stand),
                journey_id=self.journey,
                stage_id=self.stage,
                radius=RADIUS_M,
                desired_speed=0.0,
                reaction_time=REACTION_S,
                wall_buffer_distance=WALL_BUFFER_M,
                time_gap=TIME_GAP_S,
            )
            self.agent.append(self.sim.add_agent(params))
        for a, per in zip(self.agent, self.people, strict=False):
            self.sim.agent(a).target = tuple(per.stand)

    def _ok(self, xy, room=RADIUS_M + WALL_BUFFER_M):
        if not self.area.contains(Point(xy)) or self.area.exterior.distance(Point(xy)) < room:
            return False
        if any(h.distance(Point(xy)) < room for h in self.area.interiors):
            return False
        return all(np.hypot(*(xy - p.stand)) >= 2 * RADIUS_M + 0.05 for p in self.people if p.stand is not None)

    def _ring(self, centre, k):
        r = TALK_M / (2 * np.sin(np.pi / k)) if k > 2 else TALK_M / 2
        for a0 in np.radians(np.arange(0, 90, 15)):
            ring = [
                centre + r * np.array([np.cos(a0 + 2 * np.pi * m / k), np.sin(a0 + 2 * np.pi * m / k)])
                for m in range(k)
            ]
            if all(self._ok(q) for q in ring) and not _blocks(self.walkway, self.room, ring):
                return ring
        return None

    def _beside(self, xy, face, side, m):
        d = (face - xy) / max(np.hypot(*(face - xy)), 1e-6)
        for off in ((GROUP_SIDE_M * side, 0.0), (GROUP_SIDE_M * side, -0.6), (-GROUP_SIDE_M * side, -0.6), (0.0, -1.2)):
            q = xy + off[0] * np.array([-d[1], d[0]]) + off[1] * d
            if self._ok(q):
                return q
        return None

    def _stay(self, per, place, duration):
        k = next((k for k in place.free_slots() if self._ok(place.slots[k])), place.free_slot())
        place.taken.append(k)
        per.place, per.slot, per.stand = place, k, place.slots[k]
        per.face = place.xy if k else place.slots[1] if len(place.slots) > 1 else place.xy + [1.0, 0.0]
        per.until = self.t + duration if hasattr(self, "t") else duration
        per.state = "stay"

    # ---- decisions ----
    def _next_place(self, per):
        rng, here = self.rng, (per.stand if per.stand is not None else self.xy[per.i])
        cands = [
            p
            for p in self.places
            if len(p.free_slots()) > len(per.followers)
            and p.id not in per.recent[-min(len(per.recent), 3 if per.allowed else 6) :]
            and p is not per.place
            and (per.allowed is None or p.id in per.allowed)
        ]
        if per.persona == "stroller":
            same = [p for p in cands if p.area == per.area]
            if same and rng.random() < 0.85:
                cands = same
            else:
                cands = [p for p in cands if p.kind in ("park", "pitch", "cafe")] or cands
        d = np.array([np.hypot(*(p.xy - here)) for p in cands])
        if per.persona == "stroller":
            w = np.exp(-d / 60.0)
        else:
            w = (
                np.array([ERRAND_WEIGHT[p.kind] for p in cands])
                * np.exp(-np.abs(d - ERRAND_TRIP_M) / ERRAND_TRIP_M)
                * (d > 25.0)
            )
            if per.allowed is not None:  # a fixed set: any of its places, other kinds first
                w = np.array([1.0 + 2.0 * (per.place is None or p.kind != per.place.kind) for p in cands]) * (d > 8.0)
        if not w.sum():
            return None
        order = rng.choice(len(cands), size=min(6, int(np.count_nonzero(w))), replace=False, p=w / w.sum())
        return [cands[k] for k in order]

    def _depart(self, per):
        for place in self._next_place(per) or []:
            k = place.free_slot()
            route = self._route(self.xy[per.i], place.slots[k], per.i)
            if route is None:
                continue
            for who in [per] + [self.people[f] for f in per.followers]:
                if who.place is not None and who.slot is not None and who.slot in who.place.taken:
                    who.place.taken.remove(who.slot)
                who.place = who.slot = None
            place.taken.append(k)
            for f in per.followers:  # the group stands on slots of the place (_blocks)
                fol = self.people[f]
                fol.place, fol.slot, fol.stand = place, next(s for s in place.free_slots() if s != k), None
                place.taken.append(fol.slot)
            per.recent = (per.recent + [place.id])[-6:]
            if per.persona == "stroller" and place.kind == "park":
                per.area = place.area
            per.place, per.slot, per.stand, per.face = place, k, place.slots[k], None
            per.route, per.arc, per.s, per.state = route, _arc(route), 0.0, "walk"
            per.vmax = _speed_profile(route, per.arc)
            for k, f in enumerate(per.followers):
                fol = self.people[f]
                fol.offset = self._side_offsets(route, per.arc, fol.side)
                fol.s = _progress(route, per.arc, self.xy[f], 0.0) if len(route) > 1 else 0.0
                fol.behind = (k + 1) * SINGLE_FILE_M
            self.events.append(dict(t=round(self.t, 1), person=per.i, kind="depart", place=place.id))
            return True
        per.until = self.t + 20.0
        return False

    def _route(self, a, b, who=None, block=STANDING_CAR_BLOCK_M):
        """Route from `a` to `b` on the reachable walkway, around the people standing (cells within STANDING_M of
        one cost STANDING_COST more, so a route squeezes past only where nothing else leads), and around the bodies of
        standing cars (`cars`, set by awr.life.ambient.coupling: cells within STANDING_CAR_M cost STANDING_CAR_COST more,
        cells within `block` are closed; the stretch of their ways ahead, `_waiting_cells`, costs WAITING_PATH_COST
        more). Where nothing leads past standing cars at STANDING_CAR_BLOCK_M (people
        caught between the queues of a narrow two-way street), the way may pass them at STANDING_CAR_MIN_M, clear of
        a walking body's reach. Where the people standing against the walker close every way out (one standing at a
        slot between the neighbours of its place and a wall), the way is planned past them at their usual cost:
        otherwise it never leaves, and nor may they."""
        route = self._plan(a, b, who, block)
        if route is None and self.cars and block > STANDING_CAR_MIN_M:
            route = self._plan(a, b, who, STANDING_CAR_MIN_M)
        if route is None:
            route = self._plan(a, b, who, min(block, STANDING_CAR_MIN_M), contact=False)
        return route

    def _plan(self, a, b, who, block, contact=True):
        cost = self.cost
        if self.cars:
            cost = cost.copy()
            c = self.walkway.cell_m
            for _, centre, yaw, hl, hw in self.cars:
                reach = np.hypot(hl, hw) + STANDING_CAR_M
                r0, q0 = self.walkway.cell(centre - reach)
                r1, q1 = self.walkway.cell(centre + reach)
                r0, q0 = max(r0, 0), max(q0, 0)
                r1, q1 = min(r1 + 1, cost.shape[0]), min(q1 + 1, cost.shape[1])
                if r1 <= r0 or q1 <= q0:
                    continue
                yy, xx = np.mgrid[r0:r1, q0:q1]
                dx, dy = self.walkway.x0 + (xx + 0.5) * c - centre[0], self.walkway.y0 + (yy + 0.5) * c - centre[1]
                along, across = dx * np.cos(yaw) + dy * np.sin(yaw), -dx * np.sin(yaw) + dy * np.cos(yaw)
                m = (np.abs(along) < hl + STANDING_CAR_M) & (np.abs(across) < hw + STANDING_CAR_M)
                cost[r0:r1, q0:q1][m] += STANDING_CAR_COST
                m = (np.abs(along) < hl + block) & (np.abs(across) < hw + block)
                cost[r0:r1, q0:q1][m] = np.inf
            for r0, r1, q0, q1, m in self._waiting_cells():
                cost[r0:r1, q0:q1][m] += WAITING_PATH_COST
        standing = [
            p
            for p in self.people
            if p.v < 0.1
            and p.i != who
            and (who is None or p.leader != who)
            and (who is None or self.people[who].leader != p.i)
        ]
        if standing:
            cost = cost.copy() if cost is self.cost else cost
            c = self.walkway.cell_m
            k = int(np.ceil(STANDING_M / c))
            dy, dx = np.mgrid[-k : k + 1, -k : k + 1]
            dist = np.sqrt(dx * dx + dy * dy) * c
            for p in standing:
                r, q = self.walkway.cell(self.xy[p.i])
                r0, q0 = max(r - k, 0), max(q - k, 0)
                r1, q1 = min(r + k + 1, cost.shape[0]), min(q + k + 1, cost.shape[1])
                if r1 <= r0 or q1 <= q0:
                    continue
                d = dist[r0 - (r - k) : r1 - (r - k), q0 - (q - k) : q1 - (q - k)]
                cost[r0:r1, q0:q1][d <= STANDING_M] += STANDING_COST
                # somebody standing against the walker: no way through it (JuPedSim gives no speed towards a disc in
                # contact), as far as the walker's own distance from it allows a start
                if contact and np.hypot(*(self.xy[p.i] - a)) < 2 * RADIUS_M + BLOCK_M + 0.05:
                    ra, qa = self.walkway.cell(a)
                    cost[r0:r1, q0:q1][d < min(2 * RADIUS_M + 0.05, np.hypot(ra - r, qa - q) * c)] = np.inf
        pts = self.walkway.path(a, b, cost)
        if pts is None or len(pts) < 2:
            return None
        pts = _pull(pts, self.walkway, cost)
        pts[0], pts[-1] = a, b
        return _round(_densify(_round(pts, ROUND_M), 0.5), ROUND_M / 3)

    def _waiting_cells(self):
        """Cells of the stretches of the standing cars' ways ahead (WAITING_AHEAD_M from the front, WAITING_PATH_M
        beyond the half width) as (row and column windows, mask), once per tick: a car waits for everybody in its
        way, and people walking towards it along its lane, or along its way, would keep it waiting; a route passes
        it beside its way where there is room and crosses the way where it must."""
        if self._waiting_cost[0] == self.t:
            return self._waiting_cost[1]
        import shapely
        from shapely.geometry import LineString

        out, c = [], self.walkway.cell_m
        for line, hw in self.waiting.values():
            arc = _arc(line)
            pts = line[arc <= WAITING_AHEAD_M]
            if len(pts) < 2:
                continue
            poly = LineString(pts).buffer(hw + WAITING_PATH_M)
            x0, y0, x1, y1 = poly.bounds
            r0, q0 = self.walkway.cell((x0, y0))
            r1, q1 = self.walkway.cell((x1, y1))
            r0, q0 = max(r0, 0), max(q0, 0)
            r1, q1 = min(r1 + 1, self.cost.shape[0]), min(q1 + 1, self.cost.shape[1])
            if r1 <= r0 or q1 <= q0:
                continue
            yy, xx = np.mgrid[r0:r1, q0:q1]
            m = shapely.contains_xy(poly, self.walkway.x0 + (xx + 0.5) * c, self.walkway.y0 + (yy + 0.5) * c)
            if m.any():
                out.append((r0, r1, q0, q1, m))
        self._waiting_cost = (self.t, out)
        return out

    def _arrive(self, per):
        per.state, per.route, per.replans = "stay", None, 0
        lo, hi = DWELL_S[per.place.kind]
        per.until = self.t + self.rng.uniform(lo, hi)
        per.face = per.place.xy if per.slot else (per.place.slots[1] if len(per.place.slots) > 1 else None)
        self.events.append(dict(t=round(self.t, 1), person=per.i, kind="arrive", place=per.place.id))
        if per.followers or self.rng.random() > TALK_CHANCE:
            return
        for other in self.people:
            if (
                other.state == "stay"
                and other.partner is None
                and other.persona in ("errand", "stroller", "talk")
                and not other.followers
                and other.i != per.i
                and other.until - self.t > 30.0
                and np.hypot(*(self.xy[other.i] - per.stand)) < 4.0
            ):
                per.partner, other.partner = other.i, per.i
                per.face, other.face = self.xy[other.i], per.stand
                per.until = other.until = self.t + self.rng.uniform(60.0, 240.0)
                self.events.append(dict(t=round(self.t, 1), person=per.i, kind="talk", partner=other.i))
                return

    # ---- stepping ----
    def step(self):
        """Advance one tick (step_s): decisions, targets and desired speeds, then JuPedSim."""
        t, dt = self.t, self.step_s
        for per in self.people:
            # a car waits for this person: stand aside; when it has driven on, go on from there (a person back on
            # its way as soon as it is out of the car's path walks into it again, and with a few people about the
            # car never gets a clear way). Another spot when it cannot reach its spot, or stands there still in the
            # car's way (the path ahead of a car entering a junction turns, and may sweep a spot chosen before)
            if (
                per.blocking is not None
                and t - per.blocking_since >= GIVE_WAY_AFTER_S
                and (per.give is None or per.stalled > STALL_S or np.hypot(*(self.xy[per.i] - per.give)) < 0.5)
            ):
                found, per.stalled = self._give_way(per), 0.0
                per.give, per.give_path = (found[0], [found[1], _arc(found[1]), 0.0]) if found else (None, None)
                per.blocking_since = t  # (none: look again a little later)
                if found and (per.gave_for is None or per.gave_for[0] != per.blocking[0]):
                    car = next((b for b in self.bodies if b[0] == per.blocking[0]), None)
                    per.gave_for = (per.blocking[0], None if car is None else np.asarray(car[1], float), t)
            elif per.blocking is None and per.give is not None and self._passed(per, t):
                per.give = per.gave_for = None
                if per.leader is None and per.state == "walk":
                    self._reroute(per)
            # a car standing on the way ahead: plan around it (once per car)
            if (
                per.leader is None
                and per.state == "walk"
                and per.give is None
                and self.cars
                and per.route is not None
                and t - per.checked >= CAR_CHECK_S
            ):
                per.checked = t
                per.avoided &= {c[0] for c in self.cars}  # a car that drove off counts anew when it stands again
                vid = self._car_on_route(per)
                if vid is not None:
                    per.avoided.add(vid)
                    self._reroute(per)
        for per in self.people:
            if per.leader is not None:
                continue
            if per.state == "stay" and t >= per.until:
                if per.partner is not None:
                    other = self.people[per.partner]
                    if other.partner == per.i:
                        other.partner, other.until = None, min(other.until, t + self.rng.uniform(2.0, 15.0))
                    per.partner = None
                if per.persona == "talk":
                    per.persona = "errand"
                self._depart(per)
            elif (
                per.state == "walk"
                and per.route is not None
                and (
                    (
                        per.v < 0.12
                        and (
                            (
                                per.arc[-1] - per.s < ARRIVED_M
                                and np.hypot(*(self.xy[per.i] - per.route[-1])) < ARRIVED_M
                            )
                            or (np.hypot(*(self.xy[per.i] - per.route[-1])) < NEAR_END_M and per.stalled > 1.0)
                        )
                    )
                    or (
                        np.hypot(*(self.xy[per.i] - per.route[-1])) < NEAR_END_M
                        and per.v < 0.12
                        and self._taken(per.route[-1], per)
                    )
                )
            ):
                per.stand = self.xy[per.i].copy()  # a slot taken by somebody standing near it: stand here
                self._arrive(per)
        import shapely
        import shapely.ops

        # target (or its fallback when the target is off the inner area), then the turn limit on whichever it is
        commands = [self._unpin(per, *self._command(per)) for per in self.people]
        wanted = np.array([c[0] for c in commands], float)
        inside = shapely.contains_xy(self.inner, wanted[:, 0], wanted[:, 1])
        commands = [
            self._steer(per, target if ok or fallback is None else fallback, v)
            for per, (target, v, fallback), ok in zip(self.people, commands, inside, strict=False)
        ]
        targets = np.array([c[0] for c in commands], float)
        limit = np.minimum(self._headway(targets), self._wall_limit(targets))
        moves = []
        for per, a, (target, v) in zip(self.people, self.agent, commands, strict=False):
            agent = self.sim.agent(a)
            aim = target - self.xy[per.i]
            per.intent = (
                aim / max(np.hypot(*aim), 1e-6) * max(min(v, limit[per.i]), 0.0)
                if np.hypot(*aim) > 0.05
                else np.zeros(2)
            )
            want = v
            if per.hold or self.t < per.pause_until:
                v = 0.0
            free = v  # (people in the way, limit below, block a walker as much as JuPedSim does)
            v = min(v, limit[per.i])
            giving = per.give is not None
            if (
                (per.state == "walk" or giving)
                and per.v < 0.1
                and (free > 0.3 or ((per.hold_standing or (giving and per.hold)) and want > 0.3))
            ):
                per.stalled += dt
            elif per.v > 0.3:
                per.stalled = 0.0
            if per.stalled > STALL_S and per.leader is None and per.state == "walk" and per.give is None:
                self._replan(per)
            # no headway along the route for NO_PROGRESS_S (a walker that keeps stepping aside from people in its
            # way moves, but gets nowhere; waiting for a moving car does not count): plan again
            if per.leader is None and per.state == "walk" and per.give is None and per.route is not None:
                t0, s0, route = per.mark
                if route is not per.route or per.s > s0 + PROGRESS_M or (per.hold and not per.hold_standing):
                    per.mark = (self.t, per.s, per.route)
                elif self.t - t0 > NO_PROGRESS_S:
                    self._replan(per)
                    per.mark = (self.t, per.s, per.route)
            v = float(np.clip(v, per.v - DECEL_MPS2 * dt, per.v + ACCEL_MPS2 * dt))
            goal = target
            if per.v > 0.1 and np.hypot(*(goal - self.xy[per.i])) < per.v**2 / (2 * DECEL_MPS2) + per.v * dt + 0.1:
                # closer than the walker can stop: on along its motion, slowing down (JuPedSim would overrun the
                # target within the tick, turn round and stop the walker at once); a point within the area
                u = per.last / np.hypot(*per.last)
                on = [q for q in (self.xy[per.i] + u, self.xy[per.i] + 0.3 * u) if shapely.contains_xy(self.inner, *q)]
                goal = on[0] if on else goal
            if not shapely.contains_xy(self.area, *goal):
                # JuPedSim stops the whole simulation on a target off its area (the turn limit's short step for a
                # walker pressed against the edge, a target without a fallback): the nearest point within reach
                goal = np.asarray(shapely.ops.nearest_points(self.inner, shapely.Point(goal))[0].coords[0])
            moves.append((agent, goal, max(v, 0.0)))
        for (agent, goal, _), v in zip(moves, self._body_limit(np.array([m[2] for m in moves])), strict=False):
            agent.target = tuple(goal)
            agent.model.desired_speed = float(v)
        self.sim.iterate(SUBSTEPS)
        agents = [self.sim.agent(a) for a in self.agent]
        xy = np.array([ag.position for ag in agents])
        for per, ag in zip(self.people, agents, strict=False):
            per.heading = np.array(ag.orientation, float)
        delta = xy - self.xy
        speed = np.hypot(delta[:, 0], delta[:, 1]) / dt
        centres = np.array([b[1] for b in self.bodies]).reshape(-1, 2)
        by_car = (
            (np.hypot(*(xy[:, None] - centres[None]).transpose(2, 0, 1)) < 4.5).any(1)
            if len(centres)
            else np.zeros(len(xy), bool)
        )
        for per in self.people:
            if speed[per.i] > 0.1 and per.v > 0.1 and delta[per.i] @ per.last < 0:
                per.pause_until = self.t + dt + PAUSE_S  # moved back and forth: wait for the others to pass
            per.last = delta[per.i].copy()
            per.still = per.still + dt if speed[per.i] < 0.05 else 0.0
            per.v = float(speed[per.i])
            if per.v > 0.15:
                want = np.arctan2(delta[per.i, 1], delta[per.i, 0])
                rate = YAW_RATE
            elif per.face is not None and per.state == "stay":
                f = per.face - xy[per.i]
                want, rate = (np.arctan2(f[1], f[0]) if np.hypot(*f) > 0.3 else per.yaw), FACE_RATE
            else:
                want, rate = per.yaw, 0.0
            turn = (want - per.yaw + np.pi) % (2 * np.pi) - np.pi
            yaw = float((per.yaw + np.clip(turn, -rate * dt, rate * dt) + np.pi) % (2 * np.pi) - np.pi)
            # no turning into a car: a body next to one (the coupling keeps the walker's reach from closing in, not
            # its turning) turns only as far as that keeps GRAZE_M from the car's body, or no nearer than it is
            if yaw != per.yaw and by_car[per.i]:
                gap = self._clearance(xy[per.i], yaw)
                if gap < GRAZE_M and gap < self._clearance(xy[per.i], per.yaw) - 1e-3:
                    yaw = per.yaw
            per.yaw = yaw
            if per.state == "walk" and per.route is not None:
                per.s = _progress(per.route, per.arc, xy[per.i], per.s)
        self.xy, self.t = xy, t + dt

    def _steer(self, per, target, v):
        """Keep the target within TURN_LIMIT of a walker's motion and slow down for larger turns: JuPedSim relaxes the
        direction vector linearly, so a target behind a walker would flip it at once. The motion is the last step's
        (the recorded heading lags it on a turn)."""
        target = np.asarray(target, float)
        if per.v < STEER_MPS:
            return target, v
        h = per.last / max(np.hypot(*per.last), 1e-9)
        d = target - self.xy[per.i]
        dist = np.hypot(*d)
        if dist < 1e-3:
            return target, v
        ang = np.arctan2(h[0] * d[1] - h[1] * d[0], h @ d)
        v = min(v, max(per.speed * np.cos(min(abs(ang), np.pi / 2)), 0.3))
        # JuPedSim takes up most of the turn within the tick: at most TURN_MPS2 of sideways change
        most = min(TURN_LIMIT, 2 * np.arcsin(min(1.0, TURN_MPS2 * self.step_s / (2 * per.v))))
        if abs(ang) <= most:
            return target, v
        import shapely

        a = np.sign(ang) * most
        u = np.array([h[0] * np.cos(a) - h[1] * np.sin(a), h[0] * np.sin(a) + h[1] * np.cos(a)])
        q = self.xy[per.i] + max(dist, 1.0) * u
        return (q if shapely.contains_xy(self.inner, *q) else self.xy[per.i] + 0.3 * u), v

    def _clearance(self, xy, yaw):
        """How far a walking body (BODY_WALK extents) at `xy` facing `yaw` keeps from the nearest car's body: the
        widest gap along the separating axes of the two rectangles (the cars' and the body's edge directions), negative
        where they overlap; inf with no car near."""
        c, s_ = np.cos(yaw), np.sin(yaw)
        x0, x1, y0, y1 = BODY_WALK
        box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ np.array([[c, s_], [-s_, c]]) + xy
        out = np.inf
        for _, centre, cyaw, hl, hw in self.bodies:
            if np.hypot(*(centre - xy)) > hl + hw + 1.5:
                continue
            u = np.array([np.cos(cyaw), np.sin(cyaw)])
            car = centre + np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]]) @ np.array([u, [-u[1], u[0]]])
            out = min(
                out,
                max(
                    max(np.min(car @ ax) - np.max(box @ ax), np.min(box @ ax) - np.max(car @ ax))
                    for ax in (u, np.array([-u[1], u[0]]), np.array([c, s_]), np.array([-s_, c]))
                ),
            )
        return out

    def _taken(self, spot, per):
        """Whether somebody else (not of the walker's group) stands at `spot`."""
        group = {per.i, *per.followers}
        return any(
            o.i not in group and o.v < 0.1 and np.hypot(*(self.xy[o.i] - spot)) < 2 * RADIUS_M + 0.2
            for o in self.people
        )

    def _blocked(self, per, u):
        """Whether JuPedSim gives (next to) no speed in direction `u`: a disc ahead within BLOCK_M of contact whose
        centre lies within the walker's corridor (lateral offset below the sum of the radii)."""
        rel = self.xy - self.xy[per.i]
        ahead = rel @ u
        lateral = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
        near = (ahead > 0.0) & (lateral < 2 * RADIUS_M) & (np.hypot(rel[:, 0], rel[:, 1]) < 2 * RADIUS_M + BLOCK_M)
        near[per.i] = False
        return bool(near.any())

    def _unpin(self, per, target, v, fallback):
        """A walker nearly at rest whose way to its target is blocked by people in contact moves first in the free
        direction nearest to it (JuPedSim gives no speed towards a disc in contact, so a walker pinned between people
        standing, or against its own group, would stay where it is) and keeps that direction for SLIDE_S or until
        its own way is free; the turn limit (_steer) applies to it as to any target. With no free direction, or once
        it has wanted to go for PINNED_S without moving (a free direction that does not lead past the people touching
        it, as between them and a wall, or a pinning JuPedSim leaves in its own direction), the idle people touching
        it (standing, or its own followers) step aside for SLIDE_S (`room`) and stand where that leaves them."""
        import shapely

        xy = self.xy[per.i]
        if per.room is not None:  # stepping aside for a pinned walker, as far as the area lets it
            u, until = per.room
            if self.t < until and shapely.contains_xy(self.inner, *(xy + u)):
                return xy + u, SLIDE_MPS, None
            per.room = None
            if per.state == "stay" and per.leader is None:
                per.stand = xy.copy()
        moving = per.state == "walk" or per.leader is not None or per.give is not None
        d = np.asarray(target, float) - xy
        L = np.hypot(*d)
        if not moving or v < 0.1 or L < 0.05:
            per.slide = None
            return target, v, fallback
        e = d / L
        if per.stalled > PINNED_S:
            self._make_room(per)
        if per.slide is not None:
            u, until = per.slide
            if self.t < until and self._blocked(per, e):
                return xy + u, min(v, SLIDE_MPS), fallback
            per.slide = None
            return target, v, fallback
        if per.v > 0.2 or not self._blocked(per, e):
            return target, v, fallback
        base = np.arctan2(e[1], e[0])
        for da in sorted(np.radians(np.arange(-170.0, 181.0, 10.0)), key=abs):
            u = np.array([np.cos(base + da), np.sin(base + da)])
            if self._blocked(per, u):
                continue
            if self._open(xy, u):
                per.slide = (u, self.t + SLIDE_S)
                return xy + u, min(v, SLIDE_MPS), fallback
        self._make_room(per)  # no way out
        return target, v, fallback

    def _make_room(self, per):
        """The idle people touching a pinned walker (standing, or its own followers) step aside, away from it; of two
        walkers pinned against each other the one of the higher index steps back. Straight away from the other the
        step starts at once (JuPedSim turns a walker round at once, but sideways only gradually, and stays stopped
        while the other is still within its heading's corridor; two walkers stepping sideways together also step into
        each other's way). Against a wall the step is shorter: two walkers meeting in a narrow lane, each with its
        back to a wall, need only a few centimetres more between them to pass."""

        rel = self.xy - self.xy[per.i]
        dist = np.hypot(*rel.T)
        for j in np.flatnonzero((dist < 2 * RADIUS_M + BLOCK_M) & (dist > 1e-6)):
            o = self.people[j]
            pinned = o.state == "walk" and o.give is None and o.stalled > PINNED_S and j > per.i
            if o.room is None and (
                o.leader == per.i or (o.state == "stay" and o.leader is None and o.give is None) or pinned
            ):
                u = rel[j] / dist[j]
                if self._blocked(o, u):
                    continue
                for f in (1.0, 0.6, 0.3):
                    if self._open(self.xy[j], f * u):
                        o.room = (f * u, self.t + SLIDE_S)
                        break

    def _open(self, xy, u):
        """Whether the walkway is reachable 0.4 and 0.8 of the way from `xy` along `u` and the target of a slide or a
        step aside there (`xy`+`u`) lies within the area a disc's centre may reach (else the fallback replaces it)."""
        import shapely

        return bool(shapely.contains_xy(self.inner, *(xy + u))) and all(
            0 <= r < self.cost.shape[0] and 0 <= c < self.cost.shape[1] and np.isfinite(self.cost[r, c])
            for r, c in (self.walkway.cell(xy + u * f) for f in (0.4, 0.8))
        )

    def _headway(self, targets):
        """Speed limit per walker from crossing walkers (headings more than CROSSING_DEG apart, both moving or about
        to): JuPedSim lets a walker run into a crosser's corridor and stop at once, so the walker whose way the other
        crosses ahead slows down beforehand, HEADWAY_S to close the gap between the discs (continuous in the lateral
        offset); of two walkers crossing each other's way only the one with the other more nearly ahead yields.
        Others just ahead (oncoming, overtaken, standing) limit the speed as JuPedSim itself does, made continuous."""
        from scipy.spatial import cKDTree

        xy, n = self.xy, len(self.people)
        limit = np.full(n, np.inf)
        heading = targets - xy
        heading /= np.maximum(np.hypot(*heading.T), 1e-6)[:, None]
        wants = np.array([p.state == "walk" or p.leader is not None for p in self.people])
        motion = np.array([p.last / np.hypot(*p.last) if p.v > STEER_MPS else np.zeros(2) for p in self.people])
        group = np.array([p.leader if p.leader is not None else p.i for p in self.people])
        cos_cross = np.cos(np.radians(CROSSING_DEG))
        for i, j in cKDTree(xy).query_pairs(2 * RADIUS_M + HEADWAY_S * 1.6 + 0.5, output_type="ndarray"):
            if group[i] == group[j] or not (wants[i] and wants[j]) or abs(heading[i] @ heading[j]) > cos_cross:
                # anybody else just ahead (along the walker's target or its motion, which JuPedSim goes by): its
                # spacing rule, TIME_GAP_S over the gap between the discs; a neighbour just beside the corridor may
                # enter it within a tick, so the speed stays within CUT_IN_MPS of what that would leave (and not
                # below PASS_MPS: outside the corridor JuPedSim lets the walker pass)
                for a, b in ((i, j), (j, i)):
                    if not wants[a]:
                        continue
                    rel = xy[b] - xy[a]
                    base = max(np.hypot(*rel) - 2 * RADIUS_M, 0.0) / TIME_GAP_S
                    for hd in (heading[a], motion[a]):
                        lateral = abs(rel[0] * hd[1] - rel[1] * hd[0])
                        if rel @ hd > 0.0 and lateral < 2 * RADIUS_M + NEAR_LANE_M:
                            lim = (
                                base
                                if lateral < 2 * RADIUS_M
                                else max(base + min(4.0 * (lateral - 2 * RADIUS_M) / TIME_GAP_S, CUT_IN_MPS), PASS_MPS)
                            )
                            limit[a] = min(limit[a], lim)
                continue
            seen = {}
            for a, b in ((i, j), (j, i)):
                rel = xy[b] - xy[a]
                ahead = rel @ heading[a]
                lateral = abs(rel[0] * heading[a][1] - rel[1] * heading[a][0])
                if ahead > 0.2 and lateral < CORRIDOR_M:
                    gap = np.hypot(*rel) - 2 * RADIUS_M + 4.0 * max(lateral - 2 * RADIUS_M, 0.0)
                    seen[a] = (ahead / np.hypot(*rel), max(gap, 0.0) / HEADWAY_S)
            if len(seen) == 2:  # each in the other's way: one yields
                a = max(seen, key=lambda k: (seen[k][0], k))
                seen = {a: seen[a]}
            for a, (_, v) in seen.items():
                limit[a] = min(limit[a], v)
        return limit

    def _body_limit(self, v):
        """Wanted speeds `v` lowered where two bodies would touch. JuPedSim keeps the discs apart (2 RADIUS_M), but a
        walking body reaches beyond its disc at the corners, the more the faster it walks (awr.life.civil.body), so two
        walkers meeting nearly head-on, or at an angle, and stopped by JuPedSim at the discs' distance would overlap
        at a corner. Each pair is followed for BODY_LOOK_S along JuPedSim's headings at the wanted speeds (the
        recorded heading turning towards the motion at YAW_RATE); where the bodies would come within BODY_GAP_M, the
        walkers closing in on the other (both when neither does) slow down by a common factor, as far as that keeps
        them apart. Slowing one walker may let another catch up with it, so the pairs are looked at again."""
        from scipy.spatial import cKDTree

        v = np.asarray(v, float).copy()
        if not len(v) or v.max() <= 0.0:
            return v
        steps = np.arange(1, int(round(BODY_LOOK_S / self.step_s)) + 1) * self.step_s
        heading = np.array([p.heading for p in self.people], float)
        heading /= np.maximum(np.hypot(*heading.T), 1e-9)[:, None]
        yaw = np.array([p.yaw for p in self.people])
        turn = (np.arctan2(heading[:, 1], heading[:, 0]) - yaw + np.pi) % (2 * np.pi) - np.pi
        reach = np.hypot(max(-BODY_WALK[0], BODY_WALK[1]), BODY_WALK[3])

        def bodies(a, s):
            pos = self.xy[a][:, None] + heading[a][:, None] * (s[:, None] * steps[None])[..., None]
            y = yaw[a][:, None] + np.where(
                s[:, None] > 0.15, np.clip(turn[a][:, None], -YAW_RATE * steps, YAW_RATE * steps), 0.0
            )
            return person_box(pos.reshape(-1, 2), y.ravel(), np.repeat(s, len(steps)))

        def gap(i, j, si, sj):
            return box_gap(*bodies(i, si), *bodies(j, sj)).reshape(len(i), -1).min(1)

        pairs = cKDTree(self.xy).query_pairs(2 * reach + 2 * v.max() * steps[-1], output_type="ndarray")
        if not len(pairs):
            return v
        i, j = pairs[:, 0], pairs[:, 1]
        rel = self.xy[j] - self.xy[i]
        towards = np.c_[(heading[i] * rel).sum(1) > 0.0, (heading[j] * -rel).sum(1) > 0.0]
        for _ in range(4):
            bad = ((v[i] > 0.0) | (v[j] > 0.0)) & (gap(i, j, v[i], v[j]) < BODY_GAP_M)
            if not bad.any():
                break
            bi, bj = i[bad], j[bad]
            moving = np.c_[v[bi] > 0.0, v[bj] > 0.0]
            slow = moving & towards[bad]
            slow = np.where(slow.any(1)[:, None], slow, moving)

            # the common factor: bisection between a stop of the slowed walkers and their wanted speeds
            def at(f):
                return (
                    gap(bi, bj, v[bi] * np.where(slow[:, 0], f, 1.0), v[bj] * np.where(slow[:, 1], f, 1.0))
                    >= BODY_GAP_M
                )

            ok = at(np.zeros(len(bi)))
            lo, hi = np.zeros(len(bi)), np.ones(len(bi))
            for _ in range(7):
                mid = (lo + hi) / 2
                good = at(mid)
                lo, hi = np.where(good, mid, lo), np.where(good, hi, mid)
            f = np.where(ok, lo, 1.0)
            new = v.copy()
            np.minimum.at(new, bi[slow[:, 0]], (v[bi] * f)[slow[:, 0]])
            np.minimum.at(new, bj[slow[:, 1]], (v[bj] * f)[slow[:, 1]])
            if np.allclose(new, v):
                break
            v = new
        return v

    def _wall_limit(self, targets):
        """Speed limit per walker heading for a wall: within WALL_SOFT_M of a wall face the speed towards it (by the
        walker's target or motion) is at most WALL_TOWARD_MPS at WALL_HARD_M, more further off. JuPedSim takes away
        the component towards its nearest wall at once within its wall influence, a sudden turn for a walker coming at
        it fast. Every face that may become the nearest counts (the segments whose nearest point lies within them, and
        corners closer than the segments beside them): in a narrow passage, or where a wall bends towards the way, the
        nearest wall may be one the walker walks along while it approaches another ahead, which JuPedSim then turns it
        away from as soon as that one is the nearer."""
        import shapely

        limit = np.full(len(self.people), np.inf)
        idx = np.flatnonzero([p.v > STEER_MPS for p in self.people])
        if not len(idx):
            return limit
        w, s = self.walls.query(shapely.points(self.xy[idx]), predicate="dwithin", distance=WALL_SOFT_M)
        if not len(w):
            return limit
        p, a, ab = self.xy[idx[w]], self.wall_a[s], self.wall_b[s] - self.wall_a[s]
        f = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
        n = p - (a + ab * f[:, None])
        d = np.hypot(*n.T)
        m = len(self.wall_a)
        key = w * m + s
        order = np.argsort(key)
        keys, dist = key[order], d[order]

        def near(other):
            k = w * m + other
            pos = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
            return np.where(keys[pos] == k, dist[pos], np.inf)

        inner = (f > 1e-6) & (f < 1.0 - 1e-6)  # the nearest point within the segment: a face towards the walker
        face = (inner | (d <= near(self.wall_prev[s]) + 1e-9) & (d <= near(self.wall_next[s]) + 1e-9)) & (d > 1e-6)
        w, n, d = w[face], n[face], d[face]
        n /= d[:, None]  # away from the wall
        aim = targets[idx[w]] - self.xy[idx[w]]
        aim /= np.maximum(np.hypot(*aim.T), 1e-9)[:, None]
        mot = np.array([self.people[i].last for i in idx[w]]).reshape(-1, 2)
        mot /= np.maximum(np.hypot(*mot.T), 1e-9)[:, None]
        toward = np.maximum(np.maximum(-(aim * n).sum(1), -(mot * n).sum(1)), 0.0)
        allowed = WALL_TOWARD_MPS + 2.0 * np.maximum(d - WALL_HARD_M, 0.0)
        lim = np.where(toward > 0.05, allowed / np.maximum(toward, 0.05), np.inf)
        out = np.full(len(idx), np.inf)
        np.minimum.at(out, w, lim)
        limit[idx] = out
        return limit

    def _reroute(self, per):
        """Plan the way to the current destination again from where the walker is (around standing people and cars)."""
        route = self._route(self.xy[per.i], per.route[-1], per.i)
        if route is None:
            return
        per.route, per.arc, per.s = route, _arc(route), 0.0
        per.vmax = _speed_profile(route, per.arc)
        for f in per.followers:
            fol = self.people[f]
            fol.offset = self._side_offsets(route, per.arc, fol.side)
            fol.s = _progress(route, per.arc, self.xy[f], 0.0)

    def _replan(self, per):
        """A blocked walker plans its way again from where it stands, around the people standing; after REPLANS
        attempts it goes somewhere else."""
        per.stalled = 0.0
        per.replans += 1
        route = None if per.replans > REPLANS else self._route(self.xy[per.i], per.route[-1], per.i)
        self.events.append(dict(t=round(self.t, 1), person=per.i, kind="replan", attempt=per.replans))
        if route is None:
            per.replans = 0
            self._depart(per)
            return
        per.route, per.arc, per.s = route, _arc(route), 0.0
        per.vmax = _speed_profile(route, per.arc)
        for k, f in enumerate(per.followers):
            fol = self.people[f]
            fol.offset = self._side_offsets(route, per.arc, fol.side)
            fol.s = _progress(route, per.arc, self.xy[f], 0.0)

    def _give_way(self, per):
        """The nearest reachable spot within GIVE_WAY_REACH_M at least GIVE_WAY_M (plus the car's half width) from the
        path of the car that waits for this person and GIVE_WAY_OTHER_M from the paths of the other cars, clear of the
        people standing and of the standing cars' bodies (STANDING_CAR_M: routes do not lead there) and away from a
        spot it could not reach before, with the route there; None without one among the GIVE_WAY_TRIES nearest."""
        import shapely
        from shapely.geometry import LineString

        vid, line, half_w = per.blocking
        xy, c = self.xy[per.i], self.walkway.cell_m
        r0, q0 = self.walkway.cell(xy - GIVE_WAY_REACH_M)
        r1, q1 = self.walkway.cell(xy + GIVE_WAY_REACH_M)
        r0, q0 = max(r0, 0), max(q0, 0)
        r1, q1 = min(r1 + 1, self.cost.shape[0]), min(q1 + 1, self.cost.shape[1])
        rr, qq = np.nonzero(np.isfinite(self.cost[r0:r1, q0:q1]))
        if not len(rr):
            return None
        pts = np.c_[self.walkway.x0 + (qq + q0 + 0.5) * c, self.walkway.y0 + (rr + r0 + 0.5) * c]
        d = np.hypot(*(pts - xy).T)
        ok = (d < GIVE_WAY_REACH_M) & (d > 0.3)
        if per.give is not None:
            ok &= np.hypot(*(pts - per.give).T) > 1.0
        spots = shapely.points(pts)
        for other_id, other, hw in self.corridors or [(vid, line, half_w)]:
            ok &= shapely.distance(spots, LineString(other)) > hw + (
                GIVE_WAY_M if other_id == vid else GIVE_WAY_OTHER_M
            )
        others = np.array([self.xy[o.i] for o in self.people if o.i != per.i and o.v < 0.1])
        if len(others):
            from scipy.spatial import cKDTree

            ok &= cKDTree(others).query(pts)[0] > 2 * RADIUS_M + 0.2
        for _, centre, yaw, hl, hw in self.cars:
            dx, dy = pts[:, 0] - centre[0], pts[:, 1] - centre[1]
            along, across = dx * np.cos(yaw) + dy * np.sin(yaw), -dx * np.sin(yaw) + dy * np.cos(yaw)
            ok &= ~((np.abs(along) < hl + STANDING_CAR_M) & (np.abs(across) < hw + STANDING_CAR_M))
        for k in np.flatnonzero(ok)[np.argsort(d[ok])][:GIVE_WAY_TRIES]:
            route = self._route(xy, pts[k], per.i)
            if route is not None:
                self.events.append(dict(t=round(self.t, 1), person=per.i, kind="give_way", car=per.blocking[0]))
                return pts[k], route
        return None

    def _passed(self, per, t):
        """Whether the car a person stands aside for has driven on GIVE_WAY_PASS_M (or gone), or it has stood aside
        GIVE_WAY_HOLD_S."""
        if per.gave_for is None:
            return True
        vid, was, since = per.gave_for
        car = next((b for b in self.bodies if b[0] == vid), None)
        return (
            car is None
            or was is None
            or np.hypot(*(np.asarray(car[1], float) - was)) >= GIVE_WAY_PASS_M
            or t - since >= GIVE_WAY_HOLD_S
        )

    def _car_on_route(self, per):
        """A standing car (not yet planned around) within STANDING_CAR_M of the next ROUTE_CHECK_M of the route."""
        ahead = per.route[(per.arc >= per.s) & (per.arc <= per.s + ROUTE_CHECK_M)]
        for vid, centre, yaw, hl, hw in self.cars:
            if vid in per.avoided or not len(ahead):
                continue
            dx, dy = ahead[:, 0] - centre[0], ahead[:, 1] - centre[1]
            along, across = dx * np.cos(yaw) + dy * np.sin(yaw), -dx * np.sin(yaw) + dy * np.cos(yaw)
            if ((np.abs(along) < hl + STANDING_CAR_M - 0.2) & (np.abs(across) < hw + STANDING_CAR_M - 0.2)).any():
                return vid
        return None

    def _command(self, per):
        """Target point, wanted speed (before the ramp) and a fallback target inside the area for a person."""
        xy = self.xy[per.i]
        if per.give is not None:  # along the planned way (around the car's body), the last metre straight
            route, arc, s = per.give_path
            s = per.give_path[2] = _progress(route, arc, xy, s)
            d = np.hypot(*(per.give - xy))
            target = _at(route, arc, min(s + LOOKAHEAD_M, arc[-1])) if arc[-1] - s > 1.0 else per.give
            return target, (min(0.8, np.sqrt(2 * ARRIVE_MPS2 * (d - 0.2))) if d > 0.3 else 0.0), None
        if per.leader is not None:
            return self._follow(per)
        if per.state == "stay":
            d = np.hypot(*(per.stand - xy))
            if per.still > SETTLE_S and d < NEAR_END_M:
                per.stand = xy.copy()  # stand where one stopped; no shuffling between people
                d = 0.0
            return per.stand, (min(0.6, np.sqrt(2 * ARRIVE_MPS2 * (d - 0.3))) if d > 0.5 else 0.0), None
        s_end = per.arc[-1]
        target = _at(per.route, per.arc, min(per.s + LOOKAHEAD_M, s_end))
        v = min(per.speed, float(np.interp(per.s, per.arc, per.vmax)))
        if s_end - per.s < 1.0:  # the last metre: straight to the end of the route, slowing by the distance to it
            target = per.route[-1]  # (not by the progress along the route, which ends beside a walker kept off it)
            v = min(per.speed, np.sqrt(2 * ARRIVE_MPS2 * max(np.hypot(*(target - xy)) - 0.1, 0.0)))
        end = np.hypot(*(per.route[-1] - xy))
        if end < NEAR_END_M + per.v**2 / (2 * ARRIVE_MPS2) and self._taken(per.route[-1], per):
            # somebody stands on the slot: come to rest short of them (arrival, step, once nearly stopped)
            v = min(v, np.sqrt(2 * ARRIVE_MPS2 * max(end - 2 * RADIUS_M - 0.3, 0.0)))
        if per.followers:  # wait for a follower that has fallen behind (one ahead on the way waits for the leader)
            lag = max(
                [
                    np.hypot(*(self.xy[f] - xy))
                    for f in per.followers
                    if self.people[f].s <= per.s + 0.3 or self.people[f].catch is not None
                ],
                default=0.0,
            )
            if lag > 3.5:
                v = min(v, max(per.speed * (1.0 - (lag - 3.5) / 2.0), 0.3))
        return target, v, _at(per.route, per.arc, min(per.s + 0.5, s_end))

    def _follow(self, per):
        """A follower walks along its leader's route beside it (lateral offset by route point, side by side where the
        walkway has room, else in single file behind) and stands beside the leader during a stay."""
        lead = self.people[per.leader]
        lx = self.xy[lead.i]
        apart = np.hypot(*(self.xy[per.i] - lx))
        lost = apart > (LOST_M / 2 if per.catch is not None else LOST_M)
        if lead.state == "stay" or lead.route is None:
            if per.stand is None and per.place is not None and per.place is lead.place and per.slot is not None:
                per.stand = per.place.slots[per.slot]  # its slot at the leader's place
            if per.stand is None or np.hypot(*(per.stand - lead.stand)) > 2 * SLOT_SPACING_M + 0.1:
                per.stand = (
                    self._beside(lead.stand, lead.face if lead.face is not None else lead.stand + [1, 0], per.side, 0)
                    if lead.stand is not None
                    else None
                )
                if per.stand is None:
                    per.stand = self.xy[per.i].copy()
            per.state, per.face = "stay", lead.face if lead.face is not None else lx
            if lost and (caught := self._catch_up(per, per.stand)) is not None:
                return caught
            per.catch = None
            d = np.hypot(*(per.stand - self.xy[per.i]))
            if per.still > SETTLE_S and d < NEAR_END_M:
                per.stand, d = self.xy[per.i].copy(), 0.0
            return per.stand, (min(0.8, np.sqrt(2 * ARRIVE_MPS2 * (d - 0.3))) if d > 0.5 else 0.0), None
        per.state, per.stand = "walk", None
        route, arc = lead.route, lead.arc
        if lost and (caught := self._catch_up(per, lx)) is not None:
            return caught
        if per.catch is not None:  # back with its leader: on along the leader's way from beside it
            per.catch = None
            per.s = _progress(route, arc, self.xy[per.i], max(lead.s - LOST_M, 0.0))
        per.s = _progress(route, arc, self.xy[per.i], per.s)
        o = float(np.interp(per.s, arc, per.offset))
        along = -0.4 if abs(o) > 0.5 * GROUP_SIDE_M else -per.behind
        want = lead.s + along
        aim = min(per.s + LOOKAHEAD_M, arc[-1])
        p = _at(route, arc, aim)
        q0, q1 = _at(route, arc, max(aim - 0.5, 0.0)), _at(route, arc, min(aim + 0.5, arc[-1]))  # the tangent there,
        t = (q1 - q0) / max(np.hypot(*(q1 - q0)), 1e-6)  # also at the end
        target = p + float(np.interp(aim, arc, per.offset)) * np.array([-t[1], t[0]])
        v = float(np.clip(lead.v + 0.6 * (want - per.s), 0.0, lead.speed + 0.4))
        v = min(v, float(np.interp(per.s, arc, lead.vmax)) + 0.2)
        if per.s > lead.s + 0.3 and np.hypot(*(self.xy[per.i] - self.xy[lead.i])) < REJOIN_M:
            # ahead of its leader (after a turn or a new route, or where the leader's way comes back past it): out of
            # the leader's way, to its side beside it; standing still there would block the leader's way
            p = _at(route, arc, lead.s)
            q0, q1 = _at(route, arc, max(lead.s - 0.5, 0.0)), _at(route, arc, min(lead.s + 0.5, arc[-1]))
            t = (q1 - q0) / max(np.hypot(*(q1 - q0)), 1e-6)
            return (
                p + per.side * GROUP_SIDE_M * np.array([-t[1], t[0]]),
                0.6,
                _at(route, arc, min(lead.s + 1.0, arc[-1])),
            )
        return target, v, _at(route, arc, min(per.s + 0.5, arc[-1]))

    def _catch_up(self, per, goal):
        """Target, wanted speed and fallback of a follower away from its leader (held up by a car, a give-way or people
        in its way): its own way to `goal` around the people and cars standing, at up to 0.4 m/s above its pace; None
        without one. Along the leader's way it would head for points of it far off, across whatever lies between."""
        xy = self.xy[per.i]
        c = per.catch
        if c is None or self.t - c[3] > CATCH_S or np.hypot(*(c[0][-1] - goal)) > CATCH_MOVE_M:
            route = self._route(xy, goal, per.i)
            if route is None:
                per.catch = None
                return None
            arc = _arc(route)
            c = per.catch = [route, arc, 0.0, self.t, _speed_profile(route, arc)]
        route, arc, s, _, vmax = c
        s = c[2] = _progress(route, arc, xy, s)
        v = min(per.speed + 0.4, float(np.interp(s, arc, vmax)))
        return _at(route, arc, min(s + LOOKAHEAD_M, arc[-1])), v, _at(route, arc, min(s + 0.5, arc[-1]))

    def _side_offsets(self, route, arc, side):
        """Lateral offset of a follower at each point of its leader's route: GROUP_SIDE_M on its side where the
        ground there is reachable, else on the other side, else none (single file), with changes spread over
        OFFSET_BLEND_M."""
        t = np.gradient(route, axis=0)
        t /= np.maximum(np.hypot(*t.T), 1e-9)[:, None]
        nrm = np.c_[-t[:, 1], t[:, 0]]
        out = np.zeros(len(route))
        for k in range(len(route)):
            for sd in (side, -side):
                ok = True
                for f in (0.5, 1.0):
                    r, c = self.walkway.cell(route[k] + nrm[k] * sd * GROUP_SIDE_M * f)
                    if not (0 <= r < self.cost.shape[0] and 0 <= c < self.cost.shape[1]) or not np.isfinite(
                        self.cost[r, c]
                    ):
                        ok = False
                if ok:
                    out[k] = sd * GROUP_SIDE_M
                    break
        # narrow first (min over the window), then blend
        w = max(int(OFFSET_BLEND_M / max(np.median(np.diff(arc)), 1e-3)), 1)
        mag = np.abs(out)
        narrow = np.array([mag[max(k - w, 0) : k + w + 1].min() for k in range(len(mag))])
        sign = np.where(out != 0, np.sign(out), side)
        kernel = np.ones(2 * w + 1) / (2 * w + 1)
        smooth = np.convolve(np.pad(narrow * sign, w, mode="edge"), kernel, mode="valid")
        return smooth

    def poses(self):
        """(x, y), heading (rad) and speed (m/s) per person."""
        return self.xy.copy(), np.array([p.yaw for p in self.people]), np.array([p.v for p in self.people])


def _arc(pts):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]


def _at(pts, arc, s):
    k = int(np.clip(np.searchsorted(arc, s) - 1, 0, len(arc) - 2))
    f = (s - arc[k]) / max(arc[k + 1] - arc[k], 1e-9)
    return pts[k] + np.clip(f, 0.0, 1.0) * (pts[k + 1] - pts[k])


def _progress(pts, arc, xy, s):
    """Arc of the route point nearest `xy`, searched from s-1 to s+4 m (monotone progress)."""
    lo, hi = np.searchsorted(arc, s - 1.0), np.searchsorted(arc, s + 4.0) + 1
    seg = slice(max(lo - 1, 0), min(hi, len(arc)))
    P, A = pts[seg], arc[seg]
    if len(P) < 2:
        return s
    a, b = P[:-1], P[1:]
    ab = b - a
    L2 = np.maximum((ab**2).sum(1), 1e-12)
    f = np.clip(((xy - a) * ab).sum(1) / L2, 0.0, 1.0)
    d = np.hypot(*(a + f[:, None] * ab - xy).T)
    k = int(np.argmin(d))
    return max(s, float(A[k] + f[k] * (A[k + 1] - A[k])))


def _speed_profile(pts, arc):
    """Speed allowed at each route point: LATERAL_MPS2 sideways on bends (radius from the turn over neighbouring
    points), reached at ARRIVE_MPS2 beforehand, and a stop at the end."""
    n = len(pts)
    v = np.full(n, np.inf)
    for k in range(1, n - 1):
        u, w = pts[k] - pts[k - 1], pts[k + 1] - pts[k]
        lu, lw = np.hypot(*u), np.hypot(*w)
        if lu < 1e-3 or lw < 1e-3:
            continue
        turn = np.arccos(np.clip(np.dot(u, w) / (lu * lw), -1.0, 1.0))
        if turn > 1e-3:
            v[k] = np.sqrt(LATERAL_MPS2 * 0.5 * (lu + lw) / turn)
    v[-1] = 0.0
    for k in range(n - 2, -1, -1):
        v[k] = min(v[k], np.sqrt(v[k + 1] ** 2 + 2 * ARRIVE_MPS2 * (arc[k + 1] - arc[k])))
    return v


def _round(pts, cut):
    """Round off each corner: replace it by the points `cut` (at most a quarter of a side) before and after it."""
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    for a, b, c in zip(pts[:-2], pts[1:-1], pts[2:], strict=False):
        u, w = b - a, c - b
        cu, cw = min(cut, np.hypot(*u) / 4), min(cut, np.hypot(*w) / 4)
        out += [b - u / max(np.hypot(*u), 1e-9) * cu, b + w / max(np.hypot(*w), 1e-9) * cw]
    out.append(pts[-1])
    return np.array(out)


def _pull(pts, walkway, cost):
    """String-pull a cell path: from each kept point the straight segment is extended along the path as long as every
    cell under it is reachable and no dearer than the dearest cell of the path it replaces."""
    cm = walkway.cell_m

    def cells(seg):
        return ((seg[:, 1] - walkway.y0) // cm).astype(int), ((seg[:, 0] - walkway.x0) // cm).astype(int)

    on_path = cost[cells(pts)]
    out, i, n = [pts[0]], 0, len(pts)
    while i < n - 1:
        j, dear = i + 1, max(on_path[i], on_path[i + 1])
        while j + 1 < n:
            dear_next = max(dear, on_path[j + 1])
            m = int(np.hypot(*(pts[j + 1] - pts[i])) / (0.25 * cm)) + 2
            seg = pts[i] + (pts[j + 1] - pts[i]) * (np.arange(m) / (m - 1))[:, None]
            here = cost[cells(seg)]
            if not (np.isfinite(here).all() and here.max() <= dear_next):
                break
            j, dear = j + 1, dear_next
        out.append(pts[j])
        i = j
    return np.array(out)


def _densify(pts, step):
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:], strict=False):
        n = max(int(np.ceil(np.hypot(*(b - a)) / step)), 1)
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)
