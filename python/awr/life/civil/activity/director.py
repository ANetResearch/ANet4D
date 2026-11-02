"""Event-driven activity selection and transactional destination allocation.

The director never writes a pose. A motion backend executes the returned steps;
arrival/service/transfer acknowledgements must use its measured state. This is
not a collision solver. Routes (including parking manoeuvres) must be validated
by the backend. No model API, query truth or recording horizon enters decisions.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

XY = tuple[float, float]


@dataclass(frozen=True)
class Route:
    length_m: float
    duration_s: float
    evidence: str
    # Backend-owned identifier, not a trajectory manufactured by the director.
    points: tuple[XY, ...] = ()
    # Optional immutable walking geometry supplied by the planning backend.
    # Execution must revalidate it on its current geometry; this is not a
    # permission to move, and the director never generates or changes it.

    def __post_init__(self):
        points = tuple(tuple(float(v) for v in p) for p in self.points)
        if points and (len(points) < 2 or any(len(p) != 2 or not all(map(math.isfinite, p)) for p in points)):
            raise ValueError("Walking evidence needs at least two finite XY points")
        object.__setattr__(self, "points", points)


@dataclass(frozen=True)
class Place:
    id: str
    xy: XY
    kind: str
    slots: tuple[str, ...]
    opens_s: float = -math.inf
    closes_s: float = math.inf
    parking: tuple[str, ...] = ()
    access_xy: XY | None = None
    group: str | None = None
    affordances: tuple[str, ...] = ()
    semantic_source: str | None = None
    # Every parking entry names a separate, verified berth Place.

    def __post_init__(self):
        if not _feature_names(self.affordances):
            raise ValueError(f"Invalid place affordances: {self.id}")
        if (self.affordances and not self.semantic_source) or (
            self.semantic_source is not None
            and (not isinstance(self.semantic_source, str) or not self.semantic_source.strip())
        ):
            raise ValueError(f"Place affordances need an explicit semantic source: {self.id}")
        object.__setattr__(self, "affordances", tuple(self.affordances))


@dataclass(frozen=True)
class Activity:
    purpose: str
    kinds: tuple[str, ...]
    dwell_s: tuple[float, float]
    recurrence_s: float
    weight: float = 1.0
    roles: tuple[str, ...] = ("resident",)
    modes: tuple[str, ...] = ("walk",)
    readiness: float = 0.5  # soft needs can be acted on before they become overdue
    requires: tuple[str, ...] = ()
    # A tour: the car drives through waypoints of the road network on the way (the backend's route mode 'tour'),
    # and the destinations are taken in random order, not by travel time (the drive is the activity).
    tour: bool = False

    def __post_init__(self):
        if not _feature_names(self.requires):
            raise ValueError(f"Invalid activity requirements: {self.purpose}")
        object.__setattr__(self, "requires", tuple(self.requires))


def supports_activity(place, activity):
    """Kind is not a capability: an outdoor post-office approach is not an interior.

    Legacy activities with no requirements keep their existing semantics. A new
    requirement is never inferred from a place name, kind or visible building.
    Environment adapters own the evidence behind each supplied affordance.
    """
    return place.kind in activity.kinds and all(k in place.affordances for k in activity.requires)


def _feature_names(value):
    return (
        isinstance(value, (tuple, list))
        and all(isinstance(k, str) and k.strip() == k and k for k in value)
        and len(value) == len(set(value))
    )


@dataclass
class Actor:
    id: str
    home: str
    role: str = "resident"
    vehicle: str | None = None
    completed: dict[str, float] = field(default_factory=dict)
    visits_since_home: int = 0
    last_place: str | None = None
    holding: str | None = None
    retry_at: float = -math.inf
    decisions: int = 0
    activities: tuple[str, ...] | None = None  # admitted local affordances, not runtime demand deletion
    places: frozenset[str] | None = None  # destinations this actor may choose (a scenario overlay), None: all


@dataclass
class Vehicle:
    id: str
    owner: str
    berth: str
    clearance_m: float = 6.0
    in_trip: bool = False
    occupant: str | None = None


@dataclass(frozen=True)
class Step:
    kind: str
    entity: str
    xy: XY
    duration_s: float = 0.0
    route: Route | None = None
    partner_xy: XY | None = None


@dataclass
class Intent:
    actor: str
    place: str
    purpose: str
    slot: str
    steps: list[Step]
    destination_berth: str | None = None
    index: int = 0
    service_started: float | None = None
    paused_at: float | None = None
    id: str | None = None

    @property
    def step(self):
        return self.steps[self.index]


class Director:
    """A single authority for activities and resource ownership.

    route(mode, start, end, entity) returns a verified Route or None.
    available(place, actor) checks current physical occupancy (in addition to
    logical reservations). Competing actors are processed in stable ID order by
    the caller. Each actor has its own random stream, independent of roster order.
    """

    def __init__(
        self,
        places,
        activities,
        route: Callable,
        *,
        seed=0,
        retry_s=15.0,
        max_wait_s=120.0,
        home_dwell_s=(60.0, 180.0),
        errands_before_home=2,
        available=None,
        max_speed_mps=None,
        route_pruning=True,
        in_vehicle_time_factor=1.0,
        min_drive_m=0.0,
        keeps_ways=None,
        selection=None,
        parking_walk=None,
    ):
        from .selection import Selection

        self.selection = Selection(selection)
        places = list(places)
        self.places = {p.id: p for p in places}
        if len(self.places) != len(places):
            raise ValueError("Duplicate place ID")
        self.activities = list(activities)
        self.route = route
        self.max_speed_mps = dict(max_speed_mps or {})
        if any(k not in ("walk", "drive") or not math.isfinite(v) or v <= 0 for k, v in self.max_speed_mps.items()):
            raise ValueError("Invalid backend speed bound")
        self.route_pruning = route_pruning
        # Choice only: time in the car counts with this factor against walking time when journeys are compared
        # (travel-demand practice weighs walking time above in-vehicle time). Timing and opening checks use the
        # actual durations. 1 keeps every journey's score at its plain duration.
        if not math.isfinite(in_vehicle_time_factor) or not 0.0 < in_vehicle_time_factor <= 1.0:
            raise ValueError("In-vehicle time factor must lie in (0, 1]")
        self.in_vehicle_time_factor = float(in_vehicle_time_factor)
        # A car journey whose measured drive is shorter than this is not offered: the destination is walked to.
        if not math.isfinite(min_drive_m) or min_drive_m < 0:
            raise ValueError("Minimum drive length must be finite and non-negative")
        self.min_drive_m = float(min_drive_m)
        # keeps_ways(vehicle, berth, needs) -> bool: the driving backend's check that parking at a berth leaves the
        # other cars the drives they still need (needs: ways_home()). None: not checked.
        self.keeps_ways = keeps_ways
        # keeps_roads(vehicle, berth) -> bool: the driving backend's check that a car parked at a berth leaves the
        # roads passable, as every parked car of the scene must (DrivingRoutes.keeps_roads). None: not checked.
        self.keeps_roads = None
        # Lane nodes of the main road network (the driving backend's way_out_nodes): a car parked at its owner's home
        # keeps a way to one of them, so that no other car's parking shuts it in. Empty: not required.
        self.way_out = ()
        # parking_walk(vehicle, berth, start, end, person) -> Route | None (read-only, adapted from codex 454fd00): the
        # walk from the berth proven with this car parked at the berth, not with its body still at the source berth.
        # None: the walk is planned on the present geometry.
        self.parking_walk = parking_walk
        self.seed = seed
        self.retry_s = float(retry_s)
        self.max_wait_s = float(max_wait_s)
        self.home_dwell_s = tuple(home_dwell_s)
        self.errands_before_home = int(errands_before_home)
        self.available = available or (lambda place, actor: True)
        self.actors = {}
        self.vehicles = {}
        self.active = {}
        self.holders = {}  # concrete slot -> actor or vehicle ID
        # A car's home berth (the berth of its owner's home parking it starts at) -> the car. Another car ends a journey
        # there only to return to its own home, whose parking it shares, also while the car is away: home parking
        # serves the returns home (cruise5 r11f v10, two cruisers in turn ended tours at the home berth of a third car,
        # whose owner then had no berth to return to).
        self.home_berths = {}
        self.events = []
        self.choice_policy = "facility_need_then_available_early_activity.v2"
        if self.selection.kind == "diverse_feasible_v1":
            self.choice_policy = "facility_need_then_diverse_feasible.v1"
        self.decision_wall_s = 0.0  # profiling only, excluded from all decisions/events
        self.car_rejections = {}  # diagnostics only: why candidate car journeys were not offered
        self._why = {}  # diagnostics only: why the journeys of the current decision were not offered
        self._after_drive = []  # diagnostics only: car journeys of the current decision refused after their drive was found
        self.car_choice = {}
        self._rng = {}
        self._time = {}
        self._waiting = {}
        slots = [s for p in places for s in p.slots]
        if len(slots) != len(set(slots)):
            raise ValueError("Slots must be globally unique")
        for p in places:
            if not _feature_names(p.affordances) or (
                p.semantic_source is not None
                and (not isinstance(p.semantic_source, str) or not p.semantic_source.strip())
            ):
                raise ValueError(f"Invalid place affordances/source: {p.id}")
            if (
                not p.slots
                or not all(math.isfinite(v) for v in p.xy)
                or math.isnan(p.opens_s)
                or math.isnan(p.closes_s)
                or p.opens_s >= p.closes_s
                or (p.access_xy is not None and not all(math.isfinite(v) for v in p.access_xy))
            ):
                raise ValueError(f"Invalid place: {p.id}")
            if any(b not in self.places or self.places[b].kind != "berth" for b in p.parking):
                raise ValueError(f"Unmapped parking: {p.id}")
        for a in self.activities:
            if not _feature_names(a.requires):
                raise ValueError(f"Invalid activity requirements: {a.purpose}")
            if (
                not all(math.isfinite(v) for v in (*a.dwell_s, a.recurrence_s, a.weight, a.readiness))
                or a.recurrence_s <= 0
                or a.weight <= 0
                or not 0 < a.readiness <= 1
                or not a.kinds
                or not 0 < a.dwell_s[0] <= a.dwell_s[1]
                or not set(a.modes) <= {"walk", "car"}
                or (a.tour and tuple(a.modes) != ("car",))
            ):
                raise ValueError(f"Invalid activity: {a.purpose}")
        if len({a.purpose for a in self.activities}) != len(self.activities):
            raise ValueError("Activity purposes must be unique")
        if (
            not all(math.isfinite(v) for v in (*self.home_dwell_s, self.retry_s, self.max_wait_s))
            or not 0 < self.retry_s <= self.max_wait_s
            or self.errands_before_home < 1
            or not 0 < self.home_dwell_s[0] <= self.home_dwell_s[1]
        ):
            raise ValueError("Invalid decision timing")

    def emit(self, kind, actor, t, **data):
        self.events.append(dict(kind=kind, actor=actor, t_s=round(float(t), 6), **data))

    def clock(self, actor, t):
        if not math.isfinite(t) or t < self._time.get(actor, -math.inf):
            raise ValueError("Activity time must be finite and monotonic")
        self._time[actor] = t

    def add_actor(self, actor, *, t=0.0):
        if actor.id in self.actors or actor.home not in self.places:
            raise ValueError("Duplicate actor or unmapped home")
        if self.places[actor.home].kind != "home":
            raise ValueError("Actor needs an explicit home/starting activity place")
        if actor.activities is not None and (
            not actor.activities
            or not set(actor.activities) <= {a.purpose for a in self.activities if actor.role in a.roles}
        ):
            raise ValueError("Actor activity profile contains unmapped or unavailable purposes")
        home = self.places[actor.home]
        slot = self.free_slot(home, actor.id)
        if slot is None:
            raise ValueError("Home capacity exhausted")
        self.actors[actor.id] = actor
        digest = hashlib.sha256(f"{self.seed}:{actor.id}".encode()).digest()
        rng = self._rng[actor.id] = random.Random(int.from_bytes(digest, "big"))
        for a in self.activities:
            if actor.activities is not None and a.purpose not in actor.activities:
                continue
            actor.completed.setdefault(a.purpose, t - rng.uniform(0.0, a.recurrence_s))
        actor.holding = slot
        self.holders[slot] = actor.id
        self.clock(actor.id, t)

    def assign_role(self, actor_id, role, activities, *, t=0.0):
        """Give an admitted actor another role and activity profile (e.g. a car owner who becomes a cruising
        driver). Needs of the new purposes start part-way through their recurrence, as in add_actor."""
        actor = self.actors[actor_id]
        rules = {a.purpose: a for a in self.activities if role in a.roles}
        if actor_id in self.active or not activities or not set(activities) <= set(rules):
            raise ValueError("A new role needs an idle actor and admitted purposes of that role")
        actor.role, actor.activities = role, tuple(activities)
        rng = self._rng[actor_id]
        for purpose in actor.activities:
            actor.completed.setdefault(purpose, t - rng.uniform(0.0, rules[purpose].recurrence_s))

    def add_vehicle(self, vehicle):
        if (
            vehicle.id in self.vehicles
            or vehicle.id in self.actors
            or vehicle.owner not in self.actors
            or vehicle.berth not in self.places
            or self.places[vehicle.berth].kind != "berth"
            or vehicle.clearance_m <= 0
        ):
            raise ValueError("Invalid vehicle ownership or berth")
        actor = self.actors[vehicle.owner]
        if actor.vehicle != vehicle.id:
            raise ValueError("Vehicle ownership must be reciprocal")
        slot = self.free_slot(self.places[vehicle.berth], vehicle.id)
        if slot is None:
            raise ValueError("Initial berth is occupied")
        self.holders[slot] = vehicle.id
        self.vehicles[vehicle.id] = vehicle
        if vehicle.berth in self.places[actor.home].parking:
            self.home_berths[vehicle.berth] = vehicle.id

    def free_slot(self, place, owner):
        return next((s for s in place.slots if self.holders.get(s) in (None, owner)), None)

    def _route(self, mode, start, end, entity):
        return self._checked_route(self.route(mode, start, end, entity), mode, start, end)

    def _checked_route(self, r, mode, start, end):
        if r is not None and (
            not r.evidence
            or not math.isfinite(r.length_m)
            or not math.isfinite(r.duration_s)
            or r.length_m < math.dist(start, end) - 1e-6
            or r.duration_s < 0
            or (r.length_m > 0 and r.duration_s <= 0)
        ):
            raise ValueError("Backend returned an invalid or unproven route")
        if r is not None and r.points:
            length = sum(math.dist(a, b) for a, b in zip(r.points[:-1], r.points[1:], strict=False))
            if (
                mode != "walk"
                or math.dist(r.points[0], start) > 1e-6
                or math.dist(r.points[-1], end) > 1e-6
                or abs(length - r.length_m) > 1e-6
            ):
                raise ValueError("Walking geometry differs from its measured route")
        limit = self.max_speed_mps.get(mode)
        if r is not None and limit and r.duration_s + 1e-6 < r.length_m / limit:
            raise ValueError("Backend route violates its declared speed bound")
        return r

    def _reject(self, reason, car=True):
        if car:
            self.car_rejections[reason] = self.car_rejections.get(reason, 0) + 1
        self._why[reason] = self._why.get(reason, 0) + 1

    def _journey(self, actor, xy, place, mode, t, dwell, tour=False):
        if mode == "walk":
            r = self._route("walk", xy, place.xy, actor.id)
            if r is None:
                self._reject("no_walk", car=False)
                return None
            # Park once, walk between nearby activities, then recover the car.
            # Walking home still requires bringing the owned car home first;
            # other activities require a verified walk back to its real berth.
            if actor.vehicle and self.vehicles[actor.vehicle].berth not in self.places[actor.home].parking:
                vehicle = self.vehicles[actor.vehicle]
                parked = self.places[vehicle.berth]
                if (
                    place.id == actor.home
                    and not vehicle.in_trip
                    and t - self._waiting.get(actor.id, math.inf) + self.retry_s > self.max_wait_s + 1e-6
                ):
                    # The driver has found no way to bring the car home and its next attempt would come after
                    # max_wait_s (every home berth refused, cruise3-r11f v2, person-005 at 1352.6 s): it leaves the car
                    # parked where it stands and walks home. The car keeps its home berth; a later drive starts from
                    # where it stands.
                    return [Step("walk", actor.id, place.xy, route=r)], None, r.duration_s
                if place.id == actor.home or vehicle.in_trip or parked.access_xy is None:
                    self._reject("walk_home_while_car_away" if place.id == actor.home else "car_unavailable", car=False)
                    return None
                back = self._route("walk", place.xy, parked.access_xy, actor.id)
                if back is None or t + r.duration_s + dwell + back.duration_s + 4.0 > parked.closes_s:
                    self._reject("no_walk_back_to_car" if back is None else "car_berth_closes_first", car=False)
                    return None
            return [Step("walk", actor.id, place.xy, route=r)], None, r.duration_s
        if not actor.vehicle or actor.vehicle not in self.vehicles:
            return None
        rejected = self._reject
        v = self.vehicles[actor.vehicle]
        if v.in_trip:
            rejected("car_in_trip")
            return None
        source = self.places[v.berth]
        if source.access_xy is None:
            rejected("car_berth_without_access")
            return None
        approach = self._route("walk", xy, source.access_xy, actor.id)
        if approach is None:
            rejected("no_walk_to_car")
            return None
        options = []
        if not place.parking:
            rejected("destination_without_parking")
        for bid in place.parking:
            berth = self.places[bid]
            if berth.access_xy is None or bid == v.berth:
                rejected("berth_is_own_or_without_access")
                continue
            if self.free_slot(berth, v.id) is None:
                rejected("berth_held")
                continue
            if place.id != actor.home and self.home_berths.get(bid, v.id) != v.id:
                rejected("home_berth_of_another_car")
                continue
            if not self.available(berth, actor.id):
                rejected("berth_occupied_now")
                continue
            if not berth.opens_s <= t < berth.closes_s:
                rejected("berth_closed")
                continue
            drive = self._route("tour" if tour else "drive", source.xy, berth.xy, v.id)
            if drive is None:
                rejected("no_drive")
                continue

            def refused(reason):
                rejected(reason)
                if len(self._after_drive) < 12:
                    self._after_drive.append(
                        dict(place=place.id, berth=bid, reason=reason, drive_m=round(drive.length_m, 1))
                    )

            if drive.length_m < self.min_drive_m:
                refused("drive_shorter_than_minimum")
                continue
            # Before the walk from the berth, the dearest proof (a candidate berth of a cruiser's tour each needs one).
            if self.keeps_roads is not None and not self.keeps_roads(v.id, bid):
                refused("would_block_a_road")
                continue
            walk = (
                self._route("walk", berth.access_xy, place.xy, actor.id)
                if self.parking_walk is None
                else self._checked_route(
                    self.parking_walk(v.id, bid, berth.access_xy, place.xy, actor.id), "walk", berth.access_xy, place.xy
                )
            )
            if walk is None:
                refused("no_walk_from_berth" if self.parking_walk is None else "no_walk_after_parking")
                continue
            cost = approach.duration_s + drive.duration_s + walk.duration_s + 8.0
            if t + cost + dwell > berth.closes_s:
                refused("berth_closes_before_return")
                continue
            # Prove a way back before sending the owner/car into a directed dead end.
            # Route queries are read-only proofs for the current snapshot. A
            # rejected outward leg needs no return proof; viable trips still do.
            home = self.places[actor.home]
            # A trip ending at an actual home berth has already brought the
            # car home. Requiring another drive from that berth to itself
            # incorrectly rejects the return leg with exact geometry backends,
            # which rightly refuse zero-length "journeys".
            escape = bid in home.parking or any(
                self._route("drive", berth.xy, self.places[b].xy, v.id) is not None for b in home.parking
            )
            if not escape:
                refused("no_drive_back")
                continue
            # A car coming home parks at its home parking even where that shuts a car parked at its home off the main
            # roads (residents park in front of each other), but only where no place of that parking keeps the way
            # out (the options are ranked so). Any other journey, though it ends at a berth of its home parking too,
            # keeps every way.
            shuts = False
            if self.keeps_ways is not None and not self.keeps_ways(v.id, bid, self.ways_home(v.id)):
                shuts = place.id == actor.home and self.keeps_ways(v.id, bid, self.ways_home(v.id, way_out=False))
                if not shuts:
                    refused("would_close_a_way_home")
                    continue
            steps = [
                Step("walk", actor.id, source.access_xy, route=approach),
                Step("board", actor.id, source.access_xy, 4.0, partner_xy=source.xy),
                Step("drive", v.id, berth.xy, route=drive),
                Step("alight", actor.id, berth.access_xy, 4.0, partner_xy=berth.xy),
                Step("walk", actor.id, place.xy, route=walk),
            ]
            options.append((shuts, cost, bid, steps))
        if not options:
            return None
        shuts, cost, bid, steps = min(options, key=lambda x: (x[0], x[1], x[2]))
        return steps, bid, cost

    def pending_destinations(self):
        """{car: destination berth} of the admitted car journeys whose car has not left its berth yet: the berth the
        car will occupy (a future body for proofs that a way exists later, DrivingRoutes.pending)."""
        out = {}
        for w in self.vehicles.values():
            intent = self.active.get(w.owner)
            if (
                intent is not None
                and intent.destination_berth
                and not w.in_trip
                and intent.destination_berth != w.berth
            ):
                out[w.id] = intent.destination_berth
        return out

    def ways_home(self, vehicle_id, *, way_out=True):
        """{car: ((berth, target berths), ...)}: the drives the cars other than `vehicle_id` still need.

        A car with an admitted journey that has not left its berth needs its way from that berth to the journey's
        destination berth. A car that stands, or is about to park (an admitted journey), away from its owner's home
        parking needs a way from that berth to the home parking. With `way_out`, a car parked at its owner's home
        parking with no admitted journey needs a way out to the main road network (Director.way_out), for whatever
        its owner does next (_journey lifts it for a car coming home when no place of its home parking keeps it)."""
        needs = {}
        for w in self.vehicles.values():
            if w.id == vehicle_id:
                continue
            intent = self.active.get(w.owner)
            legs = []
            berth = w.berth
            if intent is not None and intent.destination_berth:
                if not w.in_trip and intent.destination_berth != w.berth:
                    legs.append((w.berth, (intent.destination_berth,)))
                berth = intent.destination_berth
            home = self.places[self.actors[w.owner].home]
            if berth not in home.parking:
                legs.append((berth, tuple(home.parking)))
            elif way_out and not w.in_trip and (intent is None or not intent.destination_berth) and self.way_out:
                legs.append((berth, tuple(b for b in self.way_out if b != berth)))
            if legs:
                needs[w.id] = tuple(legs)
        return needs

    def _perceived(self, steps, cost):
        """Journey cost for comparison: in-vehicle time weighted by in_vehicle_time_factor."""
        if self.in_vehicle_time_factor == 1.0:
            return cost
        drive = sum(s.route.duration_s for s in steps if s.kind == "drive" and s.route is not None)
        return cost - (1.0 - self.in_vehicle_time_factor) * drive

    def choose(self, actor_id, t, xy):
        self.clock(actor_id, t)
        actor = self.actors[actor_id]
        if actor_id in self.active:
            return self.active[actor_id]
        if t < actor.retry_at:
            return None
        rng = self._rng[actor_id]
        self._why = {}
        self._after_drive = []
        mode_order = self.selection.mode_order(self.seed, actor_id, actor.decisions, actor.vehicle in self.vehicles)
        mode_rank = {mode: len(mode_order) - i for i, mode in enumerate(mode_order)}
        home = self.places[actor.home]
        at_home = math.dist(xy, home.xy) <= 0.1
        must_home = actor.visits_since_home >= self.errands_before_home
        choices = []
        candidates = []
        # Bounded decision evidence for car owners. This observes evaluated
        # alternatives; it neither requests extra routes nor changes RNG draws.
        mode_best = {}
        journey_attempts = {}
        journey_rejected = {}
        # A park with 200 standing positions is one activity destination, not
        # 200 independent lottery tickets. Access positions share need utility;
        # try them nearest first until a physically feasible journey is found.
        # No top-k cut: if the closest approaches fail, later ones remain eligible.
        facility_utilities = {}
        due_rules = 0
        readiness = {}
        rules = list(self.activities)
        if not at_home:
            rules.append(
                Activity("return_home", ("home",), self.home_dwell_s, 1.0, roles=(actor.role,), modes=("walk", "car"))
            )
        selection_reason = "due_activity_or_home"
        early_attempted = False
        eligible_total = 0
        for early in (False, True):
            if early:
                if choices or due_rules == 0 or must_home:
                    break
                # A blocked errand does not require waiting at home while all
                # other activities are forbidden by a soft readiness threshold.
                # Try another genuine positive need, without completing or
                # deleting the original unmet need. All access, opening-time,
                # capacity and measured-route checks remain identical.
                early_attempted = True
                candidates = []
            for rule in sorted(rules, key=lambda r: r.purpose):
                if (
                    actor.role not in rule.roles
                    or (must_home and rule.purpose != "return_home")
                    or (
                        actor.activities is not None
                        and rule.purpose != "return_home"
                        and rule.purpose not in actor.activities
                    )
                ):
                    continue
                due = (t - actor.completed.get(rule.purpose, t)) / rule.recurrence_s
                if rule.purpose == "return_home":
                    due = 2.0 if must_home else 0.05
                if early:
                    if rule.purpose == "return_home" or not 0.0 < due < rule.readiness:
                        continue
                elif due < rule.readiness and rule.purpose != "return_home":
                    continue
                if not early and rule.purpose != "return_home":
                    due_rules += 1
                    readiness[rule.purpose] = round(due, 3)
                dwell = rng.uniform(*rule.dwell_s)
                for place in sorted(self.places.values(), key=lambda p: p.id):
                    if (
                        not supports_activity(place, rule)
                        or (place.kind == "home" and place.id != actor.home)
                        or (actor.places is not None and place.kind != "home" and place.id not in actor.places)
                        or place.id == actor.last_place
                        or (
                            place.group is not None
                            and actor.last_place is not None
                            and place.group == self.places[actor.last_place].group
                        )
                        or math.dist(xy, place.xy) <= 0.1
                        or not place.opens_s <= t < place.closes_s
                        or not self.available(place, actor_id)
                    ):
                        continue
                    slot = self.free_slot(place, actor_id)
                    if slot is None:
                        continue
                    for mode in rule.modes:
                        # Draw before routing: infeasible candidates and pruning must
                        # not shift the actor's random stream or change later choices.
                        group = ("group", place.group) if place.group is not None else ("place", place.id)
                        access_key = (rule.purpose, group, mode)
                        if access_key not in facility_utilities:
                            facility_utilities[access_key] = (
                                self.selection.utility(
                                    self.seed,
                                    actor_id,
                                    actor.decisions,
                                    rule.purpose,
                                    group,
                                    rule.weight * min(due, 4.0),
                                )
                                if mode_order
                                else rule.weight * min(due, 4.0) * rng.uniform(0.9, 1.1)
                            )
                        numerator = facility_utilities[access_key]
                        limit = (
                            self.max_speed_mps.get("walk")
                            if mode == "walk"
                            else (
                                max(self.max_speed_mps.values())
                                if {"walk", "drive"} <= self.max_speed_mps.keys()
                                else None
                            )
                        )
                        straight_distance = math.dist(xy, place.xy)
                        lower_cost = straight_distance / limit if limit and not rule.tour else 0.0
                        if mode == "car":
                            lower_cost *= self.in_vehicle_time_factor  # still a lower bound of the perceived cost
                        upper = numerator if rule.tour else self.selection.score(numerator, lower_cost)
                        candidates.append(
                            (upper, place.id, mode, numerator, rule, place, slot, dwell, access_key, straight_distance)
                        )
            # Exact branch-and-bound, not a top-k destination shortlist. Unreachable
            # high-utility places never hide a reachable lower-ranked destination.
            best_score = -math.inf
            best_rank = -1
            resolved_access = set()
            for upper, _, mode, numerator, rule, place, slot, dwell, access_key, _ in sorted(
                candidates, key=lambda c: (mode_rank.get(c[2], 0), c[0], -c[9], c[1], c[2]), reverse=True
            ):
                if access_key in resolved_access:
                    continue
                rank = mode_rank.get(mode, 0)
                if self.route_pruning and (rank < best_rank or (rank == best_rank and upper + 1e-12 < best_score)):
                    break
                if actor.vehicle:
                    journey_attempts[mode] = journey_attempts.get(mode, 0) + 1
                journey = self._journey(actor, xy, place, mode, t, dwell, tour=rule.tour)
                if journey is None:
                    if actor.vehicle:
                        journey_rejected[mode] = journey_rejected.get(mode, 0) + 1
                    continue
                steps, berth, cost = journey
                if t + cost + dwell >= place.closes_s:
                    self._reject("closes_before_activity", car=False)
                    continue
                resolved_access.add(access_key)
                score = numerator if rule.tour else self.selection.score(numerator, self._perceived(steps, cost))
                if actor.vehicle and (mode not in mode_best or score > mode_best[mode]["score"]):
                    mode_best[mode] = dict(
                        place=place.id,
                        purpose=rule.purpose,
                        score=score,
                        travel_s=cost,
                        drive_s=sum(s.route.duration_s for s in steps if s.kind == "drive"),
                        walk_s=sum(s.route.duration_s for s in steps if s.kind == "walk"),
                        transfer_s=sum(s.duration_s for s in steps if s.kind in ("board", "alight")),
                        destination_berth=berth,
                    )
                if rank > best_rank:
                    best_rank, best_score = rank, score
                elif rank == best_rank:
                    best_score = max(best_score, score)
                intent = Intent(
                    actor_id, place.id, rule.purpose, slot, steps + [Step("activity", actor_id, place.xy, dwell)], berth
                )
                choices.append((rank, score, place.id, mode, intent))
            eligible_total += len(candidates)
            if choices:
                if early:
                    selection_reason = "available_early_activity_while_due_unavailable"
                break
        if actor.vehicle:
            self.emit(
                "activity_choice_evaluated",
                actor_id,
                t,
                vehicle=actor.vehicle,
                journey_attempts=journey_attempts,
                journey_rejected=journey_rejected,
                best_evaluated_by_mode=mode_best,
                eligible_candidates=eligible_total,
                route_pruning=self.route_pruning,
                scope="Evaluated alternatives only; absent modes may be inadmissible or bounded out. No additional routes requested for diagnostics.",
            )
        actor.decisions += 1
        if not choices and at_home and due_rules == 0:
            # No unmet demand: an explicit ongoing home activity, bounded by the
            # next need. Unreachable/full destinations with due demand cannot use
            # this branch to hide a failure as 'rest'.
            next_due = min(
                (
                    actor.completed[a.purpose] + a.readiness * a.recurrence_s - t
                    for a in self.activities
                    if actor.role in a.roles and (actor.activities is None or a.purpose in actor.activities)
                ),
                default=self.retry_s,
            )
            dwell = max(0.1, min(rng.uniform(*self.home_dwell_s), next_due))
            slot = self.free_slot(home, actor_id)
            if slot is None:
                raise RuntimeError("Resident home became unavailable")
            choices.append(
                (
                    0,
                    0.0,
                    home.id,
                    "stay",
                    Intent(
                        actor_id, home.id, "pause_at_home_street", slot, [Step("activity", actor_id, home.xy, dwell)]
                    ),
                )
            )
        if not choices:
            actor.retry_at = t + self.retry_s
            since = self._waiting.setdefault(actor_id, t)
            why = dict(sorted(self._why.items()))
            self.emit(
                "decision_wait",
                actor_id,
                t,
                reason="no_due_feasible_activity",
                since_s=since,
                ready_needs=readiness,
                eligible_candidates=eligible_total,
                early_fallback_attempted=early_attempted,
                must_return_home=must_home,
                journey_rejections=why,
                refused_after_drive=list(self._after_drive),
            )
            # Waiting is explicit, never relabelled as a completed activity.
            if t - since > self.max_wait_s + 1e-6:
                self.emit(
                    "activity_failure",
                    actor_id,
                    t,
                    reason="no_feasible_continuation",
                    waited_s=t - since,
                    must_return_home=must_home,
                    eligible_candidates=eligible_total,
                    journey_rejections=why,
                )
                raise RuntimeError(
                    f"{actor_id}: no feasible continuation for {t - since:.1f}s "
                    f"(must return home: {must_home}, candidates: {eligible_total}, rejected: {why})"
                )
            return None
        best = max(choices, key=lambda x: x[:-1])
        intent = best[-1]
        if actor.vehicle:
            # Diagnostics only: how often an owner had a feasible car journey among the evaluated options, and took it.
            c = self.car_choice
            c["decisions"] = c.get("decisions", 0) + 1
            c["with_car_option"] = c.get("with_car_option", 0) + any(x[3] == "car" for x in choices)
            c["car_chosen"] = c.get("car_chosen", 0) + (best[3] == "car")
        if mode_order:
            chosen_mode = (
                "car"
                if intent.destination_berth
                else ("walk" if any(s.kind == "walk" for s in intent.steps) else "stay")
            )
            self.emit(
                "activity_selection_policy",
                actor_id,
                t,
                policy=self.selection.kind,
                decision=actor.decisions - 1,
                proposed_mode_order=list(mode_order),
                selected_mode=chosen_mode,
                used_alternative=chosen_mode not in ("stay", mode_order[0]),
                scope="Seeded demand proposal; all route/resource checks still required. "
                "Priority weights are not realised mode quotas or evidence of road capacity.",
            )
        intent.id = f"{actor_id}:{actor.decisions}"
        # All checks precede mutation: a failed multimodal plan leaks no reservation.
        self.holders[intent.slot] = actor_id
        if intent.destination_berth:
            v = self.vehicles[actor.vehicle]
            self.holders[self.free_slot(self.places[intent.destination_berth], v.id)] = v.id
        self.active[actor_id] = intent
        if (
            intent.place == actor.home
            and not intent.destination_berth
            and actor.vehicle
            and self.vehicles[actor.vehicle].berth not in home.parking
        ):
            self.emit(
                "car_left_away",
                actor_id,
                t,
                vehicle=actor.vehicle,
                berth=self.vehicles[actor.vehicle].berth,
                waited_s=round(t - self._waiting.get(actor_id, t), 6),
            )
        self._waiting.pop(actor_id, None)
        self.emit(
            "activity_selected",
            actor_id,
            t,
            purpose=intent.purpose,
            place=intent.place,
            place_affordances=list(self.places[intent.place].affordances),
            semantic_source=self.places[intent.place].semantic_source,
            choice_policy=self.choice_policy,
            selection_reason=selection_reason,
            deferred_ready_needs=readiness if early_attempted else {},
            intent_id=intent.id,
            vehicle=actor.vehicle if intent.destination_berth else None,
            destination_berth=intent.destination_berth,
            steps=[
                dict(
                    kind=s.kind,
                    entity=s.entity,
                    xy=list(s.xy),
                    duration_s=s.duration_s,
                    partner_xy=list(s.partner_xy) if s.partner_xy is not None else None,
                )
                for s in intent.steps
            ],
            mode="car"
            if intent.destination_berth
            else ("walk" if any(s.kind == "walk" for s in intent.steps) else "stay"),
            routes=[s.route.evidence for s in intent.steps if s.route],
        )
        return intent

    def observe_departure(self, actor_id, xy, vehicle_xy=None):
        """Release old slots only after the corresponding physical body cleared."""
        actor = self.actors[actor_id]
        intent = self.active.get(actor_id)
        if intent is None:
            return
        if actor.holding and actor.holding != intent.slot:
            old_place = next(p for p in self.places.values() if actor.holding in p.slots)
            if math.dist(xy, old_place.xy) > 1.0:
                self.holders.pop(actor.holding, None)
                actor.holding = None
        if actor.vehicle and vehicle_xy is not None:
            v = self.vehicles[actor.vehicle]
            source = self.places[v.berth]
            if intent.step.kind == "drive" and math.dist(vehicle_xy, source.xy) > v.clearance_m:
                for slot in source.slots:
                    if self.holders.get(slot) == v.id:
                        del self.holders[slot]
                v.in_trip = True

    def acknowledge(
        self, actor_id, t, xy, speed, *, vehicle_xy=None, vehicle_speed=0.0, radius=0.1, vehicle_radius=None
    ):
        """Report current backend poses. Returns True only when this step completes.

        Missing arrival, motion, displaced service or absent transfer partner never
        completes a step. Acknowledgements are private simulator events, not agent
        claims. Parking orientation/footprint validity is the backend's contract.
        vehicle_radius separates the carrier parking contract from pedestrian
        endpoint precision; omission preserves the existing common radius.
        """
        self.clock(actor_id, t)
        intent = self.active[actor_id]
        if intent.paused_at is not None:
            return False
        step = intent.step
        vehicle_radius = radius if vehicle_radius is None else vehicle_radius
        if (
            not math.isfinite(radius)
            or not math.isfinite(vehicle_radius)
            or radius <= 0
            or vehicle_radius <= 0
            or not all(math.isfinite(v) for v in (*xy, speed))
        ):
            raise ValueError("Invalid observation")
        if step.kind in ("board", "drive", "alight"):
            valid = (
                vehicle_xy is not None
                and all(math.isfinite(v) for v in (*vehicle_xy, vehicle_speed))
                and math.dist(vehicle_xy, step.partner_xy or step.xy) <= vehicle_radius
                and abs(vehicle_speed) <= 0.08
            )
            if step.kind == "drive":
                v = self.vehicles[self.actors[actor_id].vehicle]
                valid = valid and v.occupant == actor_id and math.dist(xy, vehicle_xy) <= radius
            else:
                valid = valid and math.dist(xy, step.xy) <= radius and abs(speed) <= 0.08
        else:
            valid = math.dist(xy, step.xy) <= radius and abs(speed) <= 0.08
        if not valid:
            intent.service_started = None
            return False
        if step.kind == "activity":
            place = self.places[intent.place]
            if not place.opens_s <= t < place.closes_s:
                raise RuntimeError(f"{actor_id}: activity place closed before service completed")
        if intent.service_started is None:
            intent.service_started = t
        if t - intent.service_started < step.duration_s:
            return False
        if step.kind in ("board", "alight"):
            v = self.vehicles[self.actors[actor_id].vehicle]
            v.occupant = actor_id if step.kind == "board" else None
        if step.kind == "drive":
            v = self.vehicles[self.actors[actor_id].vehicle]
            old = self.places[v.berth]
            for slot in old.slots:
                if self.holders.get(slot) == v.id:
                    del self.holders[slot]
            v.berth = intent.destination_berth
            v.in_trip = False
        self.emit(
            "step_completed",
            actor_id,
            t,
            step=step.kind,
            purpose=intent.purpose,
            place=intent.place,
            step_index=intent.index,
            intent_id=intent.id,
            entity=step.entity,
            xy=list(xy),
            speed_mps=speed,
            vehicle_xy=list(vehicle_xy) if vehicle_xy is not None else None,
            vehicle_speed_mps=vehicle_speed,
            service_started_s=intent.service_started,
        )
        intent.index += 1
        intent.service_started = None
        if intent.index == len(intent.steps):
            actor = self.actors[actor_id]
            actor.completed[intent.purpose] = t
            actor.last_place = intent.place
            actor.holding = intent.slot
            actor.visits_since_home = 0 if intent.place == actor.home else actor.visits_since_home + 1
            del self.active[actor_id]
            self.emit(
                "activity_completed", actor_id, t, purpose=intent.purpose, place=intent.place, intent_id=intent.id
            )
        return True

    def pause(self, actor_id, t):
        self.clock(actor_id, t)
        intent = self.active.get(actor_id)
        if intent is not None and intent.paused_at is None:
            intent.paused_at = t

    def resume(self, actor_id, t):
        self.clock(actor_id, t)
        intent = self.active.get(actor_id)
        if intent is not None and intent.paused_at is not None:
            if intent.service_started is not None:
                intent.service_started += t - intent.paused_at
            intent.paused_at = None
