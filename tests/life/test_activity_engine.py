"""Causal scheduling contracts; these tests are not 3D/world admission."""

import json
import math
import unittest

from awr.life.civil.activity import Activity, Actor, Director, Place, Route, Vehicle
from awr.life.civil.activity.grounded import GroundedActivities, ReactiveAgenda
from awr.life.civil.agenda import Occupancy
from awr.life.civil.life_audit import _person_timeline


def route(mode, start, end, entity):
    length = math.dist(start, end)
    return Route(length, length / (4 if mode == "drive" else 1), "test-geometry")


def world(*, car=False, seed=17, router=route):
    places = [
        Place("h", (0.0, 0.0), "home", ("home-slot",), parking=("h-bay",)),
        Place("shop", (100.0, 0.0), "shop", ("shop-slot",), parking=("shop-bay",)),
        Place("park", (200.0, 0.0), "park", ("park-slot",)),
        Place("h-bay", (10.0, 0.0), "berth", ("h-bay-slot",), access_xy=(10.0, 2.0)),
        Place("shop-bay", (90.0, 0.0), "berth", ("shop-bay-slot",), access_xy=(90.0, 2.0)),
    ]
    rules = [
        Activity("shopping", ("shop",), (10.0, 10.0), 200.0, modes=("car",) if car else ("walk",)),
        Activity("rest", ("park",), (20.0, 20.0), 300.0),
    ]
    d = Director(places, rules, router, seed=seed, home_dwell_s=(10.0, 10.0), errands_before_home=1)
    a = Actor("p", "h", vehicle="car" if car else None)
    d.add_actor(a)
    a.completed.update(shopping=-1000.0, rest=0.0)
    if car:
        d.add_vehicle(Vehicle("car", "p", "h-bay"))
    return d


class ActivityContracts(unittest.TestCase):
    def test_bounded_routing_matches_exhaustive_with_unreachable_candidates(self):
        places = [Place("home", (0.0, 0.0), "home", ("home-slot",))] + [
            Place(f"shop-{i:03d}", (float(i * 10), 0.0), "shop", (f"slot-{i}",)) for i in range(1, 101)
        ]
        for seed in range(10):
            counts = [0, 0]
            directors = []
            for index, prune in enumerate((False, True)):

                def router(mode, start, end, entity, index=index):
                    counts[index] += 1
                    # Nearest places are unreachable, so top-k would be wrong.
                    return None if end[0] < 150 else route(mode, start, end, entity)

                d = Director(
                    places,
                    [Activity("shop", ("shop",), (10.0, 10.0), 10.0)],
                    router,
                    seed=seed,
                    max_speed_mps={"walk": 1.0},
                    route_pruning=prune,
                )
                d.add_actor(Actor("p", "home", completed={"shop": -100.0}))
                d.choose("p", 0.0, (0.0, 0.0))
                directors.append(d)
            self.assertEqual(directors[0].active, directors[1].active)
            self.assertEqual(directors[0]._rng["p"].getstate(), directors[1]._rng["p"].getstate())
            self.assertLess(counts[1], counts[0] // 2)

    def test_declared_speed_bound_is_enforced(self):
        d = world()
        d.max_speed_mps = {"walk": 0.5}
        with self.assertRaisesRegex(ValueError, "speed bound"):
            d.choose("p", 0.0, (0.0, 0.0))

    def test_grounded_home_and_static_route_cache_invalidation(self):
        from types import SimpleNamespace

        import numpy as np

        from awr.life.civil.activity.grounded import GroundedActivities

        calls = []

        def plan(start, end, free):
            calls.append((tuple(start), tuple(end)))
            return dict(points=np.array([start, end]))

        network = SimpleNamespace(
            plan=plan, cell=lambda xy: (0, 0), obstacle_mask=lambda *a, **k: np.ones((2, 2), bool)
        )
        pools = dict(
            sites=[
                dict(
                    id="park",
                    kind="park",
                    affordances=("outdoor_stay",),
                    semantic_source="fixture:park",
                    slots=[dict(id="park-slot", xy=(100.0, 0.0))],
                )
            ],
            street=[dict(kind="household", xy=(0.0, 0.0))],
        )
        insufficient = GroundedActivities(network, pools, np.ones((2, 2), bool), {}, 17)
        self.assertEqual(insufficient.place_person("isolated", [], [], [], 0.4, 0.0), (None, None, None))
        self.assertEqual(calls, [])  # reject insufficient facility diversity before expensive routing
        self.assertEqual(insufficient.admission_stats["isolated"]["insufficient_connected_kinds"], 1)
        self.assertFalse(insufficient.admission_stats["isolated"]["budget_exhausted"])
        with self.assertRaisesRegex(ValueError, "positive integer"):
            GroundedActivities(
                network, pools, np.ones((2, 2), bool), {"activity_engine": {"min_activity_kinds": 0}}, 17
            )
        runtime = GroundedActivities(
            network, pools, np.ones((2, 2), bool), {"activity_engine": {"min_activity_kinds": 1}}, 17
        )
        rr, agenda, meta = runtime.place_person("p", [], [], [], 0.4, 0.0)
        self.assertIsNotNone(agenda)
        self.assertTrue(np.array_equal(rr["points"][0], rr["points"][1]))
        self.assertNotIn("arc", rr)  # zero-length interpolation must not divide by zero
        first = runtime.route("walk", (0.0, 0.0), (100.0, 0.0), "p")
        self.assertEqual(len(calls), 2)
        self.assertEqual(runtime.route("walk", (0.0, 0.0), (100.0, 0.0), "p"), first)
        self.assertEqual(len(calls), 2)
        runtime.invalidate_routes()
        runtime.route("walk", (0.0, 0.0), (100.0, 0.0), "p")
        self.assertEqual(len(calls), 3)
        self.assertEqual(runtime.route_stats["invalidations"], 1)

    def test_unreachable_urgent_need_can_coexist_with_feasible_upcoming_activity(self):
        def router(mode, start, end, entity):
            return None if end == (100.0, 0.0) else route(mode, start, end, entity)

        d = world(router=router)
        d.actors["p"].completed["rest"] = -180.0  # readiness .6, not yet overdue
        intent = d.choose("p", 0.0, (0.0, 0.0))
        self.assertEqual(intent.purpose, "rest")
        # The inaccessible shopping need is still unsatisfied, not relabelled.
        self.assertEqual(d.actors["p"].completed["shopping"], -1000.0)

    def test_occupied_home_entries_do_not_consume_the_route_proof_budget(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        import numpy as np

        from awr.life.civil.activity.grounded import GroundedActivities

        calls = []

        def plan(a, b, free):
            calls.append((tuple(a), tuple(b)))
            return dict(points=np.array([a, b]))

        free = np.ones((2, 2), bool)
        free[0, 0] = False
        network = SimpleNamespace(
            plan=plan, cell=lambda xy: (0, int(xy[0] == 1.0)), obstacle_mask=lambda *a, **kw: free
        )
        pools = dict(
            sites=[
                dict(
                    id="park",
                    kind="park",
                    affordances=("outdoor_stay",),
                    semantic_source="fixture:park",
                    slots=[dict(id="p", xy=(100.0, 0.0))],
                )
            ],
            street=[dict(kind="household", xy=(0.0, 0.0)) for _ in range(201)]
            + [dict(kind="household", xy=(1.0, 0.0))],
        )
        runtime = GroundedActivities(network, pools, free, {"activity_engine": {"min_activity_kinds": 1}}, 17)
        with patch("random.Random.shuffle"):
            _, agenda, _ = runtime.place_person("p", [], [], [], 0.4, 0.0)
        self.assertIsNotNone(agenda)
        self.assertEqual(len(calls), 2)  # outward/return proof for the free home only
        self.assertEqual(runtime.admission_stats["p"]["occupied"], 201)
        self.assertEqual(runtime.admission_stats["p"]["accessibility_attempts"], 1)
        self.assertTrue(runtime.admission_stats["p"]["admitted"])

    def test_initial_vehicle_activity_has_no_recording_window_or_convoy_dependency(self):
        from inspect import signature

        from awr.life.civil.activity.traffic import parking_activity

        self.assertNotIn("duration", signature(parking_activity).parameters)
        alone = parking_activity("car-1", 17, -120.0)
        roster = {a: parking_activity(a, 17, -120.0) for a in ("car-0", "car-1", "car-2")}
        self.assertEqual(alone, roster["car-1"])
        self.assertGreaterEqual(alone["duration_s"], 120.0)
        self.assertLessEqual(alone["duration_s"], 1800.0)
        self.assertAlmostEqual(alone["started_s"] + alone["duration_s"], alone["departure_due_s"])
        self.assertAlmostEqual(alone["started_s"] + alone["elapsed_s"], -120.0)

    def test_disconnected_candidates_do_not_exhaust_exact_route_budget(self):
        from types import SimpleNamespace

        import numpy as np

        calls = []

        def plan(a, b, free):
            calls.append((tuple(a), tuple(b)))
            return dict(points=np.array([a, b]))

        free = np.ones((2, 2), bool)
        network = SimpleNamespace(plan=plan, cell=lambda xy: (0, 0), obstacle_mask=lambda *a, **kw: free)
        # The map has 201 free but disconnected households before an eligible
        # public starting place. Population size must not turn this into a
        # fictitious physical capacity ceiling.
        pools = dict(
            sites=[
                dict(
                    id="park",
                    kind="park",
                    affordances=("outdoor_stay",),
                    semantic_source="fixture:park",
                    slots=[dict(id="p", xy=(100.0, 0.0))],
                )
            ],
            street=[dict(kind="household", xy=(1000.0 + i * 4, 0.0)) for i in range(201)]
            + [dict(kind="street", xy=(0.0, 0.0))],
        )
        runtime = GroundedActivities(network, pools, free, {"activity_engine": {"min_activity_kinds": 1}}, 17)
        _, agenda, _ = runtime.place_person("p", [], [], [], 0.4, 0.0)
        self.assertIsNotNone(agenda)
        self.assertEqual(len(calls), 2)
        audit = runtime.admission_stats["p"]
        self.assertEqual(audit["insufficient_connected_kinds"], 201)
        self.assertEqual(audit["accessibility_attempts"], 1)
        self.assertFalse(audit["budget_exhausted"])
        self.assertEqual(len(runtime.home_access_cache), 202)
        runtime.invalidate_routes()
        self.assertEqual(runtime.home_access_cache, {})

    def test_component_prefilter_never_rejects_a_real_route_and_invalidates(self):
        import numpy as np

        from awr.life.civil.activity.grounded import GroundedActivities
        from awr.life.civil.walking import WalkNetwork

        grid = np.ones((24, 40), bool)
        grid[:, 20] = False
        net = WalkNetwork(grid, [0.0, 0.0])
        static = grid.copy()
        runtime = GroundedActivities(net, dict(sites=[], street=[]), static, {"leg_distance_m": [0, 450]}, 17)
        for start in ((2.0, 2.0), (9.6, 2.0), (10.1, 2.0), (16.0, 6.0)):
            for goal in ((4.0, 3.0), (15.0, 4.0), (10.0, 2.0)):
                exact = net.plan(np.asarray(start), np.asarray(goal), static)
                if exact is not None:
                    self.assertTrue(runtime.route_possible(start, goal))
        self.assertFalse(runtime.route_possible((2.0, 2.0), (15.0, 4.0)))
        self.assertEqual(runtime.route_stats["component_rebuilds"], 1)
        # A static parking obstacle cuts the left component. Cached labels must
        # not silently continue claiming the old corridor remains connected.
        static[:, 10] = False
        runtime.invalidate_routes()
        self.assertFalse(runtime.route_possible((2.0, 2.0), (8.0, 2.0)))
        self.assertEqual(runtime.route_stats["component_rebuilds"], 2)

    def test_admitted_local_profile_does_not_generate_impossible_global_needs(self):
        places = [Place("home", (0.0, 0.0), "home", ("h",)), Place("park", (100.0, 0.0), "park", ("p",))]
        d = Director(
            places,
            [
                Activity("rest", ("park",), (10.0, 10.0), 300.0),
                Activity("unmapped_school", ("school",), (10.0, 10.0), 1.0),
            ],
            route,
            seed=17,
        )
        d.add_actor(Actor("local", "home", activities=("rest",)))
        self.assertNotIn("unmapped_school", d.actors["local"].completed)
        d.actors["local"].completed["rest"] = 0.0
        intent = d.choose("local", 50.0, (0.0, 0.0))
        self.assertEqual(intent.purpose, "pause_at_home_street")
        self.assertLessEqual(intent.step.duration_s, 100.0)
        with self.assertRaisesRegex(ValueError, "profile"):
            d.add_actor(Actor("bad", "home", activities=("nonexistent",)))

    def test_runtime_choice_capacity_closing_and_routes(self):
        d = world()
        d.holders["shop-slot"] = "other"
        self.assertIsNone(d.choose("p", 0.0, (0.0, 0.0)))
        self.assertNotIn("p", d.active)
        del d.holders["shop-slot"]
        self.assertEqual(d.choose("p", 15.0, (0.0, 0.0)).purpose, "shopping")
        d = world(router=lambda *args: None)
        self.assertIsNone(d.choose("p", 0.0, (0.0, 0.0)))
        self.assertEqual(d.holders, {"home-slot": "p"})
        d = world()
        d.places["shop"] = Place("shop", (100.0, 0.0), "shop", ("shop-slot",), closes_s=50.0)
        self.assertIsNone(d.choose("p", 0.0, (0.0, 0.0)))

    def test_arrival_service_and_release_require_physical_progress(self):
        d = world()
        intent = d.choose("p", 0.0, (0.0, 0.0))
        d.observe_departure("p", (0.0, 0.0))
        self.assertEqual(d.holders["home-slot"], "p")
        d.observe_departure("p", (2.0, 0.0))
        self.assertNotIn("home-slot", d.holders)
        self.assertFalse(d.acknowledge("p", 100.0, (100.0, 0.0), 1.0))
        self.assertFalse(d.acknowledge("p", 101.0, (99.0, 0.0), 0.0))
        self.assertTrue(d.acknowledge("p", 102.0, (100.0, 0.0), 0.0))
        self.assertEqual(intent.step.kind, "activity")
        self.assertFalse(d.acknowledge("p", 102.0, (100.0, 0.0), 0.0))
        self.assertFalse(d.acknowledge("p", 108.0, (101.0, 0.0), 0.0))
        self.assertFalse(d.acknowledge("p", 109.0, (100.0, 0.0), 0.0))
        self.assertFalse(d.acknowledge("p", 118.0, (100.0, 0.0), 0.0))
        self.assertTrue(d.acknowledge("p", 119.0, (100.0, 0.0), 0.0))
        self.assertNotIn("p", d.active)
        self.assertEqual(d.actors["p"].completed["shopping"], 119.0)
        self.assertEqual(d.holders["shop-slot"], "p")
        nxt = d.choose("p", 120.0, (100.0, 0.0))
        self.assertEqual(nxt.purpose, "return_home")
        d.observe_departure("p", (102.0, 0.0))
        self.assertNotIn("shop-slot", d.holders)

    def test_car_owner_transfer_parking_chain(self):
        d = world(car=True)
        p = d.choose("p", 0.0, (0.0, 0.0))
        self.assertEqual([s.kind for s in p.steps], ["walk", "board", "drive", "alight", "walk", "activity"])
        self.assertEqual(d.holders["shop-bay-slot"], "car")
        self.assertTrue(d.acknowledge("p", 12.0, (10.0, 2.0), 0.0))
        self.assertFalse(d.acknowledge("p", 13.0, (10.0, 2.0), 0.0))  # no car pose
        self.assertFalse(d.acknowledge("p", 14.0, (10.0, 2.0), 0.0, vehicle_xy=(10.0, 0.0)))
        self.assertTrue(d.acknowledge("p", 18.0, (10.0, 2.0), 0.0, vehicle_xy=(10.0, 0.0)))
        d.observe_departure("p", (17.0, 0.0), vehicle_xy=(17.0, 0.0))
        self.assertNotIn("h-bay-slot", d.holders)
        self.assertTrue(d.vehicles["car"].in_trip)
        self.assertFalse(d.acknowledge("p", 40.0, (80.0, 0.0), 0.0, vehicle_xy=(80.0, 0.0)))
        self.assertTrue(d.acknowledge("p", 42.0, (90.0, 0.0), 0.0, vehicle_xy=(90.0, 0.0)))
        self.assertEqual(d.vehicles["car"].berth, "shop-bay")
        self.assertFalse(d.acknowledge("p", 43.0, (90.0, 2.0), 0.0, vehicle_xy=(90.0, 0.0)))
        self.assertTrue(d.acknowledge("p", 47.0, (90.0, 2.0), 0.0, vehicle_xy=(90.0, 0.0)))
        self.assertTrue(d.acknowledge("p", 60.0, (100.0, 0.0), 0.0))
        d.acknowledge("p", 60.0, (100.0, 0.0), 0.0)
        self.assertTrue(d.acknowledge("p", 70.0, (100.0, 0.0), 0.0))
        nxt = d.choose("p", 71.0, (100.0, 0.0))
        self.assertEqual(nxt.destination_berth, "h-bay")
        self.assertEqual(nxt.steps[0].xy, (90.0, 2.0))  # walk back to OWN car
        self.assertEqual(d.holders["shop-bay-slot"], "car")  # not released at decision

    def test_dead_end_car_route_does_not_strand_owner_or_leak_resources(self):
        def directed(mode, start, end, entity):
            return None if mode == "drive" and end[0] < start[0] else route(mode, start, end, entity)

        d = world(car=True, router=directed)
        self.assertIsNone(d.choose("p", 0.0, (0.0, 0.0)))
        self.assertEqual(d.holders, {"home-slot": "p", "h-bay-slot": "car"})

    def test_no_verified_door_no_car_activity(self):
        d = world(car=True)
        d.places["shop-bay"] = Place("shop-bay", (90.0, 0.0), "berth", ("shop-bay-slot",))
        self.assertIsNone(d.choose("p", 0.0, (0.0, 0.0)))

    def test_pause_excludes_courtesy_time_and_clock_cannot_rewind(self):
        d = world()
        d.choose("p", 0.0, (0.0, 0.0))
        d.acknowledge("p", 100.0, (100.0, 0.0), 0.0)
        d.acknowledge("p", 100.0, (100.0, 0.0), 0.0)
        d.pause("p", 105.0)
        self.assertFalse(d.acknowledge("p", 130.0, (100.0, 0.0), 0.0))
        d.resume("p", 135.0)
        self.assertFalse(d.acknowledge("p", 139.0, (100.0, 0.0), 0.0))
        self.assertTrue(d.acknowledge("p", 140.0, (100.0, 0.0), 0.0))
        with self.assertRaises(ValueError):
            d.choose("p", 139.0, (100.0, 0.0))

    def test_seed_replay_and_actor_local_randomness(self):
        a, b = world(), world()
        b.places["other-home"] = Place("other-home", (300.0, 0.0), "home", ("other-slot",))
        b.add_actor(Actor("unrelated", "other-home"))
        a.choose("p", 0.0, (0.0, 0.0))
        b.choose("p", 0.0, (0.0, 0.0))
        self.assertEqual(a.active["p"], b.active["p"])
        self.assertEqual(a.events, b.events)

    def test_two_actors_cannot_reserve_last_slot(self):
        d = world()
        d.places["h2"] = Place("h2", (-10.0, 0.0), "home", ("h2-slot",))
        d.add_actor(Actor("q", "h2"))
        d.actors["q"].completed.update(shopping=-1000.0, rest=0.0)
        d.choose("p", 0.0, (0.0, 0.0))
        self.assertIsNone(d.choose("q", 0.0, (-10.0, 0.0)))
        self.assertEqual(d.holders["shop-slot"], "p")

    def test_failed_continuation_is_explicit(self):
        d = world(router=lambda *args: None)
        d.choose("p", 0.0, (50.0, 0.0))
        with self.assertRaisesRegex(RuntimeError, "no feasible continuation"):
            d.choose("p", 121.0, (50.0, 0.0))

    def test_grounded_adapter_appends_only_after_actual_completion(self):
        d = world()
        agenda = ReactiveAgenda("p", d.places["h"], d, 0.0, 10.0)
        occ = Occupancy()
        agenda.tick(0.0, (0.0, 0.0), 0.0, occ)
        self.assertEqual(len(agenda.visits), 1)
        self.assertEqual(agenda.tick(10.0, (0.0, 0.0), 0.0, occ), "next_destination")
        self.assertEqual(len(agenda.visits), 2)
        self.assertEqual(agenda.tick(11.0, (0.0, 0.0), 0.0, occ), "travel")
        agenda.tick(110.0, (100.0, 0.0), 0.0, occ)
        self.assertEqual(agenda.phase, "servicing")
        self.assertEqual(len(agenda.visits), 2)
        self.assertEqual(agenda.tick(120.0, (100.0, 0.0), 0.0, occ), "next_destination")
        self.assertEqual(agenda.destination.purpose, "return_home")
        self.assertFalse(any(e["kind"] == "itinerary_completed" for e in agenda.events))
        json.dumps(agenda.summary())

    def test_decision_wait_is_not_counted_as_successful_dwell(self):
        phases = _person_timeline(
            {"events": [dict(kind="visit_started", t_s=0), dict(kind="activity_decision_wait", t_s=1)]}, 30
        )
        self.assertEqual(phases[15], "unscheduled")

    def test_graph_maps_replay_and_recording_horizon_do_not_change_behavior(self):
        from pathlib import Path

        from awr.life.civil.activity.headless import simulate

        for name in ("network-a", "network-b"):
            config = json.loads(
                (Path(__file__).resolve().parent / "data/scenarios/activity-engine-v1" / f"{name}.json").read_text()
            )
            for seed in (7, 11, 17):
                short, events = simulate(config, 1500, seed)
                replay, again = simulate(config, 1500, seed)
                _, longer = simulate(config, 1800, seed)
                self.assertEqual(events, again)
                self.assertEqual(events, [e for e in longer if e["t_s"] <= 1500])
                self.assertEqual(short["event_sha256"], replay["event_sha256"])
                self.assertGreater(short["car_activities"], 0)
                self.assertGreater(short["walk_activities"], 0)
                self.assertFalse(short["admitted_for_benchmark"])

    def test_no_due_need_is_home_activity_but_failed_route_is_not(self):
        d = world()
        d.actors["p"].completed["shopping"] = 0.0
        intent = d.choose("p", 0.0, (0.0, 0.0))
        self.assertEqual(intent.purpose, "pause_at_home_street")
        self.assertEqual([s.kind for s in intent.steps], ["activity"])
        blocked = world(router=lambda *args: None)
        self.assertIsNone(blocked.choose("p", 0.0, (0.0, 0.0)))
        with self.assertRaises(RuntimeError):
            blocked.choose("p", 121.0, (0.0, 0.0))

    def test_grounded_courtesy_keeps_both_service_clocks_in_sync(self):
        d = world()
        ag = ReactiveAgenda("p", d.places["h"], d, 0.0, 10.0)
        occ = Occupancy()
        ag.tick(0.0, (0.0, 0.0), 0.0, occ)
        ag.tick(10.0, (0.0, 0.0), 0.0, occ)
        ag.tick(11.0, (0.0, 0.0), 0.0, occ)
        ag.tick(110.0, (100.0, 0.0), 0.0, occ)
        ag.pause(115.0, "yield")
        ag.tick(125.0, (101.0, 0.0), 0.0, occ)
        ag.resume(135.0)
        self.assertEqual(ag.tick(139.0, (100.0, 0.0), 0.0, occ), "wait")
        self.assertEqual(ag.tick(140.0, (100.0, 0.0), 0.0, occ), "next_destination")

    def test_delayed_arrival_does_not_complete_closed_activity(self):
        d = world()
        d.places["shop"] = Place("shop", (100.0, 0.0), "shop", ("shop-slot",), closes_s=115.0)
        d.choose("p", 0.0, (0.0, 0.0))
        d.acknowledge("p", 114.0, (100.0, 0.0), 0.0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            d.acknowledge("p", 116.0, (100.0, 0.0), 0.0)
        self.assertFalse(any(e["kind"] == "activity_completed" for e in d.events))


if __name__ == "__main__":
    unittest.main()


def test_transfer_parking_precision_is_independent_of_person_arrival():
    d = world(car=True)
    intent = d.choose("p", 0.0, (0.0, 0.0))
    assert d.acknowledge("p", 0.0, intent.step.xy, 0.0, radius=0.05)
    assert intent.step.kind == "board"
    # Actual parking is 7 cm from its berth, inside the driving 10 cm contract.
    # Person precision stays 5 cm; it must not inherit the vehicle tolerance.
    door = intent.step.xy
    car = (intent.step.partner_xy[0] + 0.07, intent.step.partner_xy[1])
    assert not d.acknowledge("p", 1.0, door, 0.0, vehicle_xy=car, radius=0.05)
    assert not d.acknowledge("p", 2.0, (door[0] + 0.06, door[1]), 0.0, vehicle_xy=car, radius=0.05, vehicle_radius=0.1)
    assert not d.acknowledge("p", 3.0, door, 0.0, vehicle_xy=car, radius=0.05, vehicle_radius=0.1)
    assert d.acknowledge("p", 7.0, door, 0.0, vehicle_xy=car, radius=0.05, vehicle_radius=0.1)
    assert intent.step.kind == "drive"
    goal = intent.step.xy
    assert d.acknowledge("p", 8.0, goal, 0.0, vehicle_xy=goal, radius=0.1)
    assert intent.step.kind == "alight"
    car = (goal[0] + 0.07, goal[1])
    door = intent.step.xy
    assert not d.acknowledge("p", 9.0, door, 0.0, vehicle_xy=car, radius=0.05, vehicle_radius=0.1)
    assert d.acknowledge("p", 13.0, door, 0.0, vehicle_xy=car, radius=0.05, vehicle_radius=0.1)
    assert intent.step.kind == "walk"
