"""Regressions for centre-line parking, unsupported merges and convoy gating."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from awr.life.civil.curb import (
    berth,
    candidates,
    connected_roads,
    manoeuvre,
    road_geometry,
    road_layout,
    sampled_route,
    valid_route,
)
from awr.life.civil.curb_allocation import Allocator, departure_times, footprint, opposing_conflict
from awr.life.civil.grounded import at
from awr.life.civil.prediction import route_at_many


class FlatRoad:
    def __init__(self, width=6.4, hole=None):
        self.width = width
        self.hole = hole

    def supported(self, points, profile):
        p = np.asarray(points)
        ok = np.abs(p[:, 1]) <= self.width / 2
        if self.hole is not None:
            ok &= ~((p[:, 0] > self.hole[0]) & (p[:, 0] < self.hole[1]))
        return ok


class CurbBehavior(unittest.TestCase):
    def setUp(self):
        self.size = np.array([4.8, 2.18, 1.55])
        self.road = dict(id="street", line=[[0, 0], [240, 0]], width_m=6.4)
        self.geometry = road_geometry(self.road)
        self.surface = FlatRoad()
        # Geometry-only fixture: the real-scene preflight uses authored tyres.
        self.contacts = patch(
            "awr.life.civil.contacts.vehicle_contacts",
            return_value=np.array([[-1.5, -0.8], [-1.5, 0.8], [1.5, -0.8], [1.5, 0.8]]),
        )
        self.contacts.start()
        self.addCleanup(self.contacts.stop)

    def test_width_does_not_magically_allow_two_lanes_beside_parking(self):
        self.assertIsNone(road_layout(4.0, self.size))
        self.assertEqual(road_layout(6.4, self.size)["lanes"], 1)
        self.assertEqual(road_layout(10.0, self.size)["lanes"], 2)

    def test_pull_out_and_pull_in_are_continuous_and_off_the_travel_line(self):
        layout = road_layout(6.4, self.size)
        r = manoeuvre(self.geometry, 10.0, 210.0, layout, self.size, self.surface, "street")
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r["points"][0, 1], -layout["parking_offset_m"])
        self.assertAlmostEqual(r["points"][-1, 1], r["points"][0, 1])
        self.assertGreater(abs(at(r, r["arc"][-1] / 2)[0][1] - r["points"][0, 1]), self.size[1])
        self.assertLess(np.max(np.linalg.norm(np.diff(r["points"], axis=0), axis=1)), 0.351)
        self.assertAlmostEqual(r["yaw"][0], 0.0, places=12)
        self.assertAlmostEqual(r["yaw"][-1], 0.0, places=12)
        p, a = route_at_many(r, np.linspace(0, r["arc"][-1], 400))
        for d, pp, aa in zip(np.linspace(0, r["arc"][-1], 400), p, a, strict=False):
            q, b = at(r, d)
            np.testing.assert_allclose(q, pp)
            self.assertAlmostEqual(b, aa)

    def test_admitted_berth_pose_hands_off_without_rotating_at_departure(self):
        from awr.life.civil.vehicle_execution import advance_job

        theta = 2.694597029417945
        turn = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        road = dict(self.road, line=(np.array(self.road["line"]) @ turn.T).tolist())
        geometry = road_geometry(road)
        layout = road_layout(6.4, self.size)
        surface = FlatRoad(1000.0)
        start = berth(geometry, 44.132429915480316, layout, self.size, surface)
        end = berth(geometry, 236.13242991548032, layout, self.size, surface)
        r = manoeuvre(geometry, start["s"], end["s"], layout, self.size, surface, "street")
        self.assertIsNotNone(r)
        for index, slot in ((0, start), (-1, end)):
            np.testing.assert_allclose(r["points"][index], slot["xy"], atol=1e-12)
            self.assertAlmostEqual(r["yaw"][index], slot["yaw"], places=12)
        j = dict(route=r, s=0.0, goal_s=r["arc"][-1], speed=0.0, direction=1, max_speed=3.0)
        state = advance_job(j, np.array(start["xy"]), start["yaw"], 0.1)
        self.assertGreater(state["arc_m"], 0.0)

    def test_exact_endpoint_heading_still_requires_steering_validation(self):
        r = sampled_route([[0, 0], [100, 0]], "bad-pose", endpoint_headings=[math.pi / 2, 0.0])
        self.assertFalse(valid_route(FlatRoad(1000), r, self.size))

    def test_resampled_route_and_motion_controller_share_the_same_arc(self):
        from awr.life.civil.driving_motion import RouteExecution

        r = sampled_route([[0, 0], [0.62, 0], [1.5, 0.2], [2.5, 0.2]], "curved")
        measured = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(r["points"], axis=0), axis=1))]
        np.testing.assert_allclose(r["arc"], measured, atol=1e-12, rtol=0.0)
        execution = RouteExecution(r)
        for _ in range(150):
            state = execution.advance(0.1)
            point, yaw = at(r, state["arc_m"])
            np.testing.assert_allclose(state["xy"], point, atol=1e-12, rtol=0.0)
            self.assertAlmostEqual(state["yaw"], yaw, places=12)

    def test_unknown_ground_and_sharp_turns_do_not_become_routes(self):
        layout = road_layout(6.4, self.size)
        self.assertIsNone(manoeuvre(self.geometry, 10, 210, layout, self.size, FlatRoad(hole=(80, 85)), "street"))
        r = sampled_route([[0, 0], [20, 0], [20, 20]], "corner")
        self.assertFalse(valid_route(FlatRoad(100), r, self.size))

    def test_delivery_stops_are_on_the_curb_and_can_resume_forward(self):
        layout = road_layout(6.4, self.size)
        r = manoeuvre(self.geometry, 10, 210, layout, self.size, self.surface, "street", stops=[110])
        self.assertIsNotNone(r)
        p, a = at(r, r["curb_stops_m"][0])
        np.testing.assert_allclose(p, [110, -layout["parking_offset_m"]], atol=0.002)
        self.assertLess(abs(a), 0.002)
        self.assertGreater(at(r, r["curb_stops_m"][0] + 15)[0][0], p[0])

    def test_nearby_but_disconnected_roads_are_not_joined(self):
        sem = {"roads": [dict(self.road, line=[[0, 0], [50, 0]]), dict(self.road, id="b", line=[[50.1, 0], [100, 0]])]}
        self.assertTrue(all(road_geometry(r)[0].length <= 50 for r in connected_roads(sem)))

    def test_allocator_reserves_endpoints_and_keeps_other_routes_clear(self):
        roads = candidates(self.surface, {"roads": [self.road]}, self.size, min_trip=60.0)
        allocator = Allocator(self.surface, lambda *args: True)
        first = allocator.trip(roads, self.size)
        self.assertIsNotNone(first)
        # An already reserved start or destination cannot become another car's
        # parking slot. The driving swept area cannot cross either parked body.
        self.assertFalse(allocator.free([first["start"]], self.size))
        self.assertFalse(allocator.free([first["end"]], self.size))
        second = allocator.trip(roads, self.size)
        if second is not None:
            for slot in (second["start"], second["end"]):
                self.assertFalse(footprint(slot, self.size).intersects(allocator.paths[0]))

    def test_pedestrian_passage_failure_is_not_ignored(self):
        roads = candidates(self.surface, {"roads": [self.road]}, self.size, min_trip=60.0)
        allocator = Allocator(self.surface, lambda *args: False)
        self.assertIsNone(allocator.trip(roads, self.size))
        self.assertIsNone(allocator.stationary(roads, self.size))

    def test_two_way_traffic_requires_separated_local_corridors(self):
        east = sampled_route([[0, 0], [100, 0]], "east")
        head_on = sampled_route([[100, 0], [0, 0]], "west")
        west_lane = sampled_route([[100, 3.2], [0, 3.2]], "west")
        crossing = sampled_route([[50, -20], [50, 20]], "crossing")
        self.assertTrue(opposing_conflict(east, self.size, head_on, self.size))
        self.assertTrue(opposing_conflict(east, self.size, crossing, self.size))
        self.assertFalse(opposing_conflict(east, self.size, west_lane, self.size))

    def test_departures_cover_the_recording_without_spending_trips_in_warmup(self):
        times = departure_times([30.0] * 20, 1500.0, 600.0, np.random.default_rng(7))
        self.assertEqual(times[0], -15.0)
        self.assertGreater(times[-1], 1400.0)
        self.assertEqual(int((times < 0).sum()), 1)
        np.testing.assert_equal(times, departure_times([30.0] * 20, 1500.0, 600.0, np.random.default_rng(7)))

    def test_a_valid_parking_space_is_not_a_valid_trip(self):
        short = dict(self.road, line=[[0, 0], [30, 0]])
        roads = candidates(self.surface, {"roads": [short]}, self.size, min_trip=60.0)
        allocator = Allocator(self.surface, lambda *args: True)
        self.assertIsNone(allocator.trip(roads, self.size))
        self.assertIsNotNone(allocator.stationary(roads, self.size))


if __name__ == "__main__":
    unittest.main()
