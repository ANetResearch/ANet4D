import unittest

import numpy as np

from awr.life.civil.itineraries import plan_outdoor_visits
from awr.life.civil.walking import WalkNetwork


class WaitingPlaces(unittest.TestCase):
    def plan(self, free, goal):
        network = WalkNetwork(free, [0.0, 0.0], 1.0)
        places = [dict(id="shop-approach", poi="shop", kind="shop", xy=goal)]
        return plan_outdoor_visits(
            network, places, np.random.default_rng(2), "p", [], [], [], 0.7, 0.0, static_free=free
        )

    def test_waiting_in_single_file_passage_is_not_admitted(self):
        free = np.zeros((21, 151), bool)
        free[10, 1:150] = True
        route, agenda = self.plan(free, [140.0, 10.0])
        self.assertIsNone(route)
        self.assertIsNone(agenda)

    def test_long_trip_with_waiting_room_is_admitted(self):
        free = np.zeros((21, 151), bool)
        free[2:19, 2:149] = True
        route, agenda = self.plan(free, [140.0, 10.0])
        self.assertIsNotNone(agenda)
        self.assertGreaterEqual(np.linalg.norm(route["points"][-1] - route["points"][0]), 80.0)

    def test_different_place_ids_cannot_reserve_the_same_waiting_space(self):
        free = np.zeros((21, 151), bool)
        free[2:19, 2:149] = True
        network = WalkNetwork(free, [0.0, 0.0], 1.0)

        def plan(place, actor):
            return plan_outdoor_visits(
                network,
                [dict(id=place, poi=place, kind="shop", xy=[140.0, 10.0])],
                np.random.default_rng(2),
                actor,
                [],
                [],
                [],
                0.7,
                0.0,
                static_free=free,
            )

        self.assertIsNotNone(plan("first-map-feature", "a")[1])
        self.assertIsNone(plan("overlapping-map-feature", "b")[1])

    def test_departure_points_are_reserved_waiting_space(self):
        free = np.zeros((21, 151), bool)
        free[2:19, 2:149] = True
        network = WalkNetwork(free, [0.0, 0.0], 1.0)

        def plan(place, actor, goal):
            return plan_outdoor_visits(
                network,
                [dict(id=place, poi=place, kind="shop", xy=goal)],
                np.random.default_rng(2),
                actor,
                [],
                [],
                [],
                0.7,
                0.0,
                static_free=free,
            )

        route, agenda = plan("east-shop", "a", [140.0, 10.0])
        self.assertIsNotNone(agenda)
        start = np.asarray(agenda.visits[-1].xy)
        self.assertTrue(any(np.linalg.norm(q - start) < 1e-9 for q in network.assigned_waiting_points))
        route2, agenda2 = plan("west-shop", "b", [8.0, 10.0])
        if agenda2 is not None:
            self.assertGreaterEqual(np.linalg.norm(np.asarray(agenda2.visits[-1].xy) - start), 3.0)
            for v in agenda2.visits:
                self.assertGreaterEqual(np.linalg.norm(np.asarray(v.xy) - start), 3.0)
