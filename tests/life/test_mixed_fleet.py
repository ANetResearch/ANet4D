"""Mixed demand must not silently turn impossible journeys into parked cars."""

import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import Polygon

from awr.life.civil.activity.traffic import parking_activity
from awr.life.civil.curb_allocation import Allocator, footprint
from awr.life.civil.fleet import composition, requests
from awr.life.civil.parking_lots import allocate, candidates
from awr.life.civil.parking_rules import ParkingRules


class FlatLot:
    def __init__(self, hole=False):
        self.hole = hole

    def supported(self, points, profile):
        points = np.asarray(points)
        ok = ((points >= [0, 0]) & (points <= [40, 30])).all(axis=1)
        return ok & (points[:, 1] < 12 if self.hole else True)


class MixedFleet(unittest.TestCase):
    def test_explicit_population_and_legacy_roster(self):
        population = dict(curb_trip=6, curb_parked=34, lot_parked=80)
        roster = requests(120, ["sedan", "van"], dict(population=population))
        self.assertEqual(composition(roster), population)
        self.assertEqual(len({r.actor_id for r in roster}), 120)
        self.assertEqual(roster[-1].initial_purpose, "parking_lot_long_stay")
        old = requests(6, ["sedan", "van"], {})
        self.assertEqual(roster[:6], old)
        for a, b in zip(old, roster, strict=False):
            self.assertEqual(parking_activity(a.actor_id, 17, -120), parking_activity(b.actor_id, 17, -120))
        self.assertEqual(
            composition(requests(10, ["sedan"], dict(trip_share=0.3))), dict(curb_trip=3, curb_parked=7, lot_parked=0)
        )

    def test_ambiguous_or_invalid_population_rejected(self):
        good = dict(curb_trip=2, curb_parked=3, lot_parked=5)
        bad = [
            dict(population=good, trip_share=0.2),
            dict(population={**good, "lot_parked": 6}),
            dict(population={**good, "curb_trip": True}),
            dict(population={**good, "lot_parked": -1}),
            dict(population={"curb_trip": 10}),
            dict(trip_share=float("nan")),
        ]
        for config in bad:
            with self.subTest(config=config), self.assertRaises(ValueError):
                requests(10, ["sedan"], config)


class ParkingLotAdmission(unittest.TestCase):
    def setUp(self):
        self.size = np.array([4.8, 2.18, 1.55])
        self.surface = FlatLot()
        self.lot = dict(
            id="surveyed-lot",
            verified=True,
            evidence="fixture-survey",
            polygon=[[0, 0], [40, 0], [40, 30], [0, 30]],
            aisles=[[[0, 13], [40, 13], [40, 21], [0, 21]]],
            entrances=[[[0, 13], [4, 13], [4, 21], [0, 21]]],
            rows=[dict(start=[6, 8], end=[34, 8], heading_rad=math.pi / 2, spacing_m=4.0, access_m=9.0)],
        )

        class Network:
            junctions = []
            walking = Polygon()
            driving = Polygon()

        self.rules = ParkingRules({}, Network())
        contacts = patch(
            "awr.life.civil.contacts.vehicle_contacts",
            return_value=np.array([[-1.5, -0.8], [-1.5, 0.8], [1.5, -0.8], [1.5, 0.8]]),
        )
        contacts.start()
        self.addCleanup(contacts.stop)

    def options(self, lot=None, surface=None):
        return candidates(
            surface or self.surface, dict(verified_parking_regions=[lot or self.lot]), self.size, self.rules
        )

    def test_allocate_supported_bays_and_keep_access_clear(self):
        options = self.options()
        self.assertEqual(len(options), 8)
        allocator = Allocator(self.surface, lambda *a: True)
        chosen = [allocate(allocator, options, self.size) for _ in range(8)]
        self.assertTrue(all(chosen))
        self.assertIsNone(allocate(allocator, options, self.size))
        self.assertEqual(len({p["slot"]["bay"] for p in chosen}), 8)
        for i, a in enumerate(chosen):
            self.assertFalse(a["slot"]["access_evidence"]["road_connection_verified"])
            for j, b in enumerate(chosen):
                if i != j:
                    self.assertFalse(footprint(a["slot"], self.size).intersects(allocator.paths[j]))

    def test_no_inference_from_parks_or_legacy_unverified_lots(self):
        self.assertEqual(
            candidates(self.surface, dict(open_areas=[self.lot], parking_lots=[self.lot]), self.size, self.rules), []
        )
        for change in (dict(verified=False), dict(evidence=None), dict(entrances=[])):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.options({**self.lot, **change})

    def test_unsupported_access_and_pedestrian_cutoffs_are_rejected(self):
        self.assertEqual(self.options(surface=FlatLot(hole=True)), [])
        self.assertIsNone(allocate(Allocator(self.surface, lambda *a: False), self.options(), self.size))

    def test_aisle_cannot_be_used_as_parking_and_rules_still_apply(self):
        bad = copy.deepcopy(self.lot)
        bad["rows"][0].update(start=[6, 17], end=[34, 17])
        self.assertEqual(self.options(bad), [])
        self.rules.walking = Polygon(self.lot["polygon"])
        self.assertEqual(self.options(), [])

    def test_reverse_access_does_not_rotate_the_vehicle(self):
        lot = copy.deepcopy(self.lot)
        lot["rows"][0].update(start=[6, 26], end=[34, 26], access_m=-9.0)
        options = self.options(lot)
        self.assertEqual(len(options), 8)
        for option in options:
            self.assertTrue(np.all(option["access_route"]["yaw"] == math.pi / 2))


if __name__ == "__main__":
    unittest.main()
