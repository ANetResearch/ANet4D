import unittest

import numpy as np
from shapely.geometry import LineString, Polygon

from awr.life.civil.parking_rules import ParkingRules


class FakeLanes:
    def __init__(self):
        # Two-way street along x: carriageway y in [-3.2, 3.2]; sidewalk y in [3.2, 5.2].
        self.driving = LineString([[-100, 0], [100, 0]]).buffer(3.2, cap_style=2)
        self.walking = LineString([[-100, 4.2], [100, 4.2]]).buffer(1.0, cap_style=2)
        self.junctions = [dict(id="j", type="priority", polygon=Polygon([[40, -6], [52, -6], [52, 6], [40, 6]]))]


class ParkingRuleTests(unittest.TestCase):
    def setUp(self):
        sem = dict(crossings=[dict(id="x", ends=[[0, -4], [0, 4]], width_m=4.0)])
        self.rules = ParkingRules(sem, FakeLanes(), crossing_margin_m=5.0, min_free_width_m=3.0)
        self.size = np.array([4.5, 1.8, 1.5])

    def test_kerbside_stop_leaving_room_is_legal(self):
        r = self.rules.evaluate([20.0, 2.2], 0.0, self.size)
        self.assertTrue(r["legal"], r)
        self.assertEqual(r["where"], "carriageway")
        self.assertGreaterEqual(r["free_width_m"], 3.0)

    def test_centre_of_narrow_street_obstructs(self):
        r = self.rules.evaluate([20.0, 0.0], 0.0, self.size)
        self.assertIn("obstructs_carriageway", r["reasons"])

    def test_crossing_margin_junction_and_sidewalk(self):
        self.assertIn("within_crossing_margin", self.rules.evaluate([5.0, 2.2], 0.0, self.size)["reasons"])
        self.assertIn("in_junction", self.rules.evaluate([45.0, 2.2], 0.0, self.size)["reasons"])
        r = self.rules.evaluate([20.0, 4.2], 0.0, self.size)
        self.assertIn("on_sidewalk", r["reasons"])
        self.assertEqual(r["where"], "roadside")

    def test_queueing_is_not_evaluated_here(self):
        # The rules object has no notion of intent; callers must only evaluate
        # intentional stops. The describe() record states that convention.
        self.assertIn("convention", self.rules.describe()["note"])


if __name__ == "__main__":
    unittest.main()
