"""Run against a compiled bridge and the pinned real-scene navigation fixture."""

import csv
import math
import os
import tempfile
import unittest
from pathlib import Path

from awr.life.civil.backends.native_crowd import NativeCrowd

LIB = os.environ.get("ROUTEB_CROWD_LIBRARY")
MESH = os.environ.get("ROUTEB_NAV_MESH")
TRACE = os.environ.get("ROUTEB_TRACE_FIXTURE")


@unittest.skipUnless(LIB and MESH and TRACE, "Native bridge and real-scene fixture explicitly required")
class NativeCrowdIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(TRACE) as f:
            rows = list(csv.DictReader(f))
        cls.rows = [r for r in rows if int(r["tick"]) == 0]

    def data(self, i=0):
        r = self.rows[i]
        return tuple(float(r[k]) for k in ("east", "north", "up")), tuple(
            float(r[k]) for k in ("target_east", "target_north", "target_up")
        )

    def test_explicit_goals_rejection_and_close(self):
        start, target = self.data()
        with NativeCrowd(LIB, MESH, capacity=4) as w:
            self.assertIsNone(w.project((0, 0, 0)))
            self.assertIsNone(w.path(start, (0, 0, 0)))
            self.assertIsNotNone(w.path(start, target))
            w.add("person", start)
            self.assertFalse(w.request("person", (0, 0, 0)))
            self.assertIsNone(w.state("person").goal)
            self.assertTrue(w.request("person", target))
            old = w.state("person").goal
            with self.assertRaisesRegex(RuntimeError, "Cancel previous goal"):
                w.request("person", start)
            self.assertEqual(w.state("person").goal, old)
            for _ in range(10):
                w.state("person")
            self.assertEqual(w.time, 0.0)
            w.step(0.1)
            self.assertEqual(w.time, 0.1)
            w.stop("person")
            self.assertIsNone(w.state("person").goal)
            self.assertTrue(w.request("person", start))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            w.state("person")

    def test_overlap_identity_and_partial_input_rejected(self):
        start, _ = self.data()
        with NativeCrowd(LIB, MESH, capacity=4) as w:
            w.add("person", start)
            with self.assertRaises(ValueError):
                w.add("person", start)
            with self.assertRaisesRegex(RuntimeError, "overlaps"):
                w.add("other", start)
            w.remove("person")
            self.assertEqual(w.add("other", start).actor, "other")
            with self.assertRaises(ValueError):
                w.step(1.0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "broken.bin"
            p.write_bytes(Path(MESH).read_bytes()[:5])
            with self.assertRaisesRegex(RuntimeError, "header"):
                NativeCrowd(LIB, p)

    def test_native_replay_preserves_goal_and_complete_identities(self):
        traces = []
        for repeat in range(2):
            with NativeCrowd(LIB, MESH, capacity=4) as w:
                for i in range(4):
                    start, goal = self.data(i)
                    w.add(str(i), start)
                    self.assertTrue(w.request(str(i), goal))
                goals = {i: w.state(i).goal for i in w.actors}
                rows = []
                for tick in range(200):
                    w.step(0.1)
                    states = [w.state(str(i)) for i in range(4)]
                    for state in states:
                        self.assertEqual(state.goal, goals[state.actor])
                    rows.append(states)
                traces.append(rows)
        self.assertEqual(traces[0], traces[1])

    def test_speed_hold_preserves_native_goal_and_resumes_without_replanning(self):
        start, target = self.data()
        with NativeCrowd(LIB, MESH, capacity=4) as w:
            w.add("person", start)
            w.request("person", target)
            before = w.state("person")
            w.speed_limit("person", 0.0)
            self.assertEqual(w.state("person"), before)
            for _ in range(30):
                w.step(0.1)
            held = w.state("person")
            self.assertEqual(held.goal, before.goal)
            self.assertLess(math.dist(held.position, start), 0.16)  # Existing spawn snap is <= 0.15 m.
            self.assertEqual(held.speed, 0.0)
            steering = w.steering("person")
            self.assertIsNotNone(steering.next_corner)
            for invalid in (-1.0, float("nan"), 2.0):
                with self.assertRaises((ValueError, RuntimeError)):
                    w.speed_limit("person", invalid)
                self.assertEqual(w.state("person"), held)
            w.speed_limit("person", steering.nominal_speed)
            for _ in range(40):
                w.step(0.1)
            self.assertGreater(math.dist(w.state("person").position, held.position), 0.5)
            self.assertEqual(w.state("person").goal, before.goal)
            w.stop("person")
            w.request("person", start)
            self.assertIsNone(w.steering("person").next_corner)  # Old path is not the new intent.

    def test_body_constraints_are_deterministic_and_do_not_move_actors(self):
        from awr.life.civil.backends.native_locomotion import NativeLocomotion, angle

        start, target = self.data()
        traces = []
        for repeat in range(2):
            with NativeCrowd(LIB, MESH, capacity=4) as w:
                w.add("person", start)
                w.request("person", target)
                path = w.path(start, target)
                direction = [path[1][i] - path[0][i] for i in range(2)]
                initial = math.atan2(direction[1], direction[0]) + math.pi
                control = NativeLocomotion(w, initial_heading=initial)
                rows = []
                previous = initial
                goal = w.state("person").goal
                for _ in range(100):
                    before = w.state("person")
                    clock = w.time
                    control.before_step(0.1)
                    self.assertEqual(w.state("person"), before)
                    self.assertEqual(w.time, clock)
                    with self.assertRaises(ValueError):
                        control.before_step(0.1)
                    w.step(0.1)
                    motion = control.sample("person")
                    self.assertLessEqual(abs(angle(motion["yaw"] - previous)), 0.150001)
                    previous = motion["yaw"]
                    self.assertEqual(w.state("person").goal, goal)
                    rows.append((w.state("person"), motion))
                self.assertGreater(math.dist(rows[-1][0].position, start), 0.5)
                traces.append(rows)
        self.assertEqual(traces[0], traces[1])


if __name__ == "__main__":
    unittest.main()
