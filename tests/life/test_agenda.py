import unittest

from awr.life.civil.agenda import Agenda, Occupancy, Visit


class AgendaContracts(unittest.TestCase):
    def test_blocking_preserves_goal_and_arrival_is_physical(self):
        a = Agenda(
            "p", "shopper", [Visit("shop", (100, 0), "collect_order", 120), Visit("home", (0, 0), "return_home", 0)]
        )
        r = Occupancy()
        self.assertEqual(a.tick(0, (0, 0), 0, r), "travel")
        for t in (5, 10, 100):
            a.blocked(t, "crossing occupied")
        self.assertEqual(a.destination.place, "shop")
        self.assertEqual(a.tick(130, (99.8, 0), 1, r), "travel")
        self.assertEqual(a.tick(140, (100, 0), 0, r), "wait")
        self.assertEqual(a.tick(200, (100, 0), 0, r), "wait")
        self.assertEqual(a.tick(260, (100, 0), 0, r), "next_destination")
        self.assertEqual(a.destination.place, "home")
        self.assertEqual(a.tick(300, (0, 0), 0, r), "finished")
        self.assertEqual(a.tick(1500, (0, 0), 0, r), "finished")
        self.assertEqual(a.index, 2)

    def test_capacity_and_interruption_do_not_fake_service(self):
        r = Occupancy()
        a = Agenda("a", "courier", [Visit("bay", (0, 0), "delivery", 60)])
        b = Agenda("b", "courier", [Visit("bay", (0, 0), "delivery", 60)])
        self.assertEqual(a.tick(0, (0, 0), 0, r), "wait")
        b.tick(0, (0, 0), 0, r)
        self.assertEqual(b.phase, "queueing")
        a.tick(30, (3, 0), 0, r)
        self.assertEqual(a.phase, "travelling")
        b.tick(30, (0, 0), 0, r)
        self.assertEqual(b.service_started_s, 30)
        self.assertEqual(b.tick(60, (0, 0), 0, r), "wait")
        self.assertEqual(b.tick(90, (0, 0), 0, r), "finished")

    def test_stepping_aside_pauses_service_without_faking_it(self):
        r = Occupancy()
        a = Agenda(
            "a",
            "resident",
            [Visit("bay", (0, 0), "wait", 60), Visit("a-departure-point", (50, 0), "finish_outdoor_journey", 0)],
        )
        self.assertEqual(a.tick(0, (0, 0), 0, r), "wait")
        self.assertEqual(a.phase, "servicing")
        self.assertTrue(a.pause(30, "obstructs_b"))
        self.assertFalse(a.pause(31, "obstructs_c"))
        # Away from the place and moving: no interruption while paused.
        self.assertEqual(a.tick(40, (2, 0), 1.0, r), "paused")
        self.assertEqual(a.phase, "servicing")
        a.resume(50)
        self.assertEqual(a.tick(60, (0, 0), 0, r), "wait")
        self.assertEqual(a.tick(79, (0, 0), 0, r), "wait")
        self.assertEqual(a.tick(80, (0, 0), 0, r), "next_destination")
        kinds = [e["kind"] for e in a.events]
        self.assertEqual(
            kinds,
            ["journey_started", "arrived", "visit_started", "stepped_aside", "returned_to_place", "visit_completed"],
        )
        self.assertEqual(a.tick(200, (50, 0), 0, r), "finished")
        self.assertTrue(a.pause(210, "obstructs_d"))
        self.assertEqual(a.tick(215, (53, 0), 1, r), "paused")
        a.resume(220)
        self.assertEqual(a.tick(221, (50, 0), 0, r), "finished")
        self.assertEqual(a.index, 2)

    def test_waiting_to_depart_can_step_aside_and_departs_after_return(self):
        r = Occupancy()
        a = Agenda(
            "a",
            "resident",
            [Visit("shop", (100, 0), "wait", 60), Visit("a-departure-point", (0, 0), "finish_outdoor_journey", 0)],
            departure_s=100,
        )
        self.assertEqual(a.tick(0, (0, 0), 0, r), "wait")
        self.assertTrue(a.pause(10, "obstructs_b"))
        self.assertEqual(a.tick(120, (3, 0), 1, r), "paused")
        self.assertEqual(a.phase, "scheduled")
        a.resume(130)
        self.assertEqual(a.tick(131, (0, 0), 0, r), "travel")
        self.assertEqual(a.phase, "travelling")
