import unittest

from awr.life.civil.life_audit import evaluate_gates, link_groups


def report(**over):
    r = dict(
        person=dict(unexplained_static_share=0.0, state_share={}, longest_stall_s=0.0, finished_itineraries=0),
        vehicle=dict(parked_purpose_share={}, trips_ineffective=[], trip_moving_s=None),
        per_minute=[dict(person_moving_min=30, vehicle_moving_max=2)],
        crowding=dict(
            person_largest_group_1_5m=dict(max=9), group_episodes=dict(blockages=0), vehicle_stops_in_trip=[]
        ),
        zones=[],
        courier=dict(board_after_s=None),
    )
    for k, v in over.items():
        r[k].update(v)
    return r


class LinkGroups(unittest.TestCase):
    def test_consecutive_samples_with_shared_members_form_one_episode(self):
        a, b = frozenset(range(7)), frozenset(range(2, 9))
        eps = link_groups([(0, a), (10, b), (20, b)], 10)
        self.assertEqual(len(eps), 1)
        self.assertEqual([k for k, _ in eps[0]["frames"]], [0, 10, 20])

    def test_a_gap_or_different_people_start_a_new_episode(self):
        a, c = frozenset(range(7)), frozenset(range(20, 27))
        self.assertEqual(len(link_groups([(0, a), (20, a)], 10)), 2)
        self.assertEqual(len(link_groups([(0, a), (10, c)], 10)), 2)

    def test_an_episode_continues_into_one_group_only(self):
        a = frozenset(range(8))
        eps = link_groups([(0, a), (10, frozenset(range(4))), (10, frozenset(range(4, 8)))], 10)
        self.assertEqual(sorted(len(e["frames"]) for e in eps), [1, 2])


class Gates(unittest.TestCase):
    def test_a_large_group_alone_is_not_a_failure_but_a_blockage_is(self):
        gates = {"person_blockages_max": dict(threshold=0, min_size=7)}
        self.assertTrue(evaluate_gates(report(), gates)["all_pass"])
        self.assertFalse(evaluate_gates(report(crowding=dict(group_episodes=dict(blockages=1))), gates)["all_pass"])

    def test_ineffective_trips_and_late_boarding_fail(self):
        gates = {
            "vehicle_ineffective_trips_max": dict(threshold=0),
            "courier_board_after_stop_s_max": dict(threshold=10),
        }
        self.assertTrue(evaluate_gates(report(courier=dict(board_after_s=dict(max=0.1))), gates)["all_pass"])
        bad = evaluate_gates(
            report(vehicle=dict(trips_ineffective=[dict(vehicle="v")]), courier=dict(board_after_s=dict(max=42.0))),
            gates,
        )
        self.assertFalse(bad["gates"]["vehicle_ineffective_trips_max"]["pass_"])
        self.assertFalse(bad["gates"]["courier_board_after_stop_s_max"]["pass_"])

    def test_unknown_gate_is_reported_not_silently_passed(self):
        g = evaluate_gates(report(), {"no_such_gate": dict(threshold=1)})
        self.assertIsNone(g["gates"]["no_such_gate"]["pass_"])


if __name__ == "__main__":
    unittest.main()
