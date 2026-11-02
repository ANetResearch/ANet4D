import numpy as np

from awr.life.civil.fleet_audit import recurring_fleet
from awr.life.civil.life_audit import evaluate_gates


def trip(vehicle="v", t=10.0, end=(100.0, 0.0), **extra):
    return dict(
        vehicle=vehicle,
        t0_s=t,
        t1_s=t + 10.0,
        status="completed",
        effective=True,
        legal=True,
        end_purpose="visit_shop",
        moving_s=8.0,
        end_xy_m=list(end),
        **extra,
    )


def audit(trips, ids=("v",), moving=None):
    return recurring_fleet(trips, ids, np.ones((180, len(ids)), bool) if moving is None else moving, tick_s=1.0)


def test_parked_population_and_one_shot_trips_cannot_inflate_recurring_count():
    ids = [f"v{i}" for i in range(120)]
    journeys = [trip(aid) for aid in ids[:30]]
    assert audit(journeys, ids)["qualified_vehicles"] == 0
    journeys += [trip(aid, t=90.0, end=(0.0, 0.0)) for aid in ids[:6]]
    assert audit(journeys, ids)["qualified_vehicles"] == 6
    journeys += [trip(aid, t=90.0, end=(0.0, 0.0)) for aid in ids[6:30]]
    assert audit(journeys, ids)["qualified_vehicles"] == 30


def test_warmup_unfinished_illegal_short_and_same_destination_do_not_qualify():
    for overrides in (
        dict(t0_s=-10.0),
        dict(t1_s=None, status="under_way"),
        dict(legal=False),
        dict(effective=False),
        dict(end_xy_m=[101.0, 0.0]),
        dict(t1_s=180.0),
        dict(end_purpose=None),
        dict(moving_s=0.0),
    ):
        second = trip(t=90.0, end=(0.0, 0.0))
        second.update(overrides)
        assert audit([trip(), second])["qualified_vehicles"] == 0


def test_observed_fleet_gap_includes_start_and_end_and_minute_counts_are_half_open():
    moving = np.zeros((180, 2), bool)
    moving[20:40, 0] = True
    moving[100:120, 1] = True
    out = audit([trip("a", t=60.0)], ids=("a", "b"), moving=moving)
    assert out["longest_no_driving_s"] == 60.0
    assert [r["unique_moving_vehicles"] for r in out["per_minute"]] == [1, 1, 0]
    assert [r["departures"] for r in out["per_minute"]] == [0, 1, 0]
    assert audit([], moving=np.zeros((180, 1), bool))["longest_no_driving_s"] == 180.0


def test_missing_audit_evidence_and_unknown_acceptance_gate_fail_closed():
    report = dict(person={}, vehicle={}, per_minute=[])
    for gates in (
        {"vehicle_recurring_vehicles_min": 30},
        {"duration_s_min": 1500},
        {"vehicle_longest_no_driving_s_max": 60},
        {"typo_in_gate": 1},
    ):
        assert evaluate_gates(report, gates)["all_pass"] is False
