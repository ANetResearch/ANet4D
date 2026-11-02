from copy import deepcopy

from awr.life.motion.timing import SlowTicks


def test_slow_samples_are_bounded_and_use_stage_deltas_not_cumulative_time():
    samples = SlowTicks(limit=2)
    events = [dict(actor="old"), dict(actor="p2"), dict(actor="p1")]
    before = {"decisions": 100.0, "motion": 200.0}
    after = {"decisions": 102.0, "motion": 200.2, "record": 0.1}
    original = deepcopy((before, after, events))
    samples.add(-1, -0.1, 2.5, before, after, events, 1)
    samples.add(0, 0.0, 0.2, {}, dict(motion=0.1), [], 0)
    samples.add(1, 0.1, 3.0, {}, dict(motion=2.0), [], 0)
    samples.add(2, 0.2, 0.1, {}, dict(motion=0.05), [], 0)
    report = samples.report()
    assert [r["tick"] for r in report] == [1, -1]
    warm = report[1]
    assert warm["phase"] == "warmup" and warm["actor_sample"] == ["p1", "p2"]
    assert warm["director_events"] == 2 and warm["stages_s"]["decisions"] == 2.0
    assert abs(warm["other_s"] - 0.2) < 1e-10
    assert (before, after, events) == original


def test_actor_diagnostics_do_not_grow_with_population():
    samples = SlowTicks()
    samples.add(0, 0.0, 1.0, {}, {}, [dict(actor=str(i)) for i in range(300)], 0)
    row = samples.report()[0]
    assert row["director_events"] == 300 and len(row["actor_sample"]) == 16
