from copy import deepcopy

import pytest

from awr.life.civil.life_audit import markdown


@pytest.mark.parametrize("family", ["person", "vehicle"])
def test_empty_family_keeps_absent_statistics_without_changing_audit(family):
    person = dict(
        moving_concurrency=dict(min=1, median=2, max=3),
        moving_time_s={"median": 100},
        moving_distance_m={"median": 120},
        state_share={"moving": 1},
        unexplained_static_share=0,
    )
    vehicle = dict(
        moving_concurrency=dict(min=1, median=2, max=3),
        moving_time_s={"median": 50},
        moving_distance_m={"median": 140},
        parked_purpose_share={},
    )
    r = dict(
        event="fixture",
        duration_s=1500,
        counts={"person": 3, "vehicle": 3},
        person=person,
        vehicle=vehicle,
        per_minute=[],
    )
    r["counts"][family] = 0
    r[family].update(moving_concurrency=dict(min=0, median=0, max=0), moving_time_s=None, moving_distance_m=None)
    before = deepcopy(r)
    text = markdown(r)
    lines = [x for x in text.splitlines() if "单体累计" in x]
    assert len(lines) == 2 and all("—" in x for x in lines)
    assert "None" not in text and r == before
