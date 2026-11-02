import copy

import pytest

from awr.life.civil.parking_survey import SurveyedLotGeometry


def semantics():
    return dict(
        verified_parking_regions=[
            dict(verified=True, evidence="survey fixture", polygon=[[0, 0], [30, 0], [30, 30], [0, 30]])
        ],
        roads=[dict(line=[[20, 0], [20, 30]], width_m=4.0)],
        paths=[dict(line=[[5, 0], [5, 30]], width_m=1.0)],
        buildings=[dict(polygon=[[10, 20], [15, 20], [15, 25], [10, 25]])],
        crossings=[dict(ends=[[25, 5], [29, 5]], width_m=1.0)],
    )


@pytest.mark.parametrize(
    "xy,reason",
    [
        ([-1, 10], "outside_verified_parking_region"),
        ([20, 15], "mapped_carriageway"),
        ([5, 15], "mapped_pedestrian_path"),
        ([12, 22], "mapped_building"),
        ([27, 5], "within_crossing_margin"),
    ],
)
def test_known_conflicts_remain_rejected(xy, reason):
    result = SurveyedLotGeometry(semantics()).evaluate(xy, 0.0, [4.0, 2.0, 1.5])
    assert not result["legal"]
    assert reason in result["reasons"]


def test_clear_candidate_does_not_certify_missing_restrictions():
    rules = SurveyedLotGeometry(semantics())
    result = rules.evaluate([12, 12], 0.0, [4.0, 2.0, 1.5])
    assert result["legal"]
    assert result["parking_legality_verified"] is False
    assert rules.describe()["release_evidence"] is False


def test_containment_checks_full_body_not_centre():
    result = SurveyedLotGeometry(semantics()).evaluate([0.5, 15], 0.0, [4.0, 2.0, 1.5])
    assert "outside_verified_parking_region" in result["reasons"]


@pytest.mark.parametrize("change", ["no_evidence", "unverified", "no_lots"])
def test_missing_survey_is_not_inferred(change):
    data = copy.deepcopy(semantics())
    if change == "no_evidence":
        data["verified_parking_regions"][0]["evidence"] = ""
    elif change == "unverified":
        data["verified_parking_regions"][0]["verified"] = False
    else:
        data["verified_parking_regions"] = []
    with pytest.raises(ValueError):
        SurveyedLotGeometry(data)
