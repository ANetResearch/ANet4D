import numpy as np
import pytest

from awr.life.civil.activity.director import Place
from awr.life.civil.activity.parking_accessibility import home_clearance_conflicts


def test_destination_body_cannot_cover_another_residents_home_even_if_driver_can_exit():
    nodes = {"destination": dict(kind="berth", pose=[0, 0, 0])}
    homes = [Place("driver-home", (20, 0), "home", ("d",)), Place("neighbour-home", (1.8, 0), "home", ("n",))]
    conflicts = home_clearance_conflicts(nodes, [4, 2, 2], homes, 0.4)
    assert len(conflicts) == 1 and conflicts[0]["home"] == "neighbour-home"
    assert conflicts[0]["body_distance_m"] == 0


def test_turn_clearance_uses_full_rotated_body_and_keeps_all_source_places():
    nodes = {"bay": dict(kind="berth", pose=[0, 0, np.pi / 2])}
    homes = [Place("near", (0, 2.3), "home", ("n",)), Place("clear", (0, 2.5), "home", ("c",))]
    assert [r["home"] for r in home_clearance_conflicts(nodes, [4, 2, 2], homes, 0.4)] == ["near"]
    assert len(homes) == 2 and nodes["bay"]["pose"] == [0, 0, np.pi / 2]


def test_observed_town_parking_regression_keeps_person_163_home_clear():
    nodes = {"arrival": dict(kind="berth", pose=[-34.46659113870153, 107.80121665032333, 0.33351080976333414])}
    home = Place("person-163-home", (-35.784000000000106, 105.80799999999999), "home", ("home",))
    conflicts = home_clearance_conflicts(
        nodes, [4.305520031185865, 2.123932246479429, 1.6556694681797515], [home], 0.4107395536595902
    )
    assert len(conflicts) == 1 and 0 < conflicts[0]["body_distance_m"] < 0.4107395536595902


@pytest.mark.parametrize("radius", [0, -1, float("nan")])
def test_invalid_clearance_cannot_admit_parking(radius):
    with pytest.raises(ValueError):
        home_clearance_conflicts({}, [4, 2, 2], [], radius)
