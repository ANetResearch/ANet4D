from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box

from awr.life.motion.curb_destinations import propose


class Support:
    def __init__(self, allow=True):
        self.allow = allow

    def supported(self, points, kind):
        return np.full(len(points), self.allow, dtype=bool)


class Rules:
    def __init__(self, legal=True):
        self.legal = legal

    def evaluate(self, xy, yaw, size):
        return dict(legal=self.legal, reasons=[] if self.legal else ["junction"])


def inputs():
    roads = [dict(id="road", line=[[0.0, 0.0], [100.0, 0.0]], width_m=6.4, oneway=False)]
    profile = SimpleNamespace(body_lo=np.array([-1.0, -1.0, 0.0]), body_hi=np.array([3.0, 1.0, 1.5]))
    return roads, profile, box(0.0, -3.2, 100.0, 3.2), Support(), Rules(), []


def test_proposals_cover_both_source_directions_and_are_repeatable():
    args = inputs()
    a = propose(*args)
    b = propose(*args)
    assert a == b
    assert {r["mapped_node"]["source_direction"] for r in a["rows"]} == {"forward", "reverse"}
    assert all(r["mapped_node"]["region"] == "curb:road" for r in a["rows"])
    assert a["candidate_only"] and not a["release_evidence"]
    # The physical box is asymmetric about its model pivot; curb offset is
    # measured at its centre, not obtained by moving the original source road.
    n = next(r["mapped_node"] for r in a["rows"] if r["mapped_node"]["source_direction"] == "forward")
    assert n["pose"][0] + 1.0 == pytest.approx(n["source_station_m"])
    assert n["pose"][1] == pytest.approx(-2.0)


def test_source_oneway_never_acquires_opposite_direction_parking():
    args = inputs()
    args[0][0]["oneway"] = True
    assert {r["mapped_node"]["source_direction"] for r in propose(*args)["rows"]} == {"forward"}


@pytest.mark.parametrize("gate", ["occupied", "support", "rules", "permission"])
def test_proposals_reject_original_obstacles_and_failed_gates(gate):
    args = list(inputs())
    if gate == "occupied":
        args[-1] = [box(-1.0, -4.0, 101.0, 4.0)]
    elif gate == "support":
        args[3] = Support(False)
    elif gate == "rules":
        args[4] = Rules(False)
    else:
        args[2] = box(0.0, -0.1, 100.0, 0.1)
    assert not propose(*args)["rows"]


def test_nomination_spreads_budget_across_named_roads():
    args = list(inputs())
    args[0].append(dict(id="second", line=[[0.0, 20.0], [100.0, 20.0]], width_m=6.4, oneway=True))
    args[2] = box(0.0, -4.0, 100.0, 24.0)
    r = propose(*args, max_berths=2)
    assert {x["source_road"] for x in r["rows"]} == {"road", "second"}
