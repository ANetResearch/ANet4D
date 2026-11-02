from types import SimpleNamespace

import numpy as np
import pytest

from awr.life.civil.activity.grounded import GroundedActivities


def runtime(settings, planner=None):
    planner = planner or (lambda a, b, free: dict(points=np.array([a, b])))
    network = SimpleNamespace(plan=planner, cell=lambda xy: (0, 0))
    pools = dict(
        sites=[
            dict(
                id="park",
                kind="park",
                affordances=("outdoor_stay",),
                semantic_source="fixture:park",
                slots=[dict(id="p", xy=[500.0, 0.0])],
            )
        ],
        street=[],
    )
    return GroundedActivities(network, pools, np.ones((2, 2), bool), settings, 17)


def test_scene_activity_range_is_independent_of_legacy_chain_and_keeps_actual_routing():
    calls = []

    def route(a, b, free):
        calls.append(tuple(b))
        return None if b[1] > 0 else dict(points=np.array([a, b]))

    d = runtime(dict(leg_distance_m=[40.0, 450.0], activity_engine=dict(walking_range_m=[0.0, 900.0])), route)
    assert d.route("walk", (0.0, 0.0), (500.0, 0.0), "any-person").length_m == 500.0
    assert d.route("walk", (0.0, 0.0), (10.0, 0.0), "another-person").length_m == 10.0
    assert d.route("walk", (0.0, 0.0), (500.0, 1.0), "any-person") is None
    assert d.route("walk", (0.0, 0.0), (901.0, 0.0), "any-person") is None
    assert (901.0, 0.0) not in calls
    old = runtime(dict(leg_distance_m=[40.0, 450.0]))
    assert old.route("walk", (0.0, 0.0), (500.0, 0.0), "p") is None


@pytest.mark.parametrize("value", [(900.0, 0.0), (-1.0, 900.0), (0.0, float("inf")), (0.0,)])
def test_invalid_activity_range_fails_before_population_admission(value):
    with pytest.raises(ValueError, match="walking range"):
        runtime(dict(activity_engine=dict(walking_range_m=value)))
