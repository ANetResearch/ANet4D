import numpy as np
import pytest

from awr.life.civil.grounded import at, route_length
from awr.life.civil.route_handoff import finish_backstep


@pytest.mark.parametrize("expired", [False, True])
def test_off_route_handoff_cannot_snap_back_when_extension_or_replan_fails(expired):
    # Ordinary route is on y=0; the body has backed 0.6 m away. Reusing its
    # old arc coordinate gives a 6 m/s jump at 10 Hz, just as in v15's failure.
    old_route = dict(points=np.array([[0.0, 0.0], [20.0, 0.0]]), arc=np.array([0.0, 20.0]), road="original")
    observed = np.array([14.0, 0.6])
    agenda = object()
    job = dict(
        route=old_route,
        s=14.0,
        goal_s=20.0,
        speed=0.1,
        agenda=agenda,
        backstep=(np.array([0.0, 1.0]), 0.2 if expired else 0.0, 92.7),
        extending=True,
        temporary_endpoint=True,
        yielding=True,
        _partial=(927, 1, 0.2),
    )
    assert np.linalg.norm(at(job["route"], job["s"])[0] - observed) / 0.1 == pytest.approx(6.0)
    finish_backstep(job, observed, 0.4, 92.7)
    assert job["agenda"] is agenda
    # A planner returning no route leaves exactly the measured body in place;
    # even an attempted stale advancement cannot restore the abandoned path.
    for attempted_s in (job["s"], 0.01, 14.0, 20.0):
        point, yaw = at(job["route"], attempted_s)
        np.testing.assert_array_equal(point, observed)
        assert yaw == 0.4
    assert route_length(job["route"]) == 0 and job["goal_s"] == 0
    assert job["needs_route"] and not job.get("extending") and not job["temporary_endpoint"]
    assert "backstep" not in job and "_partial" not in job
    # Destination intent survives; the replacement route starts at the body.
    new_route = dict(points=np.array([observed, observed + [1.0, 0.0]]), arc=np.array([0.0, 1.0]))
    point, _ = at(new_route, 0.01)
    assert np.linalg.norm(point - observed) / 0.1 == pytest.approx(0.1)


def test_hold_geometry_cannot_hide_a_real_displacement_or_stop_a_fast_walker():
    with pytest.raises(ValueError, match="zero-length"):
        at(dict(points=np.array([[0.0, 0.0], [1.0, 0.0]]), arc=np.zeros(2)), 0.0)
    with pytest.raises(ValueError, match="backstep speed"):
        finish_backstep(dict(speed=1.4), [0.0, 0.0], 0.0, 10.0)
