import life_toytown as toy
import numpy as np
from shapely.geometry import box

from awr.life.civil.activity.parking_admission import JointParkingAccess
from awr.life.civil.curb_allocation import footprint
from awr.life.civil.driving_graph import DrivingGraph, route_arrays
from awr.life.civil.vehicle_connections import PreparedPermission


def setup(bodies=(), access=lambda *a: True):
    g = DrivingGraph()
    for n, x in [("a", 2.0), ("c", 7.0), ("b", 12.0)]:
        g.add_node(n, [x, 10.0, 0.0], kind="berth")
    for a, b in [("a", "b"), ("c", "b")]:
        p, q = g.nodes[a]["pose"][:2], g.nodes[b]["pose"][:2]
        g.add_edge(a + b, a, b, route_arrays([p, q], [0.0, 0.0]), kind="travel")
        g.add_edge(b + a, b, a, route_arrays([q, p], [0.0, 0.0], [-1]), kind="travel")
    return JointParkingAccess(
        g, toy.Surface(), [1.0, 0.5, 1.0], PreparedPermission(box(0, 0, 20, 20)), bodies, {"a", "c"}, {"b"}, access
    )


def test_new_car_cannot_strand_an_already_admitted_owner():
    a = setup()
    assert a.accept("first", "a")
    # The car at c would occupy the only aisle from a to the shared destination.
    assert not a.accept("second", "c")
    assert set(a.selected) == {"first"}
    assert a.report["would_strand_owner"] == 1


def test_failed_candidate_does_not_change_population_and_walking_access_is_required():
    a = setup(access=lambda actor, *args: actor == "can_walk")
    assert not a.accept("cannot_walk", "a") and not a.selected
    assert a.accept("can_walk", "a")
    obstructing = footprint(dict(xy=np.array([7.0, 10.0]), yaw=0.0), [1.0, 0.5, 1.0])
    b = setup(bodies=[obstructing])
    assert not b.accept("blocked", "a")
    assert b.report["background_blocked_edges"] > 0
