"""Adversarial transfers through the public shared life runtime, without pose repair."""

import math

import life_toytown as toy
import numpy as np
import pytest

from awr.life.motion.crowd import DT


def alighting_world():
    w = toy.build(drivers=1)
    e = w.engine
    for k in range(4000):
        intent = w.director.active.get("person-0")
        if intent is not None and intent.step.kind == "alight" and e.riding[0]:
            return w, k * DT
        e.step(k * DT)
    pytest.fail("Fixture did not reach a physical parked arrival")


@pytest.mark.parametrize("fault", ["remote_car", "wrong_heading", "reversing_car"])
def test_person_does_not_appear_before_actual_car_is_parked(fault):
    w, t = alighting_world()
    e = w.engine
    v = e.vehicle_index["vehicle-0"]
    if fault == "remote_car":
        e.xy[v] += [0.0, 8.0]
    elif fault == "wrong_heading":
        e.yaw[v] += math.pi / 2
    else:
        e.speed[v] = -0.5
    e.xy[0] = e.xy[v]
    before = e.stats["alightings"]
    e.step(t)
    assert e.riding[0] and not e.active[0] and e.container[0] == v
    np.testing.assert_array_equal(e.xy[0], e.xy[v])
    assert e.stats["alightings"] == before
    assert w.director.active["person-0"].service_started is None
    assert 0 in e.transfer_wait_since


def test_blocked_transfer_has_bounded_wait_instead_of_indefinite_containment():
    w, t = alighting_world()
    e = w.engine
    v = e.vehicle_index["vehicle-0"]
    e.xy[v] += [0.0, 8.0]
    e.xy[0] = e.xy[v]
    e.transfer_wait_s = 0.2
    for k in range(3):
        e.step(t + k * DT)
    with pytest.raises(RuntimeError, match="Transfer blocked for too long"):
        e.step(t + 3 * DT)
    assert e.riding[0]


def test_failed_world_cannot_resume_after_partial_decision_updates(monkeypatch):
    w = toy.build(drivers=1)

    def broken(*args, **kwargs):
        raise RuntimeError("injected motion failure")

    monkeypatch.setattr(w.engine.crowd, "step", broken)
    with pytest.raises(RuntimeError, match="injected motion failure"):
        w.engine.step(0.0)
    with pytest.raises(RuntimeError, match="LifeEngine already failed"):
        w.engine.step(0.1)


def test_clock_rejects_skips_without_advancing_the_world():
    w = toy.build(drivers=1)
    e = w.engine
    e.step(-10.0)
    xy = e.xy.copy()
    for t in (-10.0, -11.0, 0.0, float("nan")):
        with pytest.raises(ValueError, match="exactly DT"):
            e.step(t)
        np.testing.assert_array_equal(xy, e.xy)
    e.step(-9.9)


def test_driving_snapshot_contains_static_facilities_as_well_as_cars():
    from awr.life.civil.activity.driving import Body, DrivingRoutes
    from awr.life.motion.engine import LifeEngine

    w = toy.build(drivers=1)
    e = w.engine
    extra = Body((70.0, 30.0, 0.0), (5.0, 2.0, 1.0))
    routes = DrivingRoutes(w.graph, w.surface, toy.CAR, w.routes.areas, dict(w.routes.occupancy, facility=extra))
    args = dict(
        ids=e.ids + ["facility"],
        kinds=np.concatenate([e.kinds, ["facility"]]),
        ext=np.concatenate([e.ext, np.array([[-2.5, 2.5, -1.0, 1.0]] * 2)[None]]),
        xy=np.vstack([e.xy, [70.0, 30.0]]),
        yaw=np.r_[e.yaw, 0.0],
        surface=w.surface,
        walking=w.walking,
        router=w.router,
        agendas=e.agendas,
        vmax=np.r_[e.vmax, 0.0],
        occupancy=e.occupancy,
        vehicle_ids=e.vehicle_ids,
        sizes=dict(e.sizes),
        driving=routes,
    )
    args["sizes"][e.n] = extra.size
    with pytest.raises(ValueError, match="every rigid scene body"):
        LifeEngine(**dict(args, driving=w.routes))
    mixed = LifeEngine(**args)
    mixed.xy[-1] = [71.0, 31.0]
    mixed._sync_driving()
    assert routes.occupancy["facility"].pose == (71.0, 31.0, 0.0)
    assert set(routes.occupancy) == {"vehicle-0", "facility"}
