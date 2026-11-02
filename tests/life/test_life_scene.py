"""Recorded failure scenes of the real-town builds on dmax (awr.life.motion.scene), replayed on the crowd alone.

scene-parked-car: life-v1-scale150/v2, t=0, person-029 stands 0.26 m behind a parked car facing it (its walking
pose would overlap the car). scene-support-corner: life-v1-scale300/v2, t=52.3 s, person-095 stands at a corner of
the walkable support where its resting pose fits only at its current heading. scene-support-corner-facing-away:
life-v1-scale300/v3, t=92.8 s, the same walker 0.4 m further on, at the corner, facing away from its way (it can
turn on the spot towards it neither way). Outside the 20 m window of a scene
there is no support, so walkers near its edge may be held there; only the recorded walker is judged. (Out of the
corner, person-095 follows person-065, who walks at 0.12 m/s, before it walks on: hence 30 s.)
scene-overtaking-abreast: life-v1-scale300/v6, rebuilt from the recorded tape at t=186.3 s, 12 s before
person-194 (1.16 m/s), overtaking person-117 (1.08 m/s) on the same street with no lateral clearance, and
person-117, turning towards its path, were both held (a stop of 10.8 m/s2).
"""

import math
from pathlib import Path

import numpy as np
import pytest

from awr.life.motion.crowd import Path as WalkPath
from awr.life.motion.reroute import Reroutes
from awr.life.motion.scene import Scene

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    "name, walker, within_s, metres",
    [
        ("scene-parked-car", "person-029", 15.0, 4.0),
        ("scene-support-corner", "person-095", 30.0, 2.0),
        ("scene-support-corner-facing-away", "person-095", 20.0, 2.0),
    ],
)
def test_recorded_stall_resolves_on_the_crowd_alone(name, walker, within_s, metres):
    sc = Scene(DATA / f"{name}.npz")
    i = sc.ids.index(walker)
    start = sc.xy[i].copy()
    held = 0
    for _ in range(int(round(within_s * 10))):
        held += i in sc.step()[3]
    assert held == 0
    assert math.dist(sc.xy[i], start) >= metres
    assert np.all(sc.speed[sc.mover] <= sc.vmax[sc.mover] + 1e-9)


def test_overtaking_walker_keeps_clear_of_the_one_it_passes():
    sc = Scene(DATA / "scene-overtaking-abreast.npz")
    pair = [sc.ids.index("person-117"), sc.ids.index("person-194")]
    start = sc.xy[pair].copy()
    for _ in range(250):
        held = sc.step()[3]
        assert not set(held) & set(pair)
    assert np.all(np.linalg.norm(sc.xy[pair] - start, axis=1) > 15.0)


def test_walker_pushed_off_the_network_beside_a_stay_reroutes_round_it():
    """scene-off-network-beside-stay: life-v1-full300/v8, t=344.2 s. person-016, on its way to a free slot of a
    group of rest places in a park, was pushed while giving way to the edge of a support gap, where its body is
    supported but the network cells (half a walking width from unsupported ground) are not; person-098, at a stay,
    stands on its route there (60 s stall). Every replan from where it stood failed, and a detour round the people
    at their stays was pulled back through them. Now it connects to the network over supported ground, its detour
    survives string pulling, and it walks round them to its slot. (The network window was exported from the same
    build inputs without street furniture, of which there is none within it.)"""
    sc = Scene(DATA / "scene-off-network-beside-stay.npz")
    router = sc.router()
    i = sc.ids.index("person-016")
    goal = sc.paths[i].at(sc.paths[i].total)
    assert not router.on_snapshot(sc.xy[i])
    assert router.plan(sc.xy[i], goal) is not None

    class Paths:
        def __contains__(self, k):
            return k in sc.paths

        def path(self, k):
            return sc.paths[k]

    events = []
    agendas = {i: type("Agenda", (), {"blocked": lambda self, t, reason: events.append(reason)})()}
    reroutes = Reroutes(router)
    stall = np.zeros(sc.n, int)
    stall[i] = 600
    out = reroutes.update(sc.t, [i], sc.xy, sc.speed, sc.active, stall, Paths(), lambda k: goal, set(), agendas, 0.1)
    assert list(out) == [i] and events == ["rerouted"]
    standing = [j for j in range(sc.n) if j != i]
    still = sc.xy[standing].copy()
    sc.paths[i] = WalkPath(out[i])
    held = 0
    for _ in range(150):
        held += i in sc.step()[3]
        if math.dist(sc.xy[i], goal) < 0.05 and sc.speed[i] < 0.05:
            break
    assert held == 0 and math.dist(sc.xy[i], goal) < 0.05
    np.testing.assert_allclose(sc.xy[standing], still)


def test_walker_grazing_a_notch_of_the_support_is_not_held():
    """scene-support-notch-overtaking: life-v1-full300/v9, t=1470.3 s. person-147 (1.24 m/s), overtaking
    person-080 (1.0 m/s) on the left, passes 1 cm from a corner where the unsupported ground juts out by 0.5 m, its
    heading 1.2 degrees towards it. Neither keeping its heading nor the selected turn is supported, and the hold
    searched only five headings within the turn rate; the supported turns span 3.7 degrees between them (a stop of
    12.4 m/s2 was recorded). It must find one, brake within the limits and walk on."""
    sc = Scene(DATA / "scene-support-notch-overtaking.npz")
    i = sc.ids.index("person-147")
    start = sc.xy[i].copy()
    held = 0
    worst = 0.0
    for _ in range(150):
        v0 = sc.speed[i]
        held += i in sc.step()[3]
        worst = max(worst, (v0 - sc.speed[i]) / 0.1)
    assert held == 0
    assert worst <= 5.0 + 1e-6
    assert math.dist(sc.xy[i], start) >= 8.0


def test_every_step_is_supported_as_the_tape_records_it():
    """scene-recorded-support-edge: life-v1-drive-demo150/v10 (the same tick in v8 and v9), t=809.6 s. person-059
    accelerates along an edge of the support. Its step at 0.94 m/s is supported in the exact pose, but a corner of
    its box lies within a millimetre of the edge, and the pose as recorded (mm, centidegrees, cm/s) leaves the
    support: the tape QA found one unsupported footprint. Every step must be supported as the tape records it."""
    from awr.life.civil.assemble import recorded_pose
    from awr.life.motion import kernels as K

    sc = Scene(DATA / "scene-recorded-support-edge.npz")
    c, i = sc.crowd, sc.ids.index("person-059")
    for _ in range(30):
        xy = sc.xy.copy()
        new_xy, new_yaw = sc.step()[:2]
        pose = recorded_pose(new_xy[i, 0], new_xy[i, 1], new_yaw[i], *((new_xy[i] - xy[i]) / 0.1))
        assert K.body_supported(*pose, c.ext[i], c.z_ok, c.walk, c.sx0, c.sy0, c.scell)
