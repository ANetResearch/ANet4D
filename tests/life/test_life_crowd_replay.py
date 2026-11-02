import json

import life_toytown as toy
import numpy as np
import pytest
from test_life_crowd import EXT, corridor
from test_life_runtime import context

from awr.life.motion.crowd import Crowd
from awr.life.motion.failure_evidence import replay_crowd, save_crowd
from awr.life.motion.runtime import run


@pytest.mark.parametrize("driven", [False, True])
def test_crowd_replay_restores_all_bodies_and_driven_predictions(tmp_path, driven):
    surface, network = corridor()
    crowd = Crowd(np.stack([EXT, EXT]), np.array(["person", "person"]), surface, network)
    xy, yaw = np.array([[3.0, 0.0], [8.0, 0.0]]), np.zeros(2)
    inputs = dict(
        xy=xy,
        yaw=yaw,
        speed=np.array([0.8, 0.2]),
        active=np.ones(2, bool),
        mover=np.array([True, not driven]),
        des_yaw=yaw.copy(),
        des_speed=np.ones(2),
        vmax=np.full(2, 1.3),
        goal_cap=np.full(2, np.inf),
        anchor=xy.copy(),
        tangent=yaw.copy(),
        lane=np.zeros(2),
        seen_standing=np.full(2, -1),
    )
    crowd.waited[:] = [2.0, 1.0]
    crowd.walked[:] = [0.0, 0.5]
    crowd.aim[:] = [0.1, -0.1]
    history = dict(waited=crowd.waited.copy(), walked=crowd.walked.copy(), aim=crowd.aim.copy())
    rows = (
        (
            np.array([1]),
            np.array([[8.02, 0.0]]),
            np.zeros(1),
            np.array([0.2]),
            (8.0 + 0.2 * crowd.times)[None],
            np.zeros((1, len(crowd.times))),
            np.zeros((1, len(crowd.times))),
        )
        if driven
        else None
    )
    xy1, yaw1, speed1, held = crowd.step(**inputs, driven=rows)
    path = tmp_path / "frame.npz"
    save_crowd(path, crowd, inputs, history, dict(xy=xy1, yaw=yaw1, speed=speed1, held=np.asarray(held, int)), rows)
    assert replay_crowd(path)["exact"]
    assert replay_crowd(path)["exact"]  # a replay never modifies its source snapshot


def test_life_gate_failure_saves_last_crowd_step_for_exact_replay(tmp_path):
    world = toy.build(drivers=0, walkers=3)
    ctx = context(world, 3.0, tmp_path, dict(world.agendas))
    ctx["cfg"]["life_gates"] = {"person_longest_stall_s_max": -1.0}  # fails at the first recorded tick
    with pytest.raises(ValueError, match="life failure"):
        run(ctx)
    row = json.loads((tmp_path / "tape.life-failure.json").read_text())
    assert row["crowd_replay"] == "tape.life-crowd.npz"
    replay = replay_crowd(tmp_path / row["crowd_replay"])
    assert replay["exact"] and replay["implementation_matches_snapshot"]
