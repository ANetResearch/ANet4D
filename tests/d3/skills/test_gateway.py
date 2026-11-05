"""指挥网关（自 CHIMERA `tests/test_sandbox_bridge.py`、`test_sandbox_http.py`、`test_realtime.py` 移入）：桥的恰好一次与取消、HTTP
的鉴权 / 公开任务 / 过期序号 409 / 取消、墙钟配速与异步决策、任务层账本钩子、像素投影。"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from awr.agentsdk.gateway import (
    DecisionBridge,
    DecisionWorker,
    Gateway,
    RealtimeClock,
    RealtimeOverrun,
    SessionCancelled,
    handler,
    load_token,
)
from awr.agentsdk.gateway.projection import camera_metadata, project_pixel


def turn(text="t=0", images=(), episode_s=0.0):
    return SimpleNamespace(text=text, images=list(images), episode_s=episode_s, sensor_metadata={})


# ---------------------------------------------------------------- bridge
def test_pending_request_and_exactly_once():
    b = DecisionBridge()
    got = []
    t = threading.Thread(target=lambda: got.append(b.complete("public system", [], turn(images=[("cam", b"jpeg")]))))
    t.start()
    ob = b.observation()
    assert ob["sequence"] == 1 and ob["status"] == "awaiting_action" and ob["images"][0]["label"] == "cam"
    time.sleep(0.05)
    assert got == [] and b.observation()["sequence"] == 1
    with pytest.raises(ValueError, match="commands/reports must be arrays"):
        b.submit(1, {"commands": "bad"})
    with pytest.raises(ValueError, match="numbers must be finite"):
        b.submit(1, {"commands": [{"uav": "uav-01", "skill": "goto", "params": {"x": float("nan"), "y": 0}}]})
    b.submit(1, {"commands": [], "done": False})
    with pytest.raises(RuntimeError, match="stale, duplicate or unavailable"):
        b.submit(1, {"done": True})
    t.join(2)
    assert not t.is_alive() and got == ['{"commands": [], "done": false}']
    assert b.usage.calls == 1
    b.finish({"score": None})
    assert b.observation(after=1) == {"status": "finished", "result": {"score": None}}


def _complete_in_background(b):
    def work():
        with contextlib.suppress(SessionCancelled):
            b.complete("", [], turn())

    threading.Thread(target=work, daemon=True).start()
    return b.observation()


def test_fleet_bound_bridge_rejects_unknown_uavs():
    b = DecisionBridge(["uav-01"])
    _complete_in_background(b)
    with pytest.raises(ValueError, match="unknown UAV"):
        b.submit(1, {"commands": [{"uav": "uav-02", "skill": "hold"}]})
    b.cancel()


def test_cancel_wakes_worker_and_cannot_be_swallowed_by_provider_retry():
    b = DecisionBridge()
    stopped = []

    def work():
        try:
            b.complete("", [], turn(""))
        except SessionCancelled:
            stopped.append(True)

    t = threading.Thread(target=work)
    t.start()
    b.observation()
    b.cancel()
    t.join(2)
    assert stopped == [True] and not issubclass(SessionCancelled, Exception)
    assert b.observation() == {"status": "cancelled"}


def test_task_layer_hooks_run_before_the_vocabulary_check():
    class Billed(DecisionBridge):
        def __init__(self):
            super().__init__(reports=lambda r: None if r.get("type") == "found" else (_ for _ in ()).throw(ValueError("invalid report type")))
            self.ledger = []

        def pre_validate(self, sequence, action):
            self.check_fresh(sequence)
            self.ledger.append((sequence, action.pop("_usage", None)))
            return action

    b = Billed()
    _complete_in_background(b)
    with pytest.raises(ValueError, match="invalid report type"):
        b.submit(1, {"reports": [{"type": "plate"}], "_usage": [{"call_id": "a"}]})
    assert b.ledger == [(1, [{"call_id": "a"}])]                  # the rejected action was still a billed call
    with pytest.raises(RuntimeError):
        b.submit(2, {"done": False})
    assert len(b.ledger) == 1                                      # a stale sequence is never billed
    b.submit(1, {"reports": [{"type": "found"}], "done": True})
    assert b.ledger[-1] == (1, None)


# ---------------------------------------------------------------- HTTP
class _Gateway(Gateway):
    def __init__(self, bridge):
        super().__init__()
        self.sessions["s"] = bridge
        self.task = dict(task_id="x", brief_zh="find a person", targets=["SECRET-ID"], checkpoints=[{"private": True}], ranked=False)

    def public_tasks(self):
        return [{k: self.task[k] for k in ("task_id", "brief_zh", "ranked")}]

    def start(self, request):
        if request.get("task_id") != "x":
            raise KeyError(request.get("task_id"))
        raise RuntimeError("GPU sandbox already has an active session")


@pytest.fixture
def http():
    bridge = DecisionBridge()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(_Gateway(bridge), "test-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    def call(path, body=None, token="test-token"):
        req = Request(url + path, None if body is None else json.dumps(body).encode(),
                      headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        with urlopen(req, timeout=2) as r:
            return r.status, json.load(r)

    yield call, bridge
    bridge.cancel()
    server.shutdown()
    server.server_close()
    thread.join()


def test_public_tasks_no_ground_truth_and_auth(http):
    call, _ = http
    status, doc = call("/v1/tasks")
    encoded = json.dumps(doc)
    assert status == 200 and "SECRET-ID" not in encoded and "checkpoints" not in encoded and doc["tasks"][0]["task_id"] == "x"
    with pytest.raises(HTTPError) as c:
        call("/v1/tasks", token="wrong")
    assert c.value.code == 401
    assert call("/v1/status")[1] == {"active_session": None}


def test_stale_action_conflict_bad_input_and_cancel(http):
    call, _ = http
    with pytest.raises(HTTPError) as c:
        call("/v1/sessions/s/actions", {"sequence": 1, "action": {"done": False}})
    assert c.value.code == 409
    with pytest.raises(HTTPError) as c:
        call("/v1/sessions/s/actions", {"sequence": "1", "action": {}})
    assert c.value.code == 400 and json.load(c.value)["error"] == "sequence and action are required"
    with pytest.raises(HTTPError) as c:
        call("/v1/sessions/zz/observation")
    assert c.value.code == 404
    with pytest.raises(HTTPError) as c:
        call("/v1/sessions", {"task_id": "x"})
    assert c.value.code == 409
    with pytest.raises(HTTPError) as c:
        call("/v1/sessions/s/project", {"sequence": 1, "uav": "uav-01", "pixel": [1, 1]})
    assert c.value.code == 400 and "projection is not available" in json.load(c.value)["error"]
    assert call("/v1/sessions/s/cancel", {})[0] == 202
    assert call("/v1/sessions/s/observation")[1]["status"] == "cancelled"


def test_token_file_is_created_once_with_owner_only_permissions(tmp_path):
    p = tmp_path / "api-token"
    tok = load_token(p)
    assert len(tok) >= 32 and (p.stat().st_mode & 0o777) == 0o600 and load_token(p) == tok


# ---------------------------------------------------------------- realtime
def test_worker_does_not_block_world_or_refresh_old_observation():
    b = DecisionBridge()
    worker = DecisionWorker()
    worker.start(lambda: b.complete("", [], turn("t=0", episode_s=0)), dict(t=0))
    first = b.observation()
    assert first["clock_policy"] == "realtime"
    for tick in range(1, 101):
        b.advance_clock(tick * 0.1)
    pending = b.observation()
    assert pending["world_time_s"] == 10.0 and pending["observation_time_s"] == 0 and not worker.ready
    b.submit(first["sequence"], dict(commands=[], reports=[], done=False))
    worker.join(1)
    assert worker.ready
    context, result = worker.take()
    assert context == dict(t=0) and json.loads(result)["done"] is False and not worker.busy


def test_wall_clock_never_runs_faster_and_lag_is_not_silently_forgiven():
    value = [100.0]
    clock = RealtimeClock(2, now=lambda: value[0], sleep=lambda dt: value.__setitem__(0, value[0] + dt))
    for t in (0, 0.1, 0.2, 1, 5):
        elapsed, lag = clock.pace(t)
        assert abs(elapsed - t) < 1e-9 and lag < 1e-9
    value[0] = 110
    with pytest.raises(RealtimeOverrun):
        clock.pace(6)
    assert clock.peak_lag_s == 4.0


def test_episode_closure_releases_pending_worker():
    b = DecisionBridge()
    worker = DecisionWorker()
    worker.start(lambda: b.complete("", [], turn("")), None)
    b.observation()
    b.end_episode()
    worker.join(1)
    assert worker.ready
    with pytest.raises(SessionCancelled):
        worker.take()
    with pytest.raises(RuntimeError, match="Only one policy call"):
        worker.start(lambda: None, None) or worker.start(lambda: None, None)


# ---------------------------------------------------------------- projection
def _frame(seq=12):
    return dict(sensor_id="eo", seq=seq, t_mid_ns=1_200_000_000, exposure_ns=2_000_000, readout_ns=10_000_000, width=1920, height=1080,
                K=[1662.8, 0, 960, 0, 1662.8, 540, 0, 0, 1], dist=[0, 0, 0, 0, 0], mount_nominal=dict(t_bs_m=[0.1, 0, -0.05], q_bs_xyzw=[0, 0, 0, 1]),
                gimbal_reading=dict(roll_deg=0, pitch_deg=-90, yaw_deg=0),
                capture_navigation=dict(t_ns=1_200_000_000, pos_enu_m=[100.0, 50.0, 70.0], vel_enu_mps=[0, 0, 0], q_wb_xyzw=[0, 0, 0, 1], pos_std_m=[1.2, 1.2, 2.0], fix="single"))


def test_camera_metadata_scales_the_intrinsics_to_the_delivered_image():
    nav = dict(t_ns=1_300_000_000, pos_enu_m=[101.0, 50.0, 70.0], vel_enu_mps=[1, 0, 0], q_wb_xyzw=[0, 0, 0, 1], omega_b_rps=[0, 0, 0], pos_std_m=[1.2, 1.2, 2.0], fix="single")
    m = camera_metadata(_frame(), nav, 960)
    assert m["width"] == 960 and m["height"] == 540 and m["native_dimensions"] == [1920, 1080]
    assert m["K"][:6] == pytest.approx([831.4, 0, 480, 0, 831.4, 270]) and m["K"][6:] == [0, 0, 1]
    assert m["navigation_time_offset_s"] == pytest.approx(0.1) and m["navigation_at_delivery"]["pos_enu_m"] == [101.0, 50.0, 70.0]
    assert m["navigation_at_capture"]["pos_enu_m"] == [100.0, 50.0, 70.0] and "omega_b_rps" not in m["navigation_at_capture"]
    assert "capture_navigation" not in m


def test_project_pixel_hits_the_terrain_below_a_nadir_camera():
    nav = dict(t_ns=1_200_000_000, pos_enu_m=[100.0, 50.0, 70.0], vel_enu_mps=[0, 0, 0], q_wb_xyzw=[0, 0, 0, 1], pos_std_m=[1.2, 1.2, 2.0], fix="single")
    m = camera_metadata(_frame(), nav, 960)
    hit = project_pixel(m, [480, 270], lambda x, y: 10.0)
    assert np.allclose(hit["position_enu_m"], [100.1, 50.0, 10.0], atol=0.05) and hit["frame_seq"] == 12 and hit["approximate"]
    # a pixel right of the centre is north of nadir? camera x axis points along -y of the image: check the sign convention is stable
    right = project_pixel(m, [960 - 1, 270], lambda x, y: 10.0)["position_enu_m"]
    assert right[2] == pytest.approx(10.0, abs=0.05) and abs(right[1] - 50.0) > 30 and abs(right[0] - 100.1) < 0.5
    with pytest.raises(ValueError, match="inside the delivered image"):
        project_pixel(m, [960, 270], lambda x, y: 10.0)
    with pytest.raises(ValueError, match="does not point down"):
        project_pixel({**m, "gimbal_reading": dict(roll_deg=0, pitch_deg=0, yaw_deg=0)}, [480, 100], lambda x, y: 10.0)
    with pytest.raises(ValueError, match="below the terrain"):
        project_pixel(m, [480, 270], lambda x, y: 80.0)
