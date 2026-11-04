"""发往 Agent 的全部消息的白名单在线审计（M26-FR-022、FR-026、FR-105；D3-AC-16、D3-AC-56）：CONFIG、OBS（含 inbox）、ERROR、
CLOSE 逐帧按线上字节校验，未知字段逐键计数；违规帧不发送并以 644 中止运行；`run_job` 把计数写入运行元数据，违规时判平台失败。"""

from __future__ import annotations

import copy
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from awr.agentsdk.env.obs_audit import ObsAudit, ObsSchemaViolation, audit_scope, jpeg_segments_ok, lidar_tag_ok
from awr.agentsdk.wire import PROTOCOL, T, WireSocket, encode_frame

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SDK = ROOT / "tools/sdk/anet4d/src"
sys.path.insert(0, str(SDK))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tests" / "contracts"))
MOUNT = {"t_bs_m": [0.1, 0, -0.05], "q_bs_xyzw": [0, 0, 0, 1]}


def _jpeg(extra: bytes = b"") -> bytes:
    Image = pytest.importorskip("PIL.Image")

    b = io.BytesIO()
    Image.fromarray(np.full((16, 16, 3), 90, np.uint8)).save(b, format="JPEG", quality=90)
    j = b.getvalue()
    return j[:2] + extra + j[2:] if extra else j


def _obs() -> dict:
    from test_d3_bench_agent import _obs as base

    o = base()
    o["obs"]["sensors"]["cam0"][0]["rgb"]["data"] = _jpeg()
    o["obs"]["sensors"]["lidar0"] = [{"sensor_id": "lidar0", "seq": 0, "t_start_ns": 0, "mount_nominal": MOUNT, "n": 2, "xyz_m": b"\0" * 24,
                                      "reflectivity": b"\x05\x06", "tag": b"\x00\x01", "offset_time_ns": b"\0" * 8}]
    o["inbox"] = [{"src": "uav-02", "seq": 0, "topic": "tracks", "t_sent_ns": 1, "t_recv_ns": 2, "payload": b"x"}]
    return o


def _body(t: int, doc) -> bytes:
    return encode_frame(t, doc)[8:]


# ---------------------------------------------------------------------------------------------------------------- 单元


def test_valid_frames_are_counted_per_kind_and_uav() -> None:
    a = ObsAudit()
    a.check_body(T.OBS, _body(T.OBS, _obs()), uav="uav-01")
    a.check_body(T.ERROR, _body(T.ERROR, {"code": 666, "why": "k_mismatch"}), uav="uav-01")
    a.check_body(T.CLOSE, _body(T.CLOSE, {"reason": "run_ended"}), uav="uav-02")
    a.check_body(T.ACT, _body(T.ACT, {"k": 1}))                                   # Agent → Env 的帧不在审计范围
    s = a.summary()
    assert s["passed"] and s["schema_checked"] == 3 and s["schema_valid"] == 3 and s["unknown_fields"] == 0
    assert s["by_kind"]["obs"] == {"checked": 1, "valid": 1, "violations": 0, "unknown_fields": 0}
    assert s["by_uav"]["uav-02"]["close"]["checked"] == 1 and s["samples"] == []
    assert s["timing"]["obs_check_ms_p99"] is not None


@pytest.mark.parametrize(("where", "n"), [
    (lambda o: o["obs"].__setitem__("run_seed", 1), 1),                                                  # 顶层观测
    (lambda o: o["obs"]["nav"].update(true_pos_enu_m=[0, 0, 0], err_m=1.0), 2),                         # 导航解（两个键）
    (lambda o: o["obs"]["sensors"]["cam0"][0].update(T_world_sensor=[0] * 16), 1),                      # oneOf 分支内（CameraObs）
    (lambda o: o["obs"]["sensors"]["lidar0"][0].update(instance=b"\0\0"), 1),                           # oneOf 分支内（LidarObs）
    (lambda o: o["inbox"][0].__setitem__("actor_id", "person_a"), 1),                                   # inbox 消息
    (lambda o: o.__setitem__("oracle", {}), 1),                                                         # 帧体顶层
])
def test_unknown_fields_are_counted_per_key_and_fail_closed(where, n: int) -> None:
    o = _obs()
    where(o)
    a = ObsAudit()
    with pytest.raises(ObsSchemaViolation) as e:
        a.check_body(T.OBS, _body(T.OBS, o), uav="uav-01")
    assert e.value.code == 644 and e.value.why == "schema" and e.value.kind == "obs"
    s = a.summary()
    assert s["unknown_fields"] == n and s["schema_violations"] == 1 and not s["passed"]
    smp = s["samples"][0]
    assert smp["why"] == "unknown_field" and smp["k"] == 3 and len([p for p in smp["paths"] if ":" not in p]) == n
    assert "person_a" not in json.dumps(s)                                          # 只记路径与键名，不记取值


def test_other_violations_jpeg_segments_and_lidar_tag() -> None:
    a = ObsAudit(fail_closed=False)
    o = _obs()
    o["obs"]["status"]["mode"] = "teleport"
    a.check_body(T.OBS, _body(T.OBS, o))
    o = _obs()
    o["obs"]["sensors"]["cam0"][0]["rgb"]["data"] = _jpeg(b"\xff\xe1\x00\x08Exif\x00\x00")                  # APP1
    a.check_body(T.OBS, _body(T.OBS, o))
    o = _obs()
    o["obs"]["sensors"]["cam0"][0]["rgb"]["data"] = _jpeg(b"\xff\xfe\x00\x06seed")                          # COM
    a.check_body(T.OBS, _body(T.OBS, o))
    o = _obs()
    o["obs"]["sensors"]["lidar0"][0]["tag"] = b"\x00\x41"                                                  # 位 6 置位
    a.check_body(T.OBS, _body(T.OBS, o))
    a.check_body(T.ERROR, _body(T.ERROR, {"code": 666, "why": "Bad Frame: seed=5"}))                       # 错误消息不允许自由文本
    a.check_body(T.CLOSE, _body(T.CLOSE, {"reason": "episode_end"}))
    s = a.summary()
    assert s["schema_checked"] == 6 and s["schema_violations"] == 6 and s["unknown_fields"] == 0
    assert s["jpeg_segment_violations"] == 2 and s["lidar_tag_violations"] == 1
    assert [x["why"] for x in s["samples"]] == ["schema", "jpeg_segment", "jpeg_segment", "lidar_tag", "schema", "schema"]
    assert s["samples"][0]["paths"] == ["obs/status/mode:enum"]


def test_config_whitelist_and_decode_limits(synth_event, tmp_path: Path) -> None:
    from test_env import _spec

    from awr.agentsdk.env.core import BenchEnv

    env = BenchEnv(_spec(synth_event, tmp_path / "r", dur=2.0))
    _obs0, infos = env.reset()
    env.close()
    a = ObsAudit(fail_closed=False)
    cfg = infos["uav-01"]["config"]
    a.check_body(T.CONFIG, _body(T.CONFIG, cfg), uav="uav-01")
    leak = copy.deepcopy(cfg)
    leak["run_seed"], leak["episode"]["tape_sha256"] = 5, "0" * 64
    a.check_body(T.CONFIG, _body(T.CONFIG, leak), uav="uav-01")
    a.check_body(T.OBS, _body(T.OBS, {"obs": {"schema": "x" * 5000}, "inbox": []}))                         # 超出解码上限（字符串 4 KiB）
    s = a.summary()
    assert s["by_kind"]["config"] == {"checked": 2, "valid": 1, "violations": 1, "unknown_fields": 2}
    assert s["decode_violations"] == 1 and s["by_kind"]["obs"]["violations"] == 1


def test_byte_checks() -> None:
    assert jpeg_segments_ok(_jpeg())
    assert not jpeg_segments_ok(_jpeg(b"\xff\xe2\x00\x04ab"))
    assert not jpeg_segments_ok(_jpeg()[:-2])                                                               # 缺 EOI
    assert not jpeg_segments_ok(b"\x89PNG")
    assert lidar_tag_ok(bytes(range(64))) and not lidar_tag_ok(b"\x80") and lidar_tag_ok(b"")


def test_audit_scope_binds_hosts(tmp_path: Path) -> None:
    from awr.agentsdk.env.remote import SocketAgentHost

    with audit_scope() as a:
        h1, h2 = SocketAgentHost("uav-01", tmp_path / "a.sock"), SocketAgentHost("uav-02", tmp_path / "b.sock")
    h3 = SocketAgentHost("uav-03", tmp_path / "c.sock")
    assert h1.audit is a and h2.audit is a and h3.audit is not a
    for h in (h1, h2, h3):
        h.close()


# ---------------------------------------------------------------------------------------------------------------- 套接字运行


def _serve(hosts: dict) -> list[subprocess.Popen]:
    env_ = {**os.environ, "PYTHONPATH": f"{SDK}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    return [subprocess.Popen([sys.executable, "-m", "anet4d.serve", "--baseline", "track_share", "--sock", str(h.path)], env=env_)
            for h in hosts.values()]


def test_socket_episode_counts_every_frame(synth_event, tmp_path: Path) -> None:
    from test_env import _spec

    from awr.agentsdk.env.core import BenchEnv
    from awr.agentsdk.env.remote import SocketAgentHost, run_remote_episode

    spec = _spec(synth_event, tmp_path / "run", dur=10.0)
    with audit_scope() as audit:
        hosts = {f.uav: SocketAgentHost(f.uav, tmp_path / "sock" / f.uav / "agent.sock") for f in spec.fleet}
    procs = _serve(hosts)
    try:
        for h in hosts.values():
            h.accept()
        _m, stats = run_remote_episode(BenchEnv(spec), hosts, step_budget_s=30.0)
    finally:
        for p in procs:
            p.wait(30)
    assert all(p.returncode == 0 for p in procs) and all(s["protocol_errors"] == 0 for s in stats.values())
    s = audit.summary()
    n_obs = sum(1 for u in hosts for _ in (tmp_path / "run/agents" / u / "obs.sha256").read_text().splitlines())
    assert s["passed"] and s["unknown_fields"] == 0 and s["schema_violations"] == 0
    assert s["by_kind"]["config"]["checked"] == 2 and s["by_kind"]["close"]["checked"] == 2
    n_sent = n_obs - len(hosts)                                                      # 终止步的观测只记录、不再下发
    assert s["by_kind"]["obs"]["checked"] == n_sent == sum(st["acts"] for st in stats.values()) and n_sent == 2 * 100
    assert s["by_uav"]["uav-01"]["obs"]["checked"] + s["by_uav"]["uav-02"]["obs"]["checked"] == n_sent


def test_leaked_field_aborts_run_before_delivery(synth_event, tmp_path: Path, monkeypatch) -> None:
    """k = 5 的观测被注入一个真值字段 → 该帧不发送，运行以 644 中止，Agent 收到的最后一帧为 k = 4，CLOSE 为 shutdown。"""
    from test_env import _spec

    from awr.agentsdk.env.core import BenchEnv
    from awr.agentsdk.env.remote import SocketAgentHost, run_remote_episode

    orig = BenchEnv._observe

    def leaky(self, final=False):
        out = orig(self, final)
        for msg in out.values():
            if msg["obs"]["k"] == 5:
                msg["obs"]["nav"]["true_pos_enu_m"] = [1.0, 2.0, 3.0]
        return out

    monkeypatch.setattr(BenchEnv, "_observe", leaky)
    spec = _spec(synth_event, tmp_path / "run", dur=5.0, fleet=_spec(synth_event, None).fleet[:1])
    sock = Path(tempfile.mkdtemp(prefix="d3o", dir="/tmp"))
    seen: list[tuple[int, object]] = []
    with audit_scope() as audit:
        host = SocketAgentHost("uav-01", sock / "agent.sock")

    def agent() -> None:
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.connect(str(host.path))
        ws = WireSocket(c)
        ws.send(T.HELLO, {"protocol": PROTOCOL, "sdk_version": "test", "prefs": {}})
        ws.recv(time.monotonic() + 10)
        ws.send(T.READY, {})
        while True:
            t, body = ws.recv(time.monotonic() + 30)
            seen.append((t, body["obs"]["k"] if t == T.OBS else body))
            if t != T.OBS:
                break
            ws.send(T.ACT, {"schema": "drone4d.bench.act.v1", "k": body["obs"]["k"]})
        c.close()

    th = threading.Thread(target=agent, daemon=True)
    th.start()
    try:
        host.accept()
        with pytest.raises(ObsSchemaViolation) as e:
            run_remote_episode(BenchEnv(spec), {"uav-01": host})
    finally:
        th.join(10)
        shutil.rmtree(sock, ignore_errors=True)
    assert e.value.code == 644 and e.value.paths == ["obs/nav/true_pos_enu_m"]
    assert [k for t, k in seen if t == T.OBS] == [0, 1, 2, 3, 4]
    assert seen[-1] == (T.CLOSE, {"reason": "shutdown"})
    s = audit.summary()
    assert s["unknown_fields"] == 1 and s["by_kind"]["obs"] == {"checked": 6, "valid": 5, "violations": 1, "unknown_fields": 1}


# ---------------------------------------------------------------------------------------------------------------- run_job 与运行元数据


@pytest.fixture()
def job(synth_event, tmp_path: Path, monkeypatch, request) -> dict:
    w, ev = synth_event
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(tmp_path / "docker-state"))
    monkeypatch.setenv("PYTHONPATH", f"{SDK}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    sock = Path(tempfile.mkdtemp(prefix="d3j", dir="/tmp"))
    request.addfinalizer(lambda: shutil.rmtree(sock, ignore_errors=True))
    monkeypatch.setenv("AWR_AGENT_SOCK_ROOT", str(sock))
    docker = tmp_path / "docker"
    docker.write_text(f"#!/bin/sh\nexec {sys.executable} {HERE / 'fake_docker.py'} \"$@\"\n")
    docker.chmod(0o755)
    rs = {"schema": "drone4d.bench.run_spec.v1", "experiment": {"name": "t"}, "world": {"id": w.name}, "event": {"id": ev.parent.name, "version": 1},
          "fleet": [{"uav_id": "uav-01", "vehicle_ref": "x500", "start": {"kind": "pose", "pos_enu_m": [-10.0, 50.0, 0.0], "yaw_deg": 0.0}},
                    {"uav_id": "uav-02", "vehicle_ref": "x500", "start": {"kind": "pose", "pos_enu_m": [10.0, 50.0, 0.0], "yaw_deg": 0.0}}],
          "policy": {"kind": "reference", "name": "track_share"}, "obs_level": "L1", "physics": "kinematic", "duration_s": 8.0, "seeds": [11]}
    return {"rs": rs, "policy": {"kind": "reference", "name": "static_guard"}, "worlds_root": w.parent, "events_root": ev.parent.parent,
            "physics": "kinematic", "docker": str(docker), "step_budget_s": 30.0}


def test_run_job_writes_audit_into_run_metadata(job: dict, tmp_path: Path) -> None:
    from awr.bench.runner import run_job

    rs = job.pop("rs")
    r = run_job(rs, seed=11, run_id="t-ok", run_dir=tmp_path / "ok", **job)
    assert r["obs_audit"]["passed"] and r["obs_audit"]["unknown_fields"] == 0 and r["obs_audit"]["schema_checked"] > 0
    doc = json.loads((tmp_path / "ok/agents/obs_audit.json").read_text())
    man = json.loads((tmp_path / "ok/manifest.json").read_text())
    assert doc["schema_checked"] == r["obs_audit"]["schema_checked"] == man["obs_audit"]["schema_checked"]
    assert man["obs_audit"]["unknown_fields"] == 0 and man["obs_audit"]["passed"] and "samples" not in man["obs_audit"]
    assert doc["by_kind"]["config"]["checked"] == 2 and doc["by_kind"]["close"]["checked"] == 2
    assert (tmp_path / "ok/evaluation.json").exists()


def test_run_job_with_unknown_field_is_platform_failure(job: dict, tmp_path: Path, monkeypatch) -> None:
    from awr.agentsdk.env.core import BenchEnv
    from awr.bench.runner import run_job

    orig = BenchEnv.agent_config

    def leaky(self, uav):
        c = orig(self, uav)
        c["episode"]["run_id"] = self.spec.run_id                                   # reset 配置泄漏运行标识
        return c

    monkeypatch.setattr(BenchEnv, "agent_config", leaky)
    rs = job.pop("rs")
    with pytest.raises(ObsSchemaViolation) as e:
        run_job(rs, seed=11, run_id="t-bad", run_dir=tmp_path / "bad", **job)
    assert e.value.code == 644 and e.value.kind == "config"
    doc = json.loads((tmp_path / "bad/agents/obs_audit.json").read_text())
    assert doc["unknown_fields"] == 1 and not doc["passed"] and doc["by_kind"]["obs"]["checked"] == 0
    assert doc["samples"][0]["paths"] == ["episode/run_id"] and "t-bad" not in json.dumps(doc)
    assert json.loads((tmp_path / "bad/manifest.json").read_text())["obs_audit"]["unknown_fields"] == 1
    assert not (tmp_path / "bad/evaluation.json").exists()                          # 平台失败：不评测


def test_l2_observations_pass_audit_with_byte_checks(synth_event, tmp_path: Path) -> None:
    """L2（相机 JPEG、LiDAR、IMU、GNSS；CPU 渲染替身）全部观测按线上字节通过审计，JPEG 段与 tag 断言 0 违规。"""
    from test_l2 import _episode

    seen, cfg = _episode(synth_event, tmp_path / "l2")
    a = ObsAudit()
    a.check_body(T.CONFIG, _body(T.CONFIG, cfg), uav="uav-01")
    for u, msg in seen:
        a.check_body(T.OBS, _body(T.OBS, msg), uav=u)
    s = a.summary()
    assert s["passed"] and s["by_kind"]["obs"]["checked"] == len(seen) and s["jpeg_segment_violations"] == 0 and s["lidar_tag_violations"] == 0


def _mutations(doc, rng, n: int) -> list:
    """随机变异：替换叶子（类型、越界、枚举外、布尔与整数互换、整值浮点）、删除键、增加键、增删数组元素。"""
    def nodes(x, path=()):
        yield path, x
        if isinstance(x, dict):
            for k, v in x.items():
                yield from nodes(v, (*path, k))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                yield from nodes(v, (*path, i))

    allp = [p for p, _ in nodes(doc) if p]
    vals = [None, True, False, 0, 1, -1, 1.0, 2.5, -1e9, 2**32, "", "x", "none", "jpeg", "Bad Why", b"\0", [], [0, 0, 0], {}, {"a": 1}]
    out = []
    for _ in range(n):
        d = copy.deepcopy(doc)
        p = allp[int(rng.integers(len(allp)))]
        parent = d
        for q in p[:-1]:
            parent = parent[q]
        op = int(rng.integers(4))
        if op == 0:
            parent[p[-1]] = copy.deepcopy(vals[int(rng.integers(len(vals)))])
        elif op == 1:
            del parent[p[-1]]
        elif op == 2 and isinstance(parent[p[-1]], dict):
            parent[p[-1]]["zz_extra"] = 1
        elif isinstance(parent[p[-1]], list):
            parent[p[-1]].append(copy.deepcopy(parent[p[-1]][0]) if parent[p[-1]] else 0)
        out.append(d)
    return out


@pytest.mark.parametrize("kind", ["obs", "config", "error", "close"])
def test_compiled_validator_agrees_with_jsonschema(kind: str, synth_event, tmp_path: Path) -> None:
    from awr.agentsdk.env.obs_audit import _fast, _validator

    if kind == "config":
        from test_env import _spec

        from awr.agentsdk.env.core import BenchEnv

        env = BenchEnv(_spec(synth_event, tmp_path / "r", dur=2.0))
        doc = env.reset()[1]["uav-01"]["config"]
        env.close()
    else:
        doc = {"obs": _obs(), "error": {"code": 666, "why": "frame"}, "close": {"reason": "run_ended"}}[kind]
    rng = np.random.default_rng(7)
    fast, ref = _fast(kind), _validator(kind)
    docs = [doc, *_mutations(doc, rng, 400)]
    verdicts = [(fast(d), ref.is_valid(d)) for d in docs]
    assert all(a == b for a, b in verdicts), [i for i, (a, b) in enumerate(verdicts) if a != b][:5]
    assert verdicts[0] == (True, True) and sum(not b for _, b in verdicts) > 100


def test_compile_rejects_unsupported_keywords() -> None:
    from awr.agentsdk.env.obs_audit import compile_schema

    with pytest.raises(NotImplementedError):
        compile_schema({"type": "object", "patternProperties": {"^a": {}}}, "", lambda u: {})
    with pytest.raises(NotImplementedError):
        compile_schema({"type": "string", "format": "date-time"}, "", lambda u: {})


def test_subworker_counts_obs_violation_as_platform_failure(synth_event, tmp_path: Path, monkeypatch, request) -> None:
    """正式评测 Run 下发了未知字段 → run_job 抛 644 → 内部提交流水线按平台原因重跑一次，仍失败则 FAILED（不计入提交方）。"""
    from test_subworker import _docker, _submit
    from test_subworker import root as root_fixture

    from awr.agentsdk.env.core import BenchEnv
    from awr.bench.submissions import SubmissionStore
    from awr.bench.subworker import process_one

    root = root_fixture.__wrapped__(synth_event, tmp_path, monkeypatch, request)
    orig = BenchEnv._observe

    def leaky(self, final=False):
        out = orig(self, final)
        if not self.spec.run_id.endswith("-smoke"):
            for msg in out.values():
                msg["obs"]["status"]["actor_ids"] = []
        return out

    monkeypatch.setattr(BenchEnv, "_observe", leaky)
    st = SubmissionStore(root / "submissions")
    sid = _submit(st)
    logs: list[str] = []
    assert process_one(root, docker=_docker(tmp_path), smoke_s=5.0, n_boot=100, log=logs.append) == sid
    d = st.get(sid)
    assert d["state"] == "FAILED" and d["reason"]["detail"] == {"why": "platform", "run": 0}
    assert sum("ObsSchemaViolation: 644 schema obs" in m for m in logs) == 2                        # 首次与平台重跑各一次
    doc = json.loads(next((root / "dataset").glob(f"*/runs/{sid}-r0/agents/obs_audit.json")).read_text())
    assert doc["unknown_fields"] >= 1 and doc["samples"][0]["paths"] == ["obs/status/actor_ids"]
