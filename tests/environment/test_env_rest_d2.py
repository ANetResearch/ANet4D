"""环境 REST 的 D2 部分（WP-09；AWR-04 §11.1、§14.3；17 §17.5 SB-60 至 SB-63、§17.6.9；M17-FR-038；M11-FR-202）。

- show-api 的 D1 REST 面不变（没有 `/api/env/calendar`，日历经 WS `call env/calendar`）；
- `sandbox_router` 以 `/api/sandbox/v1/sessions/{sid}` 前缀挂载：四个端点经请求级上下文 `ctx_for` 解析到各自会话的 Gateway
  与总线（两个会话互不串扰）；日历生效后 `env/state` 帧带 `calendar` 与 `sun.mode = clock`。
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from awr.api import deps as D
from awr.api.problem import ApiProblem, problem
from awr.api.rest import env as R
from awr.api.security import ApiPrincipal
from awr.environment.field import EnvironmentServiceImpl


class _Ch:
    payload: bytes | None = None


class _Rpc:
    def __init__(self, ctx: _Ctx) -> None:
        self.ctx = ctx
        self.calls: list[dict] = []

    def handle_call(self, caller, m: dict) -> None:
        self.calls.append(m)
        svc = self.ctx.svc
        if m["service"] == "env/query":
            rep = svc.handle_query({"args": m["args"]})
            caller.send_ctrl({"op": "result", "id": m["id"], "status": "succeeded", "code": rep["code"], "data": rep, "final": True})
            return
        rep = _cmd(svc, m["service"], m["args"], svc.t_grid_ns // svc.tick_ns + 1)
        st = rep["status"]
        msg = {"op": "result", "id": m["id"], "status": st, "code": rep["code"]}
        if st == "rejected":
            msg["detail"] = rep.get("detail")
        else:
            svc.on_env_tick_all(rep["result"]["t_apply_ns"] // svc.tick_ns)
            self.ctx.publish()
            msg["data"] = rep["result"]
        caller.send_ctrl(msg)


def _cmd(svc: EnvironmentServiceImpl, op: str, args: dict, apply_tick: int) -> dict:
    from awr.environment.weather.presets import EnvError

    try:
        r = svc.apply(svc.op_from_msg(op, args, by="p-test"), apply_tick)
    except EnvError as ex:
        return {"status": "rejected", "code": ex.code, "detail": ex.detail}
    return {"status": "accepted", "code": 0, "result": {"version": r.version, "t_apply_ns": r.t_apply_ns}}


class _Gw:
    def __init__(self, ctx: _Ctx) -> None:
        self.env = type("E", (), {"ch": _Ch()})()
        self.rpc = _Rpc(ctx)


class _Bus:
    def __init__(self, svc: EnvironmentServiceImpl) -> None:
        self.svc = svc

    async def call(self, key, msg, **kw):
        return {"v": 1, "code": 0, "frame": self.svc.state_at(msg["args"]["t_ns"])}


class _Ctx:
    def __init__(self, world: str) -> None:
        self.svc = EnvironmentServiceImpl(world_id=world, world_seed=3, load_assets=False,
                                          coordinate={"anchor": {"latDeg": 30.0, "lonDeg": 120.0, "hMslM": 10.0}})
        self.svc.on_env_tick_all(0)
        self.gateway = _Gw(self)
        self.bus = _Bus(self.svc)
        self.publish()

    def publish(self) -> None:
        self.gateway.env.ch.payload = self.svc.heartbeat_bytes()


@pytest.fixture()
def client(monkeypatch):
    show = _Ctx("synthcity")
    sessions = {"sb-aaaaaaaaaa": _Ctx("synthcity"), "sb-bbbbbbbbbb": _Ctx("shanghai")}

    def ctx_for(request: Request):
        sid = request.path_params.get("sid")
        if sid is None:
            return show
        c = sessions.get(sid)
        if c is None:
            raise ApiProblem(305, status=404)
        return c

    monkeypatch.setattr(R, "ctx_for", ctx_for)
    app = FastAPI()

    @app.middleware("http")
    async def auth(request: Request, call_next):
        request.state.principal = ApiPrincipal("p-test", request.headers.get("x-role", "viewer"), "j", 0, "loopback")
        return await call_next(request)

    @app.exception_handler(ApiProblem)
    async def _p(request: Request, exc: ApiProblem):
        return problem(exc.code, status=exc.status, detail=exc.detail, headers=exc.headers)

    def sandbox_principal(request: Request) -> ApiPrincipal:  # 沙盒 token 鉴权由 M11 覆盖（tests/api），此处只替身
        if request.path_params.get("sid") not in sessions:
            raise ApiProblem(305, status=404)
        return request.state.principal

    app.dependency_overrides[D.require_sandbox_session] = sandbox_principal
    app.include_router(R.router)
    app.include_router(R.sandbox_router, prefix="/api/sandbox/v1/sessions/{sid}")
    c = TestClient(app)
    c.show = show  # type: ignore[attr-defined]
    c.sessions = sessions  # type: ignore[attr-defined]
    return c


def test_show_rest_surface_unchanged(client):
    paths = sorted(r.path for r in R.router.routes)
    assert "/api/env/calendar" not in paths
    assert paths == sorted(["/api/env/state", "/api/env/presets", "/api/env/query", "/api/env/set", "/api/env/preset",
                            "/api/env/streamlines/{lib}/{name}"])
    assert client.get("/api/env/state").json()["config"]["sun"] == {"azimuth_deg": 150, "elevation_deg": 50}


def test_sandbox_calendar_validation(client):
    a = "/api/sandbox/v1/sessions/sb-aaaaaaaaaa"
    body = {"start_utc": "2026-10-05T14:00:00Z", "tz": "Asia/Shanghai"}
    assert client.post(f"{a}/env/calendar", json=body).status_code == 403
    r = client.post(f"{a}/env/calendar", json={**body, "id": "c-9"}, headers={"x-role": "operator"})
    assert r.status_code == 200 and r.json()["status"] == "accepted"
    rpc = client.sessions["sb-aaaaaaaaaa"].gateway.rpc
    assert rpc.calls[-1] == {"op": "call", "id": "c-9", "service": "env/calendar",
                             "args": {"start_utc": "2026-10-05T14:00:00Z", "tz": "Asia/Shanghai", "jump": True}}
    f = client.get(f"{a}/env/state").json()
    assert f["config"]["sun"]["mode"] == "clock" and f["config"]["sun"]["band"] == "night" and f["calendar"]["tz"] == "Asia/Shanghai"
    r = client.post(f"{a}/env/calendar", json={"start_utc": "2026-13-01T00:00:00Z", "tz": "UTC"}, headers={"x-role": "operator"})
    assert r.status_code == 422 and r.json()["code"] == 110
    assert client.post(f"{a}/env/calendar", json={"tz": "UTC"}, headers={"x-role": "operator"}).status_code == 422
    assert client.post(f"{a}/env/calendar", json={**body, "extra": 1}, headers={"x-role": "operator"}).status_code == 422


def test_sandbox_routes_per_session(client):
    a, b = "/api/sandbox/v1/sessions/sb-aaaaaaaaaa", "/api/sandbox/v1/sessions/sb-bbbbbbbbbb"
    s0 = client.get(f"{a}/env/state").json()
    assert s0["schema"] == "awr.env.keyframe.v1" and "calendar" not in s0
    r = client.post(f"{a}/env/calendar", json={"start_utc": "2026-10-05T02:00:00Z", "tz": "Asia/Shanghai", "jump": False},
                    headers={"x-role": "operator"})
    assert r.status_code == 200
    fa = client.get(f"{a}/env/state").json()
    fb = client.get(f"{b}/env/state").json()
    assert fa["calendar"]["start_utc"] == "2026-10-05T02:00:00Z" and fa["config"]["sun"]["band"] == "day"
    assert "calendar" not in fb and fb["world_id"] == "shanghai"  # 会话互不串扰
    r = client.post(f"{b}/env/preset", json={"name": "fog", "duration_s": 0}, headers={"x-role": "operator"})
    assert r.status_code == 200 and client.sessions["sb-bbbbbbbbbb"].svc.kf.to_preset == "fog"
    assert client.sessions["sb-aaaaaaaaaa"].svc.kf.to_preset != "fog"
    q = client.post(f"{a}/env/query", json={"points": [[0, 0, 30]], "fields": ["OPTICS"]})
    assert q.status_code == 200 and q.json()["data"]["n"] == 1
    t = client.sessions["sb-aaaaaaaaaa"].svc.t_grid_ns + 5_000_000_000
    st = client.get(f"{a}/env/state?t_ns={t}").json()
    assert st["t_ns"] == t and int(st["calendar"]["t_cal_unix_ns"]) - int(fa["calendar"]["t_cal_unix_ns"]) == t - fa["t_ns"]
    assert client.post(f"{a}/env/preset", json={"name": "fog"}).status_code == 403
    assert client.get("/api/sandbox/v1/sessions/sb-zzzzzzzzzz/env/state").status_code == 404


@pytest.mark.parametrize("name", ["d²³⁴.awsl", "d\u0663\u0666\u0660.awsl", "d0_1.awsl", "d+01.awsl"])
def test_streamlines_name_ascii_digits(client, name):
    """流线文件名只认 ASCII 数字：上标数字过去让 int() 抛错，请求变成 500；其余写法同样 404。"""
    r = client.get(f"/api/env/streamlines/default/{name}")
    assert r.status_code == 404 and r.json()["code"] == 305
