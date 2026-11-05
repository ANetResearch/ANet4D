"""本地鉴权 HTTP 网关：外部智能体轮询观测、提交决策。

    python -m <task layer> --tasks … --port 28651        # 任务层给出 Gateway 子类并调用 serve()

全部请求带 `Authorization: Bearer <token>`，POST 为 JSON（≤ 1 MiB）。路由：

| 请求 | 用途 | 网关方法 |
|---|---|---|
| `GET /v1/status` | 当前会话与任务层的状态摘要 | `status()` |
| `GET /v1/tasks` | 公开任务列表（不含真值） | `public_tasks()` |
| `POST /v1/sessions` | 创建会话 | `start(request)` |
| `GET /v1/sessions/<id>/observation?after=N` | 最多等 15 s 取尚未消费的回合 | `DecisionBridge.observation` |
| `POST /v1/sessions/<id>/actions` | `{"sequence", "action"}` | `DecisionBridge.submit` |
| `POST /v1/sessions/<id>/project` | 由当前交付帧估计像素的地面坐标 | `project(sid, request)` |
| `POST /v1/sessions/<id>/cancel` | 取消会话 | `DecisionBridge.cancel` |

错误：未授权 401；未知任务 / 会话 404；`ValueError`、`TypeError` 400；`RuntimeError`（过期序号、已有活动会话、源码已变）409。
令牌写在运行目录（`load_token`，0600），经 SSH 隧道使用；不要把令牌写进仓库。
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

__all__ = ["Gateway", "handler", "load_token", "serve"]

BODY_LIMIT = 1024 * 1024


class Gateway:
    """任务层子类实现 `start`（创建 `DecisionBridge`、登记到 `sessions`、在线程里运行回合与评分）、`public_tasks`、`status`、`project`。"""

    def __init__(self) -> None:
        self.sessions: dict[str, object] = {}
        self.active: str | None = None
        self.lock = threading.Lock()

    def status(self) -> dict:
        return dict(active_session=self.active)

    def public_tasks(self) -> list[dict]:
        return []

    def start(self, request: dict) -> dict:
        raise NotImplementedError

    def project(self, sid: str, request: dict) -> dict:
        raise ValueError("projection is not available in this gateway")

    def close(self) -> None:
        for b in self.sessions.values():
            b.cancel()


def handler(gateway, token: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, status, obj):
            body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def dispatch(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                return self.send_json(401, {"error": "authentication required"})
            parsed = urlparse(self.path)
            parts = parsed.path.strip("/").split("/")
            try:
                if self.command == "GET" and parts == ["v1", "status"]:
                    return self.send_json(200, gateway.status())
                if self.command == "GET" and parts == ["v1", "tasks"]:
                    return self.send_json(200, {"tasks": gateway.public_tasks()})
                data = {}
                if self.command == "POST":
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= BODY_LIMIT:
                        raise ValueError("body must be 1..1048576 bytes")
                    data = json.loads(self.rfile.read(length))
                    if not isinstance(data, dict):
                        raise ValueError("JSON object required")
                if self.command == "POST" and parts == ["v1", "sessions"]:
                    return self.send_json(201, gateway.start(data))
                if len(parts) == 4 and parts[:2] == ["v1", "sessions"]:
                    b = gateway.sessions[parts[2]]
                    if self.command == "GET" and parts[3] == "observation":
                        after = int(parse_qs(parsed.query).get("after", ["0"])[0])
                        return self.send_json(200, b.observation(after))
                    if self.command == "POST" and parts[3] == "actions":
                        if type(data.get("sequence")) is not int or "action" not in data:
                            raise ValueError("sequence and action are required")
                        b.submit(data["sequence"], data["action"])
                        return self.send_json(202, {"accepted": True})
                    if self.command == "POST" and parts[3] == "project":
                        return self.send_json(200, gateway.project(parts[2], data))
                    if self.command == "POST" and parts[3] == "cancel":
                        b.cancel()
                        return self.send_json(202, {"cancel_requested": True})
                return self.send_json(404, {"error": "unknown endpoint"})
            except KeyError:
                return self.send_json(404, {"error": "unknown task/session"})
            except (ValueError, TypeError) as e:
                return self.send_json(400, {"error": str(e)})
            except RuntimeError as e:
                return self.send_json(409, {"error": str(e)})

        do_GET = dispatch
        do_POST = dispatch

    return Handler


def load_token(path: Path) -> str:
    """运行目录里的令牌文件：存在则读取，否则生成（0600）。"""
    path = Path(path)
    if path.exists():
        return path.read_text().strip()
    token = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    return token


def serve(gateway: Gateway, *, runs: Path, port: int = 28651, host: str = "127.0.0.1", token_path: Path | None = None) -> None:
    """阻塞运行网关直到中断；退出时取消全部会话。"""
    runs = Path(runs)
    runs.mkdir(parents=True, exist_ok=True)
    token_path = Path(token_path) if token_path else runs / "api-token"
    token = load_token(token_path)
    server = ThreadingHTTPServer((host, port), handler(gateway, token))
    print(f"Gateway listening on {host}:{port}; token file: {token_path}", flush=True)
    try:
        server.serve_forever()
    finally:
        gateway.close()
        server.server_close()
