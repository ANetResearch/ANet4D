"""静态服务：World Package（HTTP Range）、前端构建产物与 SPA 回退（AWR-17 §5；M11-FR-082；ADR-013；AWR-03 §3.3）。

- `/worlds/{id}/**`：`world.json` 为 `no-cache` + 强 ETag（内容 sha256）+ 304；其他文件带 `?v=<当前 contentVersion>` 时
  `public, max-age=31536000, immutable`（200 与 206 相同），`?v=` 与当前 contentVersion 不同时 409 `310` + `no-store`
  （防止旧键缓存新内容），不带 `?v=` 时 `no-cache` + ETag。每个请求先 `stat` 一次该世界的 `world.json`，inode 或 mtime
  变化即重读 contentVersion（M03 以目录原子替换发布世界）。只服务 `worlds/<id>/` 内的普通文件：拒绝 `..`、隐藏路径，
  `.staging/`、`.trash/`、`.locks/`、`.status/`、`.geo-cache/` 因 id 正则与隐藏段规则不可达；`/worlds/_shared/env/**` 为跨世界共享资产。
- Range：单区间 `bytes=a-b`、`bytes=a-`、`bytes=-n` → 206 + `Content-Range` + `Content-Length`；不可满足 → 416 `309` +
  `Content-Range: bytes */size`；多区间不属于契约（返回整体 200）；禁止动态 Content-Encoding。
- 前端（`apps/web/dist`，生产模式）：`/assets/**` 哈希文件名 immutable；`/brand/**`、`/bench/**`、`/demo/**`（公开演示构建
  的落地页媒体，ADR-083）no-cache + ETag；
  `/models/{file}.glb`（构建时随前端复制的机型模型，`apps/web/public/models/`，FX-WEB1）no-cache + ETag；
  `/`、`/world/{id}`、`/worlds` 等前端路由回退到 `index.html`（no-cache）；`/api/**` 不回退（404 `305`）。
- 安全头（COOP、COEP、CORP、nosniff、Referrer-Policy）由 `middleware.AwrMiddleware` 统一添加。
文件读取在 anyio 线程中执行（默认 executor，M11 §6.2），不在事件循环内做同步大 I/O。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

import anyio
from fastapi import APIRouter, Request
from starlette.responses import Response

from .problem import problem

__all__ = ["RangeFileResponse", "WorldFiles", "build_static_router", "parse_range"]

WORLD_ID_RE = re.compile(r"^[a-z0-9-]{1,63}$")
IMMUTABLE = "public, max-age=31536000, immutable"
CHUNK = 256 * 1024
MIME = {".json": "application/json", ".geojson": "application/geo+json", ".bin": "application/octet-stream",
        ".f32": "application/octet-stream", ".awrv": "application/octet-stream", ".awsl": "application/octet-stream",
        ".u8": "application/octet-stream", ".npy": "application/octet-stream", ".glb": "model/gltf-binary",
        ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".html": "text/html; charset=utf-8",
        ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp",
        ".woff2": "font/woff2", ".woff": "font/woff", ".wasm": "application/wasm", ".txt": "text/plain; charset=utf-8",
        ".map": "application/json", ".webmanifest": "application/manifest+json", ".ico": "image/x-icon"}
SPA_RESERVED = ("api", "assets", "vehicles", "models", "demo", "build-info.json", "manifest.webmanifest")
ROOT_FILES = ("build-info.json", "manifest.webmanifest")  # D2.1（ADR-118；AWR-04 §15.8.7；d2-WP-16-to-M11 第 4 条）
MODEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
FILE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}\.glb$")


def mime_of(p: Path) -> str:
    return MIME.get(p.suffix.lower(), "application/octet-stream")


def parse_range(header: str | None, size: int) -> tuple[int, int] | str | None:
    """单区间解析：返回 (start, end_exclusive)；None 表示无 Range（或多区间，按整体返回）；"416" 表示不可满足。"""
    if not header:
        return None
    h = header.strip()
    if not h.lower().startswith("bytes="):
        return None
    spec = h[6:].strip()
    if "," in spec:
        return None
    a, _, b = spec.partition("-")
    # RFC 9110 的字节位置只有 ASCII 数字；int() 还接受下划线、正负号、空白和其他文字的数字，这里一概不认。
    if not all(x == "" or (x.isascii() and x.isdigit()) for x in (a, b)):
        return None
    try:
        if a == "":
            n = int(b)
            if n <= 0:
                return "416"
            return (max(0, size - n), size) if size > 0 else "416"
        start = int(a)
        end = int(b) + 1 if b != "" else size
    except ValueError:
        return None
    if start >= size or start < 0 or end <= start:
        return "416"
    return start, min(end, size)


class RangeFileResponse(Response):
    """流式文件响应（整体 200 或单区间 206）；分块在 anyio 线程读取。"""

    def __init__(self, path: Path, *, start: int, end: int, size: int, status: int, headers: dict[str, str],
                 media_type: str, head: bool = False) -> None:
        self.path = path
        self.start, self.end, self.size, self.head = start, end, size, head
        self.status_code = status
        self.media_type = media_type
        self.background = None
        h = dict(headers)
        h["accept-ranges"] = "bytes"
        h["content-length"] = str(end - start)
        if status == 206:
            h["content-range"] = f"bytes {start}-{end - 1}/{size}"
        self.init_headers(h)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})
        if self.head or self.end <= self.start:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return
        async with await anyio.open_file(self.path, "rb") as f:
            await f.seek(self.start)
            left = self.end - self.start
            while left > 0:
                chunk = await f.read(min(CHUNK, left))
                if not chunk:
                    break
                left -= len(chunk)
                await send({"type": "http.response.body", "body": chunk, "more_body": left > 0})
            if left > 0:
                await send({"type": "http.response.body", "body": b"", "more_body": False})


class WorldFiles:
    """每个世界 `world.json` 的 (inode, mtime, size) → contentVersion 与强 ETag 缓存。"""

    def __init__(self, worlds_dir: Path) -> None:
        self.root = Path(worlds_dir)
        self._cache: dict[str, tuple[tuple[int, int, int], str | None, str]] = {}

    def world_json(self, wid: str) -> tuple[os.stat_result, str | None, str] | None:
        p = self.root / wid / "world.json"
        try:
            st = os.stat(p)
        except OSError:
            self._cache.pop(wid, None)
            return None
        key = (st.st_ino, st.st_mtime_ns, st.st_size)
        hit = self._cache.get(wid)
        if hit is None or hit[0] != key:
            data = p.read_bytes()
            try:
                cv = str(json.loads(data).get("contentVersion"))
            except (ValueError, AttributeError):
                cv = None
            hit = (key, cv, '"' + hashlib.sha256(data).hexdigest() + '"')
            self._cache[wid] = hit
        return st, hit[1], hit[2]

    def content_version(self, wid: str) -> str | None:
        r = self.world_json(wid)
        return None if r is None else r[1]


def _safe_join(base: Path, rel: str) -> Path | None:
    parts = [x for x in rel.split("/") if x]
    if not parts or any(x in (".", "..") or x.startswith(".") or "\\" in x or "\0" in x for x in parts):
        return None
    p = base.joinpath(*parts)
    try:
        rp = p.resolve()
        if not str(rp).startswith(str(base.resolve()) + os.sep):
            return None
    except OSError:
        return None
    return p


def _etag_match(request: Request, etag: str) -> bool:
    inm = request.headers.get("if-none-match")
    if not inm:
        return False
    tags = [t.strip() for t in inm.split(",")]
    return "*" in tags or etag in tags or ("W/" + etag) in tags


def _file_response(request: Request, p: Path, st: os.stat_result, headers: dict[str, str]) -> Response:
    size = st.st_size
    head = request.method == "HEAD"
    rng = parse_range(request.headers.get("range"), size)
    if rng == "416":
        rid = getattr(request.state, "request_id", None)
        return problem(309, request_id=rid, detail={"size": size},
                       headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes",
                                "Cache-Control": "no-store"})
    if rng is None:
        return RangeFileResponse(p, start=0, end=size, size=size, status=200, headers=headers, media_type=mime_of(p),
                                 head=head)
    start, end = rng
    return RangeFileResponse(p, start=start, end=end, size=size, status=206, headers=headers, media_type=mime_of(p),
                             head=head)


def build_static_router(worlds_dir: Path, web_dist: Path | None) -> tuple[APIRouter, WorldFiles]:
    wf = WorldFiles(worlds_dir)
    r = APIRouter(include_in_schema=False)
    dist = Path(web_dist) if web_dist is not None else None

    @r.api_route("/worlds/_shared/{path:path}", methods=["GET", "HEAD"])
    async def shared(request: Request, path: str) -> Response:
        rid = getattr(request.state, "request_id", None)
        if not path.startswith("env/"):
            return problem(305, request_id=rid)
        p = _safe_join(wf.root / "_shared", path)
        st = await anyio.to_thread.run_sync(_stat_file, p) if p is not None else None
        if st is None:
            return problem(305, request_id=rid)
        etag = f'"shared-{st.st_size}-{st.st_mtime_ns}"'
        cc = IMMUTABLE if request.query_params.get("v") else "no-cache"
        if cc == "no-cache" and _etag_match(request, etag):
            return Response(status_code=304, headers={"ETag": etag, "Cache-Control": cc})
        return _file_response(request, p, st, {"ETag": etag, "Cache-Control": cc})

    @r.api_route("/worlds/{wid}/{path:path}", methods=["GET", "HEAD"])
    async def world_file(request: Request, wid: str, path: str) -> Response:
        rid = getattr(request.state, "request_id", None)
        if not WORLD_ID_RE.match(wid):
            return problem(305, request_id=rid)
        info = await anyio.to_thread.run_sync(wf.world_json, wid)
        if info is None:
            return problem(305, request_id=rid, detail={"world_id": wid})
        wj_stat, cv, wj_etag = info
        if path == "world.json":
            hdr = {"ETag": wj_etag, "Cache-Control": "no-cache"}
            if _etag_match(request, wj_etag):
                return Response(status_code=304, headers=hdr)
            return _file_response(request, wf.root / wid / "world.json", wj_stat, hdr)
        p = _safe_join(wf.root / wid, path)
        st = await anyio.to_thread.run_sync(_stat_file, p) if p is not None else None
        if st is None:
            return problem(305, request_id=rid, detail={"path": path})
        v = request.query_params.get("v")
        if v is not None:
            if v != cv:
                return problem(310, request_id=rid, detail={"requested": v, "current": cv},
                               headers={"Cache-Control": "no-store"})
            return _file_response(request, p, st, {"Cache-Control": IMMUTABLE, "ETag": f'"{cv}-{st.st_size}"'})
        etag = f'"{cv}-{st.st_size}-{st.st_mtime_ns}"'
        hdr = {"ETag": etag, "Cache-Control": "no-cache"}
        if _etag_match(request, etag):
            return Response(status_code=304, headers=hdr)
        return _file_response(request, p, st, hdr)

    if dist is not None:
        @r.api_route("/assets/{path:path}", methods=["GET", "HEAD"])
        async def assets(request: Request, path: str) -> Response:
            p = _safe_join(dist / "assets", path)
            st = await anyio.to_thread.run_sync(_stat_file, p) if p is not None else None
            if st is None:
                return problem(305, request_id=getattr(request.state, "request_id", None))
            return _file_response(request, p, st, {"Cache-Control": IMMUTABLE})

        for top in ("brand", "bench", "demo"):
            def make(top: str):
                async def pub(request: Request, path: str) -> Response:
                    p = _safe_join(dist / top, path)
                    st = await anyio.to_thread.run_sync(_stat_file, p) if p is not None else None
                    if st is None:
                        return problem(305, request_id=getattr(request.state, "request_id", None))
                    etag = f'"{st.st_size}-{st.st_mtime_ns}"'
                    if _etag_match(request, etag):
                        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "no-cache"})
                    return _file_response(request, p, st, {"ETag": etag, "Cache-Control": "no-cache"})
                return pub
            r.add_api_route(f"/{top}/{{path:path}}", make(top), methods=["GET", "HEAD"])

        for name in ROOT_FILES:
            def make_root(name: str):
                async def root_file(request: Request) -> Response:
                    p = dist / name
                    st = await anyio.to_thread.run_sync(_stat_file, p)
                    if st is None:
                        return problem(305, request_id=getattr(request.state, "request_id", None))
                    etag = f'"{st.st_size}-{st.st_mtime_ns}"'
                    if _etag_match(request, etag):
                        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "no-cache"})
                    return _file_response(request, p, st, {"ETag": etag, "Cache-Control": "no-cache"})
                return root_file
            r.add_api_route(f"/{name}", make_root(name), methods=["GET", "HEAD"])

        @r.api_route("/models/{file}", methods=["GET", "HEAD"])
        async def web_model(request: Request, file: str) -> Response:
            """前端构建自带的机型模型副本（`apps/web/public/models/*.glb` → `dist/models/`，由 `tools/vehicles/*.py --copy-web`
            生成；AWR-17 §5.1、M06-FR-035，FX-WEB1）：只暴露 `*.glb`，no-cache + ETag。此前该路径落到 SPA 回退，返回
            index.html，GLTFLoader 解析失败，P600 一律退回低模。"""
            rid = getattr(request.state, "request_id", None)
            if not FILE_RE.match(file):
                return problem(305, request_id=rid)
            p = _safe_join(dist / "models", file)
            st = await anyio.to_thread.run_sync(_stat_file, p) if p is not None else None
            if st is None:
                return problem(305, request_id=rid)
            etag = f'"{st.st_size}-{st.st_mtime_ns}"'
            if _etag_match(request, etag):
                return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "no-cache"})
            return _file_response(request, p, st, {"ETag": etag, "Cache-Control": "no-cache"})

    vehicles_dir = Path(os.environ.get("AWR_VEHICLES_DIR") or Path(__file__).resolve().parents[3] / "vehicles")

    @r.api_route("/vehicles/{model}/model/{file}", methods=["GET", "HEAD"])
    async def vehicle_model(request: Request, model: str, file: str) -> Response:
        """机型模型生成物（AWR-17 §5 第 536 行；M06-FR-035）：只暴露 `vehicles/<model>/model/*.glb`；带 `?v=` 时 immutable，
        否则 no-cache + ETag。INT-1 按 M06-to-M11 第 1 条代为实现（此前 hero 模型 404，P600 一律退回低模）。"""
        rid = getattr(request.state, "request_id", None)
        if not MODEL_RE.match(model) or not FILE_RE.match(file):
            return problem(305, request_id=rid)
        p = _safe_join(vehicles_dir / model / "model", file)
        st = await anyio.to_thread.run_sync(_stat_file, p) if p is not None else None
        if st is None:
            return problem(305, request_id=rid)
        etag = f'"{st.st_size}-{st.st_mtime_ns}"'
        if request.query_params.get("v"):
            return _file_response(request, p, st, {"Cache-Control": IMMUTABLE, "ETag": etag})
        if _etag_match(request, etag):
            return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "no-cache"})
        return _file_response(request, p, st, {"ETag": etag, "Cache-Control": "no-cache"})

    return r, wf


def spa_route(web_dist: Path | None):
    """SPA 回退：非 `/api`、`/assets`、`/vehicles`、`/models` 的 GET 返回 `index.html`（no-cache）；dist 不存在时 404 `305`。"""
    index = Path(web_dist) / "index.html" if web_dist is not None else None

    async def spa(request: Request, path: str = "") -> Response:
        first = path.split("/", 1)[0]
        rid = getattr(request.state, "request_id", None)
        if first in SPA_RESERVED or (first == "worlds" and "/" in path.strip("/")):
            return problem(305, request_id=rid)
        if request.method not in ("GET", "HEAD"):
            return problem(306, status=405, request_id=rid)
        if index is None:
            return problem(305, request_id=rid, detail={"why": "WEB_DIST_MISSING"})
        st = await anyio.to_thread.run_sync(_stat_file, index)
        if st is None:
            return problem(305, request_id=rid, detail={"why": "WEB_DIST_MISSING"})
        return _file_response(request, index, st, {"Cache-Control": "no-cache"})

    return spa


def _stat_file(p: Path | None) -> os.stat_result | None:
    if p is None:
        return None
    try:
        st = os.stat(p)
    except OSError:
        return None
    return st if stat.S_ISREG(st.st_mode) else None
