"""R04 `GET /api/worlds`、R05 `GET /api/worlds/{id}`（AWR-17 §4.1 第 6 条、§4.3.2；M11-FR-081；M03 Catalog）。

列表响应 `{items, next_cursor}`（分页：`limit` 默认 100、最大 1000，`cursor` 为不透明偏移），items 字段为 snake_case：
id、name、name_zh、status、content_version、scale_status、anchor_kind、georeferenced、points、bytes、roots、octree_bytes、
node_count、levels_points、max_height_m、first_screen、world_json_url、thumbnail_url、default_scenario_id、in_use。
详情另含 coordinate_sha256、bounds_m、camera_home、layers、qa。`in_use` = 当前运行世界。
配置了世界白名单（`AWR_WORLDS_ALLOW`，公开模式只有 synthcity，ADR-082）时列表只含白名单内的世界，其余世界的详情 404 `305`。
Catalog 读盘在 anyio 线程中执行（冷读约 10 ms，此后 1 s 缓存），不占用事件循环。
"""

from __future__ import annotations

import base64
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, Query, Request

from ..deps import Viewer, ctx_for
from ..problem import ApiProblem
from ..static import WORLD_ID_RE

router = APIRouter(prefix="/api/worlds", tags=["worlds"])

LIST_FIELDS = ("id", "name", "name_zh", "status", "content_version", "scale_status", "anchor_kind", "georeferenced",
               "points", "bytes", "roots", "octree_bytes", "node_count", "levels_points", "max_height_m", "first_screen",
               "world_json_url", "thumbnail_url", "default_scenario_id", "in_use")
DETAIL_EXTRA = ("coordinate_sha256", "bounds_m", "camera_home", "layers", "qa")


def _item(w: Any, current: str, fields: tuple[str, ...]) -> dict[str, Any]:
    d = w.to_json()
    d["in_use"] = w.id == current
    return {k: d.get(k) for k in fields}


def _cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(f"o{offset}".encode()).decode().rstrip("=")


def _offset(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode()
        if raw.startswith("o") and raw[1:].isascii() and raw[1:].isdigit():
            return int(raw[1:])
    except (ValueError, UnicodeDecodeError):
        pass
    raise ApiProblem(300, status=400, detail={"field": "cursor"})


@router.get("")
async def list_worlds(request: Request, _p: Viewer, limit: Annotated[int, Query(ge=1, le=1000)] = 100,
                      cursor: str | None = None) -> dict[str, Any]:
    ctx = ctx_for(request)
    off = _offset(cursor)
    worlds = [w for w in await anyio.to_thread.run_sync(ctx.catalog.list) if ctx.settings.world_allowed(w.id)]
    page = worlds[off:off + limit]
    nxt = _cursor(off + limit) if off + limit < len(worlds) else None
    return {"items": [_item(w, ctx.settings.world_id, LIST_FIELDS) for w in page], "next_cursor": nxt}


@router.get("/{world_id}")
async def get_world(world_id: str, request: Request, _p: Viewer) -> dict[str, Any]:
    ctx = ctx_for(request)
    if not WORLD_ID_RE.match(world_id) or not ctx.settings.world_allowed(world_id):
        raise ApiProblem(305, status=404, detail={"world_id": world_id})
    w = await anyio.to_thread.run_sync(ctx.catalog.get, world_id)
    if w is None:
        raise ApiProblem(305, status=404, detail={"world_id": world_id})
    return _item(w, ctx.settings.world_id, LIST_FIELDS + DETAIL_EXTRA)
