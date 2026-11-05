"""公开静态地形：技能执行器与像素投影用的地面高程 `terrain(x, y) -> z`，永远不是演员真值。

`GridTerrain` 读 `f32-grid@1` 栅格（世界包的 `geometry/terrain/dtm_10m.json`，或同格式的派生层，如 CHIMERA 由世界包 2 m 表面模型形态学
开运算得到的 `dtm_2m`）：元数据 `{href, width, height, originXY, cellM, method?}`，数据为小端 float32、行序自南向北；双线性插值，
越界或任一角点为 NaN 时抛 `ValueError`（文案面向外部智能体：目的地在公开地形图外 / 高程未知）。
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import numpy as np

__all__ = ["DTM_10M", "GridTerrain", "Terrain"]

DTM_10M = "geometry/terrain/dtm_10m.json"


class Terrain(Protocol):
    def __call__(self, x: float, y: float) -> float: ...


class GridTerrain:
    def __init__(self, meta_path: Path | str) -> None:
        meta_path = Path(meta_path)
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        self.source = str(meta_path)
        self.method = m.get("method")
        self.z = np.fromfile(meta_path.parent / m["href"], "<f4").reshape(int(m["height"]), int(m["width"]))
        self.origin = np.asarray(m["originXY"], float)
        self.cell = float(m["cellM"])

    @classmethod
    def open(cls, world_dir: Path | str, candidates: Iterable[Path | str] = ()) -> GridTerrain:
        """第一个存在的 `candidates`（派生层）；都没有时用世界包的 10 m DTM。"""
        for c in candidates:
            if Path(c).exists():
                return cls(c)
        return cls(Path(world_dir) / DTM_10M)

    def __call__(self, x: float, y: float) -> float:
        gx, gy = (np.array([x, y]) - self.origin) / self.cell - 0.5
        ix, iy = math.floor(gx), math.floor(gy)
        if not (0 <= ix < self.z.shape[1] - 1 and 0 <= iy < self.z.shape[0] - 1):
            raise ValueError("Destination is outside the public terrain map")
        v = self.z[iy:iy + 2, ix:ix + 2]
        if not np.isfinite(v).all():
            raise ValueError("Destination has unknown public terrain elevation")
        u, w = gx - ix, gy - iy
        return float((1 - u) * (1 - w) * v[0, 0] + u * (1 - w) * v[0, 1] + (1 - u) * w * v[1, 0] + u * w * v[1, 1])
