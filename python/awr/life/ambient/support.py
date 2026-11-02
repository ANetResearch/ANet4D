"""A window of the measured support map (awr.life.civil.surface, schema awr.life.support.v1) over the stage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from awr.life.common.schema import schema_id


@dataclass
class Support:
    x0: float  # world ENU of the lower-left corner of cell (0, 0)
    y0: float
    cell_m: float
    walk_person: np.ndarray  # [row = y, col = x] cells a person can stand on
    walk_vehicle: np.ndarray  # cells a car can stand on
    ground: np.ndarray  # measured ground height (m)

    @classmethod
    def load(cls, root, bbox, margin_m=30.0):
        """The support map at `root` (a awr.life.support.v1 directory, or an .npz window written by `save`) cut to the
        stage `bbox` (x0, y0, x1, y1) and `margin_m` around it."""
        root = Path(root)
        if root.suffix == ".npz":
            d = np.load(root)
            full = cls(
                float(d["x0"]), float(d["y0"]), float(d["cell_m"]), d["walk_person"], d["walk_vehicle"], d["ground"]
            )
        else:
            meta = json.loads((root / "meta.json").read_text())
            if schema_id(meta.get("schema")) != "awr.life.support.v1":
                raise ValueError("Unvalidated support map")
            full = cls(
                float(meta["x0"]),
                float(meta["y0"]),
                float(meta.get("cell_m", 1.0)),
                *(np.load(root / f"{k}.npy", mmap_mode="r") for k in ("walk_person", "walk_vehicle", "ground")),
            )
        return full.window((bbox[0] - margin_m, bbox[1] - margin_m, bbox[2] + margin_m, bbox[3] + margin_m))

    def window(self, bbox):
        c = self.cell_m
        i0, j0 = max(int((bbox[0] - self.x0) // c), 0), max(int((bbox[1] - self.y0) // c), 0)
        i1 = min(int((bbox[2] - self.x0) // c) + 1, self.walk_person.shape[1])
        j1 = min(int((bbox[3] - self.y0) // c) + 1, self.walk_person.shape[0])

        def cut(a):
            return np.array(a[j0:j1, i0:i1])

        return Support(
            self.x0 + i0 * c, self.y0 + j0 * c, c, cut(self.walk_person), cut(self.walk_vehicle), cut(self.ground)
        )

    def save(self, path):
        np.savez_compressed(
            path,
            x0=self.x0,
            y0=self.y0,
            cell_m=self.cell_m,
            walk_person=self.walk_person,
            walk_vehicle=self.walk_vehicle,
            ground=self.ground,
        )

    def cells(self, xy):
        p = np.asarray(xy, float).reshape(-1, 2)
        i = np.floor((p[:, 0] - self.x0) / self.cell_m).astype(int)
        j = np.floor((p[:, 1] - self.y0) / self.cell_m).astype(int)
        inside = (i >= 0) & (j >= 0) & (i < self.walk_person.shape[1]) & (j < self.walk_person.shape[0])
        return np.clip(i, 0, self.walk_person.shape[1] - 1), np.clip(j, 0, self.walk_person.shape[0] - 1), inside

    def supported(self, xy, profile="vehicle"):
        i, j, inside = self.cells(xy)
        mask = self.walk_vehicle if profile == "vehicle" else self.walk_person
        return inside & mask[j, i]
