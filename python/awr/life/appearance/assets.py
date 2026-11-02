"""Verified real-mesh assets shared by RGB, depth and instance rendering."""

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from threading import RLock
from weakref import WeakValueDictionary

import numpy as np
from PIL import Image

ROOT = Path(os.environ.get("AWR_LIFE_ASSETS", Path(__file__).resolve().parent / "catalog"))


@lru_cache(maxsize=32)
def load(asset, expected_sha=None):
    entries = json.loads((ROOT / "catalog.json").read_text())["assets"]
    spec = next((x for x in entries if x["id"] == asset), None)
    if expected_sha and (spec is None or spec["npz_sha256"] != expected_sha):
        spec = json.loads((ROOT / "history" / (expected_sha + ".json")).read_text())
        if spec["id"] != asset:
            raise ValueError("Asset identity mismatch")
    if spec is None:
        raise ValueError(f"Unregistered asset: {asset}")
    paths = {ext: ROOT / "compiled" / (spec[ext + "_sha256"] + "." + ext) for ext in ("npz", "png")}
    paths = {ext: p if p.exists() else ROOT / "compiled" / (asset + "." + ext) for ext, p in paths.items()}
    for ext, p in paths.items():
        if hashlib.sha256(p.read_bytes()).hexdigest() != spec[ext + "_sha256"]:
            raise ValueError(f"Asset integrity error: {p}")
    with np.load(paths["npz"]) as f:
        data = {k: f[k] for k in f.files}
    return spec, data, Image.open(paths["png"]).convert("RGBA")


_live_clothing_atlases = WeakValueDictionary()
_clothing_lock = RLock()


@lru_cache(maxsize=8)
def _clothing_atlas(asset, expected_sha):
    """Reuse atlases held by live meshes, plus at most eight unused cached ones.

    A roster with more than eight human assets previously evicted an atlas even
    while existing meshes still held it, repeatedly making full-resolution CPU
    copies. Weak ownership preserves sharing without pinning departed rosters.
    """
    from awr.life.appearance.identity import neutral_clothing_atlas

    key = (asset, expected_sha)
    with _clothing_lock:
        atlas = _live_clothing_atlases.get(key)
        if atlas is None:
            atlas = neutral_clothing_atlas(load(asset, expected_sha)[2])
            _live_clothing_atlases[key] = atlas
        return atlas


class Mesh:
    def __init__(self, actor, *, wheel_partition=None):
        self.actor = actor
        self.spec, self.data, self.image = load(
            actor["appearance"]["asset"].split("@")[0], actor["appearance"].get("asset_sha256")
        )
        from awr.life.appearance.identity import texture_key

        self.texture_key = (self.spec["npz_sha256"], self.spec["png_sha256"], texture_key(actor["appearance"]))
        self.wheel_rig = None
        if wheel_partition is not None:
            from awr.life.appearance.wheels import WheelRig

            if self.spec["kind"] != "vehicle":
                raise ValueError("Wheel rig requires vehicle asset")
            self.wheel_rig = WheelRig(self.spec, self.data, wheel_partition)
        if actor["appearance"].get("top_rgb") and self.spec["kind"] == "human":
            from awr.life.appearance.identity import clothing_vertices

            self.data = clothing_vertices(self.data, actor["appearance"])
            self.image = _clothing_atlas(self.spec["id"], self.spec["npz_sha256"])
            # Outfit RGB is in each actor's vertex buffer, not these atlas pixels.
            self.texture_key = ("neutral-outfit-v1", self.spec["npz_sha256"], self.spec["png_sha256"])
        if actor["appearance"].get("body_rgb") and self.spec["kind"] == "vehicle":
            from awr.life.appearance.identity import apply_paint

            self.data, self.image = apply_paint(self.data, self.image, actor["appearance"])
        if actor["appearance"].get("plate_text"):
            if self.spec["kind"] != "vehicle":
                raise ValueError("License plates require a verified vehicle")
            from awr.life.appearance.identity import add_plates

            self.data, self.image = add_plates(self.data, self.image, actor["appearance"]["plate_text"])
        self.p = self.data["p"].copy()
        self.scale = np.ones(3)
        if self.spec["kind"] == "human":
            self.scale[:] = actor["size_mm"][2] / 1000 / self.spec["dimensions_m"][2]
        # Vehicles keep authored dimensions; simulator must use catalog sizes.
        else:
            expected = np.asarray(self.spec["dimensions_m"])
            if np.max(np.abs(np.asarray(actor["size_mm"]) / 1000 - expected)) > 0.02:
                raise ValueError("Vehicle collision dimensions differ from rendered asset")
        self.last = None
        self._wheels = None
        self._human_pose = None
        # The accelerated kernel mirrors NumPy 2's float64 phase promotion.
        # Keep the original implementation for other array/scalar semantics.
        if (
            "poses" in self.data
            and "wheel_ids" not in self.data
            and len(self.p) % 3 == 0
            and all(self.data[k].dtype == np.float32 for k in ("p", "poses", "c", "uv"))
            and (np.zeros(1, np.float32) * np.float64(0.5)).dtype == np.float64
        ):
            from awr.life.appearance.human_pose import warmup

            warmup()
            self._human_pose = tuple(np.ascontiguousarray(self.data[k]) for k in ("poses", "c", "uv"))

    def vertices(self, distance=0.0, speed=0.0, *, wheel_pose=None):
        if self.wheel_rig is not None and wheel_pose is None:
            if distance or speed:
                raise ValueError("Articulated asset requires explicit wheel pose")
            return np.hstack([self.p * self.scale, self.data["n"], self.data["c"], self.data["uv"]]).astype("<f4")
        if wheel_pose is not None:
            if self.wheel_rig is None:
                raise ValueError("Explicit wheel pose requires verified partition")
            # Start from authored geometry, not legacy distance-based spin.
            out = np.hstack([self.p * self.scale, self.data["n"], self.data["c"], self.data["uv"]]).astype("<f4")
            for first, rows in self.wheel_rig.updates(self, wheel_pose):
                out[first : first + len(rows)] = rows
            return out
        p = self.p
        n = self.data["n"]
        if "poses" in self.data and speed > 0.03:
            poses = self.data["poses"]
            q = (distance / self.scale[0] / self.spec["stride_m"] * len(poses)) % len(poses)
            k = int(q)
            u = q - k
            if self._human_pose is not None:
                from awr.life.appearance.human_pose import vertices

                poses, color, uv = self._human_pose
                blend = min(speed / 0.5, 1.0)
                rest_float64 = np.result_type(self.p, 1 - blend).itemsize > 4
                return vertices(
                    self.p,
                    poses[k],
                    poses[(k + 1) % len(poses)],
                    color,
                    uv,
                    self.scale,
                    float(u),
                    float(blend),
                    rest_float64,
                )
            posed = poses[k] * (1 - u) + poses[(k + 1) % len(poses)] * u
            blend = min(speed / 0.5, 1.0)
            p = p * (1 - blend) + posed * blend
            f = p.reshape(-1, 3, 3)
            face = np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0])
            face /= np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-12)
            n = np.repeat(face, 3, axis=0)
        if "wheel_ids" in self.data:
            p = p.copy()
            n = n.copy()
            for j, (center, radius) in enumerate(
                zip(self.data["wheel_centers"], self.data["wheel_radii"], strict=False)
            ):
                ids = self.data["wheel_ids"] == j
                theta = distance / max(float(radius), 0.01)
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                p[ids] = (p[ids] - center) @ R.T + center
                n[ids] = n[ids] @ R.T
        return np.hstack([p * self.scale, n, self.data["c"], self.data["uv"]]).astype("<f4")

    def updates(self, distance=0.0, speed=0.0, *, wheel_pose=None):
        """Rows of ``vertices(distance, speed)`` that can differ from ``vertices()``, as
        ``(first_vertex, rows)`` runs. A rigid body only re-sends its wheel vertices; a body
        without wheels or poses never changes. Rows are computed by the same operations on
        the same inputs as ``vertices``, so a buffer built from ``vertices()`` plus these runs
        equals ``vertices(distance, speed)`` bit for bit."""
        if self.wheel_rig is not None and wheel_pose is None:
            raise ValueError("Articulated asset update requires explicit wheel pose")
        if wheel_pose is not None:
            if self.wheel_rig is None:
                raise ValueError("Explicit wheel pose requires verified partition")
            return self.wheel_rig.updates(self, wheel_pose)
        if "poses" in self.data:
            return [(0, self.vertices(distance, speed))]
        if "wheel_ids" not in self.data:
            return []
        if self._wheels is None:
            ids = [np.flatnonzero(self.data["wheel_ids"] == j) for j in range(len(self.data["wheel_centers"]))]
            rows = np.unique(np.concatenate(ids)) if ids else np.zeros(0, np.int64)
            out = np.zeros((len(rows), 11), "<f4")
            out[:, 6:9] = self.data["c"][rows]
            out[:, 9:11] = self.data["uv"][rows]
            cut = np.flatnonzero(np.diff(rows) > 1) + 1
            runs = list(zip(np.r_[0, cut].astype(int), np.r_[cut, len(rows)].astype(int), strict=False))
            self._wheels = (ids, [np.searchsorted(rows, i) for i in ids], rows, out, runs)
        ids, slots, rows, out, runs = self._wheels
        if not len(rows):
            return []
        p, n = self.p, self.data["n"]
        for j, (center, radius) in enumerate(zip(self.data["wheel_centers"], self.data["wheel_radii"], strict=False)):
            theta = distance / max(float(radius), 0.01)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            out[slots[j], 0:3] = ((p[ids[j]] - center) @ R.T + center).astype(p.dtype) * self.scale
            out[slots[j], 3:6] = (n[ids[j]] @ R.T).astype(n.dtype)
        return [(int(rows[a]), out[a:b]) for a, b in runs]


def assign(actor, ordinal):
    """Choose persistent geometry deterministically, before collision dimensions are used."""
    if actor["class"] == "person":
        key = f"human-{ordinal % 4 + 1}"
    elif actor["class"] == "vehicle":
        vt = actor.get("vtype", "sedan")
        key = {
            "car": "sedan",
            "sedan": "sedan",
            "suv": "suv",
            "van": "van",
            "bus": "bus",
            "truck": "truck",
            "pickup": "pickup",
            "hatchback": "hatchback",
        }.get(vt)
        if key is None:
            raise ValueError(f"No verified asset for vehicle type {vt}; do not substitute a box")
    else:
        raise ValueError(f"No verified asset for class {actor['class']}")
    spec, _, _ = load(key)
    actor["appearance"] = {**actor["appearance"], "asset": key + "@1", "asset_sha256": spec["npz_sha256"]}
    if actor["class"] == "vehicle":
        actor["size_mm"] = [round(v * 1000) for v in spec["dimensions_m"]]
    return actor
