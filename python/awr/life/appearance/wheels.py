"""Explicit wheel articulation for verified assets, separate from legacy distance spin.

Hub positions and rotations are in the authored vehicle frame (+X forward, +Z up).
This changes render geometry only; it never fits terrain or modifies a body pose.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from awr.life.common.schema import schema_id


def positions_digest(p):
    return hashlib.sha256(np.asarray(p, "<f4").tobytes()).hexdigest()


@dataclass(frozen=True, eq=False)
class WheelPose:
    hubs: np.ndarray
    rotations: np.ndarray

    def __post_init__(self):
        h = np.array(self.hubs, dtype=float, copy=True)
        r = np.array(self.rotations, dtype=float, copy=True)
        if h.ndim != 2 or h.shape[1:] != (3,) or r.shape != (len(h), 3, 3):
            raise ValueError("Invalid wheel pose shape")
        if not np.isfinite(h).all() or not np.isfinite(r).all():
            raise ValueError("Non-finite wheel pose")
        if not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-10, rtol=0) or not np.allclose(
            np.linalg.det(r), 1.0, atol=1e-10, rtol=0
        ):
            raise ValueError("Wheel rotations must be proper orthonormal matrices")
        h.flags.writeable = r.flags.writeable = False
        object.__setattr__(self, "hubs", h)
        object.__setattr__(self, "rotations", r)

    @property
    def key(self):
        return (self.hubs.tobytes(), self.rotations.tobytes())


class WheelRig:
    """Source-bound partition; disjoint complete triangles, never nearest-vertex skinning."""

    def __init__(self, spec, data, partition):
        if (
            schema_id(partition.get("schema")) != "awr.life.wheel_partition.v1"
            or partition.get("asset_sha256") != spec["npz_sha256"]
            or partition.get("positions_sha256") != positions_digest(data["p"])
        ):
            raise ValueError("Wheel partition does not match asset")
        self.centers = np.array(data["wheel_centers"], float, copy=True)
        ids = np.full(len(data["p"]), -1, np.int16)
        for row in partition["runs"]:
            start, count, wheel = row
            if (
                any(type(x) is not int for x in row)
                or start < 0
                or count <= 0
                or start % 3
                or count % 3
                or start + count > len(ids)
                or not 0 <= wheel < len(self.centers)
                or (ids[start : start + count] >= 0).any()
            ):
                raise ValueError("Invalid or overlapping wheel partition")
            ids[start : start + count] = wheel
        old = data["wheel_ids"]
        if not np.array_equal(ids[old >= 0], old[old >= 0]):
            raise ValueError("Partition drops or relabels authored wheel vertices")
        self.ids = [np.flatnonzero(ids == j) for j in range(len(self.centers))]
        if any(not len(i) for i in self.ids):
            raise ValueError("Empty wheel partition")
        self.rows = np.flatnonzero(ids >= 0)
        cut = np.flatnonzero(np.diff(self.rows) > 1) + 1
        self.runs = list(zip(np.r_[0, cut], np.r_[cut, len(self.rows)], strict=False))
        self.slots = [np.searchsorted(self.rows, i) for i in self.ids]

    def updates(self, mesh, pose):
        if not isinstance(pose, WheelPose) or len(pose.hubs) != len(self.centers):
            raise ValueError("Explicit matching WheelPose required")
        data = mesh.data
        out = np.empty((len(self.rows), 11), "<f4")
        out[:, 6:9] = data["c"][self.rows]
        out[:, 9:11] = data["uv"][self.rows]
        for j, (ids, slots) in enumerate(zip(self.ids, self.slots, strict=False)):
            out[slots, :3] = ((mesh.p[ids] - self.centers[j]) @ pose.rotations[j].T + pose.hubs[j]) * mesh.scale
            out[slots, 3:6] = data["n"][ids] @ pose.rotations[j].T
        return [(int(self.rows[a]), out[a:b]) for a, b in self.runs]
