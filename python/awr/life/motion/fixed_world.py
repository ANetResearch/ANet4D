"""Frozen, explicitly registered obstacles in a shared native activity world.

A fixed collider conservatively encloses supplied rendered world vertices. It
never substitutes those box dimensions for the rendered asset, moves a parked
actor or pretends a kinematic obstacle is driveable. Ground remains separate.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from awr.life.civil.activity.driving import Body


@dataclass(frozen=True)
class FixedCollider:
    name: str
    body: Body
    base_z: float
    vertices_sha256: str

    @classmethod
    def enclose(cls, name, vertices, anchor_xy, yaw, *, padding=0.001):
        p = np.asarray(vertices, float)
        a = np.asarray(anchor_xy, float)
        if (
            not isinstance(name, str)
            or not name
            or p.ndim != 2
            or p.shape[1:] != (3,)
            or not len(p)
            or a.shape != (2,)
            or not np.isfinite(p).all()
            or not np.isfinite(a).all()
            or not math.isfinite(yaw)
            or not 0 <= padding <= 0.02
        ):
            raise ValueError("Finite rendered vertices, original anchor and bounded margin required")
        c, s = math.cos(yaw), math.sin(yaw)
        local = (p[:, :2] - a) @ np.array([[c, -s], [s, c]])
        half = np.max(abs(local), axis=0) + padding
        low = float(p[:, 2].min() - padding)
        size = tuple(np.r_[half * 2, float(p[:, 2].max() + padding - low)])
        return cls(name, Body((*a, float(yaw)), size), low, hashlib.sha256(p.astype("<f8").tobytes()).hexdigest())


class FixedWorld:
    def __init__(self, world, colliders):
        entries = tuple(colliders)
        if not entries or not all(isinstance(c, FixedCollider) for c in entries):
            raise ValueError("Explicit fixed rendered collider roster required")
        if len({c.name for c in entries}) != len(entries):
            raise ValueError("Duplicate fixed actor")
        self.world = world
        self.colliders = entries
        self.indices = []
        self.poses = []
        for c in entries:
            pose = np.array([*c.body.pose[:2], c.base_z, c.body.pose[2]])
            index = world.add_kinematic(c.name, c.body.size, pose, max_speed_mps=0.0, max_yaw_rate=0.0) - 1000000
            self.indices.append(index)
            self.poses.append(tuple(pose))
        self.bodies = MappingProxyType({c.name: c.body for c in entries})
        self.identity = hashlib.sha256(
            json.dumps(
                [
                    dict(
                        name=c.name,
                        pose=c.body.pose,
                        size=c.body.size,
                        base_z=c.base_z,
                        vertices_sha256=c.vertices_sha256,
                    )
                    for c in entries
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.verify()

    def verify(self):
        # The underlying world owns these poses; no external caller may move
        # one while retaining its old planning/interaction certificate.
        for c, i, pose in zip(self.colliders, self.indices, self.poses, strict=False):
            if (
                self.world.kinematics[i] != c.name
                or self.world._limits[i] != (0.0, 0.0)
                or not np.array_equal(self.world._kinematic_pose[i], pose)
            ):
                raise ValueError("Fixed native obstacle changed: " + c.name)
        return self.bodies
