"""Explicit goal ownership over native DetourCrowd; no activity selection here."""

from __future__ import annotations

import ctypes as C
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CrowdState:
    actor: str
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    goal: tuple[float, float, float] | None
    motion_state: int
    target_state: int
    partial: bool

    @property
    def speed(self):
        return math.hypot(*self.velocity[:2])


@dataclass(frozen=True)
class CrowdSteering:
    desired_velocity: tuple[float, float, float]
    avoidance_velocity: tuple[float, float, float]
    next_corner: tuple[float, float, float] | None
    speed_limit: float
    nominal_speed: float


class NativeCrowd:
    """Single-owner synchronous component; callers advance world time explicitly.

    It never advances on read or waits for model reasoning. The world runner must
    schedule step independently of agent decisions. Native file and geometry
    admission are external preconditions; this adapter is not a scene certificate.
    """

    def __init__(self, library, mesh, *, capacity=256, radius=0.3, snap_xy=0.15, snap_z=0.2):
        self.library, self.mesh = Path(library).resolve(), Path(mesh).resolve()
        self.provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.library, self.mesh)}
        self._lib = C.CDLL(str(self.library))
        lib = self._lib
        fptr = C.POINTER(C.c_float)
        iptr = C.POINTER(C.c_int)
        h = C.c_void_p
        specs = {
            "abi": ([], C.c_int),
            "error": ([], C.c_char_p),
            "create": ([C.c_char_p, C.c_int, C.c_float, C.c_float, C.c_float], h),
            "destroy": ([h], None),
            "project": ([h, fptr, fptr], C.c_int),
            "path": ([h, fptr, fptr, fptr, C.c_int], C.c_int),
            "add": ([h, fptr, C.c_float, C.c_float], C.c_int),
            "request": ([h, C.c_int, fptr], C.c_int),
            "stop": ([h, C.c_int], C.c_int),
            "remove": ([h, C.c_int], C.c_int),
            "step": ([h, C.c_float], C.c_int),
            "state": ([h, C.c_int, fptr, iptr], C.c_int),
        }
        for name, (args, result) in specs.items():
            fn = getattr(lib, "rb_crowd_" + name)
            fn.argtypes = args
            fn.restype = result
        if lib.rb_crowd_abi() != 1:
            raise RuntimeError("Unsupported crowd ABI")
        self.motion_capability = hasattr(lib, "rb_crowd_motion_abi")
        if self.motion_capability:
            lib.rb_crowd_motion_abi.argtypes = []
            lib.rb_crowd_motion_abi.restype = C.c_int
            if lib.rb_crowd_motion_abi() != 1:
                raise RuntimeError("Unsupported motion capability ABI")
            lib.rb_crowd_speed_limit.argtypes = [h, C.c_int, C.c_float]
            lib.rb_crowd_speed_limit.restype = C.c_int
            lib.rb_crowd_steering.argtypes = [h, C.c_int, fptr, iptr]
            lib.rb_crowd_steering.restype = C.c_int
        self._handle = lib.rb_crowd_create(str(self.mesh).encode(), capacity, radius, snap_xy, snap_z)
        if not self._handle:
            raise RuntimeError(lib.rb_crowd_error().decode())
        self.actors = {}
        self.time = 0.0
        self.radius = float(radius)

    def _call(self, name, *args):
        if not self._handle:
            raise RuntimeError("Native crowd is closed")
        result = getattr(self._lib, "rb_crowd_" + name)(self._handle, *args)
        if result < 0:
            raise RuntimeError(self._lib.rb_crowd_error().decode())
        return result

    @staticmethod
    def _point(point):
        if len(point) != 3 or not all(math.isfinite(float(x)) for x in point):
            raise ValueError("Finite ENU xyz required")
        return (C.c_float * 3)(*point)

    def project(self, point):
        result = (C.c_float * 3)()
        return tuple(result) if self._call("project", self._point(point), result) else None

    def path(self, start, end):
        points = (C.c_float * (8192 * 3))()
        n = self._call("path", self._point(start), self._point(end), points, 8192)
        return tuple(tuple(points[3 * i : 3 * i + 3]) for i in range(n)) if n else None

    def add(self, actor, position, *, speed=1.3, acceleration=2.0):
        if not isinstance(actor, str) or not actor or actor in self.actors:
            raise ValueError("New nonempty actor identity required")
        idx = self._call("add", self._point(position), speed, acceleration)
        self.actors[actor] = idx
        return self.state(actor)

    def request(self, actor, goal):
        return bool(self._call("request", self.actors[actor], self._point(goal)))

    def stop(self, actor):
        self._call("stop", self.actors[actor])

    def speed_limit(self, actor, value):
        if not self.motion_capability:
            raise RuntimeError("Native bridge lacks motion capability")
        if not math.isfinite(value) or value < 0:
            raise ValueError("Finite nonnegative speed limit required")
        self._call("speed_limit", self.actors[actor], value)

    def steering(self, actor):
        if not self.motion_capability:
            raise RuntimeError("Native bridge lacks motion capability")
        values = (C.c_float * 11)()
        corners = C.c_int()
        self._call("steering", self.actors[actor], values, C.byref(corners))
        return CrowdSteering(
            tuple(values[:3]), tuple(values[3:6]), tuple(values[6:9]) if corners.value else None, values[9], values[10]
        )

    def remove(self, actor):
        self._call("remove", self.actors[actor])
        del self.actors[actor]

    def state(self, actor):
        values = (C.c_float * 9)()
        flags = (C.c_int * 4)()
        self._call("state", self.actors[actor], values, flags)
        return CrowdState(
            actor,
            tuple(values[:3]),
            tuple(values[3:6]),
            tuple(values[6:9]) if flags[3] else None,
            flags[0],
            flags[1],
            bool(flags[2]),
        )

    def step(self, dt):
        if not math.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("Step must be in (0,0.1] seconds")
        self._call("step", dt)
        self.time += dt

    def close(self):
        if self._handle:
            self._lib.rb_crowd_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
