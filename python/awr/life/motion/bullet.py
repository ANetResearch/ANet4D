"""Optional persistent Bullet fleet world; independent of activity policy.

All vehicles share one world clock and one collision world. The caller supplies
bounded controls and external kinematic trajectories. This module never creates
itineraries, fits heights, snaps poses, or treats a completed step as admission.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from awr.life.appearance.wheels import WheelPose

WIDTH = 43
BULLET_REVISION = "2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5"
D = C.POINTER(C.c_double)


class _Config(C.Structure):
    _fields_ = [
        ("body_lo", C.c_double * 3),
        ("body_hi", C.c_double * 3),
        ("mass_kg", C.c_double),
        ("com_z", C.c_double),
        ("base", C.c_double * 3),
        ("quaternion_xyzw", C.c_double * 4),
        *[
            (name, C.c_double)
            for name in [
                "suspension_rest",
                "suspension_loaded",
                "suspension_travel",
                "stiffness",
                "friction_slip",
                "max_steer",
                "steer_rate",
                "max_force",
                "max_brake_impulse",
            ]
        ],
        ("wheels", (C.c_double * 4) * 4),
    ]


def _array(value, shape):
    a = np.array(value, dtype=np.float64, order="C", copy=True)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"Finite array of shape {shape} required")
    return a


def _ptr(a):
    return a.ctypes.data_as(D)


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@dataclass(frozen=True)
class VehicleProfile:
    body_lo: np.ndarray
    body_hi: np.ndarray
    wheels: np.ndarray  # four rows of x/y/z/radius, front wheels have positive x
    mass_kg: float = 1250.0
    com_z: float = 0.6
    suspension_rest: float = 0.35
    suspension_loaded: float = 0.25
    suspension_travel: float = 0.1
    stiffness: float = 30.0
    friction_slip: float = 0.9
    max_steer: float = float(np.deg2rad(32.0))
    steer_rate: float = 0.6
    max_force: float = 2500.0
    max_brake_impulse: float = 100.0
    body_hull: object = None

    def __post_init__(self):
        for name, shape in [("body_lo", (3,)), ("body_hi", (3,)), ("wheels", (4, 4))]:
            a = _array(getattr(self, name), shape)
            a.flags.writeable = False
            object.__setattr__(self, name, a)
        if (
            np.any(self.body_hi <= self.body_lo)
            or np.any(self.wheels[:, 3] <= 0)
            or not (self.wheels[:, 0] > 0).any()
            or not (self.wheels[:, 0] < 0).any()
        ):
            raise ValueError("Invalid four-wheel body profile")
        for name, _ in _Config._fields_:
            if name in ["body_lo", "body_hi", "wheels", "base", "quaternion_xyzw"]:
                continue
            if not np.isfinite(getattr(self, name)):
                raise ValueError("Non-finite vehicle profile")
        if (
            self.mass_kg <= 0
            or self.suspension_loaded <= 0
            or not 0 < self.suspension_travel < self.suspension_rest
            or self.stiffness <= 0
            or self.friction_slip <= 0
            or not 0 < self.max_steer < 1.5
            or self.steer_rate <= 0
            or self.max_force <= 0
            or self.max_brake_impulse <= 0
        ):
            raise ValueError("Invalid physical limits")

        if self.body_hull is not None:
            from .body_hull import BodyHull

            if not isinstance(self.body_hull, BodyHull):
                raise ValueError("Validated body hull required")
            p = self.body_hull.vertices
            if not (
                np.allclose(p.min(0), self.body_lo, rtol=0, atol=1e-8)
                and np.allclose(p.max(0), self.body_hi, rtol=0, atol=1e-8)
            ):
                raise ValueError("Hull must retain the complete declared body extents")

    def identity_sha256(self):
        """Physical identity includes optional collision geometry, not pose."""
        raw = bytes(self.native([0, 0, 0], [0, 0, 0, 1]))
        if self.body_hull is not None:
            raw += b"\0asset_convex_hull.v1\0" + self.body_hull.vertices.astype("<f8").tobytes()
        return hashlib.sha256(raw).hexdigest()

    def native(self, base, quaternion):
        base = _array(base, (3,))
        q = _array(quaternion, (4,))
        if abs(q @ q - 1) > 1e-8:
            raise ValueError("Unit quaternion required")
        result = _Config()
        for name, _ in _Config._fields_:
            if name == "wheels":
                for j, w in enumerate(self.wheels):
                    result.wheels[j][:] = w
            elif name in ["body_lo", "body_hi"]:
                getattr(result, name)[:] = getattr(self, name)
            elif name == "base":
                result.base[:] = base
            elif name == "quaternion_xyzw":
                result.quaternion_xyzw[:] = q
            else:
                setattr(result, name, getattr(self, name))
        return result


@dataclass(frozen=True)
class Control:
    steering_rad: float = 0.0
    engine_force_n: float = 0.0  # total for four wheels; negative permits reverse
    brake_impulse_ns: float = 100.0  # Bullet per-wheel maximum brake impulse per substep


@dataclass(frozen=True)
class FleetFrame:
    names: tuple[str, ...]
    states: np.ndarray

    @property
    def time_s(self):
        return float(self.states[0, 0])

    def row(self, name):
        return self.states[self.names.index(name)]

    def wheel_pose(self, index, profile):
        data = self.states[index, 22:42].reshape(4, 5)
        hubs = profile.wheels[:, :3].copy()
        hubs[:, 2] += profile.suspension_loaded - data[:, 0]
        return WheelPose(hubs, Rotation.from_euler("ZY", data[:, 1:3]).as_matrix())

    def actor_frame(self, indices, profiles):
        from awr.sensorsvc.camera.raster import ActorFrame

        ids = np.asarray(indices)
        if (
            ids.shape != (len(self.names),)
            or not np.issubdtype(ids.dtype, np.integer)
            or len(set(ids)) != len(ids)
            or (ids < 0).any()
        ):
            raise ValueError("Unique renderer indices required")
        if len(profiles) != len(ids):
            raise ValueError("Renderer profile roster mismatch")
        return ActorFrame(
            ids,
            self.states[:, 1:4],
            self.states[:, 8],
            speed=self.states[:, 11],
            pitch=self.states[:, 9],
            roll=self.states[:, 10],
            wheel_poses={int(i): self.wheel_pose(j, profiles[j]) for j, i in enumerate(ids)},
        )


class BulletWorld:
    """One owner thread; fixed roster; no API for teleporting/reinitializing cars."""

    def __init__(self, build, triangles, *, dt=1 / 240):
        self._handle = None
        self._thread = threading.get_ident()
        self._tick = 0
        self.names = []
        self.profiles = []
        self.kinematics = []
        self._kinematic_pose = []
        self._limits = []
        self.dt = float(dt)
        if not np.isfinite(self.dt) or not 0 < self.dt <= 0.02:
            raise ValueError("Invalid fixed timestep")
        build = Path(build)
        manifest = json.loads((build / "build.json").read_text())
        if manifest.get("bullet_revision") != BULLET_REVISION or manifest.get("fleet_library", {}).get("abi") != 4:
            raise ValueError("Pinned fleet library required")
        library = manifest["fleet_library"]
        path = build / library["name"]
        if path.parent != build or _sha(path) != library["sha256"]:
            raise ValueError("Fleet library digest mismatch")
        for name, digest in manifest["source_sha256"].items():
            if _sha(Path(name)) != digest:
                raise ValueError("Fleet source digest mismatch: " + name)
        self.build_sha256 = _sha(build / "build.json")
        self._lib = C.CDLL(str(path.resolve()))

        def bind(name, restype, args):
            f = getattr(self._lib, name)
            f.restype = restype
            f.argtypes = args
            return f

        bind("d4_bullet_abi", C.c_int, [])
        bind("d4_vehicle_config_size", C.c_size_t, [])
        bind("d4_bullet_error", C.c_char_p, [])
        bind("d4_world_create", C.c_void_p, [D, C.c_size_t, C.c_double])
        bind("d4_world_destroy", None, [C.c_void_p])
        bind("d4_world_add_vehicle", C.c_int, [C.c_void_p, C.POINTER(_Config)])
        bind("d4_world_add_vehicle_hull", C.c_int, [C.c_void_p, C.POINTER(_Config), D, C.c_size_t])
        bind("d4_world_add_kinematic", C.c_int, [C.c_void_p, D, D])
        bind("d4_world_step", C.c_int, [C.c_void_p, D, C.c_size_t, D, C.c_size_t, C.c_size_t, D, C.c_size_t])
        bind("d4_world_read", C.c_int, [C.c_void_p, D, C.c_size_t])
        bind("d4_world_contacts", C.c_int, [C.c_void_p, D, C.c_size_t])
        bind("d4_world_wheel_diagnostics", C.c_int, [C.c_void_p, D, C.c_size_t])
        bind("d4_world_contact_diagnostics", C.c_int, [C.c_void_p, D, C.c_size_t])
        if self._lib.d4_bullet_abi() != 4 or self._lib.d4_vehicle_config_size() != C.sizeof(_Config):
            raise ValueError("Fleet ABI mismatch")
        triangles = np.array(triangles, dtype=np.float64, order="C", copy=True)
        if (
            triangles.ndim != 3
            or triangles.shape[1:] != (3, 3)
            or not len(triangles)
            or not np.isfinite(triangles).all()
        ):
            raise ValueError("Finite terrain triangles required")
        self.terrain_sha256 = hashlib.sha256(triangles.astype("<f8").tobytes()).hexdigest()
        self._handle = self._lib.d4_world_create(_ptr(triangles), len(triangles), self.dt)
        if not self._handle:
            raise ValueError(self._error())

    def _error(self):
        return self._lib.d4_bullet_error().decode("utf-8")

    def _check(self, result):
        if result < 0:
            raise ValueError(self._error())
        return result

    def _owned(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError("Physics world belongs to another thread")
        if not self._handle:
            raise RuntimeError("Physics world is closed")

    def _new_name(self, name):
        self._owned()
        if self._tick:
            raise ValueError("Roster is frozen after stepping")
        if not isinstance(name, str) or not name or name in self.names or name in self.kinematics:
            raise ValueError("Unique non-empty actor name required")

    def add_vehicle(self, name, profile, base, quaternion=(0.0, 0.0, 0.0, 1.0)):
        self._new_name(name)
        c = profile.native(base, quaternion)
        if profile.body_hull is None:
            index = self._check(self._lib.d4_world_add_vehicle(self._handle, C.byref(c)))
        else:
            h = profile.body_hull.vertices
            index = self._check(self._lib.d4_world_add_vehicle_hull(self._handle, C.byref(c), _ptr(h), len(h)))
        if index != len(self.names):
            raise RuntimeError("Native roster divergence")
        self.names.append(name)
        self.profiles.append(profile)
        return index

    def add_kinematic(self, name, size, base_yaw, *, max_speed_mps=3.0, max_yaw_rate=3.0):
        self._new_name(name)
        size = _array(size, (3,))
        pose = _array(base_yaw, (4,))
        if (
            (size <= 0).any()
            or not np.isfinite([max_speed_mps, max_yaw_rate]).all()
            or min(max_speed_mps, max_yaw_rate) < 0
        ):
            raise ValueError("Invalid kinematic dimensions/limits")
        index = self._check(self._lib.d4_world_add_kinematic(self._handle, _ptr(size), _ptr(pose)))
        if index != len(self.kinematics):
            raise RuntimeError("Native roster divergence")
        self.kinematics.append(name)
        self._kinematic_pose.append(pose)
        self._limits.append((max_speed_mps, max_yaw_rate))
        return 1000000 + index

    def snapshot(self):
        self._owned()
        states = np.empty((len(self.names), WIDTH))
        if not self.names:
            raise ValueError("No vehicles registered")
        self._check(self._lib.d4_world_read(self._handle, _ptr(states), states.size))
        states.flags.writeable = False
        return FleetFrame(tuple(self.names), states)

    def step(self, controls, *, kinematic_targets=None, substeps=24):
        self._owned()
        if type(substeps) is not int or not 1 <= substeps <= 240:
            raise ValueError("1–240 substeps required")
        if set(controls) != set(self.names):
            raise ValueError("Every vehicle needs an explicit control")
        commands = np.empty((len(self.names), 3))
        for i, (name, p) in enumerate(zip(self.names, self.profiles, strict=False)):
            c = controls[name]
            commands[i] = [c.steering_rad, c.engine_force_n, c.brake_impulse_ns]
            if (
                not np.isfinite(commands[i]).all()
                or abs(c.steering_rad) > p.max_steer
                or abs(c.engine_force_n) > p.max_force
                or not 0 <= c.brake_impulse_ns <= p.max_brake_impulse
            ):
                raise ValueError("Control exceeds declared limits")
        targets = self._targets(kinematic_targets, substeps)
        states = np.empty((substeps, len(self.names), WIDTH))
        self._check(
            self._lib.d4_world_step(
                self._handle,
                _ptr(commands),
                len(self.names),
                _ptr(targets),
                len(targets),
                substeps,
                _ptr(states),
                states.size,
            )
        )
        self._tick += substeps
        self._kinematic_pose = list(targets.copy())
        states.flags.writeable = False
        return states

    def validate_targets(self, kinematic_targets, *, substeps=24):
        """Read-only input check before a shared activity admission transaction."""
        self._owned()
        if type(substeps) is not int or not 1 <= substeps <= 240:
            raise ValueError("1–240 substeps required")
        self._targets(kinematic_targets, substeps)

    def _targets(self, kinematic_targets, substeps):
        targets = np.array(self._kinematic_pose, dtype=float).reshape(-1, 4)
        if kinematic_targets is not None:
            if set(kinematic_targets) != set(self.kinematics):
                raise ValueError("Complete kinematic target roster required")
            targets = _array([kinematic_targets[n] for n in self.kinematics], targets.shape)
        elapsed = substeps * self.dt
        for old, new, (speed, yawrate) in zip(self._kinematic_pose, targets, self._limits, strict=False):
            turn = abs(np.arctan2(np.sin(new[3] - old[3]), np.cos(new[3] - old[3])))
            if np.linalg.norm(new[:3] - old[:3]) > speed * elapsed + 1e-9 or turn > yawrate * elapsed + 1e-9:
                raise ValueError("Kinematic target would teleport beyond declared motion limits")
        return targets

    def contacts(self):
        self._owned()
        n = self._check(self._lib.d4_world_contacts(self._handle, None, 0))
        rows = np.empty((n, 6))
        self._check(self._lib.d4_world_contacts(self._handle, _ptr(rows), n))
        rows.flags.writeable = False
        return rows

    def wheel_diagnostics(self):
        """Read cached wheel solver state, shape (vehicles,4,22); no advancement."""
        self._owned()
        rows = np.empty((len(self.names), 4, 22))
        self._check(self._lib.d4_world_wheel_diagnostics(self._handle, _ptr(rows), rows.size))
        rows.flags.writeable = False
        return rows

    def contact_diagnostics(self):
        """Read all solver manifolds, including positive-distance contacts."""
        self._owned()
        n = self._check(self._lib.d4_world_contact_diagnostics(self._handle, None, 0))
        rows = np.empty((n, 13))
        self._check(self._lib.d4_world_contact_diagnostics(self._handle, _ptr(rows), n))
        rows.flags.writeable = False
        return rows

    def close(self):
        if self._handle:
            self._owned()
            self._lib.d4_world_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_handle", None):
            self._lib.d4_world_destroy(self._handle)
            self._handle = None


class RouteFollower:
    """Forward-only generic controller, retargetable from the actual stopped pose.

    Activity choice and route planning belong to the caller. This controller is
    not a traffic planner; speed_limit can carry the live safety controller cap.
    """

    def __init__(self, profile, points, observed, *, speed=2.0, lookahead=1.0, settle_s=0.0):
        self.profile = profile
        self.speed = float(speed)
        self.lookahead = float(lookahead)
        self.settle_s = float(settle_s)
        if not np.isfinite([speed, lookahead, settle_s]).all() or min(speed, lookahead) <= 0 or settle_s < 0:
            raise ValueError("Invalid controller parameters")
        self.set_route(points, observed)

    def set_route(self, points, observed):
        points = np.array(points, float, copy=True)
        state = _array(observed, (WIDTH,))
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("Finite route points required")
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if (lengths <= 0).any():
            raise ValueError("Duplicate route point")
        if np.linalg.norm(points[0] - state[1:3]) > 0.2 or abs(state[11]) > 0.1:
            raise ValueError("New route must start at the actual stopped vehicle")
        heading = np.array([np.cos(state[8]), np.sin(state[8])])
        if (points[1] - points[0]) @ heading <= 0:
            raise ValueError("Forward controller needs a forward departure")
        self.points = points
        self.arc = np.r_[0.0, np.cumsum(lengths)]
        self.progress = 0.0
        self.arrived = False
        self.park_steer = None
        self.origin_time = float(state[0])
        self.last_time = None
        self.last_command = None

    def command(self, state, *, dt=1 / 240, speed_limit=None):
        row = _array(state, (WIDTH,))
        time = float(row[0])
        if self.last_time is not None and time <= self.last_time:
            raise ValueError("Controller requires a fresh world frame")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Positive physics timestep required")
        limit = self.speed if speed_limit is None else float(speed_limit)
        if not np.isfinite(limit) or not 0 <= limit <= self.speed:
            raise ValueError("Invalid live speed limit")
        if self.arrived:
            # A parked driver holds the actual steering angle. Continuing to
            # chase the passed endpoint steers the tyres while brake holding.
            self.last_time = time
            self.last_command = Control(self.park_steer, 0.0, self.profile.max_brake_impulse)
            return self.last_command
        a, b = self.points[:-1], self.points[1:]
        delta = b - a
        u = np.clip(np.sum((row[1:3] - a) * delta, axis=1) / np.sum(delta * delta, axis=1), 0, 1)
        distance = np.linalg.norm(row[1:3] - a - u[:, None] * delta, axis=1)
        valid = (self.arc[1:] >= self.progress - 0.5) & (self.arc[:-1] <= self.progress + 8)
        distance[~valid] = np.inf
        i = int(np.argmin(distance))
        self.progress = max(self.progress, self.arc[i] + u[i] * (self.arc[i + 1] - self.arc[i]))
        target_arc = min(self.arc[-1], self.progress + self.lookahead)
        j = min(int(np.searchsorted(self.arc[1:], target_arc)), len(self.points) - 2)
        target = self.points[j] + (target_arc - self.arc[j]) / (self.arc[j + 1] - self.arc[j]) * (
            self.points[j + 1] - self.points[j]
        )
        alpha = np.arctan2(target[1] - row[2], target[0] - row[1]) - row[8]
        wheelbase = np.ptp(self.profile.wheels[:, 0])
        steer = float(
            np.clip(
                np.arctan2(2 * wheelbase * np.sin(alpha), max(1.0, np.linalg.norm(target - row[1:3]))),
                -self.profile.max_steer,
                self.profile.max_steer,
            )
        )
        remaining = self.arc[-1] - self.progress
        desired = min(limit, 0.8 * remaining)
        settling = time < self.origin_time + self.settle_s
        if settling or self.arrived:
            desired = 0.0
        if (
            not settling
            and remaining < 0.1
            and np.linalg.norm(row[1:3] - self.points[-1]) < 0.15
            and abs(row[11]) < 0.1
        ):
            self.arrived = True
            self.park_steer = float(row[33])
            steer = self.park_steer
        force = float(
            np.clip(
                self.profile.mass_kg * (4 * (desired - row[11]) - 9.81 * np.sin(row[9])), -3000, self.profile.max_force
            )
        )
        if settling or self.arrived:
            command = Control(steer, 0.0, self.profile.max_brake_impulse)
        else:
            command = Control(steer, max(0.0, force), min(self.profile.max_brake_impulse, max(0.0, -force) * dt / 4))
        self.last_time = time
        self.last_command = command
        return command
