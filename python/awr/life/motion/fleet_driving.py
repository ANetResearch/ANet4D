"""Shared Bullet execution behind the existing activity/berth transaction API.

Environment adapters supply frozen rear-axle templates, per-edge permissions,
an explicit preflight gate and a full-frame safety gate. Neither elapsed
schedule time nor a nominal endpoint can complete an actual drive. This is
opt-in and does not make a map, population or tape admitted for publication.
"""

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from awr.life.civil.activity.driving import Body, DrivingRoutes
from awr.life.civil.curb_allocation import swept
from awr.life.civil.driving_motion import Limits
from awr.life.civil.vehicle_connections import angle_delta

from .bullet import Control
from .maneuver import Maneuver, ManeuverFollower, axle


@dataclass
class _Execution:
    follower: ManeuverFollower
    observed: np.ndarray
    started_s: float
    body_arc: np.ndarray
    brake_count: int = 0
    limited: bool = False

    @property
    def arrived(self):
        return self.follower.arrived

    @property
    def time_s(self):
        return float(self.observed[0] - self.started_s)

    @property
    def arc_m(self):
        return float(np.interp(self.follower.progress, self.follower.route.arc, self.body_arc))

    def state(self):
        row = self.observed
        return dict(
            xy=row[1:3].copy(),
            yaw=float(row[8]),
            speed_mps=float(abs(row[11])),
            signed_speed_mps=float(row[11]),
            arc_m=self.arc_m,
            arrived=self.arrived,
        )


class BulletDrivingRoutes(DrivingRoutes):
    """One physics tick for the complete fleet, irrespective of driver count.

    Current catalog supports one physical profile; this is explicit rather
    than resizing differently authored vehicles to fit an admitted template.
    The environment must provide edge_gate(edge, maneuver, area, profile) ->
    (accepted, reason), and frame_gate(before, samples, backend) -> None, raising
    on failure. These must bind terrain/permission/asset evidence externally.
    Native discrete contact/support checks are mandatory and cannot be replaced
    by those callbacks. Real-map continuous checks remain an adapter duty.
    """

    shared_world = True
    candidate_only = True

    def __init__(
        self,
        world,
        graph,
        templates,
        size,
        areas,
        *,
        geometry_revision,
        edge_gate,
        frame_gate,
        limits=Limits(forward_mps=2.0, reverse_mps=1.0, lateral_mps2=0.25),
        fixed_world=None,
    ):
        if (
            not isinstance(geometry_revision, str)
            or len(geometry_revision) != 64
            or any(c not in "0123456789abcdef" for c in geometry_revision)
            or not callable(edge_gate)
            or not callable(frame_gate)
        ):
            raise ValueError("Versioned environment geometry and explicit gates required")
        self.world = world
        self.current = world.snapshot()
        self.failed = None
        self.samples = None
        self.fixed_world = fixed_world
        if fixed_world is not None:
            from .fixed_world import FixedWorld

            if not isinstance(fixed_world, FixedWorld) or fixed_world.world is not world:
                raise ValueError("Fixed obstacles must be registered in this native world")
            fixed_world.verify()
        self.profile = world.profiles[0]

        def key(p):
            return p.identity_sha256()

        if any(key(p) != key(self.profile) for p in world.profiles):
            raise ValueError("Catalog requires one actual physical profile")
        self.profile_key = key(self.profile)
        self.geometry_revision = geometry_revision
        self.edge_gate = edge_gate
        self.frame_gate = frame_gate
        self.templates = dict(templates)
        self.batches = 0
        if set(self.templates) != set(graph.edges) or not all(isinstance(r, Maneuver) for r in templates.values()):
            raise ValueError("Every driving edge needs an explicit axle maneuver")
        if np.asarray(size).shape != (3,) or np.any(
            np.asarray(size) < self.profile.body_hi - self.profile.body_lo - 1e-7
        ):
            raise ValueError("Declared body dimensions cannot shrink the physical chassis")
        rear, _ = axle(self.profile)
        for name, r in self.templates.items():
            q = graph.edges[name]["route"]
            body = np.array([r.body_pose(i) for i in range(len(r.points))])
            if (
                abs(r.rear_x - rear) > 1e-7
                or q["points"].shape != body[:, :2].shape
                or not np.allclose(q["points"], body[:, :2], rtol=0, atol=1e-7)
                or np.max(np.abs((q["yaw"] - body[:, 2] + np.pi) % (2 * np.pi) - np.pi)) > 1e-7
                or not np.array_equal(q["gear"], r.gear)
            ):
                raise ValueError("Driving graph and axle template geometry differ")
        occupancy = {
            name: Body(tuple(row[[1, 2, 8]]), tuple(size))
            for name, row in zip(world.names, self.current.states, strict=False)
        }
        occupancy.update(self.fixed_bodies())
        super().__init__(graph, None, size, areas, occupancy, limits=limits)
        catalog = dict(
            terrain=world.terrain_sha256,
            revision=geometry_revision,
            profile=self.profile_key,
            fixed_world=None if fixed_world is None else fixed_world.identity,
            physics_dt_s=world.dt,
            size=self.size.tolist(),
            edges={
                e: dict(
                    area_sha256=hashlib.sha256(self.areas[e].wkb).hexdigest(),
                    rear_sha256=hashlib.sha256(
                        self.templates[e].points.tobytes()
                        + self.templates[e].yaw.tobytes()
                        + self.templates[e].gear.tobytes()
                    ).hexdigest(),
                    resources=graph.edges[e]["traffic_resources"],
                )
                for e in sorted(graph.edges)
            },
        )
        self.catalog_sha256 = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
        self._native_gate(self.current.states[None])

    def fixed_bodies(self):
        return {} if self.fixed_world is None else dict(self.fixed_world.verify())

    def _ready(self):
        self.fixed_bodies()
        if self.failed is not None:
            raise RuntimeError("Rejected shared activity world cannot continue")
        frame = self.world.snapshot()
        if frame.names != self.current.names or not np.array_equal(frame.states, self.current.states):
            raise ValueError("Physics was advanced outside the authoritative fleet batch")
        return frame

    def _edge(self, edge):
        if edge in self.geometry:
            self.stats["edge_cache_hits"] += 1
            return self.geometry[edge]
        self.stats["edge_checks"] += 1
        accepted, reason = self.edge_gate(edge, self.templates[edge], self.areas[edge], self.profile)
        if not isinstance(accepted, (bool, np.bool_)) or not isinstance(reason, str):
            raise ValueError("Environment gate must return a boolean and explicit reason")
        corridor = swept(self.graph.edges[edge]["route"], self.size) if accepted else None
        self.geometry[edge] = (bool(accepted), reason, corridor)
        return self.geometry[edge]

    def plan(self, vehicle, source, target):
        self._ready()
        if vehicle not in self.current.names:
            raise ValueError("Fixed obstacle cannot receive a driving plan")
        route = super().plan(vehicle, source, target)
        if route is None:
            return None
        from awr.life.civil.activity.director import Route

        # Bind the alternative physical model and environment revision, rather
        # than labeling this as the old fitted-terrain kinematic contract.
        old = route.evidence
        plan = self.plans.pop(old)
        signature = "|".join([old, self.catalog_sha256, self.world.build_sha256])
        evidence = "bullet-drive:" + hashlib.sha256(signature.encode()).hexdigest()
        self.plans[evidence] = plan
        return Route(route.length_m, route.duration_s, evidence)

    def update_occupancy(self, occupancy):
        fixed = self.fixed_bodies()
        if set(occupancy) != set(self.current.names) | set(fixed):
            raise ValueError("Physical occupancy roster mismatch")
        for name, body in occupancy.items():
            if name in fixed:
                expected = fixed[name]
                if (
                    not isinstance(body, Body)
                    or body.size != expected.size
                    or body.pose[:2] != expected.pose[:2]
                    or abs(angle_delta(body.pose[2], expected.pose[2])) > 1e-12
                ):
                    raise ValueError("Fixed occupancy changed: " + name)
                continue
            row = self.current.row(name)
            if (
                body is None
                or not isinstance(body, Body)
                or not np.allclose(body.size, self.size, rtol=0, atol=1e-8)
                or np.linalg.norm(np.asarray(body.pose[:2]) - row[1:3]) > 1e-8
                or abs(angle_delta(body.pose[2], row[8])) > 1e-8
            ):
                raise ValueError("Occupancy must come from the accepted physical frame")
        super().update_occupancy({**occupancy, **fixed})

    def _maneuver(self, edges):
        parts = [self.templates[e] for e in edges]
        if not parts:
            raise ValueError("Empty physical journey")
        for a, b in zip(parts, parts[1:], strict=False):
            if np.linalg.norm(a.points[-1] - b.points[0]) > 1e-7 or abs(angle_delta(a.yaw[-1], b.yaw[0])) > 1e-7:
                raise ValueError("Axle templates have a disconnected join")
        return Maneuver(
            np.concatenate([r.points[1 if i else 0 :] for i, r in enumerate(parts)]),
            np.concatenate([r.yaw[1 if i else 0 :] for i, r in enumerate(parts)]),
            np.concatenate([r.gear for r in parts]),
            parts[0].rear_x,
        )

    def validate_batch(self, t, observations, dt, *, speed_limits=None, kinematic_targets=None):
        frame = self._ready()
        if not math.isfinite(t) or abs(t - frame.time_s) > 1e-8 or not math.isfinite(dt) or not 0 < dt <= 0.100000001:
            raise ValueError("Activity batch must use the actual shared world clock")
        count = round(dt / self.world.dt)
        if count < 1 or abs(count * self.world.dt - dt) > 1e-10:
            raise ValueError("Whole physics substeps required")
        self.world.validate_targets(kinematic_targets, substeps=count)
        caps = {} if speed_limits is None else speed_limits
        if set(caps) - set(frame.names) or any(
            not math.isfinite(v) or not 0 <= v <= self.limits.forward_mps for v in caps.values()
        ):
            raise ValueError("Invalid complete-fleet safety speed limits")
        vehicles = set()
        for o in observations.values():
            if o.container not in frame.names or o.container in vehicles:
                raise ValueError("Driving vehicle is absent or duplicated")
            vehicles.add(o.container)
            row = frame.row(o.container)
            p = np.asarray(o.vehicle_pose)
            if (
                np.linalg.norm(p[:2] - row[1:3]) > 1e-8
                or abs(angle_delta(p[2], row[8])) > 1e-8
                or abs(o.vehicle_speed - abs(row[11])) > 1e-8
            ):
                raise ValueError("Driver observation differs from authoritative physics")
        if set(self.active) - vehicles:
            raise ValueError("An active physical journey has no observed driver")

    def advance_motion(self, *args, **kwargs):
        raise ValueError("Shared physics cannot advance an individual car")

    def speed_limit_for_stop(self, evidence, stop_before_arc_m):
        """Convert the existing live body-route lookahead to a conservative cap.

        This is a local prediction, not a safety certificate. Full measured
        physics and mixed-body gates still decide whether the tick is valid.
        """
        if stop_before_arc_m is None:
            return self.limits.forward_mps
        if not math.isfinite(stop_before_arc_m):
            raise ValueError("Finite stopping arc required")
        name = self.plans[evidence]["vehicle"]
        row = self.current.row(name)
        v = abs(row[11])
        execution = self.executions.get(name)
        arc = 0.0 if execution is None else execution.arc_m
        braking = min(
            self.limits.braking_mps2, self.profile.max_force / self.profile.mass_kg - 9.81 * abs(math.sin(row[9]))
        )
        if braking <= 0.1:
            raise ValueError("Live grade leaves insufficient declared braking authority")
        clearance = max(0.0, stop_before_arc_m - arc - v * 0.35 - 0.1)
        return min(self.limits.forward_mps, math.sqrt(2 * braking * clearance))

    def _native_gate(self, samples):
        for i, p in enumerate(self.world.profiles):
            x = samples[:, i]
            w = x[:, 22:42].reshape(-1, 4, 5)
            if (
                (x[:, 18] > 0.001).any()
                or (x[:, 20] != 4).any()
                or (w[:, :, 4] != 0).any()
                or (np.abs(w[:, :, 0] - p.suspension_loaded) > p.suspension_travel).any()
                or (x[:, 21] > 0.001).any()
            ):
                raise ValueError("Shared physical contact/support gate rejected " + self.world.names[i])

    def advance_batch(self, t, admitted, dt, *, speed_limits=None, kinematic_targets=None):
        before = self._ready()
        caps = {} if speed_limits is None else speed_limits
        if not math.isfinite(t) or not math.isfinite(dt) or not 0 < dt <= 0.100000001:
            raise ValueError("Finite bounded shared physics tick required")
        count = round(dt / self.world.dt)
        if abs(before.time_s - t) > 1e-8 or count < 1 or abs(count * self.world.dt - dt) > 1e-10:
            raise ValueError("Shared physics tick mismatch")
        vehicles = {self.plans[e]["vehicle"]: e for e in admitted.values()}
        if len(vehicles) != len(admitted) or vehicles != self.active:
            raise ValueError("Batch must execute every admitted physical journey exactly once")
        try:
            controls = {
                name: Control(float(row[33]), 0.0, self.world.profiles[i].max_brake_impulse)
                for i, (name, row) in enumerate(zip(before.names, before.states, strict=False))
            }
            for name, evidence in sorted(vehicles.items()):
                row = before.row(name)
                execution = self.executions.get(name)
                if execution is None:
                    f = ManeuverFollower(
                        self.profile,
                        self._maneuver(self.plans[evidence]["edges"]),
                        row,
                        speed=self.limits.forward_mps,
                        reverse_speed=self.limits.reverse_mps,
                        gear_dwell_s=self.limits.gear_dwell_s,
                    )
                    execution = _Execution(f, row.copy(), t, self.plans[evidence]["schedule"].route["arc"].copy())
                    self.executions[name] = execution
                f = execution.follower
                gear = int(f.route.gear[int(f.bounds[f.section])])
                ceiling = self.limits.forward_mps if gear > 0 else self.limits.reverse_mps
                cap = min(ceiling, caps.get(name, ceiling))
                traffic = self.traffic_limit(name, execution.arc_m, abs(float(row[11])))
                if traffic is not None:
                    cap = min(cap, self.speed_limit_for_stop(evidence, traffic))
                limited = cap < ceiling
                if limited and not execution.limited:
                    execution.brake_count += 1
                execution.limited = limited
                controls[name] = f.command(row, dt=self.world.dt, speed_limit=cap)
            samples = self.world.step(controls, kinematic_targets=kinematic_targets, substeps=count)
            self.samples = samples
            self._native_gate(samples)
            if self.frame_gate(before, samples, self) is not None:
                raise ValueError("Safety gate must raise on failure, not return an ignored verdict")
            after = self.world.snapshot()
            for name, execution in self.executions.items():
                execution.observed = after.row(name).copy()
            self.current = after
            self.update_occupancy(
                {
                    **{
                        name: Body(tuple(row[[1, 2, 8]]), tuple(self.size))
                        for name, row in zip(after.names, after.states, strict=False)
                    },
                    **self.fixed_bodies(),
                }
            )
            self.batches += 1
            return after
        except Exception as exc:
            self.failed = dict(t_s=t, error=repr(exc), accepted_batches=self.batches)
            raise

    def can_arrive(self, *args, **kwargs):
        self._ready()
        return super().can_arrive(*args, **kwargs)
