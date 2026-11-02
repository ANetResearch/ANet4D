"""Rest-to-rest timing for admitted car geometry, including parking reversals.

This is a nominal motion schedule, not a collision controller. A backend must
admit traffic before departure and retain live body checks while executing it.
No actor identities, recording horizon or query conditions enter this module.
"""

import math
from dataclasses import dataclass

import numpy as np

from .driving_graph import route_arrays


@dataclass(frozen=True)
class Limits:
    forward_mps: float = 6.0
    reverse_mps: float = 1.4
    acceleration_mps2: float = 1.5
    braking_mps2: float = 2.0
    lateral_mps2: float = 1.8
    yaw_rps: float = 0.5
    gear_dwell_s: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in self.__dict__.values()):
            raise ValueError("Driving limits must be finite and positive")


class Schedule:
    """Piecewise constant acceleration along the exact route interpolation.

    All ends and gear changes have zero speed. Intermediate speed limits bound
    reverse speed, yaw rate and lateral acceleration. A forward/backward pass
    enforces acceleration/braking before every future low-speed segment.
    Sampling absolute elapsed time is independent of frame rate and duration.
    Holding an already moving schedule is NOT a valid emergency-stop policy.
    """

    def __init__(self, route, limits=Limits()):
        self.route = route_arrays(route["points"], route["yaw"], route.get("gear"))
        self.limits = limits
        r = self.route
        # Include a midpoint in each geometric interval: even adjacent zero-
        # speed cusp endpoints need a triangular velocity profile between them.
        arc = r["arc"]
        s = np.sort(np.r_[arc, (arc[:-1] + arc[1:]) / 2])
        ds = np.diff(s)
        gears = np.repeat(r["gear"], 2)
        curvature = np.repeat(np.abs(np.diff(r["yaw"])) / np.diff(arc), 2)
        caps = np.where(gears > 0, limits.forward_mps, limits.reverse_mps)
        turning = curvature > 1e-12
        caps[turning] = np.minimum(
            caps[turning],
            np.minimum(np.sqrt(limits.lateral_mps2 / curvature[turning]), limits.yaw_rps / curvature[turning]),
        )
        speed = np.r_[caps[0], np.minimum(caps[:-1], caps[1:]), caps[-1]]
        stops = np.flatnonzero(np.diff(gears)) + 1
        speed[np.r_[0, stops, len(speed) - 1]] = 0.0
        for i in range(1, len(speed)):
            speed[i] = min(speed[i], math.sqrt(speed[i - 1] ** 2 + 2 * limits.acceleration_mps2 * ds[i - 1]))
        for i in range(len(speed) - 2, -1, -1):
            speed[i] = min(speed[i], math.sqrt(speed[i + 1] ** 2 + 2 * limits.braking_mps2 * ds[i]))
        if np.any(speed[:-1] + speed[1:] <= 0):
            raise ValueError("Car segment cannot be traversed under motion limits")
        self.phases = []
        clock = 0.0
        stop_set = set(stops.tolist())
        for i, distance in enumerate(ds):
            duration = 2 * distance / (speed[i] + speed[i + 1])
            self.phases.append((clock, clock + duration, s[i], s[i + 1], speed[i], speed[i + 1], int(gears[i])))
            clock += duration
            if i + 1 in stop_set:
                self.phases.append((clock, clock + limits.gear_dwell_s, s[i + 1], s[i + 1], 0.0, 0.0, 0))
                clock += limits.gear_dwell_s
        self.phases = np.asarray(self.phases, float)
        self.duration_s = float(clock)
        self.length_m = float(arc[-1])
        self.gear_stops = [(float(p[0]), float(p[1]), float(p[2])) for p in self.phases if p[-1] == 0]

    def at(self, elapsed_s):
        if not math.isfinite(elapsed_s):
            raise ValueError("Elapsed driving time must be finite")
        if elapsed_s <= 0:
            s, speed, gear = 0.0, 0.0, 0
        elif elapsed_s >= self.duration_s:
            s, speed, gear = self.length_m, 0.0, 0
        else:
            i = int(np.searchsorted(self.phases[:, 1], elapsed_s, side="right"))
            t0, t1, s0, s1, v0, v1, direction = self.phases[i]
            u = elapsed_s - t0
            acceleration = (v1 - v0) / (t1 - t0)
            s = float(np.clip(s0 + v0 * u + 0.5 * acceleration * u * u, s0, s1))
            speed, gear = float(max(0.0, v0 + acceleration * u)), int(direction)
        r = self.route
        xy = np.array([np.interp(s, r["arc"], r["points"][:, k]) for k in range(2)])
        yaw = float(np.interp(s, r["arc"], r["yaw"]))
        return dict(
            xy=xy,
            yaw=yaw,
            arc_m=s,
            speed_mps=speed,
            gear=gear,
            signed_speed_mps=gear * speed,
            arrived=elapsed_s >= self.duration_s,
        )


class UnsafeStop(ValueError):
    """A newly observed obstruction is inside the physical stopping distance."""


class RouteExecution:
    """Tick-driven execution with physical braking and rest-to-rest resumption.

    The caller supplies a safe stopping arc measured by its live body/ground
    lookahead, *after* body clearance has been subtracted. This class does not
    detect obstacles. An impossible stop is an explicit failure, never a pose
    hold at driving speed. It finishes a braking manoeuvre even if an obstacle
    clears early, then rebuilds the remaining schedule from the actual stopped
    pose. World time advances through braking, waiting and gear changes.

    A backend must validate/commit each proposed pose in its authoritative body
    state. Calling this without that backend is only a kinematic component test.
    """

    def __init__(self, route, limits=Limits()):
        self.route = route_arrays(route["points"], route["yaw"], route.get("gear"))
        self.limits = limits
        self.schedule = Schedule(self.route, limits)
        self.offset_m = 0.0
        self.schedule_s = 0.0
        self.schedule_error = 0.0
        self.time_s = 0.0
        self.time_error = 0.0
        self.mode = "scheduled"
        self.arc_m = 0.0
        self.speed_mps = 0.0
        self.gear = 0
        self.last_gear = 0
        self.hold_s = 0.0
        self.arrived = False
        self.brake_count = 0

    def state(self):
        r = self.route
        s = self.arc_m
        return dict(
            xy=np.array([np.interp(s, r["arc"], r["points"][:, k]) for k in range(2)]),
            yaw=float(np.interp(s, r["arc"], r["yaw"])),
            arc_m=s,
            speed_mps=self.speed_mps,
            gear=self.gear,
            signed_speed_mps=self.gear * self.speed_mps,
            arrived=self.arrived,
            elapsed_s=self.time_s,
            mode=self.mode,
        )

    def _next_gear(self):
        index = min(
            int(np.searchsorted(self.route["arc"], self.arc_m + 1e-9, side="right")) - 1, len(self.route["gear"]) - 1
        )
        return int(self.route["gear"][index])

    @staticmethod
    def _clock(value, error, dt):
        increment = dt - error
        result = value + increment
        return result, (result - value) - increment

    def _advance_clock(self, dt):
        self.time_s, self.time_error = self._clock(self.time_s, self.time_error, dt)

    def _resume(self):
        r = self.route
        if self.arc_m >= r["arc"][-1] - 1e-9:
            self.arc_m = float(r["arc"][-1])
            self.arrived = True
            self.mode = "arrived"
            return
        # Exactly measured pose, followed by untouched future route vertices.
        future = np.flatnonzero(r["arc"] > self.arc_m + 1e-9)
        here = self.state()
        points = np.vstack([here["xy"], r["points"][future]])
        yaw = np.r_[here["yaw"], r["yaw"][future]]
        gear = r["gear"][future - 1]
        self.schedule = Schedule(route_arrays(points, yaw, gear), self.limits)
        self.offset_m = self.arc_m
        self.schedule_s = 0.0
        self.schedule_error = 0.0
        self.mode = "scheduled"

    def advance(self, dt, *, stop_before_arc_m=None):
        """Advance one control tick (<=100 ms); stop location is not a target.

        None means the backend's entire relevant braking horizon is clear.
        A finite obstruction may be beyond the trip, but never behind the body.
        The instantaneous speed in the result is distinct from frame-average
        displacement/dt, particularly during a partial-tick braking stop.
        """
        if not math.isfinite(dt) or not 0 < dt <= 0.1 + 1e-12:
            raise ValueError("Driving execution requires a finite tick in (0, 0.1] s")
        stop = math.inf if stop_before_arc_m is None else float(stop_before_arc_m)
        if stop_before_arc_m is not None and (not math.isfinite(stop) or stop < self.arc_m - 1e-9):
            raise ValueError("Finite stopping arc cannot be behind the measured body")
        b = self.limits.braking_mps2
        # Check before any mutation; a caller may inspect the unchanged state.
        if self.arc_m + self.speed_mps**2 / (2 * b) > stop + 1e-8:
            error = UnsafeStop("Live obstacle is inside the validated braking distance")
            error.diagnostic = dict(
                execution=self.state(),
                stop_before_arc_m=stop,
                stopping_arc_m=self.arc_m + self.speed_mps**2 / (2 * b),
                braking_mps2=b,
            )
            raise error
        if self.arrived:
            self._advance_clock(dt)
            return self.state()
        if self.mode == "waiting":
            if self.hold_s > 1e-9:
                self.hold_s = max(0.0, self.hold_s - dt)
                self._advance_clock(dt)
                return self.state()
            if stop - self.arc_m < 0.25:
                self._advance_clock(dt)
                return self.state()
            self._resume()
            if self.arrived:
                self._advance_clock(dt)
                return self.state()
        if self.mode == "scheduled":
            next_clock, clock_error = self._clock(self.schedule_s, self.schedule_error, dt)
            target = self.schedule.at(next_clock)
            candidate_arc = self.offset_m + target["arc_m"]
            if candidate_arc + target["speed_mps"] ** 2 / (2 * b) + 0.25 > stop:
                self.mode = "braking"
                self.brake_count += 1
                # Interrupting a nominal cusp dwell must not erase its remainder.
                for phase in self.schedule.phases:
                    if phase[-1] == 0 and phase[0] <= self.schedule_s < phase[1]:
                        self.hold_s = max(self.hold_s, float(phase[1] - self.schedule_s))
                        break
            else:
                self.schedule_s = next_clock
                self.schedule_error = clock_error
                self.arc_m = candidate_arc
                self.speed_mps = target["speed_mps"]
                self.gear = target["gear"]
                self.arrived = target["arrived"]
                if self.gear:
                    self.last_gear = self.gear
                if self.arrived:
                    self.mode = "arrived"
                self._advance_clock(dt)
                return self.state()
        if self.mode == "braking":
            moving_s = min(dt, self.speed_mps / b)
            self.arc_m += self.speed_mps * moving_s - 0.5 * b * moving_s * moving_s
            self.speed_mps = max(0.0, self.speed_mps - b * dt)
            if self.speed_mps <= 1e-10:
                self.speed_mps = 0.0
                self.gear = 0
                self.mode = "waiting"
                if self.arc_m < self.route["arc"][-1] - 1e-9 and self.last_gear and self._next_gear() != self.last_gear:
                    self.hold_s = max(self.hold_s, self.limits.gear_dwell_s)
            elif self.gear == 0:
                raise RuntimeError("Moving vehicle cannot have neutral gear")
        self._advance_clock(dt)
        return self.state()
