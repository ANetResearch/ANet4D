"""Deterministic truth queries over a compiled Drone4D event (tape.bin + plan.json + anchors.json).

Everything here is a pure function of the tape: no learned component, no judgement. The checkpoint
engine of the benchmark only consumes these queries plus run telemetry.

Tape state codes (drone4d.tape v1): 0 absent, 1 moving, 2 waiting, 3 in vehicle, 4 in building, 5 departed.
States 1 and 2 are "present" (observable from the air).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NS = 1_000_000_000
TICK_NS = 100_000_000
PRESENT = (1, 2)


def point_in_polygon(poly: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Even-odd rule, vectorised over points. ``poly`` is (K, 2)."""
    P = np.asarray(poly, np.float64)
    inside = np.zeros(len(x), bool)
    j = len(P) - 1
    for i in range(len(P)):
        xi, yi, xj, yj = P[i, 0], P[i, 1], P[j, 0], P[j, 1]
        c = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi)
        inside ^= c
        j = i
    return inside


def intervals(mask: np.ndarray, dt_s: float) -> list[tuple[float, float]]:
    """Contiguous True runs of a per-tick mask as [t0, t1) in seconds."""
    out: list[tuple[float, float]] = []
    if not mask.any():
        return out
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=False):
        out.append((float(a) * dt_s, float(b) * dt_s))
    return out


@dataclass
class EventTruth:
    event_dir: Path
    plan: dict
    anchors: dict[str, dict]
    tape: object
    idx: dict[str, int]
    n_ticks: int
    _pos: dict[str, np.ndarray]
    _state: dict[str, np.ndarray]

    # ------------------------------------------------------------------ construction
    @classmethod
    def open(cls, event_dir: Path) -> EventTruth:
        from awr.sim.events.tape import open_tape

        event_dir = Path(event_dir)
        plan = json.loads((event_dir / "plan.json").read_text(encoding="utf-8"))
        anchors = {a["id"]: a for a in json.loads((event_dir / "anchors.json").read_text(encoding="utf-8"))["anchors"]}
        tape = open_tape(event_dir)
        info = tape.actors_info()
        idx = {a["id"]: a["idx"] for a in info}
        return cls(event_dir, plan, anchors, tape, idx, tape.n_ticks, {}, {})

    @property
    def duration_s(self) -> float:
        return self.n_ticks * TICK_NS / NS

    def actor_ids(self, *, scripted_only: bool = False) -> list[str]:
        return [a["id"] for a in self.tape.actors_info(scripted_only=scripted_only)]

    def actor(self, actor_id: str) -> dict:
        return next(a for a in self.tape.actors_info() if a["id"] == actor_id)

    # ------------------------------------------------------------------ per-tick tracks
    def _load(self, actor_id: str) -> None:
        if actor_id in self._pos:
            return
        i = self.idx[actor_id]
        pos = np.zeros((self.n_ticks, 3))
        st = np.zeros(self.n_ticks, np.uint8)
        for k in range(self.n_ticks):
            r = self.tape.frame(k)[i]
            pos[k] = (r["x_mm"] / 1000.0, r["y_mm"] / 1000.0, r["z_mm"] / 1000.0)
            st[k] = r["state"]
        self._pos[actor_id], self._state[actor_id] = pos, st

    def positions(self, actor_id: str) -> np.ndarray:
        """(n_ticks, 3) world-frame metres at tick starts."""
        self._load(actor_id)
        return self._pos[actor_id]

    def states(self, actor_id: str) -> np.ndarray:
        self._load(actor_id)
        return self._state[actor_id]

    def present(self, actor_id: str) -> np.ndarray:
        return np.isin(self.states(actor_id), PRESENT)

    def position_at(self, actor_id: str, t_s: float) -> tuple[np.ndarray, int]:
        """Interpolated position and state at time ``t_s`` (tape API; Hermite between ticks)."""
        st = self.tape.actor_states(int(round(t_s * NS)), [self.idx[actor_id]])
        return st.pos_enu_m[0], int(st.state[0])

    # ------------------------------------------------------------------ zones and anchors
    def zone_polygon(self, zone: str | list) -> np.ndarray:
        if isinstance(zone, str):
            a = self.anchors[zone]
            if "polygon" in a:
                return np.asarray(a["polygon"], np.float64)
            x, y = a["xy"]
            r = float(a.get("radius_m", 3.0))
            return np.array([[x - r, y - r], [x + r, y - r], [x + r, y + r], [x - r, y + r]])
        return np.asarray(zone, np.float64)

    def anchor_xy(self, anchor: str) -> np.ndarray:
        a = self.anchors[anchor]
        if "xy" in a:
            return np.asarray(a["xy"], np.float64)
        p = np.asarray(a["polygon"], np.float64)
        return p.mean(0)

    def in_zone_mask(self, actor_id: str, zone, *, present_only: bool = True) -> np.ndarray:
        P = self.zone_polygon(zone)
        pos = self.positions(actor_id)
        m = point_in_polygon(P, pos[:, 0], pos[:, 1])
        if present_only:
            m &= self.present(actor_id)
        return m

    def zone_intervals(self, actor_id: str, zone) -> list[tuple[float, float]]:
        """Time intervals [t0, t1) in seconds during which the actor is present inside the zone."""
        return intervals(self.in_zone_mask(actor_id, zone), TICK_NS / NS)

    def zone_pass_times(self, actor_id: str, zone) -> list[tuple[float, float]]:
        """Entries that are also exits (the actor passes through rather than ending inside)."""
        out = []
        for t0, t1 in self.zone_intervals(actor_id, zone):
            if t1 < self.duration_s - TICK_NS / NS:
                out.append((t0, t1))
        return out

    def near_anchor_times(self, actor_id: str, anchor: str, radius_m: float) -> list[tuple[float, float]]:
        xy = self.anchor_xy(anchor)
        pos = self.positions(actor_id)
        m = (np.hypot(pos[:, 0] - xy[0], pos[:, 1] - xy[1]) <= radius_m) & self.present(actor_id)
        return intervals(m, TICK_NS / NS)

    # ------------------------------------------------------------------ interaction truths
    def encounter_intervals(self, a: str, b: str, dist_m: float = 2.0, min_s: float = 3.0) -> list[tuple[float, float]]:
        """Both present and within ``dist_m`` for at least ``min_s`` (proximity_encounter)."""
        pa, pb = self.positions(a), self.positions(b)
        m = (np.linalg.norm(pa[:, :2] - pb[:, :2], axis=1) <= dist_m) & self.present(a) & self.present(b)
        return [(t0, t1) for t0, t1 in intervals(m, TICK_NS / NS) if t1 - t0 >= min_s]

    def state_transitions(self, actor_id: str, from_states: tuple[int, ...], to_state: int) -> list[float]:
        """Times (s) at which the actor's state changes from one of ``from_states`` to ``to_state``."""
        st = self.states(actor_id)
        k = np.flatnonzero(np.isin(st[:-1], from_states) & (st[1:] == to_state)) + 1
        return [float(x) * TICK_NS / NS for x in k]

    def board_times(self, person: str) -> list[float]:
        return self.state_transitions(person, PRESENT, 3)

    def alight_times(self, person: str) -> list[float]:
        return self.state_transitions(person, (3,), 1) + self.state_transitions(person, (3,), 2)

    def enter_building_times(self, person: str) -> list[float]:
        return self.state_transitions(person, PRESENT, 4)

    def exit_building_times(self, person: str) -> list[float]:
        return self.state_transitions(person, (4,), 1) + self.state_transitions(person, (4,), 2)

    def count_passing(self, zone, actors: list[str]) -> int:
        return sum(1 for a in actors if self.zone_pass_times(a, zone))

    def first_pass_time(self, zone, actors: list[str]) -> float | None:
        ts = [iv[0][0] for a in actors if (iv := self.zone_pass_times(a, zone))]
        return min(ts) if ts else None

    # ------------------------------------------------------------------ environment
    def env_at(self, t_s: float) -> dict:
        e = self.tape.env_at(int(round(t_s * NS)))
        return {
            "visibility_m": e.visibility_m,
            "fog_top_agl_m": e.fog_top_agl_m,
            "wind_speed_mps": e.wind_speed_mps,
            "rain_mm_h": e.rain_mm_h,
            "cloud": e.cloud,
        }


def describe_scripted(truth: EventTruth) -> list[dict]:
    """Compact per-actor summary used by the demo page and the task author."""
    out = []
    for a in truth.tape.actors_info(scripted_only=True):
        pr = truth.present(a["id"])
        iv = intervals(pr, TICK_NS / NS)
        out.append(
            {
                "id": a["id"],
                "class": a["cls"],
                "appearance": a["appearance"],
                "size_m": list(a["size_m"]),
                "present_s": [[round(t0, 1), round(t1, 1)] for t0, t1 in iv],
            }
        )
    return out
