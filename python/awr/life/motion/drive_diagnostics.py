"""Bounded observations of actual braking decisions; never motion authority.

Retain one-second samples and the latest slow blocked lookahead per vehicle.
The latter can be replayed without an expensive whole-world build. It describes
the controller's decision, not proof that the obstacle physically cannot yield.
"""

import copy
import hashlib
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from awr.life.common.schema import schema_id

from .vehicles import stop_arc_detail

INPUT_NAMES = (
    "me",
    "pts",
    "yaw",
    "arc",
    "s0",
    "speed",
    "length",
    "ext",
    "px",
    "py",
    "pyaw",
    "pspeed",
    "is_person",
    "active",
    "ignore",
    "pred_x",
    "pred_y",
    "pred_yaw",
    "times",
    "person_room",
    "person_room_s",
    "vehicle_room",
    "step",
)


class DriveDiagnostics:
    def __init__(self, ids):
        self.ids = tuple(ids)
        self.last = {}
        self.history = deque(maxlen=4096)
        self.snapshots = {}  # at most one per admitted vehicle, not per tick
        self.last_sample = {}

    def observe(self, t, inputs, detail, execution, admitted, *, reservation_snapshot=None):
        i = int(inputs[0])
        stop, blocker, sample, predicted_s = detail
        row = dict(
            t_s=round(t, 6),
            vehicle=self.ids[i],
            arc_m=float(inputs[4]),
            speed_mps=float(inputs[5]),
            stop_before_arc_m=None if math.isinf(stop) else float(stop),
            obstacle=None if blocker < 0 else self.ids[blocker],
            obstacle_kind=None if blocker < 0 else ("person" if inputs[12][blocker] else "rigid"),
            clearance_limited=bool(blocker < 0 and math.isfinite(stop)),
            conflict_arc_m=None if math.isinf(sample) else float(sample),
            prediction_s=None if blocker < 0 else float(predicted_s),
            admitted=bool(admitted),
            mode=execution.mode,
            next_speed_mps=execution.speed_mps,
            arrived=execution.arrived,
        )
        self.last[i] = row
        if t - self.last_sample.get(i, -math.inf) < 1.0 - 1e-7:
            return
        self.last_sample[i] = t
        if reservation_snapshot is not None:
            # Adapted from Claude 781b959's causal reservation journal idea.
            # Sample the existing read-only diagnostic at 1 Hz, not on every
            # motion tick; never import another backend's reservation policy.
            row["reservation_snapshot"] = dict(sampled_at_s=round(t, 6), data=copy.deepcopy(reservation_snapshot()))
        self.history.append(row)
        if admitted and inputs[5] < 0.2 and blocker >= 0:
            # Copy only sampled, slow-blocked inputs. No terrain or unbounded history.
            self.snapshots[i] = dict(
                t_s=round(t, 6),
                inputs=tuple(v.copy() if isinstance(v, np.ndarray) else v for v in inputs),
                expected=tuple(float(v) for v in detail),
            )

    def save(self, out):
        out = Path(out)
        data = {}
        entries = []
        for i, snapshot in sorted(self.snapshots.items()):
            prefix = str(i) + "_"
            for name, value in zip(INPUT_NAMES, snapshot["inputs"], strict=False):
                data[prefix + name] = np.asarray(value)
            entries.append(dict(index=i, vehicle=self.ids[i], t_s=snapshot["t_s"], expected=snapshot["expected"]))
        meta = dict(
            schema="awr.life.drive_lookahead.v1",
            ids=self.ids,
            snapshots=entries,
            source_sha256={
                n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest()
                for n in ("vehicles.py", "kernels.py", "drive_diagnostics.py")
            },
            note="Decision-input replay only, not a full-world checkpoint or a movement counterfactual.",
        )
        data["metadata_json"] = np.array(json.dumps(meta))
        path = out.parent / (out.name + ".drive-lookahead.npz")
        np.savez_compressed(path, **data)
        summary = dict(
            schema="awr.life.drive_diagnostics.v1",
            last=list(self.last.values()),
            samples=list(self.history),
            replay=path.name,
            snapshots=entries,
            admitted=False,
        )
        (out.parent / (out.name + ".drive-diagnostics.json")).write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def replay(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        if schema_id(meta["schema"]) != "awr.life.drive_lookahead.v1":
            raise ValueError("Unknown drive lookahead schema")
        rows = []
        for entry in meta["snapshots"]:
            values = [data[str(entry["index"]) + "_" + n] for n in INPUT_NAMES]
            args = [a.item() if a.ndim == 0 else a for a in values]
            actual = stop_arc_detail(*args)
            rows.append(dict(entry, actual=list(actual), exact=bool(np.array_equal(actual, entry["expected"]))))
    return dict(metadata=meta, rows=rows, all_exact=bool(rows) and all(r["exact"] for r in rows), admitted=False)
