"""Horizon-independent initial parking activities for the curb motion adapter.

This is a departure-demand adapter, not the missing resident driver/return-road
backend. Initial parking is observed part-way through an ongoing stay. Neither
recording duration nor roster size enters a vehicle's private random stream.
"""

import hashlib
import math
import random


def parking_activity(actor_id, seed, world_start_s, dwell_s=(120.0, 1800.0)):
    lo, hi = map(float, dwell_s)
    if not all(math.isfinite(x) for x in (lo, hi, world_start_s)) or not 0 < lo <= hi:
        raise ValueError("Invalid initial parking activity interval")
    rng = random.Random(int.from_bytes(hashlib.sha256(f"{seed}:parking-activity:{actor_id}".encode()).digest(), "big"))
    # Equilibrium renewal sampling: a random observer is more likely to see a
    # longer stay. Sample length-biased Uniform(lo,hi), then its current phase.
    duration = math.sqrt(lo * lo + rng.random() * (hi * hi - lo * lo))
    elapsed = rng.random() * duration
    return dict(
        purpose="resident_parking",
        duration_s=duration,
        elapsed_s=elapsed,
        started_s=world_start_s - elapsed,
        departure_due_s=world_start_s + duration - elapsed,
        trigger="actual completion of ongoing parking activity",
        limitations="verified outward curb trip only; resident drivers and return routing unavailable",
    )
