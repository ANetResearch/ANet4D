"""Seeded activity demand, separate from physical feasibility and travel optimisation.

Weighted exponential races order modes and facilities without a global RNG,
recording horizon, map coordinates or query identity. A priority is a proposal,
never permission to bypass a route, resource, opening-time or return check.
"""

import hashlib
import json
import math


class Selection:
    def __init__(self, config=None):
        self.kind = "travel_utility_v2"
        self.weights = {}
        if config is None:
            return  # preserve historical trajectories for explicit comparisons
        if (
            not isinstance(config, dict)
            or set(config) != {"kind", "mode_weights"}
            or config["kind"] != "diverse_feasible_v1"
        ):
            raise ValueError("Selection requires diverse_feasible_v1 and mode_weights")
        weights = config["mode_weights"]
        if (
            not isinstance(weights, dict)
            or set(weights) != {"walk", "car"}
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in weights.values())
        ):
            raise ValueError("Selection mode weights require positive finite walk and car values")
        self.kind = config["kind"]
        self.weights = dict(weights)

    @staticmethod
    def race(seed, actor, decision, domain, key):
        raw = json.dumps([seed, actor, decision, domain, key], ensure_ascii=False, separators=(",", ":")).encode()
        # 52-bit midpoint avoids log(0) and rounding to 1; independent of Python hash seed.
        n = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") >> 12
        return -math.log((n + 0.5) / 2**52)

    def mode_order(self, seed, actor, decision, has_car):
        if self.kind == "travel_utility_v2":
            return ()
        modes = ("walk", "car") if has_car else ("walk",)
        return tuple(sorted(modes, key=lambda m: (self.race(seed, actor, decision, "mode", m) / self.weights[m], m)))

    def utility(self, seed, actor, decision, purpose, group, weight):
        return weight / self.race(seed, actor, decision, "facility", [purpose, group])

    def score(self, numerator, travel_s):
        return numerator if self.kind == "diverse_feasible_v1" else numerator / (1.0 + travel_s / 300.0)
