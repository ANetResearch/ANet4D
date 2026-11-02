"""Bounded diagnostic samples; wall time never influences simulated decisions."""

import heapq


class SlowTicks:
    def __init__(self, limit=16):
        self.limit = limit
        self.samples = []

    def add(self, tick, t_s, elapsed, before, after, events, first_event):
        if len(self.samples) >= self.limit and elapsed <= self.samples[0][0]:
            return
        stages = {k: max(0.0, v - before.get(k, 0.0)) for k, v in after.items()}
        recent = events[first_event:]
        row = dict(
            tick=tick,
            t_s=round(t_s, 3),
            phase="warmup" if tick < 0 else "recording",
            wall_s=elapsed,
            stages_s=stages,
            other_s=max(0.0, elapsed - sum(stages.values())),
            director_events=len(recent),
            actor_sample=sorted({e["actor"] for e in recent if e.get("actor")})[:16],
        )
        item = (elapsed, tick, row)
        if len(self.samples) < self.limit:
            heapq.heappush(self.samples, item)
        else:
            heapq.heapreplace(self.samples, item)

    def report(self):
        return [row for _, _, row in sorted(self.samples, key=lambda p: (-p[0], p[1]))]
