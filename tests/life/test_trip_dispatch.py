from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from awr.life.motion.trip_dispatch import Trip, TripDispatcher


class Backend:
    """Transactions only; native motion is covered by actual shared-world runs."""

    def __init__(self):
        self.graph = SimpleNamespace(
            nodes={
                n: dict(kind="berth", pose=np.array([x, 0.0, 0.0]))
                for n, x in [("A", 0), ("B", 100), ("C", 200), ("D", 300)]
            }
        )
        self.rows = {n: np.zeros(43) for n in ("a", "b")}
        self.rows["b"][1] = 300
        self.current = SimpleNamespace(names=("a", "b"), time_s=4.0, row=lambda n: self.rows[n])
        self.active = {}
        self.plans = {}
        self.calls = Counter()
        self.finished = set()

    def _at(self, actual, target, speed):
        return np.allclose(actual, target) and speed < 0.08

    def plan(self, vehicle, source, target):
        self.calls[vehicle] += 1
        e = f"{vehicle}:{source}>{target}"
        self.plans[e] = (vehicle, source, target)
        return SimpleNamespace(evidence=e)

    def acquire(self, evidence, pose, speed):
        v, s, t = self.plans[evidence]
        target = self.graph.nodes[t]["pose"]
        if any(n != v and np.allclose(row[[1, 2, 8]], target) for n, row in self.rows.items()):
            return False
        if any(self.plans[e][2] == t for e in self.active.values()):
            return False
        self.active[v] = evidence
        return True

    def can_arrive(self, evidence, pose, speed, *, require_executed=False):
        assert require_executed
        v, s, t = self.plans[evidence]
        return v in self.finished and self._at(pose, self.graph.nodes[t]["pose"], speed)

    def arrive(self, evidence, pose, speed):
        v, _, _ = self.plans[evidence]
        del self.active[v]
        self.finished.remove(v)
        return True

    def finish(self, name):
        _, _, target = self.plans[self.active[name]]
        self.rows[name][[1, 2, 8]] = self.graph.nodes[target]["pose"]
        self.current.time_s += 10
        self.finished.add(name)


def test_oneway_parking_retains_occupancy_and_never_invents_a_return():
    b = Backend()
    d = TripDispatcher(b, {"a": "A", "b": "D"})
    d.submit(Trip("a1", "a", "A", "B", 4, "visit"))
    d.submit(Trip("b1", "b", "D", "B", 4, "visit"))
    assert set(d.dispatch()) == {"a"}
    for t in (5, 10, 20):
        b.current.time_s = t
        d.dispatch()
        d.observe()
    assert b.calls == {"a": 1, "b": 1} and not d.completed
    b.finish("a")
    assert d.observe()[0].id == "a1"
    assert d.parked["a"] == "B" and "a" not in d.pending and "a" not in d.active
    d.dispatch()
    assert not b.active and b.calls["b"] == 2  # parked body still blocks B
    b.current.time_s = 1500
    d.dispatch()
    d.observe()
    assert d.parked["a"] == "B" and b.calls["b"] == 2
    d.submit(Trip("a2", "a", "B", "C", 1500, "next_need"))
    d.dispatch()
    b.finish("a")
    d.observe()
    assert set(d.dispatch()) == {"b"}  # B becomes available through actual motion


def test_future_departure_and_completion_use_authoritative_observations():
    b = Backend()
    d = TripDispatcher(b, {"a": "A"})
    d.submit(Trip("a1", "a", "A", "B", 100, "visit"))
    assert d.dispatch() == {} and not b.calls
    b.current.time_s = 100
    d.dispatch()
    b.current.time_s = 1000
    assert d.observe() == [] and "a" in d.active
    b.finish("a")
    d.observe()
    assert d.completed[0]["t_s"] == 1010


def test_duplicate_requests_wrong_sources_and_external_execution_are_rejected():
    b = Backend()
    d = TripDispatcher(b, {"a": "A"})
    for trip in [
        Trip("x", "a", "B", "C", 4, "visit"),
        Trip("x", "a", "A", "A", 4, "visit"),
        Trip("x", "a", "A", "B", float("nan"), "visit"),
    ]:
        with pytest.raises(ValueError):
            d.submit(trip)
    t = Trip("a1", "a", "A", "B", 4, "visit")
    d.submit(t)
    with pytest.raises(ValueError):
        d.submit(t)
    d.dispatch()
    with pytest.raises(ValueError):
        d.submit(Trip("a2", "a", "A", "C", 4, "visit"))
    b.active["a"] = "foreign-evidence"
    with pytest.raises(ValueError, match="owner"):
        d.observe()


def test_clock_cannot_go_backwards_and_initial_berth_cannot_be_invented():
    b = Backend()
    with pytest.raises(ValueError, match="observed"):
        TripDispatcher(b, {"a": "B"})
    d = TripDispatcher(b, {"a": "A"})
    b.current.time_s = 3
    with pytest.raises(ValueError, match="clock"):
        d.dispatch()
