import json
from types import SimpleNamespace

import shapely
from test_activity_driving import contacts, fixture  # noqa: F401 (contacts: autouse fixture)

from awr.life.motion.traffic_evidence import record_traffic, snapshot


def test_failure_snapshot_retains_exact_reservations_without_refresh(tmp_path, monkeypatch):
    routes, starts, _ = fixture()
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(a.evidence, starts["a"], 0.0)
    assert routes.acquire(b.evidence, starts["b"], 0.0)
    ra, rb = (routes.traffic.layout(p.evidence).runs[0] for p in (a, b))
    assert routes.traffic_limit("b", rb.stop - 1.0, 0.0) is None
    assert routes.traffic_limit("a", ra.stop - 1.0, 0.0) == ra.stop
    before = dict(routes.traffic.stats)

    def forbidden(*a, **kw):
        raise AssertionError("Evidence must not replan, refresh or grant a reservation")

    for name in ("course", "layout", "_refresh", "_try", "limit"):
        monkeypatch.setattr(routes.traffic, name, forbidden)
    routes.graph.freeze()
    first = snapshot(routes)
    name = record_traffic(tmp_path / "tape", SimpleNamespace(driving=routes))
    assert json.loads((tmp_path / name).read_text()) == first
    assert routes.traffic.stats == before
    wait = first["trips"]["a"]["waiting"]
    assert wait["reason"] == "conflict"
    waiting_run = first["resources"][wait["run"]]
    assert waiting_run["stop_m"] == ra.stop
    for raw, (edge, body) in zip(waiting_run["parts"], ra.parts, strict=False):
        assert raw["edge"] == edge
        assert shapely.from_wkb(bytes.fromhex(raw["wkb_hex"])).equals_exact(body, 0.0)
    assert first["trips"]["b"]["held"]
    assert first["courses"][a.evidence]["edges"] == list(routes.traffic.courses[a.evidence].edges)
    first["trips"]["a"]["held"].append(999)
    first["courses"][a.evidence]["route"]["points"][0][0] = 999.0
    assert snapshot(routes) != first  # detached output, no source aliases


def test_no_driving_does_not_create_a_traffic_artifact(tmp_path):
    assert record_traffic(tmp_path / "tape", SimpleNamespace(driving=None)) is None
    assert not list(tmp_path.iterdir())
