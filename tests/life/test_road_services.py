import hashlib
import json

import numpy as np
import pytest

from awr.life.civil.driving_graph import DrivingGraph, load_candidate, route_arrays
from awr.life.civil.road_services import DirectedServices, load_bound_services


def graph(links, berths=()):
    g = DrivingGraph()
    for i, node in enumerate(sorted({n for pair in links for n in pair})):
        g.add_node(node, [i * 10.0, 0.0, 0.0], kind="berth" if node in berths else "lane")
    for a, b in links:
        route = route_arrays(np.array([g.nodes[a]["pose"][:2], g.nodes[b]["pose"][:2]]), np.zeros(2))
        g.add_edge(a + b, a, b, route, kind="travel")
    return g


def service(g, interfaces=None, connections=None):
    return DirectedServices(
        g,
        interfaces or {"west": ["a"], "east": ["d"]},
        connections or [dict(id="eastbound", source="west", target="east")],
    )


def test_alternative_waypoints_can_disappear_but_actual_path_is_preserved():
    g = graph([("a", "b"), ("b", "d"), ("a", "c"), ("c", "d")])
    audit = service(g).compare(after=["ab", "bd"])
    row = audit["rows"][0]
    assert row["status"] == "preserved" and row["after"]["edges"] == ["ac", "cd"]
    assert audit["cut_services"] == 0 and not audit["admitted"]
    # Actual graph nodes survive unchanged; no identity erasure to satisfy a check.
    assert "b" in g.nodes
    assert service(g).compare(after=["ab", "ac"])["rows"][0]["status"] == "cut"


def test_direction_and_actual_occupied_snapshot_not_undirected_connectivity():
    g = graph([("a", "b"), ("b", "d"), ("d", "c"), ("c", "a")])
    s = service(
        g, connections=[dict(id="east", source="west", target="east"), dict(id="west", source="east", target="west")]
    )
    rows = {r["id"]: r for r in s.compare(before=["ab"], after=["ab", "dc"])["rows"]}
    assert rows["east"]["status"] == "unavailable" and rows["west"]["status"] == "cut"
    assert all(r["status"] == "preserved" for r in s.compare()["rows"])  # no occupancy mutation


def test_interface_alternatives_never_become_connecting_edges():
    g = graph([("a", "b"), ("c", "d")])
    s = service(
        g,
        {"west": ["a"], "junction": ["b", "c"], "east": ["d"]},
        [
            dict(id="enter", source="west", target="junction"),
            dict(id="exit", source="junction", target="east"),
            dict(id="whole", source="west", target="east"),
        ],
    )
    rows = {r["id"]: r for r in s.compare()["rows"]}
    assert rows["enter"]["status"] == rows["exit"]["status"] == "preserved"
    assert rows["whole"]["status"] == "unavailable"


def test_berths_are_terminal_and_actual_owner_access_is_independent():
    g = graph([("a", "p"), ("p", "d"), ("a", "b"), ("b", "d")], berths=["p"])
    s = service(
        g,
        {"west": ["a"], "east": ["d"], "owner": ["p"]},
        [dict(id="road", source="west", target="east"), dict(id="owner_exit", source="owner", target="east")],
    )
    rows = {r["id"]: r for r in s.compare(after=["ab"])["rows"]}
    assert rows["road"]["status"] == "cut"  # cannot use a parked bay as a bridge
    assert rows["owner_exit"]["status"] == "preserved"
    rows = {r["id"]: r for r in s.compare(after=["pd"])["rows"]}
    assert rows["road"]["status"] == "preserved" and rows["owner_exit"]["status"] == "cut"


def test_local_service_scope_cannot_be_preserved_by_unrelated_detour():
    g = graph([("a", "b"), ("b", "d"), ("a", "c"), ("c", "d")])
    s = service(g, connections=[dict(id="local", source="west", target="east", allowed_edges=["ab", "bd"])])
    assert s.compare(after=["ab"])["rows"][0]["status"] == "cut"


def test_whole_interface_loss_does_not_pass_vacuously():
    g = graph([("a", "d"), ("b", "c")])
    s = service(g, {"west": ["a", "b"], "east": ["c", "d"]})
    assert s.compare(after=["ad", "bc"])["cut_services"] == 1
    with pytest.raises(ValueError, match="disjoint"):
        service(g, {"west": ["a"], "east": ["a", "d"]})
    with pytest.raises(ValueError, match="Unknown blocked"):
        s.compare(after=["wrong-map-edge"])


def test_topology_snapshot_and_results_are_deterministic():
    g = graph([("a", "b"), ("b", "d"), ("a", "c"), ("c", "d")])
    s = service(g)
    before = s.compare()
    g.edges.clear()
    g.nodes.clear()
    assert s.compare() == before


def test_bound_contract_and_explicit_source_junction_survive_roundtrip():
    g = graph([("a", "d")])
    g.edges["ad"]["kind"] = "join"
    g.edges["ad"]["source_junction"] = "actual-junction"
    raw = json.dumps(
        dict(schema="awr.life.driving_map_candidate.v1", nodes=g.nodes, edges=g.edges), default=lambda v: v.tolist()
    ).encode()
    assert load_candidate(json.loads(raw)).edges["ad"]["source_junction"] == "actual-junction"
    contract = dict(
        schema="awr.life.directed_service_contract.v1",
        driving_map_sha256=hashlib.sha256(raw).hexdigest(),
        feature_evidence={"source": "reviewed fixture"},
        interfaces={"west": ["a"], "east": ["d"]},
        connections=[dict(id="eastbound", source="west", target="east")],
    )
    assert load_bound_services(raw, json.dumps(contract).encode()).compare()["cut_services"] == 0
    with pytest.raises(ValueError, match="Exact map"):
        load_bound_services(raw + b" ", json.dumps(contract).encode())
    contract.pop("feature_evidence")
    with pytest.raises(ValueError, match="feature evidence"):
        load_bound_services(raw, json.dumps(contract).encode())


def test_alternative_interface_search_matches_independent_terminal_paths():
    # Compare against the existing one-origin/one-target planner, not another
    # grouped search. In particular, reaching a different candidate origin
    # must not create an unrecorded teleport between its alternatives.
    for seed in range(30):
        rng = np.random.default_rng(seed)
        names = list("abcdefgh")
        links = [(a, b) for a in names for b in names if a != b and rng.random() < 0.4]
        # Keep all chosen endpoint names present, without guaranteeing a route.
        links.extend([("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")])
        links = sorted(set(links))
        g = graph(links, berths=["b", "f"])
        interfaces = {"start": ["a", "b"], "end": ["g", "h"]}
        blocked = [e for e in g.edges if rng.random() < 0.4]
        report = service(g, interfaces, [dict(id="trip", source="start", target="end")]).compare(after=blocked)
        path = report["rows"][0]["after"]
        expected = any(
            g.path(a, b, blocked_edges=blocked) is not None for a in interfaces["start"] for b in interfaces["end"]
        )
        assert (path is not None) == expected
        if path is not None:
            node = path["source_node"]
            assert node in interfaces["start"]
            for edge in path["edges"]:
                assert edge not in blocked and g.edges[edge]["source"] == node
                node = g.edges[edge]["target"]
            assert node == path["target_node"] and node in interfaces["end"]
            assert all(g.nodes[g.edges[e]["target"]]["kind"] != "berth" for e in path["edges"][:-1])
