from copy import deepcopy

import numpy as np
import pytest

from awr.life.civil.activity import Place
from awr.life.civil.activity.access import side_access_candidates
from awr.life.civil.driving_graph import DrivingGraph, route_arrays
from awr.life.motion.cycle_opportunities import nominate
from awr.life.motion.purpose_screen import PurposePolicy, PurposeScreen


def fixture(*, near=False):
    g = DrivingGraph()
    places = [
        Place("home", (0.0, 4.0), "home", ("h",)),
        Place("service", (120.0, 4.0), "shop", ("s",), group="real-shop"),
    ]
    for name, x in [("a", 0.0), ("b", 120.0)] + ([("near", 30.0)] if near else []):
        g.add_node(name, [x, 0.0, 0.0], kind="berth", region=name)
    rows = []
    for name, node in g.nodes.items():
        anchor = next(side_access_candidates(node["pose"], [4, 2, 1.5], 0.6))
        xy = anchor["xy"]

        def route(a, b):
            return dict(points=[list(a), list(b)], arc=[0.0, float(np.linalg.norm(np.asarray(b) - a))])

        links = places if name == "a" else places[1:]
        rows.append(
            dict(
                berth=name,
                pose=node["pose"],
                access=[
                    dict(
                        **anchor,
                        connections=[
                            dict(
                                place=p.id,
                                kind=p.kind,
                                group=p.group or p.id,
                                xy=p.xy,
                                outward=route(xy, p.xy),
                                returning=route(p.xy, xy),
                            )
                            for p in links
                        ],
                    )
                ],
            )
        )
    for a in g.nodes:
        for b in g.nodes:
            if a != b:
                g.add_edge(
                    a + "-" + b,
                    a,
                    b,
                    route_arrays([g.nodes[a]["pose"][:2], g.nodes[b]["pose"][:2]], [0, 0], [1]),
                    kind="travel",
                )
    report = dict(
        schema="awr.life.berth_activity_access_candidate.v1",
        vehicle_asset="car",
        size_m=[4, 2, 1.5],
        human_turn_radius_m=0.6,
        rows=rows,
    )
    return g, places, report


def screen(g, places, report):
    return PurposeScreen(g, places, report, asset="car", size=[4, 2, 1.5])


def test_service_identity_walk_cost_and_reverse_cost_are_part_of_nomination():
    g, places, r = fixture()
    s = screen(g, places, r)
    result = s.evaluate("a", "b", [120, 120])
    assert result["eligible"] and not result["admitted"]
    row = result["options"][0]
    assert row["service_group"] == "real-shop"
    assert row["known_walk_roundtrip_s"] > row["car_outward_lower_s"] + row["car_return_lower_s"]
    dominated = s.evaluate("a", "b", [120, 1000])
    assert not dominated["eligible"] and dominated["rejected"]["known_walk_dominates_optimistic_drive"] == 1


def test_missing_walking_alternative_is_unknown_not_unreachable_or_admitted():
    g, places, r = fixture()
    r["rows"][0]["access"][0]["connections"] = r["rows"][0]["access"][0]["connections"][:1]
    row = screen(g, places, r).evaluate("a", "b", [120, 120])["options"][0]
    assert row["known_walk_roundtrip_s"] is None
    assert row["mode_evidence"] == "unknown_walking_alternative" and row["requires_direct_walk_comparison"]


def test_same_facility_slots_do_not_inflate_number_of_purposes():
    g, places, r = fixture()
    places.append(Place("service-2", (120.0, 4.0), "shop", ("s2",), group="real-shop"))
    for x in r["rows"]:
        c = deepcopy(next(c for c in x["access"][0]["connections"] if c["place"] == "service"))
        c["place"] = "service-2"
        x["access"][0]["connections"].append(c)
    result = screen(g, places, r).evaluate("a", "b", [120, 120])
    assert result["possible_options"] == 1 and len(result["options"]) == 1


def test_no_cross_side_borrowing_and_no_stale_berth_pose():
    g, places, r = fixture()
    r["rows"][1]["pose"] = np.array(r["rows"][1]["pose"]) + [0.1, 0, 0]
    with pytest.raises(ValueError, match="pose changed"):
        screen(g, places, r)
    g, places, r = fixture()
    r["rows"].append(deepcopy(r["rows"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        screen(g, places, r)
    with pytest.raises(ValueError, match="policy"):
        PurposePolicy(walk_mps=0)


def test_purpose_filter_precedes_top_k_and_preserves_graph():
    g, places, r = fixture(near=True)
    before = set(g.edges)
    # The short cycle includes a small real detour, still cheaper in topology
    # than the far destination, but it saves less walking than driving costs.
    g.edges["a-near"]["route"] = route_arrays([[0, 0], [15, 20], [30, 0]], [0, 0, 0], [1, 1])
    g.edges["near-a"]["route"] = route_arrays([[30, 0], [15, 20], [0, 0]], [0, 0, 0], [1, 1])
    # Near berth leaves a long walk to exactly the same service; the longer car
    # cycle is the credible first candidate under the declared cost model.
    result = nominate(g, {}, (), max_results=1, purpose_screen=screen(g, places, r))
    assert result["purpose_screened"] and result["topology_pairs"] == 3
    assert result["rows"][0]["pair"] == ["a", "b"]
    assert result["excluded"]["no_credible_activity_option"] == 2
    assert result["rows"][0]["purpose_screen"]["eligible"]
    assert set(g.edges) == before and not result["executable"]


def test_walk_alternatives_reuse_exact_shared_endpoints_and_cached_directed_trees():
    g, places, r = fixture()
    r["rows"][0]["access"][0]["connections"] = r["rows"][0]["access"][0]["connections"][:1]
    park = Place("shared-path", (60.0, 5.0), "park", ("p",), group="park")
    places.append(park)
    for row in r["rows"]:
        xy = np.asarray(row["access"][0]["xy"])
        length = float(np.linalg.norm(np.asarray(park.xy) - xy))
        row["access"][0]["connections"].append(
            dict(
                place=park.id,
                kind=park.kind,
                group=park.group,
                xy=park.xy,
                outward=dict(points=[xy.tolist(), list(park.xy)], arc=[0, length]),
                returning=dict(points=[list(park.xy), xy.tolist()], arc=[0, length]),
            )
        )
    s = screen(g, places, r)
    w = s.walking_alternative("home", "shop", "real-shop")
    assert w and len(w["outward_links"]) == 4
    assert w["outward_links"][0][0] == ("place", "home")
    assert w["outward_links"][-1][1] == ("place", "service")
    for path in (w["outward_links"], w["return_links"]):
        assert all(a[1] == b[0] for a, b in zip(path, path[1:], strict=False))
    assert w["return_links"][0][0] == ("place", "service") and w["return_links"][-1][1] == ("place", "home")
    assert len(s.trees) == 2
    assert s.walking_alternative("home", "shop", "real-shop") == w and len(s.trees) == 2
