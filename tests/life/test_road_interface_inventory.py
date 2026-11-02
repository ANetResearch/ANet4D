from copy import deepcopy

from awr.life.civil.road_interface_inventory import inventory


def fixture():
    def road(name, start, end, xy, oneway=False):
        return dict(
            properties=dict(
                feature="edge", edge_id=name, from_node=start, to_node=end, oneway=oneway, agents=["vehicle"]
            ),
            geometry=dict(coordinates=xy),
        )

    src = dict(
        features=[
            road("a", "west", "j", [[0, 0], [10, 0]]),
            road("b", "j", "east", [[10, 0], [20, 0]]),
            road("c", "other", "north", [[10, 1], [10, 20]]),
        ]
    )
    nodes = {
        n: dict(kind="lane", pose=[x, y, 0], traffic_resource="road:" + r, direction=d)
        for n, r, d, x, y in [
            ("a", "a", "forward", 9, 0),
            ("b", "b", "forward", 11, 0),
            ("reverse", "b", "reverse", 11, 0),
            ("c", "c", "forward", 10, 1),
        ]
    }
    doc = dict(nodes=nodes, edges={n: dict(source="a", target=n, kind="join") for n in ["b", "reverse", "c"]})
    return doc, src


def test_nearby_roads_and_wrong_direction_are_not_inferred_junctions():
    doc, src = fixture()
    old = deepcopy(doc)
    r = inventory(doc, src)
    assert r["join_counts"] == dict(
        source_incident_turn_candidate=1, source_direction_not_a_junction_turn=1, no_shared_source_junction=1
    )
    assert len(r["candidate_interfaces"]) == 2 and not r["admitted"] and doc == old


def test_bad_declared_junction_and_oneway_do_not_gain_authority():
    doc, src = fixture()
    doc["edges"]["b"]["source_junction"] = "nearby"
    src["features"][1]["properties"]["oneway"] = True
    r = inventory(doc, src)
    assert r["join_counts"]["explicit_junction_conflicts_with_source"] == 1
    assert r["join_counts"]["against_source_oneway"] == 1
    assert not r["candidate_interfaces"]


def test_source_node_identity_alone_does_not_override_coordinate_mismatch():
    doc, src = fixture()
    src["features"][1]["geometry"]["coordinates"][0] = [100, 100]
    r = inventory(doc, src)
    assert r["join_counts"]["source_junction_coordinates_disagree"] == 1 and not r["candidate_interfaces"]
