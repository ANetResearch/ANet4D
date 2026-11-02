import copy
import json

import pytest

from awr.life.civil.road_service_gaps import inspect


def fixture():
    a = json.dumps(["road", "west", "outgoing", "forward"])
    b = json.dumps(["road", "east", "incoming", "forward"])
    nodes = {
        n: dict(kind="lane", pose=[x, 0.0, 0.0], traffic_resource="road:road", direction="forward")
        for n, x in [("a", 0), ("b", 10), ("c", 12), ("d", 20)]
    }
    doc = dict(
        nodes=nodes,
        edges={n: dict(source=i, target=j, kind="travel") for n, i, j in [("ab", "a", "b"), ("cd", "c", "d")]},
    )
    src = dict(
        features=[
            dict(
                properties=dict(feature="edge", agents=["vehicle"], edge_id="road"),
                geometry=dict(coordinates=[[0, 0], [20, 0]]),
            )
        ]
    )
    contract = dict(interfaces={a: ["a"], b: ["d"]}, connections=[dict(id="trip", source=a, target=b)])
    return doc, src, contract


def test_small_gap_is_not_an_edge_or_capacity_claim():
    args = fixture()
    old = copy.deepcopy(args)
    r = inspect(*args)
    row = r["rows"][0]
    pair = row["nearest"][0]
    assert row["status"] == "missing_transition"
    assert pair["source_node"] == "b" and pair["target_node"] == "c" and pair["distance_m"] == 2
    assert pair["approach_edges"] == ["ab"] and pair["departure_edges"] == ["cd"]
    assert not pair["physically_verified"] and not r["admitted"] and args == old


def test_reverse_search_never_crosses_a_berth_to_claim_connectivity():
    doc, src, contract = fixture()
    doc["nodes"]["p"] = dict(kind="berth", pose=[11, 0, 0])
    doc["edges"].update(bp=dict(source="b", target="p"), pc=dict(source="p", target="c"))
    row = inspect(doc, src, contract)["rows"][0]
    assert row["status"] == "missing_transition"
    assert row["nearest"][0]["departure_edges"] == ["cd"]
    assert row["same_road_destination_nodes"] == 2


def test_existing_detour_is_reported_as_an_actual_path():
    doc, src, contract = fixture()
    doc["nodes"]["other"] = dict(kind="lane", pose=[10, 10, 0], traffic_resource="road:elsewhere")
    doc["edges"].update(bo=dict(source="b", target="other"), oc=dict(source="other", target="c"))
    row = inspect(doc, src, contract)["rows"][0]
    assert row["status"] == "whole_network_path" and row["edges"] == ["ab", "bo", "oc", "cd"]


def test_backward_leg_is_returned_in_executable_order():
    doc, src, contract = fixture()
    doc["nodes"]["e"] = dict(doc["nodes"]["d"], pose=[18, 0, 0])
    del doc["edges"]["cd"]
    doc["edges"].update(ce=dict(source="c", target="e"), ed=dict(source="e", target="d"))
    pair = inspect(doc, src, contract)["rows"][0]["nearest"][0]
    assert pair["departure_edges"] == ["ce", "ed"]


def test_reverse_source_road_keeps_positive_progress_in_driving_direction():
    doc, src, contract = fixture()
    for n in doc["nodes"].values():
        n["direction"] = "reverse"
        n["pose"] = [20 - n["pose"][0], 0, 3.141592653589793]
    aliases = {}
    for key in contract["interfaces"]:
        r, j, side, _ = json.loads(key)
        aliases[key] = json.dumps([r, "east" if j == "west" else "west", side, "reverse"])
    contract["interfaces"] = {aliases[k]: v for k, v in contract["interfaces"].items()}
    for c in contract["connections"]:
        c["source"], c["target"] = aliases[c["source"]], aliases[c["target"]]
    pair = inspect(doc, src, contract)["rows"][0]["nearest_positive_station"][0]
    assert pair["directed_station_delta_range_m"] == [2.0, 2.0]


def test_zero_distance_duplicate_pose_remains_disconnected():
    doc, src, contract = fixture()
    doc["nodes"]["c"]["pose"] = [10, 0, 0]
    row = inspect(doc, src, contract)["rows"][0]
    assert row["status"] == "missing_transition" and row["nearest"][0]["distance_m"] == 0
    assert row["nearest_positive_station"][0]["distance_m"] > 0


def test_deterministic_bound_does_not_mix_in_other_road_poses():
    doc, src, contract = fixture()
    doc["nodes"]["near"] = dict(kind="lane", pose=[11, 0, 0], traffic_resource="road:elsewhere", direction="forward")
    doc["edges"]["bn"] = dict(source="b", target="near")
    first = inspect(doc, src, contract, closest=1)
    doc["nodes"] = dict(reversed(list(doc["nodes"].items())))
    doc["edges"] = dict(reversed(list(doc["edges"].items())))
    assert inspect(doc, src, contract, closest=1) == first
    assert first["rows"][0]["nearest"][0]["source_node"] == "b"
    assert len(first["rows"][0]["nearest"]) == 1


@pytest.mark.parametrize("closest", [0, -1, 21, True])
def test_diagnostic_work_is_bounded(closest):
    with pytest.raises(ValueError):
        inspect(*fixture(), closest=closest)
