from copy import deepcopy

import numpy as np
import pytest

from awr.life.motion.native_road_windows import extend_chain


class Surface:
    def supported(self, points, kind):
        return np.ones(len(points), dtype=bool)


class Clearance:
    def pose(self, p):
        return p[0] not in (1, 3), "mesh" if p[0] in (1, 3) else None


def fixture():
    nodes = {
        str(i): dict(pose=[i, 0, 0], road="road", direction="forward", guide_offset_m=1.5, geometry_reference="body")
        for i in range(5)
    }
    edges = {
        str(i): dict(source=str(i), target=str(i + 1), kind="travel", traffic_resources=["road"], route={"arc": [0, 1]})
        for i in range(4)
    }
    return dict(nodes=nodes, edges=edges)


def test_unsafe_artificial_seam_extends_only_through_existing_directed_edges():
    doc = fixture()
    before = deepcopy(doc)
    chain, info = extend_chain(doc, "1", Surface(), Clearance())
    assert chain == ["0", "1"] and info["before_m"] == 1 and info["after_m"] == 0
    chain, info = extend_chain(doc, "2", Surface(), Clearance())
    assert chain == ["2", "3"] and info["after_m"] == 1
    assert doc == before


@pytest.mark.parametrize("change", ["road", "direction", "offset", "resource", "reverse", "budget"])
def test_extension_cannot_cross_road_direction_lane_resource_or_budget(change):
    doc = fixture()
    kwargs = {}
    if change == "road":
        doc["nodes"]["4"]["road"] = "other"
    elif change == "direction":
        doc["nodes"]["4"]["direction"] = "reverse"
    elif change == "offset":
        doc["nodes"]["4"]["guide_offset_m"] = 0
    elif change == "resource":
        doc["edges"]["3"]["traffic_resources"] = ["other"]
    elif change == "reverse":
        doc["edges"]["3"].update(source="4", target="3")
    elif change == "budget":
        kwargs["max_extra_m"] = 0.5
    chain, info = extend_chain(doc, "2", Surface(), Clearance(), **kwargs)
    assert chain is None and info["reason"] == "no_safe_source_port"
