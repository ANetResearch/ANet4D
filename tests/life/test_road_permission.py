import copy

import pytest
from shapely.geometry import Point

from awr.life.civil.road_permission import driving_area


def fixture():
    roads = [
        dict(id="a", width_m=6.4, line=[[-30, 0], [0, 0]]),
        dict(id="b", width_m=6.4, line=[[0, 0], [0, 30]]),
        dict(id="near", width_m=3.2, line=[[0.1, 0.1], [30, 0.1]]),
    ]
    source = dict(
        features=[
            dict(
                properties=dict(
                    feature="edge",
                    edge_id=r["id"],
                    oneway=True,
                    agents=["vehicle"],
                    from_node=a,
                    to_node=b,
                    osm_id=r["id"],
                ),
                geometry=dict(coordinates=r["line"]),
            )
            for r, (a, b) in zip(roads, [("west", "j"), ("j", "north"), ("unrelated", "east")], strict=False)
        ]
    )
    sem = dict(
        roads=roads,
        verified_junction_regions=[
            dict(
                verified=True,
                source_junction="j",
                source_roads=["a", "b"],
                polygon=list(Point(0, 0).buffer(3.2).exterior.coords),
                evidence="reviewed ground image and physical candidate checks",
            )
        ],
    )
    return sem, source


def test_reviewed_shared_junction_does_not_round_terminal_road_ends():
    sem, source = fixture()
    baseline = driving_area(dict(sem, verified_junction_regions=[]))
    assert not baseline.covers(Point(1.9, -1.9))
    patched = driving_area(sem, source)
    assert patched.covers(Point(1.9, -1.9))
    assert not patched.covers(Point(-31.0, 0.0))
    assert not patched.covers(Point(0.0, 31.0))
    assert source["features"][0]["properties"]["oneway"] is True


def test_unreviewed_regions_and_parking_never_grant_driving_permission():
    sem, source = fixture()
    sem["verified_junction_regions"][0]["verified"] = False
    sem["verified_parking_regions"] = [dict(verified=False, polygon=[[100, 100], [110, 100], [110, 110], [100, 110]])]
    area = driving_area(sem)
    assert not area.covers(Point(1.9, -1.9))
    assert not area.covers(Point(105, 105))


def test_patch_requires_true_source_incidence_evidence_and_width_bound():
    sem, source = fixture()
    with pytest.raises(ValueError, match="source road graph"):
        driving_area(sem)
    for change in [
        dict(source_roads=["a", "near"]),
        dict(source_roads=["a", "a"]),
        dict(evidence=""),
        dict(polygon=list(Point(0, 0).buffer(3.3).exterior.coords)),
    ]:
        bad = copy.deepcopy(sem)
        bad["verified_junction_regions"][0].update(change)
        with pytest.raises(ValueError):
            driving_area(bad, source)
    bad = copy.deepcopy(sem)
    bad["verified_junction_regions"] *= 2
    with pytest.raises(ValueError):
        driving_area(bad, source)
