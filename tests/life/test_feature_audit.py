import json

import numpy as np
import pytest

from awr.life.motion.feature_audit import SCHEMA, audit, digest


def fixture(tmp_path, change=None):
    docs = dict(
        world={"id": "test", "geometryOnlyDiagnostic": True},
        coordinate={
            "worldId": "test",
            "frame": {"units": "m", "type": "ENU", "upAxis": "+Z"},
            "anchor": {"georeferenced": False},
        },
        semantics={"roads": [], "pois": [], "crossings": [], "doors": []},
        support_meta={"ny": 4, "nx": 4, "cell_m": 0.5, "unknown_is_blocked": True},
        places=[{"id": "home", "kind": "home", "xy": [0.0, 0.0], "parking": []}],
        initial={"actors": [{"id": "p1", "class": "person"}], "positions": [[0.0, 0.0]]},
    )
    arrays = dict(
        ground=np.zeros((4, 4)),
        walk_person=np.ones((4, 4), bool),
        z_person=np.zeros((4, 4)),
        walk_vehicle=np.zeros((4, 4), bool),
        z_vehicle=np.zeros((4, 4)),
    )
    if change:
        change(docs, arrays)
    layers = {}
    for name, data in docs.items():
        p = tmp_path / (name + ".json")
        p.write_text(json.dumps(data))
        layers[name] = dict(path=p.name, sha256=digest(p))
    for name, data in arrays.items():
        p = tmp_path / (name + ".npy")
        np.save(p, data)
        layers[name] = dict(path=p.name, sha256=digest(p))
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps(dict(schema=SCHEMA, id="test", layers=layers)))
    return bundle


def statuses(report):
    return {r["code"]: r["status"] for r in report["checks"]}


def test_small_pedestrian_environment_is_not_rejected_for_old_population_goal(tmp_path):
    b = fixture(tmp_path)
    r = audit(b, tmp_path)
    assert r["structurally_consistent"]
    assert r["facts"]["initial_recipe"]["count"] == 1
    assert statuses(r)["joint_capacity"] == "unverified"
    assert statuses(r)["facility_entrances"] == "unverified"
    assert r["admitted"] is False and r["release_evidence"] is False
    assert audit(b, tmp_path) == r


def test_tampered_layer_is_not_used_as_verified_feature(tmp_path):
    b = fixture(tmp_path)
    (tmp_path / "world.json").write_text('{"id":"another"}')
    r = audit(b, tmp_path)
    assert not r["structurally_consistent"]
    assert statuses(r)["digest:world"] == "error"


@pytest.mark.parametrize(
    "change,code",
    [
        (lambda d, a: a["z_person"].__setitem__((0, 0), np.nan), "support_mask:person"),
        (lambda d, a: a.update(walk_person=np.ones((3, 4), bool)), "support_shapes"),
        (lambda d, a: d["places"][0].update(parking=["nonexistent"]), "place_parking_references"),
        (lambda d, a: d["coordinate"]["frame"].update(units="feet"), "local_frame"),
        (lambda d, a: d["initial"]["actors"].append(dict(d["initial"]["actors"][0])), "actor_identity_alignment"),
    ],
)
def test_structural_faults_are_errors(tmp_path, change, code):
    r = audit(fixture(tmp_path, change), tmp_path)
    assert not r["structurally_consistent"] and statuses(r)[code] == "error"


def test_missing_required_layer_and_legacy_drift(tmp_path):
    b = fixture(
        tmp_path,
        lambda d, a: d.update(
            legacy_manifest={"layers": {"semantics": {"path": "old.json"}}, "capabilities": {"drive": {"drivers": 300}}}
        ),
    )
    r = audit(b, tmp_path)
    assert statuses(r)["legacy_manifest_drift"] == "unverified"
    assert not r["admitted"]  # Even a large historical declaration is not capacity evidence.
    (tmp_path / "ground.npy").unlink()
    r = audit(b, tmp_path)
    assert statuses(r)["missing:ground"] == "missing" and not r["structurally_consistent"]
