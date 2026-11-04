"""Scene principle package: pack, check, resolve, admit and the command line."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from awr.life.principle import PrincipleError, admit, check, demand_errors, pack, resolve

NX, NY, CELL = 40, 30, 1.0


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def legacy_inputs(tmp: Path, *, walk_vehicle=True) -> dict:
    """A minimal legacy scene: support grids, semantics, world reference, one SUMO net and one places file."""
    nav = tmp / "support"
    nav.mkdir()
    ground = np.zeros((NY, NX), np.float32)
    ground[:, :4] = np.nan  # a strip without ground
    person = np.isfinite(ground)
    vehicle = person.copy()
    vehicle[:10] = False
    if not walk_vehicle:
        vehicle[:] = False
    np.save(nav / "ground.npy", ground)
    np.save(nav / "walk_person.npy", person)
    np.save(nav / "walk_vehicle.npy", vehicle)
    np.save(nav / "z_person.npy", ground)
    np.save(nav / "z_vehicle.npy", ground)
    (nav / "meta.json").write_text(
        json.dumps(
            {
                "schema": "awr.life.support.v1",
                "x0": 0.0,
                "y0": 0.0,
                "nx": NX,
                "ny": NY,
                "cell_m": CELL,
                "unknown_is_blocked": True,
                "method": "test",
            }
        )
    )
    sem = {
        "schema": "chimera.town_semantics.v1",
        "bounds": {"min": [0, 0], "max": [NX, NY]},
        "roads": [{"id": "r1", "line": [[5, 15], [35, 15]], "width_m": 6.0, "kind": "residential"}],
        "buildings": [
            {"id": "b1", "polygon": [[10, 20], [14, 20], [14, 24], [10, 24]], "kind": "house", "source": "derived"}
        ],
        "pois": [
            {
                "id": "p1",
                "kind": "school",
                "xy": [12, 19],
                "building": "b1",
                "verified_entrance": False,
                "source": "assigned",
            }
        ],
        "open_areas": [{"id": "o1", "polygon": [[20, 20], [30, 20], [30, 28], [20, 28]]}],
    }
    sem_path = tmp / "semantics.json"
    sem_path.write_text(json.dumps(sem))
    world = tmp / "world"
    (world / "geometry/nav").mkdir(parents=True)
    (world / "geometry/mesh/raycast").mkdir(parents=True)
    (world / "geometry/mesh/raycast/source.json").write_text(json.dumps({"mesh_sha256": "0" * 64}))
    (world / "geometry/nav/person.json").write_text(json.dumps({"bin": {"sha256": _sha(nav / "walk_person.npy")}}))
    (world / "geometry/nav/vehicle.json").write_text(json.dumps({"bin": {"sha256": _sha(nav / "walk_vehicle.npy")}}))
    (world / "world.json").write_text(
        json.dumps(
            {"id": "toytown", "contentVersion": "cv-1", "name": "Toy Town", "bounds": {"min": [0, 0], "max": [NX, NY]}}
        )
    )
    lanes = tmp / "net.net.xml"
    lanes.write_text("<net/>")
    places = tmp / "places.json"
    places.write_text(
        json.dumps(
            {"schema": "chimera.outdoor_places.v1", "places": [{"id": "p1-approach", "poi": "p1", "xy": [12, 18]}]}
        )
    )
    config = tmp / "config.json"
    config.write_text(
        json.dumps(
            {
                "lane_network": str(lanes),
                "outdoor_places": str(places),
                "life": {"unavailable": [{"activity": "indoor_stays", "reason": "no verified entrance"}]},
            }
        )
    )
    return {"world": world, "nav": nav, "semantics": sem_path, "config": config, "lanes": lanes, "places": places}


@pytest.fixture
def packed(tmp_path):
    legacy = legacy_inputs(tmp_path)
    out = tmp_path / "principle"
    doc = pack(
        out,
        world=legacy["world"],
        nav=legacy["nav"],
        semantics=legacy["semantics"],
        config=legacy["config"],
        sources={"support": {"source": "measured", "verified": True, "method": "lidar"}},
    )
    return legacy, out, doc


def test_pack_then_check_passes(packed):
    legacy, out, doc = packed
    assert doc["schema"] == "awr.life.principle.v1" and doc["id"] == "toytown"
    assert set(doc["layers"]) == {"support", "semantics", "lane_network", "outdoor_places", "world"}
    assert doc["layers"]["support"]["source"] == "measured" and doc["layers"]["support"]["verified"] is True
    assert doc["capabilities"]["walking"]["status"] == "supported"
    assert doc["capabilities"]["vehicles_parked"]["status"] == "supported"
    assert doc["capabilities"]["indoor"] == {"status": "unsupported", "reason": "no verified entrance"}
    assert doc["capabilities"]["driving"]["status"] == "unknown"
    rep = check(out)
    assert rep["pass"], rep["errors"]
    assert rep["stats"]["support"]["person_m2"] == pytest.approx((NX - 4) * NY * CELL * CELL)
    assert rep["stats"]["semantics"]["pois"] == 1 and rep["stats"]["semantics"]["road_km"] == pytest.approx(0.03)
    # the package carries everything: the legacy inputs can go away
    pkg = resolve(out)
    assert pkg.world == out / "world" and pkg.support == out / "support" and pkg.semantics == out / "semantics.json"
    assert pkg.lane_network == out / "lanes/net.net.xml" and pkg.outdoor_places == out / "places.json"
    assert not (out / "world/geometry/mesh/source.glb").exists()


def test_pack_without_vehicle_cells_declares_unsupported(tmp_path):
    legacy = legacy_inputs(tmp_path, walk_vehicle=False)
    doc = pack(tmp_path / "p", world=legacy["world"], nav=legacy["nav"], semantics=legacy["semantics"])
    assert doc["capabilities"]["vehicles_parked"]["status"] == "unsupported"
    assert "lane_network" not in doc["layers"]
    assert check(tmp_path / "p")["pass"]


def test_check_detects_tampered_layer(packed):
    _, out, _ = packed
    np.save(out / "support/walk_person.npy", np.zeros((NY, NX), bool))
    rep = check(out)
    assert not rep["pass"]
    assert any("walk_person.npy sha256" in e for e in rep["errors"])


def test_check_detects_semantic_and_world_inconsistencies(packed):
    _, out, doc = packed
    sem = json.loads((out / "semantics.json").read_text())
    sem["pois"][0]["building"] = "nope"
    sem["roads"].append({"id": "r1", "line": [[0, 0]], "width_m": 0, "kind": "x"})
    (out / "semantics.json").write_text(json.dumps(sem))
    doc["layers"]["semantics"]["sha256"] = _sha(out / "semantics.json")
    from awr.life.principle import _content_version

    doc["content_version"] = _content_version(doc["layers"])
    w = json.loads((out / "world/world.json").read_text())
    w["id"] = "other"
    (out / "world/world.json").write_text(json.dumps(w))
    (out / "principle.json").write_text(json.dumps(doc))
    errors = check(out)["errors"]
    assert any("not a building id" in e for e in errors)
    assert any("duplicate id r1" in e for e in errors)
    assert any("2 points and width_m" in e for e in errors)
    assert any("differs from principle id" in e for e in errors)


def test_check_rejects_capability_without_evidence(packed):
    _, out, doc = packed
    doc["capabilities"]["driving"] = {"status": "supported"}
    del doc["layers"]["lane_network"]
    from awr.life.principle import _content_version

    doc["content_version"] = _content_version(doc["layers"])
    (out / "principle.json").write_text(json.dumps(doc))
    errors = check(out)["errors"]
    assert any("driving is supported but no lane_network" in e for e in errors)
    doc["capabilities"]["cycling"] = {"status": "unsupported"}
    (out / "principle.json").write_text(json.dumps(doc))
    assert any("unsupported needs a reason" in e for e in check(out)["errors"])


def test_check_refuses_non_package(tmp_path):
    rep = check(tmp_path)
    assert not rep["pass"] and "not a principle package" in rep["errors"][0]


def test_admit_binds_config_to_package_files(packed):
    legacy, out, _ = packed
    cfg = {"lane_network": str(legacy["lanes"]), "outdoor_places": str(legacy["places"])}
    with pytest.raises(PrincipleError, match="scene files come from the package"):
        admit(out, cfg, people=3, vehicles=1)
    cfg = {}
    pkg, rep = admit(out, cfg, people=3, vehicles=1)
    assert (
        rep["pass"]
        and cfg["lane_network"] == str(out / "lanes/net.net.xml")
        and cfg["outdoor_places"] == str(out / "places.json")
    )
    with pytest.raises(PrincipleError, match="driving is unknown"):
        admit(out, {"life": {"vehicles": {"population": {"curb_trip": 2}}}}, people=3, vehicles=2)


def test_demand_errors():
    caps = {
        "walking": {"status": "supported"},
        "vehicles_parked": {"status": "unsupported", "reason": "x"},
        "driving": {"status": "unsupported", "reason": "y"},
    }
    assert demand_errors(caps, people=10, vehicles=0) == []
    assert len(demand_errors(caps, people=10, vehicles=2)) == 1
    assert len(demand_errors({"walking": {"status": "unknown"}}, people=1, vehicles=0)) == 1


def test_cli_check_and_describe(packed):
    _, out, _ = packed
    r = subprocess.run([sys.executable, "-m", "awr.life.principle", "check", str(out)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("PASS toytown"), r.stdout + r.stderr
    (out / "semantics.json").write_text("{}")
    r = subprocess.run(
        [sys.executable, "-m", "awr.life.principle", "check", str(out), "--json"], capture_output=True, text=True
    )
    assert r.returncode == 2 and json.loads(r.stdout)["pass"] is False


def test_manifest_validates_against_contract(packed):
    jsonschema = pytest.importorskip("jsonschema")
    from awr.contracts._paths import schema_path

    schema = json.loads(schema_path("life/principle.v1.schema.json").read_text())
    _, out, doc = packed
    jsonschema.Draft202012Validator(schema).validate(doc)
    bad = dict(doc, frame=dict(doc["frame"], units="ft"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(bad)
