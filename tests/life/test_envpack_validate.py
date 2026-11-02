"""Environment package validation on a synthetic 100 x 50 m package (awr.life.envpack.validate)."""

import hashlib
import json

import numpy as np
import pytest

from awr.life.envpack.validate import markdown, validate

CELL = 0.5
NX, NY = 200, 100


def write_package(root, *, road_y=25.0, nodata_x=None, places_nav="support"):
    """A world package, support grids, semantics and places: a road along y=25, a park, a lot, a building."""
    world = root / "world"
    (world / "geometry/terrain").mkdir(parents=True)
    (world / "geometry/nav").mkdir(parents=True)
    (world / "semantic").mkdir(parents=True)
    (world / "coordinate.json").write_text(
        json.dumps(
            dict(
                frame=dict(type="ENU", units="m", upAxis="+Z", handedness="right"),
                anchor=dict(georeferenced=False, uncertaintyM=dict(horizontal=5.0)),
            )
        )
    )
    (world / "world.json").write_text(json.dumps(dict(id="toy", contentVersion="c0")))
    for name, dtype, value in (("dsm_2m", "float32", 0.1), ("quality_2m", "uint8", 0)):
        data = np.full((25, 50), value, dtype)
        if name == "quality_2m" and nodata_x is not None:
            data[:, int(nodata_x // 2) :] = 1
        data.tofile(world / "geometry/terrain" / f"{name}.f32")
        (world / "geometry/terrain" / f"{name}.json").write_text(
            json.dumps(
                dict(
                    href=f"{name}.f32",
                    dtype=dtype,
                    width=50,
                    height=25,
                    cellM=2.0,
                    originXY=[0.0, 0.0],
                    rowOrder="south-to-north",
                )
            )
        )
    (world / "semantic/quality.geojson").write_text(json.dumps(dict(features=[])))
    road = [[5.0, road_y], [95.0, road_y]]
    (world / "geometry/nav/graph.geojson").write_text(
        json.dumps(
            dict(
                features=[
                    dict(
                        type="Feature",
                        id="e-1",
                        geometry=dict(type="LineString", coordinates=road),
                        properties=dict(
                            feature="edge",
                            edge_id="e-1",
                            from_node="n-1",
                            to_node="n-2",
                            kind="road",
                            agents=["person", "vehicle"],
                            width_m=6.4,
                            oneway=False,
                            length_m=90.0,
                            highway="residential",
                        ),
                    )
                ]
            )
        )
    )
    support = root / "support"
    support.mkdir()
    yy, xx = np.mgrid[0:NY, 0:NX]
    x, y = (xx + 0.5) * CELL, (yy + 0.5) * CELL
    person = ~((x > 60) & (x < 70) & (y > 35) & (y < 45))
    vehicle = (np.abs(y - 25) <= 3) | ((x > 70) & (x < 90) & (y > 5) & (y < 15))
    np.save(support / "ground.npy", np.zeros((NY, NX), np.float32))
    np.save(support / "walk_person.npy", person)
    np.save(support / "walk_vehicle.npy", vehicle)
    (support / "meta.json").write_text(
        json.dumps(
            dict(
                schema="awr.life.support.v1",
                x0=0.0,
                y0=0.0,
                nx=NX,
                ny=NY,
                cell_m=CELL,
                unknown_is_blocked=True,
                world="world",
                person_cells=int(person.sum()),
                vehicle_cells=int(vehicle.sum()),
            )
        )
    )
    sem = dict(
        schema="awr.life.town_semantics.v1",
        bounds=[0, 0, 100, 50],
        roads=[dict(id="e-1", line=[[5.0, 25.0], [95.0, 25.0]], width_m=6.4, kind="road")],
        buildings=[dict(id="way/1", polygon=[[60, 35], [70, 35], [70, 45], [60, 45], [60, 35]], kind="building")],
        pois=[dict(id="node/2", kind="school", xy=[55.0, 40.0]), dict(id="way/3", kind="park", xy=[25.0, 40.0])],
        open_areas=[
            dict(id="way/3", kind="park", xy=[25.0, 40.0], polygon=[[10, 30], [40, 30], [40, 48], [10, 48], [10, 30]])
        ],
        verified_parking_regions=[
            dict(id="way/4", verified=True, polygon=[[70, 5], [90, 5], [90, 15], [70, 15], [70, 5]])
        ],
        curb_slots=[],
        parking_lots=[],
    )
    (root / "semantics.json").write_text(json.dumps(sem))
    places = dict(
        schema="awr.life.outdoor_places.v1",
        nav=places_nav,
        network=dict(free_area_m2=4800.0, components=1),
        places=[
            dict(
                id="node/2-approach-1",
                poi="node/2",
                kind="school",
                xy=[55.0, 32.0],
                component=1,
                function="outdoor_approach",
                verified_entrance=False,
            ),
            dict(
                id="way/3-approach-1",
                poi="way/3",
                kind="park",
                xy=[25.0, 40.0],
                component=1,
                function="public_open_space",
                verified_entrance=False,
            ),
        ],
    )
    (root / "places.json").write_text(json.dumps(places))
    manifest = dict(
        schema="awr.life.environment.v2",
        id="toy",
        layers=dict(
            world=dict(path="world"),
            support=dict(path="support"),
            semantics=dict(path="semantics.json"),
            places=dict(path="places.json"),
        ),
        capabilities=dict(walk=dict(residents=10), drive=dict(drivers=0, destinations=0)),
        features=dict(frame=dict(tick_hz=10, clock="recorded window seconds")),
        capability_status={
            c: dict(status="unknown", reason="toy")
            for c in (
                "walk_activity",
                "vehicle_traffic",
                "parking",
                "recurrent_trips",
                "driver_chain",
                "boundary_entry_exit",
            )
        },
    )
    (root / "env.json").write_text(json.dumps(manifest))
    queries = dict(
        queries=[
            dict(
                id="Q1",
                type="pedestrian_search",
                text="t",
                requires=dict(capabilities=["walk_activity"], place_kinds=["school"]),
            ),
            dict(
                id="Q2",
                type="vehicle_counting",
                text="t",
                requires=dict(capabilities=["parking"], public_regions=["parking_lot"]),
            ),
        ]
    )
    (root / "queries.json").write_text(json.dumps(queries))
    return root / "env.json"


def finding(report, feature, check):
    return next(f for f in report["findings"] if f["feature"] == feature and f["check"] == check)


def bound_public_regions(root, manifest):
    doc = json.loads(manifest.read_text())
    doc["layers"]["public_regions"] = dict(path="public.json")
    # A broken explicit layer must never fall back to this legacy declaration.
    doc["features"]["regions"] = dict(public_regions=[dict(id="legacy", kind="parking")])
    manifest.write_text(json.dumps(doc))
    regions = dict(
        schema="awr.life.public_regions.v1",
        world_json_sha256=hashlib.sha256((root / "world/world.json").read_bytes()).hexdigest(),
        semantics_sha256=hashlib.sha256((root / "semantics.json").read_bytes()).hexdigest(),
        regions=[
            dict(
                id="lot",
                label_zh="停车场",
                source=dict(kind="semantic_polygon", collection="verified_parking_regions", id="way/4"),
            )
        ],
    )
    (root / "public.json").write_text(json.dumps(regions))
    queries = dict(
        queries=[
            dict(
                id="C",
                type="vehicle_counting",
                text="Count cars in the parking lot",
                requires=dict(public_regions=["parking"]),
            )
        ]
    )
    (root / "queries.json").write_text(json.dumps(queries))
    return regions


def test_bound_public_layer_uses_runtime_polygons_and_query_condition(tmp_path):
    manifest = write_package(tmp_path)
    bound_public_regions(tmp_path, manifest)
    report = validate(manifest, tmp_path, tmp_path / "queries.json")
    regions = finding(report, "regions", "public_regions")
    assert regions["status"] == "pass"
    assert [r["id"] for r in regions["value"]["declared"]] == ["lot"]
    assert regions["value"]["declared"][0]["polygon"][0] == [70.0, 5.0]
    assert regions["value"]["definition"]["admitted"] is False
    assert report["queries"][0]["status"] == "feature_candidates_present"
    assert report["admitted"] is False
    assert str(tmp_path / "public.json") in report["source_inventory"]


@pytest.mark.parametrize("change", ["world", "semantics", "polygon", "use", "empty"])
def test_bad_public_layer_cannot_be_replaced_by_legacy_labels(tmp_path, change):
    manifest = write_package(tmp_path)
    regions = bound_public_regions(tmp_path, manifest)
    if change in ("world", "semantics"):
        regions[change + "_json_sha256" if change == "world" else "semantics_sha256"] = "0" * 64
    elif change == "polygon":
        regions["regions"][0]["polygon"] = [[0, 0], [1, 0], [1, 1]]
    elif change == "use":
        regions["regions"][0]["kind"] = "park"
    else:
        regions["regions"] = []
    (tmp_path / "public.json").write_text(json.dumps(regions))
    report = validate(manifest, tmp_path, tmp_path / "queries.json")
    regions = finding(report, "regions", "public_regions")
    assert regions["status"] == "fail"
    assert regions["value"]["declared"] == []
    assert report["queries"][0]["status"] == "overlay_required"
    assert report["admitted"] is False


def test_unavailable_public_layer_is_unknown_not_legacy_pass(tmp_path):
    manifest = write_package(tmp_path)
    bound_public_regions(tmp_path, manifest)
    (tmp_path / "public.json").unlink()
    report = validate(manifest, tmp_path, tmp_path / "queries.json")
    assert finding(report, "regions", "public_regions")["status"] == "unknown"
    assert finding(report, "regions", "public_regions")["value"]["declared"] == []
    assert report["admitted"] is False


def test_aligned_package_passes_its_geometric_checks(tmp_path):
    report = validate(write_package(tmp_path), tmp_path, tmp_path / "queries.json")
    for feature, check in (
        ("frame", "world_axes"),
        ("frame", "layers_in_support_extent"),
        ("frame", "height_datum"),
        ("frame", "time_convention"),
        ("geometry", "support_grids"),
        ("geometry", "support_over_reconstruction"),
        ("regions", "stable_ids"),
        ("regions", "open_areas_supported"),
        ("regions", "parking_regions_supported"),
        ("regions", "building_footprints"),
        ("passage", "road_support"),
        ("passage", "road_registration"),
        ("versions", "derived_layer_sources"),
        ("versions", "capability_declaration"),
    ):
        assert finding(report, feature, check)["status"] == "pass", (feature, check, finding(report, feature, check))
    width = finding(report, "passage", "road_width_measured")["value"]
    assert 5.5 <= width["median_m"] <= 7.0  # the drivable strip is 6 m wide
    assert finding(report, "regions", "public_regions")["status"] == "missing"
    caps = report["capabilities"]
    assert caps["walk_activity"]["status"] == "static_only"  # no run measures it
    assert caps["boundary_entry_exit"]["status"] == "unsupported"
    rows = {q["id"]: q["status"] for q in report["queries"]}
    # Walking is not measured by a run; parking regions are verified but the lot a query names needs an overlay.
    assert rows == {"Q1": "blocked", "Q2": "overlay_required"}
    assert "环境 feature 核验报告" in markdown(report)


def test_misregistered_road_is_reported_by_the_lateral_shift(tmp_path):
    (tmp_path / "near").mkdir()
    report = validate(write_package(tmp_path / "near", road_y=27.0), tmp_path / "near")
    assert finding(report, "passage", "road_support")["status"] == "pass"  # 2 m off centre, still on the 6 m strip
    assert finding(report, "passage", "road_registration")["status"] == "pass"
    (tmp_path / "off").mkdir()
    report = validate(write_package(tmp_path / "off", road_y=29.5), tmp_path / "off")
    assert finding(report, "passage", "road_support")["status"] == "fail"
    shift = finding(report, "passage", "road_registration")
    assert shift["status"] == "fail" and shift["value"]["gain"] > 0.5


def test_support_without_reconstruction_and_stale_places_fail(tmp_path):
    report = validate(write_package(tmp_path, nodata_x=80.0, places_nav="old-support"), tmp_path)
    assert finding(report, "geometry", "support_over_reconstruction")["status"] == "fail"
    derived = finding(report, "versions", "derived_layer_sources")
    assert derived["status"] == "fail" and any(not d["match"] for d in derived["value"])


def test_historical_failed_run_never_certifies_current_environment(tmp_path):
    manifest = write_package(tmp_path)
    run = tmp_path / "old-run"
    run.mkdir()
    (run / "life-audit.json").write_text(
        json.dumps(
            {
                "duration_s": 1500,
                "counts": {"person": 300},
                "gates": {"gates": {"person_blockages_max": {"pass_": False}}},
            }
        )
    )
    doc = json.loads(manifest.read_text())
    doc["evidence"] = [{"id": "old", "kind": "run", "path": "old-run"}]
    manifest.write_text(json.dumps(doc))
    report = validate(manifest, tmp_path, tmp_path / "queries.json")
    assert report["evidence"]["runs"][0]["audit"]["gates_failed"] == ["person_blockages_max"]
    assert report["capabilities"]["walk_activity"]["status"] == "static_only"
    assert report["historical_evidence_used_for_capabilities"] is False
    assert report["admitted"] is False
    assert all(c["admitted"] is False for c in report["capabilities"].values())
    assert all(q["admitted"] is False for q in report["queries"])


def test_missing_hashed_array_is_unknown_not_verified(tmp_path):
    manifest = write_package(tmp_path)
    doc = json.loads(manifest.read_text())
    doc["layers"]["support"]["array_sha256"] = {"missing.npy": "0" * 64}
    manifest.write_text(json.dumps(doc))
    report = validate(manifest, tmp_path)
    versions = finding(report, "versions", "layer_digests")
    assert versions["status"] == "unknown"
    assert "support/missing.npy" in versions["value"]["absent"]


def test_changes_during_measurement_invalidate_results(tmp_path, monkeypatch):
    from awr.life.envpack import validate as module

    manifest = write_package(tmp_path)
    original = module.check_geometry

    def change_source(pkg, sup, rep):
        original(pkg, sup, rep)
        p = tmp_path / "semantics.json"
        p.write_text(p.read_text() + "\n")

    monkeypatch.setattr(module, "check_geometry", change_source)
    report = module.validate(manifest, tmp_path, tmp_path / "queries.json")
    assert finding(report, "versions", "inputs_stable_during_measurement")["status"] == "fail"
    assert all(c["status"] == "unknown" for c in report["capabilities"].values())
    assert all(q["status"] == "blocked" for q in report["queries"])


def test_measurement_hashes_static_inputs_without_modifying_them(tmp_path):
    from awr.life.envpack.validate import sha256

    manifest = write_package(tmp_path)
    before = {str(p): sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    report = validate(manifest, tmp_path)
    after = {str(p): sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
    assert (
        report["source_inventory"][str(tmp_path / "support/walk_person.npy")]
        == before[str(tmp_path / "support/walk_person.npy")]
    )
    assert report["source_inventory_after"] is None


def test_explicit_derived_graph_overrides_original_without_changing_world(tmp_path):
    manifest = write_package(tmp_path, road_y=29.5)
    graph = tmp_path / "world/geometry/nav/graph.geojson"
    original = graph.read_bytes()
    derived = json.loads(original)
    derived["features"][0]["geometry"]["coordinates"] = [[5, 25], [95, 25]]
    (tmp_path / "registered-graph.json").write_text(json.dumps(derived))
    doc = json.loads(manifest.read_text())
    doc["layers"]["road_graph"] = {"path": "registered-graph.json"}
    manifest.write_text(json.dumps(doc))
    report = validate(manifest, tmp_path)
    assert finding(report, "passage", "road_support")["status"] == "pass"
    assert str(tmp_path / "registered-graph.json") in report["source_inventory"]
    assert graph.read_bytes() == original


def test_no_registration_features_are_unknown_not_a_zero_offset_proof(tmp_path):
    manifest = write_package(tmp_path)
    sem = json.loads((tmp_path / "semantics.json").read_text())
    sem["roads"], sem["buildings"] = [], []
    (tmp_path / "semantics.json").write_text(json.dumps(sem))
    report = validate(manifest, tmp_path)
    assert finding(report, "frame", "feature_registration")["status"] == "unknown"


def test_stale_declared_source_cannot_be_reported_as_current_capacity(tmp_path):
    manifest = write_package(tmp_path)
    doc = json.loads(manifest.read_text())
    doc["layers"]["places"]["built_on"] = {"support": "old-support"}
    manifest.write_text(json.dumps(doc))
    report = validate(manifest, tmp_path, tmp_path / "queries.json")
    f = finding(report, "versions", "derived_layer_sources")
    assert f["status"] == "fail"
    assert any(r.get("source") == "support" and not r["match"] for r in f["value"])
    assert all(c["status"] == "unknown" for c in report["capabilities"].values())
    assert all(q["status"] == "blocked" for q in report["queries"])
    assert all(
        any(i["item"] == "environment_input_consistency" and not i["ok"] for i in q["items"]) for q in report["queries"]
    )


def test_wrong_layer_digest_does_not_report_static_candidate_capacity(tmp_path):
    manifest = write_package(tmp_path)
    doc = json.loads(manifest.read_text())
    doc["layers"]["semantics"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(doc))
    report = validate(manifest, tmp_path, tmp_path / "queries.json")
    assert finding(report, "versions", "layer_digests")["status"] == "fail"
    assert all(c["status"] == "unknown" for c in report["capabilities"].values())
    assert all(q["status"] == "blocked" for q in report["queries"])
