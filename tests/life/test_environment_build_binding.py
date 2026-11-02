import json

import pytest

from awr.life.envpack.binding import bind_inputs
from awr.life.envpack.validate import sha256
from awr.life.motion.capability import load_environment


def fixture(tmp_path):
    for name in ("world", "support"):
        (tmp_path / name).mkdir()
    for name in ("semantics", "places", "lane_network"):
        (tmp_path / (name + ".json")).write_text("{}")
    (tmp_path / "support/walk_person.npy").write_bytes(b"frozen-grid")
    layers = {n: {"path": n} for n in ("world", "support")}
    layers.update(
        {
            n: {"path": n + ".json", "sha256": sha256(tmp_path / (n + ".json"))}
            for n in ("semantics", "places", "lane_network")
        }
    )
    layers["support"].update(
        local=False, array_sha256={"walk_person.npy": sha256(tmp_path / "support/walk_person.npy")}
    )
    manifest = {"schema": "awr.life.environment.v2", "id": "test", "layers": layers}
    args = dict(
        world=tmp_path / "world",
        support=tmp_path / "support",
        semantics=tmp_path / "semantics.json",
        config={"outdoor_places": str(tmp_path / "places.json"), "lane_network": str(tmp_path / "lane_network.json")},
        root=tmp_path,
    )
    return manifest, args


def test_v2_is_explicitly_plan_only(tmp_path):
    m, args = fixture(tmp_path)
    p = tmp_path / "environment.json"
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="validation_only"):
        load_environment(p, tmp_path)
    assert load_environment(p, tmp_path, validation_only=True) == m
    r = bind_inputs(m, **args)
    assert r["identity_bound"] and not r["admitted"]


@pytest.mark.parametrize("key", ["world", "support", "semantics"])
def test_cli_cannot_use_a_different_layer(tmp_path, key):
    m, args = fixture(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    args[key] = other
    with pytest.raises(ValueError, match=key + " mismatch"):
        bind_inputs(m, **args)


def test_config_cannot_mix_old_places_with_new_support(tmp_path):
    m, args = fixture(tmp_path)
    old = tmp_path / "old.json"
    old.write_text("{}")
    args["config"]["outdoor_places"] = str(old)
    with pytest.raises(ValueError, match="places mismatch"):
        bind_inputs(m, **args)


def test_consumed_lane_network_must_be_declared(tmp_path):
    m, args = fixture(tmp_path)
    del m["layers"]["lane_network"]
    with pytest.raises(ValueError, match="bind consumed layer lane_network"):
        bind_inputs(m, **args)


@pytest.mark.parametrize("missing", [False, True])
def test_remote_label_does_not_bypass_used_array_validation(tmp_path, missing):
    m, args = fixture(tmp_path)
    p = tmp_path / "support/walk_person.npy"
    p.unlink() if missing else p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest input missing" if missing else "digest mismatch"):
        bind_inputs(m, **args)


def test_symlink_alias_to_same_source_is_allowed(tmp_path):
    m, args = fixture(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "support", target_is_directory=True)
    args["support"] = alias
    assert bind_inputs(m, **args)["identity_bound"]


def test_builder_rejects_v2_recording_before_any_output(tmp_path):
    from awr.life.civil.grounded import build

    m, args = fixture(tmp_path)
    manifest = tmp_path / "environment.json"
    manifest.write_text(json.dumps(m))
    cfg = tmp_path / "scenario.json"
    cfg.write_text(json.dumps(dict(environment=str(manifest), environment_root=str(tmp_path))))
    with pytest.raises(ValueError, match="validation_only"):
        build(args["world"], args["support"], args["semantics"], tmp_path / "v1", config=cfg)
    assert not (tmp_path / "v1.initial-inputs.json").exists()


def test_builder_plan_only_still_rejects_mixed_layers_before_placement(tmp_path):
    from awr.life.civil.grounded import build

    m, args = fixture(tmp_path)
    manifest = tmp_path / "environment.json"
    manifest.write_text(json.dumps(m))
    cfg = tmp_path / "scenario.json"
    cfg.write_text(json.dumps(dict(environment=str(manifest), environment_root=str(tmp_path), **args["config"])))
    wrong = tmp_path / "old-support"
    wrong.mkdir()
    with pytest.raises(ValueError, match="support mismatch"):
        build(args["world"], wrong, args["semantics"], tmp_path / "v1", config=cfg, plan_only=True)
    assert not (tmp_path / "v1.initial-inputs.json").exists()


def test_resident_driving_map_is_a_bound_layer(tmp_path):
    m, args = fixture(tmp_path)
    (tmp_path / "driving.json").write_text("{}")
    args["config"] = dict(args["config"], life={"driving": {"map": str(tmp_path / "driving.json")}})
    with pytest.raises(ValueError, match="does not bind consumed layer driving_map"):
        bind_inputs(m, **args)
    m["layers"]["driving_map"] = {"path": "stale.json", "sha256": sha256(tmp_path / "driving.json")}
    (tmp_path / "stale.json").write_text("{}")
    with pytest.raises(ValueError, match="driving_map mismatch"):
        bind_inputs(m, **args)
    m["layers"]["driving_map"] = {"path": "driving.json", "sha256": sha256(tmp_path / "driving.json")}
    assert {"layer": "driving_map", "path": str((tmp_path / "driving.json").resolve())} in bind_inputs(m, **args)[
        "inputs"
    ]


def test_valid_hashes_do_not_allow_old_driving_sources(tmp_path):
    m, args = fixture(tmp_path)
    old = tmp_path / "old-map.json"
    old.write_text("{}")
    m["layers"]["driving_map"] = {"path": old.name, "sha256": sha256(old), "built_on": {"support": "old-support"}}
    with pytest.raises(ValueError, match="derived layer source mismatch: driving_map <- support"):
        bind_inputs(m, **args)


@pytest.mark.parametrize("dependencies", [None, [], {"missing_layer": "x"}, {"support": None}])
def test_malformed_or_dangling_dependencies_fail_closed(tmp_path, dependencies):
    m, args = fixture(tmp_path)
    m["layers"]["places"]["built_on"] = dependencies
    with pytest.raises(ValueError, match="derived layer source mismatch"):
        bind_inputs(m, **args)


def test_derived_source_symlink_is_same_input_without_granting_admission(tmp_path):
    m, args = fixture(tmp_path)
    (tmp_path / "support-alias").symlink_to(tmp_path / "support", target_is_directory=True)
    m["layers"]["places"]["built_on"] = {"support": "support-alias"}
    result = bind_inputs(m, **args)
    assert result["declared_sources"][0]["match"]
    assert not result["admitted"]


def test_valid_place_digest_cannot_hide_a_different_walking_policy(tmp_path):
    m, args = fixture(tmp_path)
    policy = dict(radius_m=0.713, step_m=0.5, shared_streets=None)
    p = tmp_path / "places.json"
    p.write_text(json.dumps(dict(walking_policy=policy)))
    m["layers"]["places"]["sha256"] = sha256(p)
    assert bind_inputs(m, **args)["places_walking_policy"] == policy
    args["config"]["shared_streets"] = dict(carriageway_cost=2.0)
    with pytest.raises(ValueError, match="walking policy differs"):
        bind_inputs(m, **args)
    policy["shared_streets"] = args["config"]["shared_streets"]
    p.write_text(json.dumps(dict(walking_policy=policy)))
    m["layers"]["places"]["sha256"] = sha256(p)
    assert bind_inputs(m, **args)["identity_bound"]
    args["config"]["walking_step_m"] = 0.25
    with pytest.raises(ValueError, match="walking policy differs"):
        bind_inputs(m, **args)


def test_explicit_public_regions_must_be_a_bound_environment_input(tmp_path):
    m, args = fixture(tmp_path)
    p = tmp_path / "public-regions.json"
    p.write_text("{}")
    args["config"]["public_regions"] = str(p)
    with pytest.raises(ValueError, match="does not bind consumed layer public_regions"):
        bind_inputs(m, **args)
    m["layers"]["public_regions"] = dict(
        path=p.name, sha256=sha256(p), built_on={"semantics": "semantics.json", "world": "world"}
    )
    assert any(r["layer"] == "public_regions" for r in bind_inputs(m, **args)["inputs"])
    p.write_text('{"regions":[]}')
    with pytest.raises(ValueError, match="digest mismatch"):
        bind_inputs(m, **args)


@pytest.mark.parametrize("layer,key", [("driving_map", "map"), ("road_graph", "source_graph")])
def test_resident_driving_inputs_must_match_environment(tmp_path, layer, key):
    m, args = fixture(tmp_path)
    named = tmp_path / (layer + ".json")
    named.write_text("{}")
    m["layers"][layer] = {"path": named.name, "sha256": sha256(named)}
    other = tmp_path / "other.json"
    other.write_text("{}")
    args["config"]["life"] = {"driving": {key: str(other)}}
    with pytest.raises(ValueError, match=layer + " mismatch"):
        bind_inputs(m, **args)
    args["config"]["life"]["driving"][key] = str(named)
    assert any(r["layer"] == layer for r in bind_inputs(m, **args)["inputs"])
