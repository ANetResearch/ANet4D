import hashlib
import json

import pytest

from awr.life.civil.public_regions import SCHEMA, load


def files(tmp_path):
    world = tmp_path / "world"
    world.mkdir()
    (world / "world.json").write_text(json.dumps(dict(id="town", contentVersion="v1")))
    sem = tmp_path / "semantics.json"
    sem.write_text(
        json.dumps(dict(open_areas=[dict(id="park", kind="park", polygon=[[0, 0], [20, 0], [20, 20], [0, 20]])]))
    )
    spec = dict(
        schema=SCHEMA,
        world_json_sha256=hashlib.sha256((world / "world.json").read_bytes()).hexdigest(),
        semantics_sha256=hashlib.sha256(sem.read_bytes()).hexdigest(),
        regions=[
            dict(
                id="park-public",
                label_zh="公园",
                source=dict(kind="semantic_polygon", collection="open_areas", id="park"),
            ),
            dict(
                id="north",
                label_zh="北街区",
                kind="zone",
                polygon=[[0, 0], [30, 0], [30, 30], [0, 30]],
                source=dict(kind="authored_region", rationale="Fixed public search district for this scene"),
            ),
        ],
    )
    path = tmp_path / "public-regions.json"
    path.write_text(json.dumps(spec))
    return path, world, sem, spec


def test_fixed_regions_are_source_bound_and_do_not_use_population_or_horizon(tmp_path):
    path, world, sem, spec = files(tmp_path)
    result = load(path, world=world, semantics=sem)
    assert result["anchors"][1]["polygon"] == [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]]
    assert result["anchors"][1]["kind"] == "park"
    assert not result["definition"]["admitted"]
    # Source order is irrelevant to published anchors; no trajectory is an input.
    spec["regions"].reverse()
    path.write_text(json.dumps(spec))
    assert load(path, world=world, semantics=sem)["anchors"] == result["anchors"]
    assert all("source" not in a and "actor" not in a for a in result["anchors"])


@pytest.mark.parametrize("file", ["world", "semantics"])
def test_same_names_cannot_hide_changed_scene_files(tmp_path, file):
    path, world, sem, _ = files(tmp_path)
    selected = world / "world.json" if file == "world" else sem
    selected.write_text(selected.read_text() + "\n")
    with pytest.raises(ValueError, match="differs from the selected scene"):
        load(path, world=world, semantics=sem)


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s["regions"][0].update(polygon=[[0, 0], [1, 0], [1, 1]]),
        lambda s: s["regions"][0].update(kind="parking"),
        lambda s: s["regions"][0]["source"].update(id="absent"),
        lambda s: s["regions"][1].update(kind="parking"),
        lambda s: s["regions"][1].update(polygon=[[0, 0], [1, 1], [0, 1], [1, 0]]),
        lambda s: s["regions"][1]["source"].update(rationale=""),
        lambda s: s["regions"][1].update(id="park-public"),
        lambda s: s["regions"][1].update(label_zh="公园"),
        lambda s: s.update(regions=[]),
    ],
)
def test_invalid_or_relabelled_regions_fail_without_automatic_grid_fallback(tmp_path, change):
    path, world, sem, spec = files(tmp_path)
    change(spec)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        load(path, world=world, semantics=sem)


def test_builder_validates_region_definition_before_initial_planning(tmp_path):
    from awr.life.civil.grounded import build

    path, world, sem, spec = files(tmp_path)
    spec["semantics_sha256"] = "0" * 64
    path.write_text(json.dumps(spec))
    cfg = tmp_path / "scene.json"
    cfg.write_text(json.dumps(dict(public_regions=str(path))))
    with pytest.raises(ValueError, match="differs from the selected scene"):
        build(world, tmp_path / "unneeded-support", sem, tmp_path / "v1", config=cfg, plan_only=True)
    assert not (tmp_path / "v1.initial-inputs.json").exists()
