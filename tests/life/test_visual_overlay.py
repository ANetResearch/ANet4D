from copy import deepcopy

import pytest

from awr.life.civil.visual_overlay import apply


def inputs():
    actors = [
        dict(
            id="v",
            **{"class": "vehicle"},
            size_mm=[4000, 2000, 1500],
            owner="p",
            appearance=dict(
                asset="hatchback@1",
                asset_sha256="mesh",
                colour_name="银色",
                body_rgb=[190, 192, 196],
                plate_text="SIM 0033",
            ),
            description="银色两厢车",
        ),
        dict(
            id="p",
            **{"class": "person"},
            size_mm=[500, 500, 1700],
            appearance=dict(
                asset="human-1@1",
                asset_sha256="human",
                top_name="黑色",
                top_rgb=[25, 25, 28],
                bottom_name="黑色",
                bottom_rgb=[25, 25, 28],
                age="adult",
            ),
        ),
    ]
    overlay = dict(
        schema="awr.life.visual_overlay.v1",
        edits=[
            dict(
                actor="v",
                expected_asset="hatchback@1",
                expected_appearance=deepcopy(actors[0]["appearance"]),
                colours=dict(vehicle_colour="白色"),
            )
        ],
    )
    return actors, overlay


def test_colour_layout_changes_actual_material_metadata_not_geometry_or_motion():
    actors, overlay = inputs()
    original = deepcopy(actors)
    result = apply(actors, overlay)
    assert actors == original
    assert result[0]["appearance"] == dict(actors[0]["appearance"], colour_name="白色", body_rgb=[235, 235, 232])
    assert "description" not in result[0]
    for key in ("id", "class", "size_mm", "owner"):
        assert result[0][key] == actors[0][key]
    assert result[1] == actors[1]
    overlay["edits"].append(
        dict(
            actor="p",
            expected_asset="human-1@1",
            expected_appearance=deepcopy(actors[1]["appearance"]),
            colours=dict(top="橄榄色", bottom="卡其色"),
        )
    )
    person = apply(actors, overlay)[1]
    assert person["appearance"]["top_rgb"] == [100, 110, 60]
    assert person["appearance"]["bottom_rgb"] == [195, 180, 140]
    assert person["size_mm"] == actors[1]["size_mm"]


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "asset", "old_colour", "class", "unknown", "geometry", "plate"]
)
def test_bad_overlay_is_atomic_and_cannot_change_physics_or_plate(damage):
    actors, overlay = inputs()
    original = deepcopy(actors)
    edit = overlay["edits"][0]
    if damage == "missing":
        edit["actor"] = "absent"
    if damage == "duplicate":
        overlay["edits"].append(deepcopy(edit))
    if damage == "asset":
        edit["expected_asset"] = "bus@1"
    if damage == "old_colour":
        edit["expected_appearance"]["colour_name"] = "红色"
    if damage == "class":
        edit["colours"] = {"top": "白色"}
    if damage == "unknown":
        edit["colours"] = {"vehicle_colour": "mystery"}
    if damage == "geometry":
        edit["size_mm"] = [1, 1, 1]
    if damage == "plate":
        edit["colours"] = {"plate_text": "SIM 9999"}
    with pytest.raises(ValueError):
        apply(actors, overlay)
    assert actors == original
