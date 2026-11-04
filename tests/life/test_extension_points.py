"""The three extension points: entity classes, activity recipes, motion backends."""

import json
import sys
import types

import pytest

from awr.life import runtimes
from awr.life.civil.activity.recipes import load_activities
from awr.life.classes import EntityClassError, describe, load, population

RUNNER = {
    "label_zh": "跑步者",
    "tape_class": "person",
    "profile": "person",
    "body": {"kind": "articulated", "assets": "humans", "height_m": {"base": 1.68, "step": 0.04, "cycle": 4}},
    "motion": {"max_speed_mps": {"base": 2.6, "step": 0.15, "cycle": 4}},
    "role": "runner",
    "activities": ["jog"],
    "min_activity_kinds": 2,
    "semantic_type": "pedestrian",
    "description": "跑步的人",
}


def test_builtin_catalog_reproduces_legacy_formulas():
    cat = load({"human_assets": 4})
    r = cat["resident"]
    for j in range(13):
        assert r.asset(j) == f"human-{j % 4 + 1}"
        assert r.height(j) == pytest.approx(1.60 + 0.04 * (j % 6))
        assert r.max_speed(j) == pytest.approx(1.0 + 0.08 * (j % 4))
        assert r.actor_id(j) == f"person-{j:03d}"
    v = cat["local_traffic"]
    assert v.max_speed(5) == pytest.approx(3.0 + 0.2 * (5 % 4)) and v.assets == (
        "sedan",
        "hatchback",
        "suv",
        "van",
        "pickup",
    )
    assert cat["street_object"].max_speed(0) == 0.0 and not cat["street_object"].moving
    assert load({"human_assets": 12})["resident"].asset(11) == "human-12"
    assert load({"vehicle_assets": ["sedan"]})["local_traffic"].assets == ("sedan",)


def test_population_legacy_and_declared():
    assert [(c.id, n) for c, n in population({}, 24)] == [("resident", 24)]
    assert [(c.id, n) for c, n in population({"people": 7}, 24)] == [("resident", 7)]
    cfg = {"classes": {"runner": RUNNER}, "population": {"runner": 8, "resident": 150}}
    pop = population(cfg, 0)
    assert [(c.id, n) for c, n in pop] == [("resident", 150), ("runner", 8)]  # residents keep their indices
    runner = pop[1][0]
    assert runner.role == "runner" and runner.actor_id(3) == "runner-003" and runner.max_speed(1) == pytest.approx(2.75)
    assert runner.activities == ("jog",) and runner.min_activity_kinds == 2
    pop = population({"classes": {"runner": RUNNER}, "population": {"runner": 2}}, 0)
    assert [(c.id, n) for c, n in pop] == [("resident", 0), ("runner", 2)]


def test_catalog_file_and_inline_override(tmp_path):
    f = tmp_path / "classes.json"
    f.write_text(json.dumps({"schema": "awr.life.entity_classes.v1", "classes": {"runner": RUNNER}}))
    cat = load({"entity_classes": str(f), "classes": {"runner": {**RUNNER, "role": "jogger"}}})
    assert cat["runner"].role == "jogger"  # inline wins over the file
    d = describe({"entity_classes": str(f), "population": {"runner": 2}}, 0)
    assert d["population"] == {"resident": 0, "runner": 2} and d["classes"]["runner"]["max_speed_mps"] == [2.6, 3.05]


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"classes": {"x": {**RUNNER, "tape_class": "ghost"}}}, "tape_class"),
        ({"classes": {"x": {**RUNNER, "body": {"kind": "static", "assets": "street"}}}}, "static body cannot"),
        ({"classes": {"x": {**RUNNER, "body": {"kind": "articulated", "assets": "humans"}}}}, "height_m"),
        ({"classes": {"x": {**RUNNER, "profile": "vehicle"}}}, "person profile"),
        ({"classes": {"x": {**RUNNER, "id_prefix": "person"}}}, "share the actor id prefix"),
        (
            {"classes": {"x": {**RUNNER, "motion": {"max_speed_mps": {"base": 1.0, "step": -0.5, "cycle": 3}}}}},
            "positive",
        ),
        ({"population": {"ghost": 1}}, "unknown entity class"),
        ({"population": {"local_traffic": 1}}, "articulated person classes"),
        ({"population": {"resident": -1}}, "non-negative"),
    ],
)
def test_catalog_rejections(bad, match):
    with pytest.raises(EntityClassError, match=match):
        population(bad, 1)


def test_recipes_take_roles_and_refuse_duplicates():
    rows = [
        {
            "purpose": "jog",
            "kinds": ["park", "pitch"],
            "dwell_s": [15, 45],
            "recurrence_s": 60,
            "requires": ["outdoor_stay"],
            "roles": ["runner"],
        },
        {"purpose": "rest", "kinds": ["park"], "dwell_s": [60, 120], "recurrence_s": 600, "requires": ["outdoor_stay"]},
    ]
    acts = load_activities(rows, ("walk",))
    assert acts[0].roles == ("runner",) and acts[1].roles == ("resident",)
    with pytest.raises(ValueError, match="unique"):
        load_activities(rows[:1], ("walk",), existing={"jog"})
    with pytest.raises(ValueError, match="unsupported fields"):
        load_activities([{**rows[0], "speed": 3}], ("walk",))


def test_runtime_registry_resolves_and_extends():
    assert runtimes.resolve({}) == ("grounded_v1", None)
    name, fn = runtimes.resolve({"runtime": "life_v1"})
    assert name == "life_v1" and callable(fn) and fn.__module__ == "awr.life.motion.runtime"
    with pytest.raises(ValueError, match="Unknown runtime"):
        runtimes.resolve({"runtime": "ghost"})
    mod = types.ModuleType("awr_life_test_backend")
    mod.run = lambda ctx: {"ran": True}
    sys.modules["awr_life_test_backend"] = mod
    try:
        name, fn = runtimes.resolve({"runtime": "frozen", "runtimes": {"frozen": "awr_life_test_backend:run"}})
        assert name == "frozen" and fn({}) == {"ran": True}
        assert "frozen" in runtimes.names({"runtimes": {"frozen": "awr_life_test_backend:run"}})
        with pytest.raises(ValueError, match="module:callable"):
            runtimes.resolve({"runtime": "x", "runtimes": {"x": "nocolon"}})
    finally:
        del sys.modules["awr_life_test_backend"]
    runtimes.register("inline_test", lambda ctx: 1)
    try:
        assert runtimes.resolve({"runtime": "inline_test"})[1]({}) == 1
    finally:
        del runtimes.REGISTRY["inline_test"]
    with pytest.raises(ValueError):
        runtimes.register("grounded_v1", lambda ctx: 1)
