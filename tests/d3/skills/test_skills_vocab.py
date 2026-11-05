"""技能词表与决策校验（ADR-227，第 6 步）：`validate_command` / `validate_decision` 的拒绝文案与 CHIMERA 网关一致，并与契约
`bench/decision.v1` 的判定一致。"""

from __future__ import annotations

import math

import pytest

from awr.agentsdk.schemas import bench_validator
from awr.agentsdk.skills import SKILLS, SkillError, validate_command, validate_decision

UAVS = {"uav-01", "uav-02"}

VALID = [
    {"uav": "uav-01", "skill": "goto", "params": {"x": 10, "y": -5.5, "agl": 60, "speed": 4}},
    {"uav": "uav-01", "skill": "sweep", "params": {"x": 0, "y": 0, "size": 120, "agl": 50}},
    {"uav": "uav-01", "skill": "sweep", "params": {"polygon": [[0, 0], [100, 0], [100, 80]], "lane": 20}},
    {"uav": "uav-02", "skill": "orbit", "params": {"x": 1, "y": 2, "radius": 40, "period": 60}},
    {"uav": "uav-02", "skill": "orbit", "params": {"x": 1, "y": 2}},
    {"uav": "uav-02", "skill": "follow", "params": {"x": 1, "y": 2, "standoff": 15}},
    {"uav": "uav-01", "skill": "look_at", "params": {"x": 3, "y": 4}},
    {"uav": "uav-01", "skill": "hold", "params": {}},
    {"uav": "uav-01", "skill": "rtl"},
    {"uav": "uav-01", "skill": "zoom", "params": {"factor": 12}},
]

INVALID = [
    ("invalid command", "not a dict"),
    ("invalid command", {"uav": 1, "skill": "hold"}),
    ("invalid command", {"uav": "uav-01", "skill": "goto", "params": [1, 2]}),
    ("unknown UAV", {"uav": "uav-09", "skill": "hold"}),
    ("unknown skill", {"uav": "uav-01", "skill": "land"}),
    ("unknown skill", {"uav": "uav-01", "skill": "takeoff"}),
    ("missing skill parameter", {"uav": "uav-01", "skill": "goto", "params": {"x": 1}}),
    ("missing skill parameter", {"uav": "uav-01", "skill": "zoom", "params": {}}),
    ("skill coordinates/limits must be numeric", {"uav": "uav-01", "skill": "goto", "params": {"x": "1", "y": 2}}),
    ("skill coordinates/limits must be numeric", {"uav": "uav-01", "skill": "goto", "params": {"x": 1, "y": 2, "agl": True}}),
    ("skill coordinates/limits must be numeric", {"uav": "uav-01", "skill": "goto", "params": {"x": 1, "y": 2, "speed": "fast"}}),
    ("invalid orbit", {"uav": "uav-01", "skill": "orbit", "params": {"x": 1, "y": 2, "radius": 10}}),
    ("invalid orbit", {"uav": "uav-01", "skill": "orbit", "params": {"x": 1, "y": 2, "period": 0}}),
    ("zoom must be 1..12", {"uav": "uav-01", "skill": "zoom", "params": {"factor": 13}}),
    ("zoom must be 1..12", {"uav": "uav-01", "skill": "zoom", "params": {"factor": 0.5}}),
    ("invalid sweep polygon", {"uav": "uav-01", "skill": "sweep", "params": {"polygon": [[0, 0], [1, 1]]}}),
    ("invalid sweep polygon", {"uav": "uav-01", "skill": "sweep", "params": {"polygon": [[0, 0], [1, 1], [1, "2"]]}}),
    ("sweep needs polygon or x,y", {"uav": "uav-01", "skill": "sweep", "params": {"x": 1}}),
]


def test_vocabulary_is_closed():
    assert set(SKILLS) == {"goto", "sweep", "orbit", "follow", "look_at", "hold", "rtl", "zoom"}
    assert not SKILLS["look_at"].motion and not SKILLS["zoom"].motion and SKILLS["goto"].motion


@pytest.mark.parametrize("cmd", VALID, ids=lambda c: c["skill"])
def test_valid_commands_pass_the_validator_and_the_contract(cmd):
    assert validate_command(cmd, uavs=UAVS) is cmd
    assert not list(bench_validator("decision").iter_errors({"commands": [cmd], "reports": [], "done": False}))


@pytest.mark.parametrize(("msg", "cmd"), INVALID, ids=[f"{m}-{i}" for i, (m, _) in enumerate(INVALID)])
def test_invalid_commands_are_rejected_with_the_gateway_text(msg, cmd):
    with pytest.raises(SkillError, match=f"^{msg}$"):
        validate_command(cmd, uavs=UAVS)
    if msg != "unknown UAV":                      # the contract does not know the fleet
        assert list(bench_validator("decision").iter_errors({"commands": [cmd], "reports": [], "done": False}))


def test_decision_shape_and_finite_numbers():
    ok = {"thought": "scan", "commands": [VALID[0]], "reports": [{"type": "found"}], "done": False, "summary": ""}
    assert validate_decision(ok, uavs=UAVS) is ok
    with pytest.raises(SkillError, match="action must be an object"):
        validate_decision([], uavs=UAVS)
    with pytest.raises(SkillError, match="commands/reports must be arrays"):
        validate_decision({"commands": "bad"}, uavs=UAVS)
    with pytest.raises(SkillError, match="done must be boolean"):
        validate_decision({"done": 1}, uavs=UAVS)
    with pytest.raises(SkillError, match="invalid report"):
        validate_decision({"reports": ["found"]}, uavs=UAVS)
    with pytest.raises(SkillError, match="numbers must be finite"):
        validate_decision({"commands": [{"uav": "uav-01", "skill": "goto", "params": {"x": math.nan, "y": 0}}]}, uavs=UAVS)
    with pytest.raises(SkillError, match="numbers must be finite"):
        validate_decision({"reports": [{"type": "found", "position_enu_m": [math.inf, 0]}]}, uavs=UAVS)


def test_task_layer_report_validation_is_injected():
    def reports(r):
        if r.get("type") not in ("found", "count"):
            raise ValueError("invalid report type")

    validate_decision({"reports": [{"type": "found"}]}, reports=reports)
    with pytest.raises(ValueError, match="invalid report type"):
        validate_decision({"reports": [{"type": "plate"}]}, reports=reports)
