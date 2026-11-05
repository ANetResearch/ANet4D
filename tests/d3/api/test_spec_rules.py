"""Run 规格排名拒绝规则（D3-AC-34、D3-AC-36、D3-AC-41、D3-AC-69；AWR-17 §18.5.4；M27 V20、§14.9；M25-FR-194、FR-201）。

逐条对照 AWR-05 §11 原文：`physics = fast` 的排名请求返回 682（`why = ranked_requires_realistic`）；SITL Run 的 `ranked = false`，
提交排行榜返回 633；排名运行只接受 `appearance = mesh`，`appearance_source ∈ {mesh_distilled, generative}` 的排名提交返回 682
（`why = ranked_appearance`）；Run 规格、帧元数据与榜单分轨的外观枚举一致。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from awr.bench.spec_rules import (
    CODE_PRECEDENCE,
    JOB_SEEDED_RULES,
    check,
    check_ranked_eval,
    check_submission_eval,
    first_layer,
    load_rules,
    problem,
    reason,
)
from awr.contracts._paths import schema_path
from awr.contracts.reasons import REASONS

REPO = Path(__file__).resolve().parents[3]
PROTO_SPEC = REPO / "configs/bench.d3/crop-handoff-001.l1-3uav.run_spec.json"


def spec(**kw) -> dict:
    """外部请求形态的最小合法规格（种子由平台抽取：`seed_count`）。"""
    s = {"schema": "drone4d.bench.run_spec.v1", "experiment": {"name": "ranked-gate"}, "world": {"id": "oryahovo-crop1"},
         "event": {"id": "crop-handoff-001"},
         "fleet": [{"uav_id": "uav-01", "vehicle_ref": "x500", "start": {"kind": "pose", "pos_enu_m": [76.0, -78.0, 0.0], "yaw_deg": 90.0}}],
         "policy": {"kind": "image", "image_digest": "sha256:" + "c" * 64}, "obs_level": "L1", "seed_count": 10}
    s.update(kw)
    return s


def _schema(rel: str) -> dict:
    return json.loads(schema_path(rel).read_text(encoding="utf-8"))


def _codes(s: dict, **kw) -> set[tuple[int, str]]:
    return {(v.code, v.why) for v in check(s, **kw)}


# ---- D3-AC-34：Fast 档不排名 -------------------------------------------------------------------------------------------------

def test_ranked_fast_returns_682_ranked_requires_realistic() -> None:
    assert check(spec(ranked=True)) == []                       # physics 缺省 realistic
    assert check(spec(ranked=True, physics="realistic")) == []
    vs = check(spec(ranked=True, physics="fast"))
    assert {(v.code, v.why) for v in vs} == {(682, "ranked_requires_realistic")}
    assert {v.path for v in vs} == {"physics", "ranked"}         # ranked_physics 与 fast_unranked 同时指出
    code, detail = problem(vs)
    assert code == 682 and detail["why"] == "ranked_requires_realistic"
    assert {(x["path"], x["why"], x["code"]) for x in detail["violations"]} == {("physics", "ranked_requires_realistic", 682),
                                                                                 ("ranked", "ranked_requires_realistic", 682)}
    assert check(spec(physics="fast")) == []                    # 非排名的 Fast 运行允许（Env 接受 physics = fast）
    assert check(spec(physics="fast", ranked=False)) == []
    assert check(spec(physics="gz")) == []                      # gz 锁步交叉验证档（ADR-226）：不排名时允许
    assert _codes(spec(physics="gz", ranked=True)) == {(682, "ranked_requires_realistic")}


def test_ranked_gz_profile_and_vehicle_collision_are_realistic_only() -> None:
    assert _codes(spec(ranked=True, physics_opts={"gz_profile": "xval"})) == {(682, "ranked_requires_realistic")}
    assert _codes(spec(ranked=True, physics_opts={"vehicle_collision": True})) == {(682, "ranked_requires_realistic")}
    assert check(spec(physics_opts={"gz_profile": "xval", "vehicle_collision": True})) == []


# ---- D3-AC-36：SITL 档 ranked = false，排名提交 633 ------------------------------------------------------------------------------

def test_sitl_run_is_unranked_and_ranked_request_returns_633() -> None:
    assert check(spec(physics="sitl")) == []                    # ranked 缺省 false
    assert check(spec(physics="sitl", ranked=False)) == []
    vs = check(spec(physics="sitl", ranked=True))
    assert {(v.code, v.why) for v in vs} == {(633, "sitl_ranked"), (682, "ranked_requires_realistic")}
    code, detail = problem(vs)                                  # 第 ④ 层先于第 ⑤ 层：首个失败层为 633
    assert code == 633 and detail == {"why": "sitl_ranked", "violations": [{"path": "ranked", "why": "sitl_ranked", "code": 633}]}
    r = reason(vs)
    assert r == {"code": 633, "name": "SITL_NOT_RANKED", "detail": detail}
    assert REASONS[633].http == 422


def test_sitl_board_submission_returns_633() -> None:
    rs = json.loads(PROTO_SPEC.read_text(encoding="utf-8"))
    cfg = {"run_spec": rs, "seeds": rs["seeds"], "physics": "realistic"}
    assert check_submission_eval(cfg) == []                     # 正式轮次配置（显式秘密种子、Realistic 档）可受理提交
    for physics_at in ("cfg", "spec"):                          # SITL 档由轮次配置或规格给出，结果相同
        c = copy.deepcopy(cfg)
        if physics_at == "cfg":
            c["physics"] = "sitl"
        else:
            c["run_spec"]["physics"] = "sitl"
            del c["physics"]
        vs = check_submission_eval(c)
        assert problem(vs)[0] == 633 and reason(vs)["name"] == "SITL_NOT_RANKED"
    vs = check_submission_eval({**cfg, "physics": "fast"})
    assert problem(vs) == (682, {"why": "ranked_requires_realistic",
                                 "violations": [{"path": "physics", "why": "ranked_requires_realistic", "code": 682},
                                                {"path": "ranked", "why": "ranked_requires_realistic", "code": 682}]})
    assert problem(check_submission_eval({**cfg, "physics": "kinematic"}))[1]["why"] == "ranked_requires_realistic"   # 开发档


def test_ranked_eval_keeps_job_seeds_and_eval_ref() -> None:
    rs = json.loads(PROTO_SPEC.read_text(encoding="utf-8"))
    ev = {**rs, "eval_ref": {"submission_id": "s1", "board_version": "b1", "track": "std"}}
    assert {v.why for v in check(ev)} == {"ranked_seed_rules", "eval_ref_reserved"}     # 外部请求：种子由平台抽取、eval_ref 保留
    assert check_ranked_eval(ev) == []                                                   # 作业：秘密种子由轮次配置注入
    assert {r["id"] for r in load_rules()} >= JOB_SEEDED_RULES
    assert {v.why for v in check_ranked_eval(rs, policy={"kind": "sdk"})} == {"sdk_ranked"}
    assert rs.get("physics") == "realistic" and "ranked" in rs   # 覆盖不修改输入
    assert check_ranked_eval(rs, physics="sitl") and rs["physics"] == "realistic"


# ---- D3-AC-41 / D3-AC-69：排名运行只收 mesh 外观 ------------------------------------------------------------------------------------

@pytest.mark.parametrize("source", ["aerial", "mesh_distilled", "generative"])
def test_ranked_3dgs_returns_682_ranked_appearance(source: str) -> None:
    assert check(spec(appearance="3dgs", appearance_source=source)) == []       # 非排名运行可用（643 由 M25 按世界外观层判定）
    vs = check(spec(ranked=True, appearance="3dgs", appearance_source=source))
    assert {(v.code, v.why) for v in vs} == {(682, "ranked_appearance")}
    assert problem(vs)[0] == 682 and problem(vs)[1]["why"] == "ranked_appearance"


@pytest.mark.parametrize("source", ["mesh_distilled", "generative"])
def test_distilled_and_generative_stay_unranked_after_aerial_registration(source: str) -> None:
    """P3 起榜单版本登记 `aerial` 后，`ranked_appearance` 放开 `3dgs`，`mesh_distilled` 与 `generative` 仍返回 682（ADR-187）。"""
    p3 = tuple({**r, "allow": {"appearance": ["mesh", "3dgs"]}} if r["id"] == "ranked_appearance" else r for r in load_rules())
    assert check(spec(ranked=True, appearance="3dgs", appearance_source="aerial"), p3) == []
    vs = check(spec(ranked=True, appearance="3dgs", appearance_source=source), p3)
    assert [(v.rule, v.code, v.why, v.path) for v in vs] == [("ranked_appearance_source", 682, "ranked_appearance", "ranked")]
    cfg = {"run_spec": spec(appearance="3dgs", appearance_source=source), "physics": "realistic"}
    assert problem(check_submission_eval(cfg))[1]["why"] == "ranked_appearance"   # 以该外观提交排名运行返回 682（D3-AC-69 (b)）


def test_appearance_source_pairing() -> None:
    assert check(spec(ranked=True, appearance="mesh")) == []
    assert _codes(spec(appearance="3dgs")) == {(682, "appearance_source_required")}
    assert _codes(spec(appearance_source="mesh_distilled")) == {(682, "appearance_source_forbidden")}
    assert _codes(spec(ranked=True, appearance_source="mesh_distilled")) == {(682, "appearance_source_forbidden"), (682, "ranked_appearance")}


# ---- 层次与主原因码 ---------------------------------------------------------------------------------------------------------------

def test_first_layer_precedence() -> None:
    rules = load_rules()
    assert {r["code"] for r in rules} <= set(CODE_PRECEDENCE)    # 每条规则的原因码都有层次
    assert all(c in REASONS for c in CODE_PRECEDENCE)
    vs = check(spec(physics="sitl", ranked=True, obs_level="T0", seed_count=3, policy={"kind": "sdk"}))
    layer = first_layer(vs)
    assert [v.code for v in layer] == [633, 662, 664]             # 第 ④ 层全部问题，按 630、633、643、662、664 排序
    assert problem(vs)[0] == 633 and len(problem(vs)[1]["violations"]) == 3
    assert problem([]) is None and reason([]) is None and first_layer([]) == []
    assert problem(check(spec(clock_profile="sandbox100", ranked=True)))[0] == 637


# ---- 契约枚举同步 ----------------------------------------------------------------------------------------------------------------

def test_appearance_enums_are_in_sync() -> None:
    rs = _schema("bench/run_spec.v1.schema.json")["properties"]
    meta = _schema("sensor/v2/frame.v1.schema.json")["properties"]["internal"]["properties"]
    tracks = _schema("bench/board.v1.schema.json")["properties"]["tracks"]["items"]["properties"]
    assert rs["appearance"]["enum"] == meta["appearance"]["enum"] == ["mesh", "3dgs"]
    assert meta["appearance_source"]["enum"] == ["mesh", *rs["appearance_source"]["enum"]]
    assert rs["appearance_source"]["enum"] == ["aerial", "mesh_distilled", "generative"]
    assert meta["appearance_version"]["type"] == ["string", "null"]
    assert tracks["appearance"]["enum"] == ["mesh", "3dgs-aerial"]   # 分轨只登记 mesh 与（P3）aerial
    assert tracks["physics"] == {"const": "realistic"}
    gate = {r["id"]: r for r in load_rules()}
    assert gate["ranked_appearance"]["allow"] == {"appearance": ["mesh"]}
    assert set(gate["ranked_appearance_source"]["if"]["appearance_source"]) == set(rs["appearance_source"]["enum"]) - {"aerial"}
    assert gate["sitl_unranked"]["code"] == 633 and gate["fast_unranked"]["why"] == "ranked_requires_realistic"
    assert set(rs["physics"]["enum"]) == {"realistic", "fast", "gz", "sitl"}
