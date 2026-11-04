"""门 1 错误目录一次全报与 CLI 退出码（D3-AC-06 第一句；M23-AC-002、M23-NFR-008；M23 §6.4、§7.2）。

故意写错的事件 `bad-001`（crop 世界包 + 事件锚点覆盖层）同时含 5 类错误：未知锚点、时间不单调、上车窗口、类别与 profile 不符、
未知参与者；编译一次报出全部 5 类（原因码 611、只有门 1、不装载舞台），`python -m awr.sim.events compile|preflight` 退出码 2。
另覆盖上车窗口静态检查与门 2 `I-BOARD` 的分工、限额 618 与阶段不支持 619 的原因码、YAML 别名与超限文本的拒绝。
门 1 不需要世界几何：世界包只复制 crop 的三个标定证据文件（world.json、coordinate.json、qa/report.json）。
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
import yaml

pytest.importorskip("jsonschema")
from awr.sim.events import __main__ as cli
from awr.sim.events.compile import CompileOptions, compile_event
from awr.sim.events.dsl import gate1, load_doc, stop_windows
from awr.sim.events.errors import EventError

ROOT = Path(__file__).resolve().parents[3]
CROP = ROOT / "worlds" / "oryahovo-crop1"
pytestmark = pytest.mark.skipif(not CROP.is_dir(), reason="needs worlds/oryahovo-crop1 (make worlds)")


def _sq(cx: float, cy: float, r: float) -> list[list[float]]:
    return [[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]]


OVERLAY = {"schema": "awr.anchors.v1", "extends": "world", "world_id": "oryahovo-crop1", "anchors": [
    {"id": "building_X_exit", "kind": "door", "profile": "person", "xy": [-80.0, -20.0], "building": "bX"},
    {"id": "building_Y_exit", "kind": "door", "profile": "person", "xy": [-40.0, 60.0], "building": "bY"},
    {"id": "building_Z_entrance", "kind": "door", "profile": "person", "xy": [90.0, 10.0], "building": "bZ"},
    {"id": "north_street", "kind": "point", "profile": "person", "xy": [0.0, 40.0]},
    {"id": "east_street", "kind": "point", "profile": "person", "xy": [40.0, 40.0]},
    {"id": "plaza", "kind": "zone", "profile": "person", "polygon": _sq(60.0, 0.0, 12.0)},
    {"id": "plaza_curb", "kind": "curb", "profile": "person", "xy": [75.0, -15.0], "pair": "plaza_curb_road"},
    {"id": "plaza_curb_road", "kind": "point", "profile": "vehicle", "xy": [79.0, -15.0]},
    {"id": "road_west", "kind": "point", "profile": "vehicle", "xy": [-120.0, -15.0]},
    {"id": "road_east", "kind": "point", "profile": "vehicle", "xy": [140.0, -15.0]},
    {"id": "town_center", "kind": "zone", "profile": "person", "polygon": _sq(0.0, 0.0, 100.0)},
]}

# 原型 r6 bad-001 的 crop 改写：每处错误旁注明期望的错误码
BAD_001 = """\
schema: drone4d.event.v1
event_id: bad-001
version: 1
world: {id: oryahovo-crop1}
anchors_ref: "event:anchors.json"
seed: 20261011
duration_s: 360
actors:
  - id: person_A
    class: person
    appearance: {preset: civilian, top_rgb: [42, 93, 176], height_m: 1.76}
    route:
      - {t: 0,   at: building_X_exit, do: spawn}
      - {t: 95,  at: north_streeet}                              # E-ANCHOR-UNKNOWN
      - {t: 215, at: plaza, dwell_s: 20, do: wait}
      - {t: 300, at: building_Z_entrance, do: enter_building}
  - id: person_B
    class: person
    route:
      - {t: 60,  at: building_Y_exit, do: spawn}
      - {t: 20,  at: east_street}                                # E-TIME-ORDER
      - {t: 210, at: plaza, dwell_s: 12, do: wait}
      - {t: 335, at: plaza_curb, do: board, target: vehicle_C}   # E-BOARD-TARGET：车辆 330 s 已须到达 road_east
  - id: vehicle_C
    class: vehicle
    appearance: {preset: sedan, body_rgb: [230, 230, 230], length_m: 4.5}
    route:
      - {t: 150, at: road_west, do: spawn}
      - {t: 235, at: plaza_curb, dwell_s: 30, do: wait}          # E-CLASS-PROFILE：车辆使用 person 锚点
      - {t: 330, at: road_east, do: despawn}
background:
  - {class: person, count: 40, zone: town_center, pause_s: [0, 20], stream: 1}
environment:
  keyframes:
    - {t: 0,   wind_mps: 2.0, rain_mm_h: 0}
    - {t: 180, wind_mps: 7.0, rain_mm_h: 4, cloud: 0.9}
annotations:
  - {id: encounter_AB, type: proximity_encounter, participants: [person_A, person_D], window: [195, 250],
     params: {dist_m: 3.0, min_dur_s: 5}}                        # E-PARTICIPANT-UNKNOWN
  - {id: board_BC, type: board_vehicle, participants: [person_B, vehicle_C], window: [240, 275]}
relations:
  - {src: encounter_AB, rel: before, dst: board_BC}
"""

EXPECTED = {
    "E-ANCHOR-UNKNOWN": ["/actors/0/route/1/at"],
    "E-TIME-ORDER": ["/actors/1/route/1/t"],
    "E-BOARD-TARGET": ["/actors/1/route/3/t"],
    "E-CLASS-PROFILE": ["/actors/2/route/1/at"],
    "E-PARTICIPANT-UNKNOWN": ["/annotations/0/participants/1"],
}


@pytest.fixture()
def crop(tmp_path: Path) -> Path:
    """crop 世界包的标定证据副本（门 1 只读 world.json 与米制证据，失败时不装载舞台）。"""
    d = tmp_path / CROP.name
    (d / "qa").mkdir(parents=True)
    for rel in ("world.json", "coordinate.json", "qa/report.json"):
        shutil.copyfile(CROP / rel, d / rel)
    return d


def _write(tmp: Path, text: str, name: str = "bad-001.yaml") -> Path:
    d = tmp / "ev"
    d.mkdir(exist_ok=True)
    (d / "anchors.json").write_text(json.dumps(OVERLAY), encoding="utf-8")
    (d / name).write_text(text, encoding="utf-8")
    return d / name


def _bad_doc() -> dict:
    return yaml.safe_load(BAD_001)


def _anchors() -> dict[str, dict]:
    return {a["id"]: a for a in OVERLAY["anchors"]}


def _by_code(errs: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for e in errs:
        out.setdefault(e["code"], []).append(e["path"])
    return {k: sorted(v) for k, v in out.items()}


# ------------------------------------------------------------------------------------------------- 一次全报（D3-AC-06）
def test_bad_event_reports_all_five_classes_at_once(crop: Path, tmp_path: Path) -> None:
    """一个坏事件一次报出 5 类错误，每类恰好一处、路径精确；原因码 611；门 1 失败后不进入门 2（不装载舞台）。"""
    ev = _write(tmp_path, BAD_001)
    r = compile_event(ev, crop, opts=CompileOptions(out=tmp_path / "out", cache_dir=tmp_path / "cache"))
    assert r.status == "failed" and r.reason == 611 and r.event_id == "bad-001"
    assert [g["gate"] for g in r.gates] == [1] and r.gates[0]["status"] == "fail"
    errs = r.gates[0]["errors"]
    assert _by_code(errs) == EXPECTED
    assert all(e["reason"] == 611 and e["message_zh"] and e["message_en"] for e in errs)
    board = next(e for e in errs if e["code"] == "E-BOARD-TARGET")
    assert board["detail"] == {"vehicle": "vehicle_C", "why": "window", "t_s": 335.0, "windows_s": [[235.0, 330.0]]}
    assert next(e for e in errs if e["code"] == "E-ANCHOR-UNKNOWN")["detail"] == {"anchor": "north_streeet"}
    assert next(e for e in errs if e["code"] == "E-CLASS-PROFILE")["detail"] == {"anchor": "plaza_curb"}
    assert r.as_dict()["reason"] == {"code": 611, "name": "EVENT_COMPILE_FAILED"}
    assert not (tmp_path / "out").exists() and not (tmp_path / "cache").exists()


def test_each_error_is_independent_of_the_others(crop: Path, tmp_path: Path) -> None:
    """逐一修正 4 类错误后只剩第 5 类：每类错误各自可检出，不依赖其余错误同时存在（排除"一错掩一错"）。"""
    fixes = {
        "E-ANCHOR-UNKNOWN": lambda d: d["actors"][0]["route"][1].update(at="north_street"),
        "E-TIME-ORDER": lambda d: d["actors"][1]["route"][1].update(t=120),
        "E-BOARD-TARGET": lambda d: d["actors"][1]["route"][3].update(t=290),
        "E-CLASS-PROFILE": lambda d: d["actors"][2]["route"][1].update(at="plaza_curb_road"),
        "E-PARTICIPANT-UNKNOWN": lambda d: d["annotations"][0]["participants"].__setitem__(1, "person_B"),
    }
    for keep in EXPECTED:
        doc = _bad_doc()
        for code, fix in fixes.items():
            if code != keep:
                fix(doc)
        plan, errs, _w = gate1(doc, _anchors())
        assert plan is None and _by_code([e.as_dict() for e in errs]) == {keep: EXPECTED[keep]}, keep
    doc = _bad_doc()
    for fix in fixes.values():
        fix(doc)
    plan, errs, _w = gate1(doc, _anchors())
    assert errs == [] and plan is not None and plan["event_id"] == "bad-001"


# ------------------------------------------------------------------------------------------------------- CLI 退出码
@pytest.mark.parametrize("cmd", ["compile", "preflight"])
def test_cli_exit_code_2_on_gate1_failure(cmd: str, crop: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`python -m awr.sim.events compile|preflight` 对门 1 失败返回 2，标准输出为完整编译报告（5 类错误、611）。"""
    ev = _write(tmp_path, BAD_001)
    rc = cli.main([cmd, str(ev), "--world", str(crop), "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "cache")])
    assert rc == 2
    rep = json.loads(capsys.readouterr().out)
    assert rep["status"] == "failed" and rep["reason"] == {"code": 611, "name": "EVENT_COMPILE_FAILED"}
    assert _by_code(rep["gates"][0]["errors"]) == EXPECTED and len(rep["gates"]) == 1


def test_cli_overlay_flag_and_unparsable_input_exit_2(crop: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`--anchors` 显式给出覆盖层时结果相同；无法解析的文档（YAML 别名）同属门 1，退出码 2。"""
    ev = _write(tmp_path, BAD_001.replace('anchors_ref: "event:anchors.json"\n', ""))
    assert cli.main(["compile", str(ev), "--world", str(crop), "--anchors", str(ev.parent / "anchors.json")]) == 2
    assert _by_code(json.loads(capsys.readouterr().out)["gates"][0]["errors"]) == EXPECTED
    alias = _write(tmp_path, "base: &b {t: 0}\nother: *b\n", "alias.yaml")
    assert cli.main(["compile", str(alias), "--world", str(crop)]) == 2
    rep = json.loads(capsys.readouterr().out)
    assert rep["reason"]["code"] == 611 and [e["code"] for e in rep["gates"][0]["errors"]] == ["E-SCHEMA"]
    assert "alias" in rep["gates"][0]["errors"][0]["message_en"]


def test_cli_exit_code_mapping(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """退出码表（§7.2）：通过 0；门 1 失败 2（含 618、619 与 600）；门 2 不可行 3；其他编译错误与 EventError 为 4。"""
    g1f, g1p = {"gate": 1, "status": "fail"}, {"gate": 1, "status": "pass"}
    assert cli.exit_code("compiled", [g1p]) == 0 and cli.exit_code("preflight", [g1p]) == 0
    assert cli.exit_code("failed", [g1f]) == 2
    assert cli.exit_code("infeasible", [g1p, {"gate": 2, "status": "fail"}]) == 3
    assert cli.exit_code("failed", [g1p, {"gate": 2, "status": "fail"}]) == 4
    import awr.sim.events.compile as comp

    def boom(*_a, **_k):
        raise EventError(616, "navmesh unavailable", {"profile": "vehicle"})

    monkeypatch.setattr(comp, "compile_event", boom)
    assert cli.main(["compile", str(tmp_path / "e.yaml"), "--world", str(tmp_path)]) == 4
    err = json.loads(capsys.readouterr().err)
    assert err["code"] == 616 and err["reason"] == "EVENT_NAVMESH_UNAVAILABLE" and err["detail"] == {"profile": "vehicle"}


# ------------------------------------------------------------------------------------------- 上车窗口：门 1 与门 2 的分工
def test_board_window_static_check() -> None:
    """门 1 只用声明时刻：车辆停靠窗口为 [t_wait, 下一路点声明时刻)；上车允许乘客先到等候，下车须落在窗口内。
    晚于等待结束但早于车辆下一路点的上车留给门 2（I-BOARD 与延长停留建议），门 1 不报。"""
    route = [{"t": 0, "at": "road_west", "do": "spawn"}, {"t": 60, "at": "plaza_curb_road", "do": "wait", "dwell_s": 40},
             {"t": 130, "at": "road_east", "do": "despawn"}]
    assert stop_windows(route, 150) == [(60.0, 130.0, False)]
    assert stop_windows(route[:2], 150) == [(60.0, 150.0, True)]
    assert stop_windows([*route[:2], {"t": 130, "at": "road_east", "do": "despawn", "dwell_s": 5}], 150) == [(60.0, 130.0, False),
                                                                                                           (130.0, 135.0, True)]
    assert stop_windows([route[0], {**route[1], "do": "pass", "dwell_s": 0}, route[2]], 150) == []

    def doc(do: str, t: float, vroute: list[dict]) -> dict:
        return {"schema": "drone4d.event.v1", "event_id": "board-window", "world": {"id": "oryahovo-crop1"}, "seed": 1, "duration_s": 150,
                "actors": [{"id": "p", "class": "person", "route": [{"t": 0, "at": "north_street", "do": "spawn"},
                                                                   {"t": t, "at": "plaza_curb", "do": do, "target": "v"}]},
                           {"id": "v", "class": "vehicle", "route": vroute}]}

    def board_errs(do: str, t: float, vroute: list[dict] = route) -> list[dict]:
        _plan, errs, _w = gate1(doc(do, t, vroute), _anchors())
        return [e.as_dict() for e in errs if e.code == "E-BOARD-TARGET"]

    assert board_errs("board", 10) == [] and board_errs("board", 80) == [] and board_errs("board", 110) == []
    (e,) = board_errs("board", 130)
    assert e["path"] == "/actors/0/route/1/t" and e["detail"]["why"] == "window" and e["detail"]["windows_s"] == [[60.0, 130.0]]
    assert board_errs("alight", 80) == [] and [x["detail"]["why"] for x in board_errs("alight", 40)] == ["window"]
    (e,) = board_errs("board", 80, [route[0], {**route[1], "do": "pass", "dwell_s": 0}, route[2]])
    assert e["path"] == "/actors/0/route/1/target" and e["detail"] == {"vehicle": "v", "why": "no_wait"}


# --------------------------------------------------------------------------------------------- 限额 618、阶段不支持 619
def test_limits_618_and_unsupported_619(crop: Path, tmp_path: Path) -> None:
    """§6.16 限额的 schema 违例报 E-LIMIT 618（不报 E-SCHEMA），且不阻断其余跨字段检查；不支持的类别报 619。"""
    doc = _bad_doc()
    doc["actors"] += [{"id": f"m{i}", "class": "machine", "route": [{"t": 0, "at": "north_street", "do": "spawn"}]} for i in range(28)]
    plan, errs, _w = gate1(doc, _anchors())
    codes = _by_code([e.as_dict() for e in errs])
    assert codes == {**EXPECTED, "E-LIMIT": ["/actors"]} and plan is None
    assert {e.reason for e in errs if e.code == "E-LIMIT"} == {618}
    doc = _bad_doc()
    doc["duration_s"] = 3601
    doc["background"][0]["count"] = 1001
    _plan, errs, _w = gate1(doc, _anchors())
    assert _by_code([e.as_dict() for e in errs if e.code == "E-LIMIT"]) == {"E-LIMIT": ["/background", "/duration_s"]}
    assert "E-SCHEMA" not in {e.code for e in errs} and "E-PARTICIPANT-UNKNOWN" in {e.code for e in errs}
    ev = _write(tmp_path, yaml.safe_dump(doc), "limit.yaml")
    r = compile_event(ev, crop)
    assert r.status == "failed" and r.reason == 618
    doc = _bad_doc()
    doc["actors"][2]["appearance"]["top_rgb"] = [1, 2, 3, 4]             # 结构性错误仍提前返回 E-SCHEMA
    _plan, errs, _w = gate1(doc, _anchors())
    assert [(e.code, e.path) for e in errs] == [("E-SCHEMA", "/actors/2/appearance/top_rgb")]
    doc = copy.deepcopy(_bad_doc())
    doc["actors"].append({"id": "uav_1", "class": "uav", "route": [{"t": 0, "at": "north_street", "do": "spawn"}]})
    ev = _write(tmp_path, yaml.safe_dump(doc), "uav.yaml")
    r = compile_event(ev, crop)
    assert r.status == "failed" and r.reason == 619
    assert {(e["code"], e["reason"]) for e in r.gates[0]["errors"] if e["path"] == "/actors/3/class"} == {("E-CLASS-UNSUPPORTED", 619)}
    assert set(EXPECTED) <= {e["code"] for e in r.gates[0]["errors"]}


# ------------------------------------------------------------------------------------------------ 输入安全（NFR-008）
def test_yaml_input_safety() -> None:
    """拒绝别名引用、> 1 MiB 文本、> 100,000 节点、> 4 KiB 标量；单纯的锚点定义与上限内的输入可解析。"""
    with pytest.raises(ValueError, match="aliases"):
        load_doc("a: &x [1, 2]\nb: *x\n")
    with pytest.raises(ValueError, match="aliases"):                    # 别名放大（billion laughs）在构造对象前被拒绝
        load_doc("a: &a [1, 1]\nb: &b [*a, *a]\nc: &c [*b, *b]\nd: [*c, *c]\n")
    assert load_doc("a: &x [1, 2]\nb: 3\n") == {"a": [1, 2], "b": 3}
    with pytest.raises(ValueError, match="1 MiB"):
        load_doc("a: " + "x" * (1 << 20) + "\n")
    with pytest.raises(ValueError, match="nodes"):
        load_doc("a: [" + "0," * 100_000 + "0]\n")
    assert len(load_doc("a: [" + "0," * 99_996 + "0]\n")["a"]) == 99_997     # 映射、键、序列各 1 个节点 + 99,997 个标量 = 100,000
    with pytest.raises(ValueError, match="4 KiB"):
        load_doc("a: " + "y" * 4097 + "\n")
    assert load_doc("a: " + "y" * 4096 + "\n")["a"] == "y" * 4096
