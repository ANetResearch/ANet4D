"""G-3DGS 决策门 G5 补测工具 `tools/bench/d3/g5_eval.py`（D3-AC-64；docs/05 §6.4.2、§6.4.4）。

不需要 GPU 的部分：r11 世界到产品世界的平移、深度反投影与街区掩膜、取景与演员放置（平地桩世界）、C3 合成、AP 与匹配规则、
幻觉审计、汇总与判定；另以随包定稿权重在合成视角上跑一次 eval 阶段的单视角流程（效果链、JPEG、ref_det_v1），核对"3DGS 与网格
逐像素相同则检测逐项相同"。views 与 gs 阶段需要 EGL 与 CUDA（dmax），不在本机运行。
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("g5_eval", REPO / "tools/bench/d3/g5_eval.py")
g5 = importlib.util.module_from_spec(_spec)
sys.modules["g5_eval"] = g5
_spec.loader.exec_module(g5)

K = (100.0, 100.0, 80.0, 60.0)


class FlatWorld:
    """平地桩世界：DTM = DSM = 0；`walls` 内的格子柱顶为 5 m（模拟建筑）。"""

    def __init__(self, walls: list[tuple[float, float, float, float]] | None = None) -> None:
        self.walls = walls or []

    def _h(self, xy: np.ndarray, r: float = 0.0) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, np.float64))
        h = np.zeros(len(xy))
        for x0, x1, y0, y1 in self.walls:
            h[(xy[:, 0] >= x0 - r) & (xy[:, 0] <= x1 + r) & (xy[:, 1] >= y0 - r) & (xy[:, 1] <= y1 + r)] = 5.0
        return h

    def ground_dtm(self, xy):
        return np.zeros(len(np.atleast_2d(xy)), np.float32)

    def height_dsm(self, xy, raw: bool = False):
        return self._h(xy)

    def column_max_within(self, xy, radius_m: float = 0.0):
        return self._h(xy, radius_m)


def _nadir(x: float, y: float, z: float) -> tuple[np.ndarray, np.ndarray]:
    from awr.sensorsvc.camera.proj import look_at_R

    p = np.array([x, y, z])
    return look_at_R(p, np.array([x, y, 0.0])), p


# ---------------------------------------------------------------------------------------------------------------- 坐标与取景
@pytest.mark.skipif(not (REPO / "worlds/oryahovo-crop1").is_dir(), reason="needs worlds/oryahovo-crop1 (make worlds)")
def test_gs_world_offset_matches_product_world_and_rejects_rotation() -> None:
    coord = json.loads((REPO / "worlds/oryahovo-crop1/coordinate.json").read_text(encoding="utf-8"))
    assert g5.gs_world_offset(coord) == [0.0, 0.0, 1.094441]      # 产品世界与 r11 只差高程平移
    bad = json.loads(json.dumps(coord))
    bad["source"]["T_world_source"][0][0] += 1e-3
    with pytest.raises(ValueError, match="rotation"):
        g5.gs_world_offset(bad)


def test_unproject_and_block_mask_on_flat_ground() -> None:
    R, p = _nadir(-150.0, 140.0, 50.0)
    depth = np.full((120, 160), 50.0, np.float32)
    X = g5.unproject(depth, K, R, p)
    assert np.allclose(X[..., 2], 0.0, atol=1e-9)
    # 像素中心 u = i + 0.5：中心像素 (59.5, 79.5) 落在相机正下方 ±半像素足迹
    assert np.allclose(X[60, 80, :2], [-150.0, 140.0], atol=0.5 * 50.0 / 100.0 + 1e-9)
    blk = g5.block_mask(depth, K, R, p, half=10.0)
    span = np.abs(X[..., :2] - np.array([-150.0, 140.0])).max(-1)
    assert np.array_equal(blk, span <= 10.0) and 0 < blk.mean() < 1
    depth[:5] = 0.0                                                    # 天空与几何外
    assert not g5.block_mask(depth, K, R, p)[:5].any()


@pytest.mark.parametrize("band", g5.BANDS, ids=[b[0] for b in g5.BANDS])
def test_sample_pose_band_geometry_and_determinism(band) -> None:
    q = FlatWorld()
    for k in range(20):
        R, p, meta = g5.sample_pose(np.random.default_rng([64, 0, k]), q, band)
        (a0, a1), (e0, e1) = band[1], band[2]
        assert a0 <= meta["agl_m"] <= a1 and a0 <= p[2] <= a1
        assert e0 <= meta["pitch_deg"] <= e1
        d = np.array(meta["target"]) - p
        assert np.allclose(R[:, 0], d / np.linalg.norm(d), atol=1e-4)             # 光轴指向注视点（meta 取 3 位小数）
        assert math.degrees(math.asin(-R[2, 0])) == pytest.approx(meta["pitch_deg"], abs=1e-3)
        assert abs(meta["target"][0] + 150.0) <= 50.0 and abs(meta["target"][1] - 140.0) <= 50.0
        R2, p2, meta2 = g5.sample_pose(np.random.default_rng([64, 0, k]), q, band)
        assert np.array_equal(R, R2) and np.array_equal(p, p2) and meta == meta2


def test_sample_pose_rejects_cameras_inside_buildings() -> None:
    q = FlatWorld(walls=[(-1e4, 1e4, -1e4, 1e4)])                    # 处处 5 m 柱顶，低档相机净空不足
    band = ("low", (5.0, 6.0), (89.0, 89.0))
    assert g5.sample_pose(np.random.default_rng(1), q, band) is None


def test_actor_spots_and_placement_rules() -> None:
    wall = (-160.0, -140.0, 130.0, 150.0)
    q = FlatWorld(walls=[wall])
    R, p = _nadir(-150.0, 140.0, 60.0)
    H, W = 240, 320
    Kc = (160.0, 160.0, 160.0, 120.0)
    depth = np.full((H, W), 60.0, np.float32)
    spots = g5.actor_spots(depth, Kc, R, p, q, stride=4)
    for cls, r in (("person", 1.5), ("vehicle", 3.0)):
        P = spots[cls]
        assert len(P) > 50
        assert not ((P[:, 0] >= wall[0] - r) & (P[:, 0] <= wall[1] + r) & (P[:, 1] >= wall[2] - r) & (P[:, 1] <= wall[3] + r)).any()
        assert np.allclose(P[:, 2], 0.0)
    assert len(spots["vehicle"]) < len(spots["person"])
    roster = g5.actor_roster([{"id": "car", "class": "vehicle", "size_mm": [4600, 1800, 1500]}]
                             + [{"id": f"p{i}", "class": "person", "size_mm": [500, 300, 1700]} for i in range(12)])
    assert [a["idx"] for a in roster] == list(range(18)) and sum(a["class"] == "vehicle" for a in roster) == 1 + g5.VEHICLE_COPIES
    placed = g5.place_actors(np.random.default_rng(3), spots, roster)
    ids = [i for i, _p, _y in placed]
    assert len(set(ids)) == len(ids) == g5.N_PERSON + g5.N_VEHICLE
    xy = np.array([p_[:2] for _i, p_, _y in placed])
    cls = [roster[i]["class"] for i in ids]
    for a in range(len(placed)):
        for b in range(a + 1, len(placed)):
            gap = 6.0 if "vehicle" in (cls[a], cls[b]) else 2.5
            assert np.hypot(*(xy[a] - xy[b])) >= gap - 1e-9
    assert placed == g5.place_actors(np.random.default_rng(3), spots, roster)


def test_gt_boxes_ignore_rule_matches_training_data() -> None:
    objs = [{"class": 32, "px_visible": 40, "visible_ratio": 0.9, "bbox_xywh": [1, 2, 3, 4]},
            {"class": 32, "px_visible": 11, "visible_ratio": 0.9, "bbox_xywh": [5, 6, 7, 8]},
            {"class": 33, "px_visible": 400, "visible_ratio": 0.19, "bbox_xywh": [9, 9, 9, 9]},
            {"class": 34, "px_visible": 400, "visible_ratio": 1.0, "bbox_xywh": [0, 0, 1, 1]},
            {"class": 33, "px_visible": 0, "visible_ratio": 0.0, "bbox_xywh": [0, 0, 0, 0]}]
    assert g5.gt_boxes(objs) == [[0, 1, 2, 3, 4, 0], [0, 5, 6, 7, 8, 1], [1, 9, 9, 9, 9, 1]]


# ---------------------------------------------------------------------------------------------------------------- C3 合成
def test_composite_replaces_only_static_block_pixels() -> None:
    rng = np.random.default_rng(0)
    bg = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    act = bg.copy()
    inst = np.zeros((6, 8), np.uint16)
    inst[2, 3] = 7
    act[2, 3] = (255, 0, 0)
    blk = np.zeros((6, 8), bool)
    blk[:, :4] = True
    gs = np.full((6, 8, 3), 100, np.uint8)
    a = np.full((6, 8), 255, np.uint8)
    a[0, 0] = 0                                                         # 空洞以网格回退（黑色背景渲染：alpha = 0 处颜色为 0）
    gs[0, 0] = 0
    mesh = g5.composite(bg, blk, mesh_act=act, inst=inst)
    assert np.array_equal(mesh, act.astype(np.float32) / 255)          # 网格对照即演员遍本身
    out = g5.composite(bg, blk, mesh_act=act, inst=inst, gs_rgb=gs, gs_a=a)
    assert np.allclose(out[2, 3], [1.0, 0.0, 0.0])                     # 演员像素恒取网格
    assert np.allclose(out[:, 4:], bg[:, 4:] / 255)                    # 街区外取网格
    assert np.allclose(out[1, 1], 100 / 255) and np.allclose(out[0, 0], bg[0, 0] / 255)
    half = g5.composite(bg, blk, gs_rgb=gs // 2, gs_a=np.full((6, 8), 128, np.uint8))
    assert np.allclose(half[3, 0], np.clip((gs[3, 0] // 2) / 255 + (1 - 128 / 255) * bg[3, 0] / 255, 0, 1), atol=1e-6)
    assert np.allclose(g5.srgb_to_linear(np.array([0.0, 0.04045, 1.0])), [0.0, 0.04045 / 12.92, 1.0])


# ---------------------------------------------------------------------------------------------------------------- AP 与审计
def test_average_precision_reference_values() -> None:
    assert g5.average_precision([(0.9, 1), (0.8, 1)], 2) == pytest.approx(1.0)
    assert g5.average_precision([(0.9, 0), (0.8, 0)], 2) == 0.0
    assert g5.average_precision([], 3) == 0.0
    assert math.isnan(g5.average_precision([(0.5, 0)], 0))
    # TP、FP、TP（npos 2）：召回 0.5 处精度 1，召回 1 处精度 2/3 → 101 点：51 个点取 1、50 个点取 2/3
    assert g5.average_precision([(0.9, 1), (0.8, 0), (0.7, 1)], 2) == pytest.approx((51 * 1.0 + 50 * (2 / 3)) / 101)
    # 只召回一半：召回 > 0.5 的 50 个点为 0
    assert g5.average_precision([(0.9, 1)], 2) == pytest.approx(51 / 101)


def test_match_frame_rules() -> None:
    boxes = [[0, 0, 0, 10, 20, 0], [0, 100, 100, 10, 20, 0], [0, 200, 200, 10, 20, 1], [1, 300, 300, 40, 20, 0]]
    dets = [[0, 0.9, 0, 0, 10, 20],         # TP
            [0, 0.8, 1, 1, 10, 20],         # 重复检测 → FP
            [0, 0.7, 200, 200, 10, 20],     # 落在忽略区域 → 不计
            [0, 0.6, 300, 300, 40, 20],     # 类别不符 → FP
            [1, 0.5, 300, 300, 40, 20]]     # 车辆类，不计入人员
    scored, npos = g5.match_frame(dets, boxes, 0, 0.5)
    assert npos == 2 and scored == [(0.9, 1), (0.8, 0), (0.6, 0)]
    scored, npos = g5.match_frame(dets, boxes, 1, 0.5)
    assert npos == 1 and scored == [(0.5, 1)]
    # IoU 0.33：0.3 档为 TP，0.5 档为 FP
    off = [[0, 0.9, 0, 10, 10, 20]]
    assert g5.match_frame(off, boxes, 0, 0.3)[0] == [(0.9, 1)]
    assert g5.match_frame(off, boxes, 0, 0.5)[0] == [(0.9, 0)]


def test_audit_counts_new_detections_against_mesh_frame() -> None:
    mesh = [[0, 0.9, 10, 10, 8, 16], [1, 0.4, 50, 50, 30, 15], [0, 0.2, 90, 90, 8, 16]]
    var = [[0, 0.8, 11, 10, 8, 16],       # 与网格同一检测
           [0, 0.6, 50, 50, 30, 15],      # 位置相同但类别不同 → 新增
           [0, 0.5, 90, 90, 8, 16],       # 网格中只在工作阈值以下 → 新增
           [1, 0.1, 400, 400, 10, 10]]    # 工作阈值以下，不计
    assert g5.audit_frame(var, mesh) == {"n": 3, "new": 2}
    assert g5.audit_frame(mesh, mesh) == {"n": 2, "new": 0}
    boxes = [[0, 10, 10, 8, 16, 0], [0, 90, 90, 8, 16, 1]]
    assert g5.false_positives(var, boxes) == 1                          # 忽略区域上的检测不算误检


def _synthetic(n_views: int = 12, seed: int = 1) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    views, recs = [], {}
    for i in range(n_views):
        band = g5.BANDS[i % 3][0]
        boxes = [[0, 40 * j, 10, 10, 20, 0] for j in range(5)] + [[1, 400, 400, 40, 20, 0], [0, 900, 900, 5, 5, 1]]
        dets = [[0, round(float(rng.uniform(0.5, 1.0)), 4), 40 * j, 10, 10, 20] for j in range(5)] + [[1, 0.9, 400, 400, 40, 20]]
        dets.append([0, 0.45, 600, 600, 10, 20])                       # 每帧 1 个低分误检
        vid = f"v{i:04d}"
        views.append({"id": vid, "band": band, "boxes": boxes})
        for var in ("mesh", "base", "lock"):
            d = [list(x) for x in dets]
            bg = [[0, 0.5, 700, 700, 10, 20]]
            if var == "lock" and i % 2 == 0:
                d = d[1:]                                               # 一半视角丢失一个人员 → AP 下降
                bg = [*bg, [1, 0.6, 800, 100, 30, 15]] # 一半视角新增一个背景"车辆"
            recs[(vid, var, "act")] = {"dets": d, "psnr_vs_mesh_block": None if var == "mesh" else 33.0}
            recs[(vid, var, "bg")] = {"dets": bg}
    return views, recs


def test_summarize_gate_identical_variant_passes_degraded_fails() -> None:
    views, recs = _synthetic()
    res = g5.summarize(views, recs, ["mesh", "base", "lock"], n_boot=50, seed=0)
    assert res["views"] == 12 and res["views_by_band"] == {"agl30": 4, "agl60": 4, "agl120": 4}
    assert res["positives"] == {"person": 60, "vehicle": 12} and res["ignored"] == 12
    for key in ("iou0.5", "iou0.3"):
        assert res["ap"][key]["mesh"]["all"]["mAP"] == pytest.approx(1.0)
        assert res["ap_delta"][key]["base"]["drop_pp"] == {"person": 0.0, "vehicle": 0.0, "mAP": 0.0}
        assert res["ap_delta"][key]["base"]["ci95_pp"]["mAP"] == [0.0, 0.0]
        lock = res["ap_delta"][key]["lock"]
        assert lock["drop_pp"]["person"] == pytest.approx(100 * (1 - 54 / 60), abs=0.6)  # 召回 0.9，101 点插值
        lo, hi = lock["ci95_pp"]["mAP"]
        assert lo <= lock["drop_pp"]["mAP"] <= hi
    assert res["audit"]["base"]["bg_new_per_frame"] == 0.0 and res["audit"]["base"]["bg_det_delta_per_frame"] == 0.0
    assert res["audit"]["lock"]["bg_new_per_frame"] == 0.5 and res["audit"]["lock"]["bg_new_per_frame_vehicle"] == 0.5
    assert res["audit"]["mesh"]["act_fp_per_frame"] == 1.0
    assert res["audit"]["lock"]["bg_new_vs_base_per_frame"] == 0.5 and "bg_new_vs_base_per_frame" not in res["audit"]["base"]
    assert res["gate"]["base"]["pass"] is True
    assert res["gate"]["base"]["hallucination_gated"] is False and res["gate"]["base"]["hallucination_pass"] is None
    assert res["gate"]["lock"]["ap_pass"] is False and res["gate"]["lock"]["hallucination_pass"] is False
    assert res["g5_pass"] is False
    assert "mesh_reseed" not in res["audit"]
    assert res["psnr_vs_mesh_block_median_db"] == {"base": 33.0, "lock": 33.0}
    res2 = g5.summarize(views, recs, ["mesh", "base"], n_boot=50, seed=0)
    assert res2["g5_pass"] is True and list(res2["gate"]) == ["base"]


def test_summarize_hallucination_gate_only_for_generative_and_noise_floor() -> None:
    views, recs = _synthetic()
    for v in views:                                                    # 产物 A 在一半视角上出现一个与网格帧不重合的检测（重渲染抖动）
        if int(v["id"][1:]) % 2:
            recs[(v["id"], "base", "bg")] = {"dets": [[0, 0.5, 700, 700, 10, 20], [0, 0.35, 10, 900, 6, 12]]}
    reseed = {(v["id"], "mesh", "bg"): {"dets": [[0, 0.5, 700, 700, 10, 20]] + ([[0, 0.31, 5, 5, 6, 12]] if i < 3 else [])}
              for i, v in enumerate(views)}
    res = g5.summarize(views, recs, ["mesh", "base"], n_boot=20, seed=0, reseed=reseed)
    assert res["audit"]["base"]["bg_new_per_frame"] == 0.5
    assert res["gate"]["base"]["pass"] is True                         # 产物 A 不判幻觉审计
    assert res["audit"]["mesh_reseed"] == {"bg_det_per_frame": 1.25, "bg_new_per_frame": 0.25}


# ---------------------------------------------------------------------------------------------------------------- eval 单视角流程
def test_eval_view_identical_gs_gives_identical_detections(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    vd, gd = tmp_path / "views", tmp_path / "gs" / "base"
    vd.mkdir(parents=True)
    gd.mkdir(parents=True)
    rng = np.random.default_rng(5)
    H, W = g5.H_PX, g5.W_PX
    bg = np.clip(rng.normal(120, 25, (H // 8, W // 8, 3)), 0, 255).astype(np.uint8)
    bg = np.asarray(Image.fromarray(bg).resize((W, H), Image.BILINEAR))
    act = bg.copy()
    inst = np.zeros((H, W), np.uint16)
    for j, (x, y) in enumerate(((400, 300), (800, 500), (1200, 700))):  # 三个"人员"色块
        act[y:y + 40, x:x + 16] = (42, 93, 176)
        act[y + 20:y + 40, x:x + 16] = (40, 40, 40)
        inst[y:y + 40, x:x + 16] = j + 1
    blk = np.zeros((H, W), bool)
    blk[:, : W // 2] = True
    g5._png(vd / "v0000_bg.png", bg)
    g5._png(vd / "v0000_act.png", act)
    g5._png(vd / "v0000_inst.png", inst)
    g5._png(vd / "v0000_block.png", blk)
    assert np.array_equal(g5._read_png(vd / "v0000_inst.png"), inst)
    assert np.array_equal(g5._read_png(vd / "v0000_block.png") > 0, blk)
    g5._png(gd / "v0000_rgb.png", bg)                                  # 3DGS 与网格逐像素相同、不透明
    g5._png(gd / "v0000_a.png", np.full((H, W), 255, np.uint8))
    g5._worker_init(str(vd), str(tmp_path / "gs"), {})
    v = {"id": "v0000", "band": "agl30", "boxes": []}
    recs = g5.eval_view((0, v, ["mesh", "base"], 64))
    assert [(r["variant"], r["mode"]) for r in recs] == [("mesh", "act"), ("mesh", "bg"), ("base", "act"), ("base", "bg")]
    by = {(r["variant"], r["mode"]): r for r in recs}
    for mode in ("act", "bg"):
        assert by[("mesh", mode)]["dets"] == by[("base", mode)]["dets"]
        assert by[("mesh", mode)]["jpeg_bytes"] == by[("base", mode)]["jpeg_bytes"]
        assert by[("mesh", mode)]["gain"] == by[("base", mode)]["gain"]
    assert by[("base", "act")]["psnr_vs_mesh_block"] == 100.0
    assert by[("mesh", "act")]["psnr_vs_mesh_block"] is None
    assert by[("mesh", "act")]["jpeg_bytes"] != by[("mesh", "bg")]["jpeg_bytes"]
    for r in recs:
        for d in r["dets"]:
            assert d[0] in (0, 1) and g5.DET_THR_AP <= d[1] <= 1.0 and len(d) == 6
