"""Run 调度器与准入（D3-AC-22 调度四项；AWR-05 §4.7、ADR-163、ADR-178；M27 §6.7.2、§6.7.3、FR-019、FR-021、FR-022、FR-061）。

1. 单卡并发正式 Run ≤ 3 且各自份额 ≥ 1/3 卡：同卡份额合计 ≤ 1，第 4 个排队；多卡时分布到各独占卡；共享开发卡不承载正式评测。
2. 评测队列非空时正式评测优先准入：评测预留余量，预览、在线与批作业只用余量之外的资源（`eval_priority`），已准入运行不被中断；
   卡上负载回落后评测接续；评测队列清空后生产类按 §4.7 份额上限恢复。
3. `d3.slice`：正式评测核数 ≤ 30，评测队列非空时其他批作业合计 ≤ 8 核、FAST-LIVO2 并行度 ≤ 4；评测队列空闲时受 40 核总量约束；
   cgroup 子组与 docker 资源参数（单卡绑定、`--memory`、`--cgroup-parent`）。
4. 写带宽：预算 `min(300 MB/s, 实测 × 50%)`；按估计写速率之和准入，超出排队并记 686；`io.stat` 实测超预算 10% 暂停新准入；
   `/data/d3` 超 95% 配额排队并记 685。
5. 成本估算：标准轨正式 Run（3 机 × `rgb1080-rs` + `therm512` + MID-360，360 s）的写速率、持久体积、CPU、内存与 GPU 时间与
   AWR-05 §4.7、§5.6.4 口径一致；3 个并发正式 Run 合计约 50 MB/s。
6. GPU 日历：验收窗口只准入验收作业且整卡排他；训练窗口生产上限降至 50%、预览暂停；租户轮转与实测修正。
"""

from __future__ import annotations

import time

import pytest

from awr.bench.scheduler import (
    Channel,
    CostVector,
    GpuSlot,
    Job,
    SchedConfig,
    Scheduler,
    Telemetry,
    Window,
    cgroup_parent,
    channels_of,
    estimate_cost,
    make_job,
    resource_args,
    slice_quota,
    write_budget_mbps,
)

EXCL3 = GpuSlot(3, exclusive=True)
EXCL2 = GpuSlot(2, exclusive=True)
DEV2 = GpuSlot(2)
EVAL_COST = CostVector("online", 1 / 3, 1.2, 0, 10.0, 30.0, 15_000_000_000, 0, 15.5, 970.0)


def ev(i: int, *, tenant: str = "t-eval", t: float = 0.0, cost: CostVector = EVAL_COST) -> Job:
    return Job(f"eval-{i:02d}", tenant, "eval", cost, t)


def job(jid: str, cls: str = "research", *, tenant: str = "t-a", t: float = 0.0, gpu: str = "online", share: float = 0.3,
        cpu: float = 2.0, mem: float = 4.0, write: float = 5.0, persist: int = 1_000_000_000, kind: str = "realistic_run",
        wall: float = 600.0, vram: float = 1.0, acceptance: bool = False) -> Job:
    c = CostVector(gpu, share if gpu != "none" else 0.0, vram if gpu != "none" else 0.0, 0, cpu, mem, persist, 0, write, wall)
    return Job(jid, tenant, cls, c, t, kind, acceptance)


def admitted(ds) -> list[str]:
    return [d.job_id for d in ds if d.admitted]


def by_id(ds) -> dict:
    return {d.job_id: d for d in ds}


def card_share(s: Scheduler, gpu: int) -> float:
    return sum(r.share for r in s.running.values() if r.gpu == gpu)


def rs(n: int = 3, *, ranked: bool = True, therm: bool = True, level: str = "L2", physics: str = "realistic", **kw) -> dict:
    sensors = [{"sensor_id": "therm0"}] if therm else []
    return {"schema": "drone4d.bench.run_spec.v1", "world": {"id": "town-01"}, "event": {"id": "ev-01"},
            "fleet": [{"uav_id": f"uav-{i + 1:02d}", "vehicle_ref": "x500", "start": {"kind": "pose", "pos_enu_m": [0, 0, 30], "yaw_deg": 0},
                       "sensors": list(sensors)} for i in range(n)],
            "obs_level": level, "physics": physics, "policy": {"kind": "reference", "name": "hover"}, "duration_s": 360, "ranked": ranked, **kw}


# ── 1. 单卡并发正式 Run ≤ 3、份额 ≥ 1/3 ───────────────────────────────────────────────────────────────────────────────

def test_eval_at_most_three_per_card_each_one_third():
    s = Scheduler([EXCL3])
    for i in range(5):
        s.submit(ev(i, t=i))
    ds = s.tick(10.0)
    assert admitted(ds) == ["eval-00", "eval-01", "eval-02"]
    for d in ds[:3]:
        assert d.gpu == 3 and d.share >= 1 / 3 - 1e-12
    assert card_share(s, 3) <= 1.0 + 1e-12
    assert "gpu" in by_id(ds)["eval-03"].blocked_by             # 第 4 个同时超出单卡 3 个与 30 核
    s.finish("eval-01")
    ds = s.tick(20.0)
    assert admitted(ds) == ["eval-03"] and "gpu" in by_id(ds)["eval-04"].blocked_by
    assert sum(1 for r in s.running.values() if r.gpu == 3 and r.job.is_eval) == 3


def test_eval_spreads_over_exclusive_cards_and_never_on_shared_card():
    s = Scheduler([EXCL2, EXCL3])
    for i in range(3):
        s.submit(ev(i, t=i))
    ds = s.tick(0.0)
    assert sorted(d.gpu for d in ds if d.admitted) == [2, 2, 3] or sorted(d.gpu for d in ds if d.admitted) == [2, 3, 3]
    assert max(sum(1 for r in s.running.values() if r.gpu == g) for g in (2, 3)) <= 2

    dev = Scheduler([DEV2])
    dev.submit(ev(0))
    assert dev.tick(0.0)[0].blocked_by == ("gpu",)
    proto = Scheduler([DEV2], config=SchedConfig(allow_shared_eval=True))
    proto.submit(ev(0))
    d = proto.tick(0.0)[0]
    assert d.admitted and d.condition == "shared_card"


def test_eval_cpu_pool_caps_formal_concurrency_at_three():
    """`d3-eval.slice` ≤ 30 核：标准轨 10 核/Run，两张独占卡时也只并发 3 个正式 Run。"""
    s = Scheduler([EXCL2, EXCL3])
    for i in range(4):
        s.submit(ev(i, t=i))
    ds = s.tick(0.0)
    assert len(admitted(ds)) == 3 and by_id(ds)["eval-03"].blocked_by == ("cpu",)
    assert sum(r.job.cost.cpu_cores for r in s.running.values() if r.job.is_eval) <= 30


# ── 2. 评测队列非空时正式评测优先 ─────────────────────────────────────────────────────────────────────────────────────

def test_eval_admitted_before_earlier_research_and_batch():
    s = Scheduler([EXCL3])
    s.submit(job("res-1", t=0.0, share=0.6))
    s.submit(job("bat-1", "batch", t=1.0, share=0.3, gpu="offline"))
    for i in range(3):
        s.submit(ev(i, t=100.0 + i))
    ds = s.tick(200.0)
    assert admitted(ds) == ["eval-00", "eval-01", "eval-02"]
    assert by_id(ds)["res-1"].blocked_by == ("gpu",) and by_id(ds)["bat-1"].blocked_by == ("gpu",)


def test_running_production_is_not_preempted_and_eval_takes_over_when_it_drains():
    s = Scheduler([EXCL3])
    s.submit(job("prod-1", "batch", share=0.6))
    assert admitted(s.tick(0.0)) == ["prod-1"]
    for i in range(3):
        s.submit(ev(i, t=10.0 + i))
    s.submit(job("prod-2", "batch", t=11.0, share=0.05, gpu="offline"))
    ds = s.tick(20.0)
    assert admitted(ds) == ["eval-00"]                         # 0.6 + 1/3 ≤ 1；已准入的生产不被中断
    assert by_id(ds)["eval-01"].blocked_by == ("gpu",)
    assert by_id(ds)["prod-2"].blocked_by == ("eval_priority",)  # 0.05 原本放得下，但须留给排队中的评测
    assert "prod-1" in s.running
    s.finish("prod-1")
    ds = s.tick(30.0)
    assert admitted(ds) == ["eval-01", "eval-02"]
    assert by_id(ds)["prod-2"].blocked_by == ("gpu",)          # 卡被 3 个正式 Run 占满
    assert card_share(s, 3) <= 1.0 + 1e-12


def test_preview_cannot_starve_queued_eval():
    """预览类优先级最高，但在评测预留的余量之外才能准入：2 个正式 Run 运行、1 个排队时，预览不得占用第 3 份。"""
    s = Scheduler([EXCL3])
    s.submit(job("prod-1", "batch", share=0.30))
    s.tick(0.0)
    for i in range(3):
        s.submit(ev(i, t=1.0 + i))
    ds = s.tick(2.0)
    assert admitted(ds) == ["eval-00", "eval-01"]
    s.submit(job("pv-1", "preview", gpu="preview", share=0.02, cpu=0.0, mem=0.5, write=0.0, persist=0))
    ds = s.tick(3.0)
    assert by_id(ds)["pv-1"].blocked_by == ("eval_priority",) and by_id(ds)["eval-02"].blocked_by == ("gpu",)
    s.finish("prod-1")
    ds = s.tick(4.0)
    assert admitted(ds) == ["eval-02"]
    assert by_id(ds)["pv-1"].blocked_by == ("eval_priority",)   # 本轮判定预览时 eval-02 仍在排队
    assert by_id(s.tick(4.5))["pv-1"].blocked_by == ("gpu",)    # 3 个正式 Run 占满整卡
    s.finish("eval-00")
    assert admitted(s.tick(5.0)) == ["pv-1"]                   # 评测队列已空，预览在回收后的余量内准入


def test_production_caps_restored_when_eval_queue_empty_and_reclaimed_proportionally():
    s = Scheduler([EXCL3])
    s.submit(ev(0))
    s.tick(0.0)
    s.submit(job("on-1", "batch", share=0.45))                 # online 上限 0.6 × (1 − 1/3) = 0.4
    assert s.tick(1.0)[0].blocked_by == ("gpu",)
    s.finish("eval-00")
    assert admitted(s.tick(2.0)) == ["on-1"]
    s.submit(job("on-2", "batch", share=0.2))                  # 0.45 + 0.2 > 0.6
    assert s.tick(3.0)[0].blocked_by == ("gpu",)


def test_eval_reserve_covers_write_and_memory():
    cfg = SchedConfig(measured_disk_mbps=100.0)                # 预算 50 MB/s
    s = Scheduler([EXCL3], config=cfg)
    s.submit(ev(0, cost=CostVector("online", 1 / 3, 1.2, 0, 10.0, 30.0, 10**9, 0, 30.0, 900.0)))
    s.submit(ev(1, cost=CostVector("online", 1 / 3, 1.2, 0, 10.0, 30.0, 10**9, 0, 30.0, 900.0)))
    s.submit(job("cpu-job", "batch", gpu="none", cpu=2.0, write=15.0))
    ds = s.tick(0.0)
    assert admitted(ds) == ["eval-00"]
    assert by_id(ds)["eval-01"].blocked_by == ("write",) and by_id(ds)["eval-01"].reasons == (686,)
    assert by_id(ds)["cpu-job"].blocked_by == ("eval_priority",)   # 15 MB/s 放得下，但须留给排队中的评测
    s.finish("eval-00")
    ds = s.tick(1.0)
    assert admitted(ds) == ["eval-01", "cpu-job"]              # 30 + 15 ≤ 50


# ── 3. d3.slice 配额与容器参数 ────────────────────────────────────────────────────────────────────────────────────────

def test_other_batch_jobs_limited_to_8_cores_while_eval_queue_nonempty():
    s = Scheduler([EXCL3])
    s.submit(ev(0))
    for i in range(4):
        s.submit(job(f"livo-{i}", "batch", gpu="none", cpu=3.0, mem=16.0, kind="slam_eval", t=i))
    ds = s.tick(0.0)
    assert admitted(ds) == ["eval-00", "livo-0", "livo-1"]     # 3 + 3 ≤ 8 < 9
    assert by_id(ds)["livo-2"].blocked_by == ("cpu",)
    other = sum(r.job.cost.cpu_cores for r in s.running.values() if not r.job.is_eval)
    assert other <= 8
    s.finish("eval-00")
    ds = s.tick(1.0)                                           # 评测队列空：受 40 核总量约束
    assert admitted(ds) == ["livo-2", "livo-3"]


def test_fast_livo2_parallelism_limited_to_4_while_eval_active():
    s = Scheduler([EXCL3], config=SchedConfig(other_cpu_when_eval=100.0))
    s.submit(ev(0))
    for i in range(6):
        s.submit(job(f"livo-{i}", "batch", gpu="none", cpu=1.0, mem=8.0, kind="slam_eval", t=i))
    ds = s.tick(0.0)
    assert len([d for d in ds if d.admitted and d.job_id.startswith("livo")]) == 4
    assert by_id(ds)["livo-4"].blocked_by == ("kind_limit",)


def test_total_slice_cpu_40_when_eval_idle():
    s = Scheduler([EXCL3])
    for i in range(6):
        s.submit(job(f"b-{i}", "batch", gpu="none", cpu=7.0, mem=8.0, t=i))
    ds = s.tick(0.0)
    assert len(admitted(ds)) == 5 and by_id(ds)["b-5"].blocked_by == ("cpu",)   # 35 + 7 > 40


def test_slice_quota_and_container_args():
    assert slice_quota(True) == {"d3.slice": 40.0, "d3-eval.slice": 30.0, "other_jobs": 8.0}
    assert slice_quota(False)["other_jobs"] == 40.0
    s = Scheduler([EXCL3])
    e = ev(0)
    s.submit(e)
    d = s.tick(0.0)[0]
    args = resource_args(e, d)
    assert args[:6] == ["--cpus", "10", "--memory", "30g", "--cgroup-parent", "d3.slice/d3-eval.slice"]
    assert args.count("--gpus") == 1 and "device=3" in args and "NVIDIA_DRIVER_CAPABILITIES=all" in args
    cpu_job = job("x", "batch", gpu="none", cpu=1.5, mem=2.5, kind="slam_eval")
    s.submit(cpu_job)
    d2 = by_id(s.tick(1.0))["x"]
    a2 = resource_args(cpu_job, d2)
    assert "--gpus" not in a2 and a2[:6] == ["--cpus", "1.5", "--memory", "3g", "--cgroup-parent", "d3.slice/d3-livo.slice"]
    assert cgroup_parent(job("y", kind="export")) == "d3.slice/d3-jobs.slice"
    assert cgroup_parent(job("z")) == "d3.slice/d3-run.slice"
    with pytest.raises(ValueError):
        resource_args(cpu_job, type(d2)("x", False))


# ── 4. 写带宽与存储准入 ───────────────────────────────────────────────────────────────────────────────────────────────

def test_write_budget_formula():
    assert write_budget_mbps(400.0) == 200.0
    assert write_budget_mbps(800.0) == 300.0
    assert write_budget_mbps(None) == 300.0
    assert SchedConfig().write_budget_source == "cap_only" and SchedConfig(measured_disk_mbps=1.0).write_budget_source == "measured"
    with pytest.raises(ValueError):
        write_budget_mbps(0.0)


def test_write_rate_admission_queues_and_records_686():
    s = Scheduler([EXCL3], config=SchedConfig(measured_disk_mbps=400.0))   # 预算 200 MB/s
    for i in range(3):
        s.submit(job(f"w-{i}", "batch", gpu="none", cpu=1.0, write=80.0, t=i))
    ds = s.tick(0.0)
    assert admitted(ds) == ["w-0", "w-1"]
    d = by_id(ds)["w-2"]
    assert d.blocked_by == ("write",) and d.reasons == (686,)
    assert [(e[1], e[2]) for e in s.events] == [("w-2", 686)]
    s.tick(1.0)
    assert len(s.events) == 1                                  # 阻塞维度不变时不重复记录
    assert sum(r.job.cost.write_mbps for r in s.running.values()) <= 200.0
    s.finish("w-0")
    assert admitted(s.tick(2.0)) == ["w-2"]


def test_measured_io_over_budget_pauses_new_admission():
    s = Scheduler([EXCL3], config=SchedConfig(measured_disk_mbps=400.0))
    s.submit(job("w", "batch", gpu="none", cpu=1.0, write=10.0))
    s.submit(job("ro", "batch", gpu="none", cpu=1.0, write=0.0, persist=0, t=1.0))
    ds = s.tick(0.0, Telemetry(write_mbps_1min=221.0))        # > 1.1 × 200
    assert by_id(ds)["w"].blocked_by == ("write_paused",) and by_id(ds)["w"].reasons == (686,)
    assert by_id(ds)["ro"].admitted                            # 不写盘的作业不受影响
    assert admitted(s.tick(1.0, Telemetry(write_mbps_1min=219.0))) == ["w"]


def test_storage_over_95_percent_queues_and_records_685():
    cfg = SchedConfig(storage_quota_bytes=1000 * 10**9)
    s = Scheduler([EXCL3], config=cfg)
    s.submit(job("big", "batch", gpu="none", persist=20 * 10**9))
    ds = s.tick(0.0, Telemetry(store_used_bytes=940 * 10**9))
    assert ds[0].blocked_by == ("storage",) and ds[0].reasons == (685,)
    assert admitted(s.tick(1.0, Telemetry(store_used_bytes=900 * 10**9))) == ["big"]


def test_running_jobs_remaining_bytes_count_against_storage():
    cfg = SchedConfig(storage_quota_bytes=100 * 10**9)
    s = Scheduler([EXCL3], config=cfg)
    s.submit(job("a", "batch", gpu="none", persist=60 * 10**9, wall=100.0))
    s.submit(job("b", "batch", gpu="none", persist=60 * 10**9, wall=100.0, t=1.0))
    ds = s.tick(0.0)
    assert admitted(ds) == ["a"] and by_id(ds)["b"].blocked_by == ("storage",)
    assert by_id(s.tick(50.0, Telemetry(store_used_bytes=30 * 10**9)))["b"].blocked_by == ("storage",)   # 30 + 30 + 60 > 95
    assert admitted(s.tick(100.0, Telemetry(store_used_bytes=30 * 10**9))) == ["b"]                     # a 已写完：30 + 60 ≤ 95


# ── 5. 成本估算 ───────────────────────────────────────────────────────────────────────────────────────────────────────

def test_standard_track_cost_matches_design_budget():
    """AWR-05 §4.7、§5.6.4：标准轨正式 Run 10 核、约 30 GB、约 0.85 GPU·s/仿真 s，RTF 1.0 时约 41 MB/s，完整录制 13–17 GB。"""
    c = estimate_cost(rs(), priority_class="eval")
    assert c.cpu_cores == 10.0 and c.mem_gb == pytest.approx(30.0)
    assert c.gpu_s_per_sim_s == pytest.approx(0.85, abs=0.03)
    assert 13e9 <= c.bytes_persist <= 17e9
    assert c.bytes_persist / 360 / 1e6 == pytest.approx(41.0, rel=0.10)
    assert c.gpu_share == pytest.approx(1 / 3) and c.vram_gb <= 1.2 + 1e-9
    assert 2.0 <= c.wall_s_est / 360 <= 3.0                    # 1/3 卡时传感器实时因子约 0.39
    three = 3 * c.write_mbps
    assert 40.0 <= three <= 60.0                               # 3 个并发正式 Run 约 50–60 MB/s
    r = estimate_cost(rs(ranked=False), priority_class="research")
    assert r.gpu_share == pytest.approx(0.6) and r.write_mbps > c.write_mbps


def test_cost_channels_and_levels():
    ch = channels_of(rs(2, therm=False))
    assert [(c.uav, c.kind) for c in ch] == [(u, k) for u in ("uav-01", "uav-02") for k in ("camera", "lidar", "aux", "aux")]
    assert ch[0].width == 1920 and ch[0].fps == 10.0
    spec = rs(1, therm=False)
    spec["fleet"][0]["sensors"] = [{"sensor_id": "eo", "overrides": {"width_px": 1280, "height_px": 720, "fps": "15"}},
                                   {"sensor_id": "lidar0", "enabled": False}]
    ch = channels_of(spec)
    assert [c.kind for c in ch] == ["camera", "aux", "aux"] and (ch[0].width, ch[0].fps) == (1280, 15.0)
    assert [c.kind for c in channels_of({**spec, "obs_level": "L2-depth"})] == ["camera", "depth", "aux", "aux"]
    depth = estimate_cost({**spec, "obs_level": "L2-depth"}, priority_class="research")
    plain = estimate_cost(spec, priority_class="research")
    assert depth.bytes_persist > 5 * plain.bytes_persist      # L2-depth 深度 f32 主导体积
    l1 = estimate_cost(rs(3, level="L1", physics="fast"))
    assert l1.gpu_class == "none" and l1.bytes_persist < 2_000_000 and l1.wall_s_est < 30
    assert estimate_cost(rs(3, level="L1"), priority_class="research").gpu_class == "none"
    bag = estimate_cost({**rs(1), "outputs": ["bag", "rrd"]}, priority_class="research")
    assert bag.bytes_derived == pytest.approx((430 + 560) * 1e6 * 6, rel=1e-6)
    assert estimate_cost(rs(1), priority_class="research").gz_instances == 0                  # realistic: unified L2 engine, no gz instance (ADR-226)
    xval = estimate_cost({**rs(1), "physics": "gz", "physics_opts": {"gz_profile": "xval"}}, priority_class="research")
    assert xval.gz_instances == 1 and xval.vram_gb == pytest.approx(0.4 + 2.5) and xval.cpu_cores == 2 + 4 + 2
    custom = estimate_cost(rs(1), channels=[Channel("uav-01", "eo", "camera", 1280, 720, 10.0)], priority_class="research")
    assert custom.bytes_persist < plain.bytes_persist * 2


def test_make_job_routes_ranked_runs_to_eval_class():
    j = make_job("run-aaaaaaaaaa", rs(), tenant="p1", submitted=0.0)
    assert j.priority_class == "eval" and j.cost.gpu_share == pytest.approx(1 / 3)
    assert make_job("run-bbbbbbbbbb", rs(ranked=False), tenant="p1", submitted=0.0).priority_class == "research"
    with pytest.raises(ValueError):
        Job("x", "t", "public", EVAL_COST, 0.0)


def test_end_to_end_three_standard_eval_runs_within_budgets():
    s = Scheduler([EXCL3], config=SchedConfig(measured_disk_mbps=600.0))
    for i in range(4):
        s.submit(make_job(f"run-e{i:09d}", rs(), tenant="sub-1", submitted=float(i)))
    ds = s.tick(10.0)
    assert len(admitted(ds)) == 3
    st = s.status(10.0)
    assert st["gpus"][3]["eval_runs"] == 3 and st["gpus"][3]["share_used"] <= 1.0
    assert st["cpu"]["eval"] == 30.0 and st["cpu"]["quota"]["other_jobs"] == 8.0
    assert st["write"]["est_mbps"] <= st["write"]["budget_mbps"] == 300.0 and st["write"]["budget_source"] == "measured"


# ── 6. GPU 日历、轮转与实测修正 ───────────────────────────────────────────────────────────────────────────────────────

def test_acceptance_window_exclusive_lock():
    win = Window("acceptance", 100.0, 200.0, (3,))
    s = Scheduler([EXCL3, DEV2], windows=[win])
    s.submit(job("acc", "research", share=0.5, acceptance=True))
    s.submit(job("dev", "research", share=0.2, t=1.0))
    s.submit(ev(0, t=2.0))
    ds = s.tick(50.0)
    assert by_id(ds)["acc"].blocked_by == ("calendar",)        # 窗口外不准入验收作业
    assert by_id(ds)["eval-00"].admitted and by_id(ds)["dev"].gpu == 2
    s.finish("eval-00")
    s.submit(ev(1, t=3.0))
    ds = s.tick(150.0)
    a = by_id(ds)["acc"]
    assert a.admitted and a.gpu == 3 and a.share == 1.0
    assert by_id(ds)["eval-01"].blocked_by == ("gpu",)         # 验收窗口内评测暂停
    s.finish("acc")
    s.submit(job("dev2", "research", share=0.2, t=4.0))
    assert by_id(s.tick(160.0))["dev2"].gpu == 2               # 窗口内独占卡不给其他作业
    assert by_id(s.tick(210.0))["eval-01"].gpu == 3


def test_research_runs_never_bind_exclusive_card():
    """§4.7 独占期：开发容器不得绑定独占卡；批量生产与预览可在独占卡的生产窗口内运行。"""
    s = Scheduler([EXCL3, DEV2])
    s.submit(job("res", "research", share=0.3))
    s.submit(job("res-2", "research", share=0.5, t=1.0))
    s.submit(job("bat", "batch", share=0.3, t=2.0))
    ds = by_id(s.tick(0.0))
    assert ds["res"].gpu == 2 and ds["res-2"].blocked_by == ("gpu",) and ds["bat"].gpu == 3


def test_unadmittable_eval_does_not_hold_reservation():
    s = Scheduler([EXCL3], config=SchedConfig(measured_disk_mbps=100.0))
    s.submit(ev(0, cost=CostVector("online", 1 / 3, 1.2, 0, 10.0, 30.0, 10**9, 0, 80.0, 900.0)))   # 80 > 预算 50
    s.submit(job("bat", "batch", share=0.3, write=10.0, t=1.0))
    ds = by_id(s.tick(0.0))
    assert ds["eval-00"].blocked_by == ("write",) and ds["bat"].admitted


def test_acceptance_waits_for_card_to_drain():
    s = Scheduler([EXCL3])
    s.submit(job("prod", "batch", share=0.2))
    s.tick(0.0)
    s.windows = (Window("acceptance", 10.0, 1e9),)
    s.submit(job("acc", acceptance=True, t=1.0))
    s.submit(job("late", "batch", share=0.1, t=2.0))
    ds = s.tick(20.0)
    assert by_id(ds)["acc"].blocked_by == ("gpu",) and by_id(ds)["late"].blocked_by == ("gpu",)
    s.finish("prod")
    assert admitted(s.tick(30.0)) == ["acc"]


def test_training_window_halves_production_and_pauses_preview():
    s = Scheduler([EXCL3], windows=[Window("training", 0.0, 100.0)])
    s.submit(job("on", "batch", share=0.35))
    s.submit(job("pv", "preview", gpu="preview", share=0.05, t=1.0))
    ds = s.tick(10.0)
    assert by_id(ds)["on"].blocked_by == ("gpu",) and by_id(ds)["pv"].blocked_by == ("gpu",)
    s.submit(ev(0, t=2.0))
    assert by_id(s.tick(11.0))["eval-00"].admitted            # 评测不受训练窗口影响
    s.finish("eval-00")
    assert sorted(admitted(s.tick(120.0))) == ["on", "pv"]


def test_tenant_round_robin_within_class():
    s = Scheduler([EXCL3])
    for i in range(4):
        s.submit(job(f"a-{i}", tenant="A", gpu="none", cpu=4.0, t=float(i)))
    s.submit(job("b-0", tenant="B", gpu="none", cpu=4.0, t=10.0))
    s.submit(job("c-0", tenant="C", gpu="none", cpu=4.0, t=11.0))
    assert [j.job_id for j in s.ordered_queue()][:4] == ["a-0", "b-0", "c-0", "a-1"]
    s.cfg = SchedConfig(slice_cpu=8.0)
    assert admitted(s.tick(0.0)) == ["a-0", "b-0"]
    s.cfg = SchedConfig(slice_cpu=12.0)
    assert admitted(s.tick(1.0)) == ["c-0"]                    # 指针在 B 之后


def test_measured_usage_overrides_estimate():
    s = Scheduler([EXCL3])
    s.submit(job("cpu", "batch", gpu="none", cpu=4.0))
    assert by_id(s.tick(0.0, Telemetry(cpu_cores=38.0)))["cpu"].blocked_by == ("cpu",)
    s.submit(job("vram", "batch", share=0.1, vram=4.0, t=1.0))
    ds = s.tick(1.0, Telemetry(vram_gb={3: 22.0}))
    assert by_id(ds)["vram"].blocked_by == ("gpu",) and by_id(ds)["cpu"].admitted


def test_tick_scales_to_10000_queued_jobs():
    """M27-NFR-001 口径：10,000 个排队作业下单次调度循环（本机 E5-2603 v4 宽松上限 2 s）。"""
    s = Scheduler([EXCL2, EXCL3])
    for i in range(10_000):
        s.submit(job(f"q-{i:05d}", ("research", "batch")[i % 2], tenant=f"t{i % 50}", t=float(i), gpu="none", cpu=1.0, write=0.5))
    for i in range(20):
        s.submit(ev(i, t=float(i)))
    t0 = time.perf_counter()
    ds = s.tick(0.0)
    dt = time.perf_counter() - t0
    assert len(ds) == 10_020 and dt < 2.0
    assert sum(1 for r in s.running.values() if r.job.is_eval) == 3
