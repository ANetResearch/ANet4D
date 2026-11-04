"""Realistic 档机型映射（`awr.phys.gz.vehicles`；M24-FR-083；AWR-20 §15.16.1；F-09）：`p600_bench` 的质量与最大转速与目录同源，
机型引用解析、混合机队拒绝，以及 Env 的 `_GzTier` 把派生模型写入运行目录、只读挂载进 gz 实例并改写 run.sdf（以替身代替 gz 实例）。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from awr.phys.gz import vehicles as V

ROOT = Path(__file__).resolve().parents[3]


def test_p600_bench_matches_catalog_variant_and_base_motor() -> None:
    import yaml

    from awr.sim.catalog.loader import Catalog

    v = Catalog.load(ROOT / "vehicles", d3=True).overlay.variant("p600_bench")
    q = V.p600_bench_params()
    assert q.mass_kg == pytest.approx(v.derived["mass_kg"], abs=1e-9)                     # 变体派生质量（AWR-20 §15.16.1）
    base = yaml.safe_load((ROOT / "vehicles" / "p600" / "params.yaml").read_text(encoding="utf-8"))
    omega = base["motor"]["omega_max_rad_s"]
    assert q.omega_max_rad_s == float(omega["value"] if isinstance(omega, dict) else omega)
    assert q.thrust_to_weight == pytest.approx(v.derived["twr"], abs=0.01)                 # 推重比 1.99（VH-1）
    s = (q.mass_kg / V.X500_SIH["mass_kg"]) * (V.P600_WHEELBASE_M / V.X500_WHEELBASE_M) ** 2
    assert q.inertia_kgm2 == pytest.approx(tuple(x * s for x in V.X500_SIH["inertia_kgm2"]), abs=1e-6)


def test_quad_for_resolves_refs_and_rejects_mixed_fleets() -> None:
    assert V.quad_for("p600_bench").name == V.quad_for("p600_bench@1.0.0").name == "p600_bench"
    assert V.quad_for("x500").model == V.X500_BASE and not V.quad_for("x500").mapped
    assert not V.quad_for("p600_mid360_inv").mapped                                       # 未登记映射：x500 原型（接入前的行为）
    assert V.quad_for_fleet(["p600_bench"] * 3).name == "p600_bench"
    with pytest.raises(ValueError, match="one vehicle mapping"):
        V.quad_for_fleet(["p600_bench", "x500"])


class _Inst:
    made: ClassVar[list[_Inst]] = []

    def __init__(self, run_dir, name, *, image, launcher, world_export, extra_mounts) -> None:
        self.run_dir, self.name, self.launcher, self.mounts = Path(run_dir), name, launcher, dict(extra_mounts)
        self.gz_endpoint, self.sdf, self.stopped = "ipc:///run/d4d/ctl.sock", None, False
        _Inst.made.append(self)

    def start(self, sdf: str) -> None:
        self.sdf = sdf

    def wait_socket(self, timeout_s: float) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True


class _Backend:
    def __init__(self, inst, uavs, **kw) -> None:
        self.name = "gz"

    def close(self) -> dict:
        return {"pose_sha256": "x"}


@pytest.mark.parametrize("ref", ["p600_bench", "x500"])
def test_gz_tier_writes_mounts_and_rewrites_run_sdf(ref: str, tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("google.protobuf")
    import awr.phys.gz.backend as B
    import awr.phys.gz.instance as I
    import awr.world.export.gz as X
    from awr.agentsdk.env.core import _GzTier

    monkeypatch.setattr(I, "GzInstance", _Inst)
    monkeypatch.setattr(B, "SupervisedGzBackend", _Backend)
    monkeypatch.setattr(X, "export_world", lambda *a, **k: SimpleNamespace(root=tmp_path / "exp", uri="model://world_static", spherical=None))
    fleet = [SimpleNamespace(uav=f"uav-0{i}", vehicle_ref=ref) for i in (1, 2, 3)]
    spec = SimpleNamespace(gz={"launcher": "docker"}, world_dir=str(tmp_path / "w"), out_dir=str(tmp_path / "run"), run_id="r1", fleet=fleet)
    spawn = {f.uav: ((10.0 * i, 0.0, 30.0), 0.0) for i, f in enumerate(fleet)}
    _Inst.made.clear()
    h = _GzTier.create(spec, spawn, [], world=None)
    inst = _Inst.made[-1]
    info = h.close()
    assert inst.stopped and info["pose_sha256"] == "x" and info["vehicle"]["name"] == ref
    if ref == "x500":
        assert inst.mounts == {} and "model://x500_d4d</uri>" in inst.sdf and not (tmp_path / "run" / "phys" / "models").exists()
        return
    md = tmp_path / "run" / "phys" / "models" / "x500_d4d_p600_bench"
    assert inst.mounts == {str(md): "/opt/d4d/models/x500_d4d_p600_bench"}
    assert (md / "model.sdf").read_text(encoding="utf-8").count("<model name='x500_d4d_p600_bench'>") == 1
    assert inst.sdf.count("<uri>model://x500_d4d_p600_bench</uri>") == 3 and "model://x500_d4d</uri>" not in inst.sdf
    assert inst.sdf.count("<maxRotVelocity>1500.0</maxRotVelocity>") == 12
