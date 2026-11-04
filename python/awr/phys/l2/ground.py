"""Vehicle package families of the unified L2 engine beyond the air catalog (``awr.phys.l2.ground``): ground vehicles
(``vehicles/ground``, ``awr.vehicle.ground.v1``) and surface vessels (``vehicles/usv``, ``awr.vehicle.usv.v1``).

Each family is its own catalog + ``<dir>/params.yaml`` tree, kept apart from the air catalog (``awr.sim.fleet.profiles``)
so the sandbox's nest, mission and UI machinery, which is written for multirotors and fixed wings, is untouched: nothing
but the unified L2 engine (``awr.phys.l2``) and the device ladder (``awr.sim.device``) reads these directories. Every
physical value carries ``value / conf / src`` like the air profiles; documents are validated against their schema in
``packages/contracts/vehicle/`` when they are loaded.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from awr.phys import PhysError

__all__ = ["FAMILIES", "GROUND", "USV", "Family", "GroundTable", "PackageTable", "UgvParams", "UsvParams", "UsvTable", "ground_dir",
           "load_family_docs", "load_ground_docs", "read_family_catalog", "read_ground_catalog", "schema_errors"]

G = 9.80665


@dataclass(frozen=True)
class Family:
    name: str                      # ladder domain name: ugv, usv
    dir_name: str                  # vehicles/<dir_name>
    env_var: str                   # overrides the directory
    schema: str                    # params.yaml `schema`
    catalog_schema: str            # catalog.json `schema`
    schema_file: str               # packages/contracts/vehicle/<file>
    kind: str                      # params.yaml `kind`

    def dir(self) -> Path:
        from awr.sim.fleet.profiles import vehicles_dir

        return Path(os.environ.get(self.env_var) or (vehicles_dir() / self.dir_name))


GROUND = Family("ugv", "ground", "AWR_GROUND_VEHICLES_DIR", "awr.vehicle.ground.v1", "awr.device_catalog.ground.v1", "ground_profile.schema.json", "ugv")
USV = Family("usv", "usv", "AWR_USV_VEHICLES_DIR", "awr.vehicle.usv.v1", "awr.device_catalog.usv.v1", "usv_profile.schema.json", "usv")
FAMILIES = {GROUND.name: GROUND, USV.name: USV}


def read_family_catalog(family: Family, fdir: Path | None = None) -> dict | None:
    f = (fdir or family.dir()) / "catalog.json"
    if not f.exists():
        return None
    cat = json.loads(f.read_text(encoding="utf-8"))
    if not isinstance(cat, dict) or cat.get("schema") != family.catalog_schema or not isinstance(cat.get("models"), list):
        raise PhysError(630, f"{f}: not an {family.catalog_schema} catalog")
    return cat


@cache
def _validator(schema_file: str):
    from jsonschema import Draft202012Validator

    from awr.contracts._paths import contracts_root
    from awr.sim.schemas import contracts_registry

    schema = json.loads((contracts_root() / "vehicle" / schema_file).read_text(encoding="utf-8"))
    return Draft202012Validator(schema, registry=contracts_registry())


def schema_errors(doc: dict, family: Family = GROUND) -> list[str]:
    return [f"schema {'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message[:160]}" for e in _validator(family.schema_file).iter_errors(doc)]


def load_family_docs(family: Family, fdir: Path | None = None, *, validate: bool = True) -> dict[str, tuple[dict, Path]]:
    """{profile_id: (document, path)} for the packages the family's catalog lists; an empty dict when there is no catalog."""
    from awr.sim.fleet.profiles import yaml_load

    fdir = Path(fdir) if fdir else family.dir()
    cat = read_family_catalog(family, fdir)
    if cat is None:
        return {}
    out: dict[str, tuple[dict, Path]] = {}
    for m in cat["models"]:
        f = fdir / str(m.get("dir") or m["id"]) / "params.yaml"
        if not f.exists():
            raise PhysError(630, f"{family.name} catalog lists {m.get('id')!r} but {f} is missing")
        doc = yaml_load(f.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or doc.get("schema") != family.schema:
            raise PhysError(630, f"{f}: not an {family.schema} document")
        if validate:
            errs = schema_errors(doc, family)
            if errs:
                raise PhysError(630, f"{f}: {'; '.join(errs[:5])}", {"errors": errs})
        pid = str(doc["id"])
        if pid != m["id"] or pid in out:
            raise PhysError(630, f"{f}: id {pid!r} does not match the catalog or repeats")
        out[pid] = (doc, f)
    return out


def _val(d):
    return d.get("value") if isinstance(d, dict) and "value" in d else d


class PackageTable:
    """The packages of one family by id (``ref`` may carry ``@version``, which is ignored like the air table does)."""

    family: Family = GROUND

    def __init__(self, fdir: Path | None = None) -> None:
        self.docs = load_family_docs(self.family, fdir)
        self.params: dict[str, Any] = {pid: self.params_from_doc(doc) for pid, (doc, _p) in self.docs.items()}

    @staticmethod
    def params_from_doc(doc: dict):
        return doc

    @property
    def ids(self) -> list[str]:
        return sorted(self.params)

    @staticmethod
    def key(ref: str) -> str:
        return str(ref).split("@", 1)[0]

    def has(self, ref: str) -> bool:
        return self.key(ref) in self.params

    def get(self, ref: str):
        k = self.key(ref)
        if k not in self.params:
            raise PhysError(630, f"vehicle {ref!r} has no {self.family.name} package")
        return self.params[k]


# ------------------------------------------------------------------------------------------------------ ground vehicles
def ground_dir() -> Path:
    return GROUND.dir()


def read_ground_catalog(gdir: Path | None = None) -> dict | None:
    return read_family_catalog(GROUND, gdir)


def load_ground_docs(gdir: Path | None = None, *, validate: bool = True) -> dict[str, tuple[dict, Path]]:
    return load_family_docs(GROUND, gdir, validate=validate)


@dataclass(frozen=True)
class UgvParams:
    """Physical and actuation parameters of one ground vehicle (the single-track model's parameter set plus resistances)."""
    id: str
    mass_kg: float
    izz: float
    lf: float
    lr: float
    track_m: float
    cg_h: float
    wheel_r: float
    collision_radius_m: float
    mu: float
    c_sf: float
    c_sr: float
    steer_max: float
    steer_rate_max: float
    v_max: float
    v_min: float
    a_max: float
    a_brake: float
    v_switch: float
    k_speed: float
    c_rr: float
    cda: float
    drive: str

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr

    @classmethod
    def from_doc(cls, doc: dict) -> UgvParams:
        g, t, a = doc["geometry"], doc["tire"], doc["actuation"]
        res = doc.get("resistance") or {}
        return cls(id=str(doc["id"]), mass_kg=float(_val(doc["mass_kg"])), izz=float(_val(doc["inertia_zz_kgm2"])),
                   lf=float(_val(g["lf_m"])), lr=float(_val(g["lr_m"])), track_m=float(_val(g["track_m"])), cg_h=float(_val(g["cg_height_m"])),
                   wheel_r=float(_val(g["wheel_radius_m"])), collision_radius_m=float(_val(g["collision_radius_m"])),
                   mu=float(_val(t["mu"])), c_sf=float(_val(t["c_sf"])), c_sr=float(_val(t["c_sr"])),
                   steer_max=float(_val(a["steer_max_rad"])), steer_rate_max=float(_val(a["steer_rate_max_rad_s"])),
                   v_max=float(_val(a["v_max_mps"])), v_min=float(_val(a["v_min_mps"])), a_max=float(_val(a["a_max_mps2"])),
                   a_brake=float(_val(a["a_brake_mps2"])), v_switch=float(_val(a["v_switch_mps"])),
                   k_speed=float(_val(a.get("k_speed_hz", 2.0))), c_rr=float(_val(res.get("c_rr", 0.0))), cda=float(_val(res.get("cda_m2", 0.0))),
                   drive=str(doc["drive"]["type"]))


class GroundTable(PackageTable):
    """The ground vehicle packages by id."""

    family = GROUND

    @staticmethod
    def params_from_doc(doc: dict) -> UgvParams:
        return UgvParams.from_doc(doc)


# ------------------------------------------------------------------------------------------------------ surface vessels
@dataclass(frozen=True)
class UsvParams:
    """Physical and actuation parameters of one surface vessel: the Fossen 3-DoF manoeuvring model (surge, sway, yaw) with twin
    fixed thrusters, plus what the tier needs for buoyancy, grounding and collision."""
    id: str
    mass_kg: float
    izz: float
    length_m: float
    beam_m: float
    draft_m: float
    hull_beam_m: float
    x_g: float
    cg_above_water_m: float
    collision_radius_m: float
    a_u: float                                                # added mass, surge (−X_udot)
    a_v: float                                                # added mass, sway (−Y_vdot)
    a_r: float                                                # added inertia, yaw (−N_rdot)
    d_u: float                                                # linear damping (−X_u, −Y_v, −N_r)
    d_v: float
    d_r: float
    d_uu: float                                               # quadratic damping (−X_|u|u, −Y_|v|v, −N_|r|r)
    d_vv: float
    d_rr: float
    cd_crossflow: float                                       # 2-D cross-flow drag coefficient (0: none)
    rho_water: float
    cda_wind: float
    thr_x: float                                              # thruster position (x, ±y) in the body frame
    thr_y: float
    k_pos: float                                              # thrust = k_pos·n|n| (n > 0), k_neg·n|n| (n < 0)
    k_neg: float
    n_max: float
    n_min: float
    tau_n: float                                              # thruster first-order lag, s
    v_max: float
    a_max: float
    r_max: float
    k_speed: float
    k_yaw: float
    propulsion: str

    @property
    def m_surge(self) -> float:
        return self.mass_kg + self.a_u

    @property
    def thrust_max(self) -> float:
        """Full forward thrust of both thrusters, N."""
        return 2.0 * self.k_pos * self.n_max * abs(self.n_max)

    @classmethod
    def from_doc(cls, doc: dict) -> UsvParams:
        g, h, t = doc["geometry"], doc["hydro"], doc["thrusters"]
        return cls(id=str(doc["id"]), mass_kg=float(_val(doc["mass_kg"])), izz=float(_val(doc["inertia_zz_kgm2"])),
                   length_m=float(_val(g["length_m"])), beam_m=float(_val(g["beam_m"])), draft_m=float(_val(g["draft_m"])),
                   hull_beam_m=float(_val(g["hull_beam_m"])), x_g=float(_val(g["x_g_m"])), cg_above_water_m=float(_val(g["cg_above_water_m"])),
                   collision_radius_m=float(_val(g["collision_radius_m"])),
                   a_u=float(_val(h["a_u_kg"])), a_v=float(_val(h["a_v_kg"])), a_r=float(_val(h["a_r_kgm2"])),
                   d_u=float(_val(h["d_u"])), d_v=float(_val(h["d_v"])), d_r=float(_val(h["d_r"])),
                   d_uu=float(_val(h["d_uu"])), d_vv=float(_val(h["d_vv"])), d_rr=float(_val(h["d_rr"])),
                   cd_crossflow=float(_val(h.get("cd_crossflow", 0.0))), rho_water=float(_val(h.get("rho_water_kgm3", 1025.0))),
                   cda_wind=float(_val(h.get("cda_wind_m2", 0.0))),
                   thr_x=float(_val(t["x_m"])), thr_y=float(_val(t["y_m"])), k_pos=float(_val(t["k_pos"])), k_neg=float(_val(t["k_neg"])),
                   n_max=float(_val(t["n_max_rad_s"])), n_min=float(_val(t["n_min_rad_s"])), tau_n=float(_val(t["tau_s"])),
                   v_max=float(_val(t["v_max_mps"])), a_max=float(_val(t["a_max_mps2"])), r_max=float(_val(t["r_max_rad_s"])),
                   k_speed=float(_val(t.get("k_speed_hz", 1.0))), k_yaw=float(_val(t.get("k_yaw_hz", 2.0))),
                   propulsion=str(doc["propulsion"]["type"]))


class UsvTable(PackageTable):
    """The surface vessel packages by id."""

    family = USV

    @staticmethod
    def params_from_doc(doc: dict) -> UsvParams:
        return UsvParams.from_doc(doc)
