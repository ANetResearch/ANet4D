"""Ground vehicle packages (``awr.phys.l2.ground``): ``vehicles/ground/catalog.json`` + ``vehicles/ground/<dir>/params.yaml``.

The land counterpart of the air catalog (``awr.sim.fleet.profiles``), kept as its own family (``awr.vehicle.ground.v1``,
``awr.device_catalog.ground.v1``) so the sandbox's nest, mission and UI machinery, which is written for multirotors and
fixed wings, is untouched: nothing but the unified L2 engine (``awr.phys.l2``) and the device ladder (``awr.sim.device``)
reads this directory. Every physical value carries ``value / conf / src`` like the air profiles; documents are validated
against ``packages/contracts/vehicle/ground_profile.schema.json`` when they are loaded.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from awr.phys import PhysError

__all__ = ["GroundTable", "UgvParams", "ground_dir", "load_ground_docs", "read_ground_catalog", "schema_errors"]

SCHEMA = "awr.vehicle.ground.v1"
CATALOG_SCHEMA = "awr.device_catalog.ground.v1"
G = 9.80665


def ground_dir() -> Path:
    from awr.sim.fleet.profiles import vehicles_dir

    return Path(os.environ.get("AWR_GROUND_VEHICLES_DIR") or (vehicles_dir() / "ground"))


def read_ground_catalog(gdir: Path | None = None) -> dict | None:
    f = (gdir or ground_dir()) / "catalog.json"
    if not f.exists():
        return None
    cat = json.loads(f.read_text(encoding="utf-8"))
    if not isinstance(cat, dict) or cat.get("schema") != CATALOG_SCHEMA or not isinstance(cat.get("models"), list):
        raise PhysError(630, f"{f}: not an {CATALOG_SCHEMA} catalog")
    return cat


@cache
def _validator():
    from jsonschema import Draft202012Validator

    from awr.contracts._paths import contracts_root
    from awr.sim.schemas import contracts_registry

    schema = json.loads((contracts_root() / "vehicle" / "ground_profile.schema.json").read_text(encoding="utf-8"))
    return Draft202012Validator(schema, registry=contracts_registry())


def schema_errors(doc: dict) -> list[str]:
    return [f"schema {'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message[:160]}" for e in _validator().iter_errors(doc)]


def load_ground_docs(gdir: Path | None = None, *, validate: bool = True) -> dict[str, tuple[dict, Path]]:
    """{profile_id: (document, path)} for the packages the catalog lists; an empty dict when there is no catalog."""
    from awr.sim.fleet.profiles import yaml_load

    gdir = Path(gdir) if gdir else ground_dir()
    cat = read_ground_catalog(gdir)
    if cat is None:
        return {}
    out: dict[str, tuple[dict, Path]] = {}
    for m in cat["models"]:
        f = gdir / str(m.get("dir") or m["id"]) / "params.yaml"
        if not f.exists():
            raise PhysError(630, f"ground catalog lists {m.get('id')!r} but {f} is missing")
        doc = yaml_load(f.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            raise PhysError(630, f"{f}: not an {SCHEMA} document")
        if validate:
            errs = schema_errors(doc)
            if errs:
                raise PhysError(630, f"{f}: {'; '.join(errs[:5])}", {"errors": errs})
        pid = str(doc["id"])
        if pid != m["id"] or pid in out:
            raise PhysError(630, f"{f}: id {pid!r} does not match the catalog or repeats")
        out[pid] = (doc, f)
    return out


def _val(d):
    return d.get("value") if isinstance(d, dict) and "value" in d else d


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


class GroundTable:
    """The ground vehicle packages by id (``ref`` may carry ``@version``, which is ignored like the air table does)."""

    def __init__(self, gdir: Path | None = None) -> None:
        self.docs = load_ground_docs(gdir)
        self.params = {pid: UgvParams.from_doc(doc) for pid, (doc, _p) in self.docs.items()}

    @property
    def ids(self) -> list[str]:
        return sorted(self.params)

    @staticmethod
    def key(ref: str) -> str:
        return str(ref).split("@", 1)[0]

    def has(self, ref: str) -> bool:
        return self.key(ref) in self.params

    def get(self, ref: str) -> UgvParams:
        k = self.key(ref)
        if k not in self.params:
            raise PhysError(630, f"vehicle {ref!r} has no ground vehicle package")
        return self.params[k]
