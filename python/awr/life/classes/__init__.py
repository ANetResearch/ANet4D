"""Entity class catalog ``awr.life.entity_classes.v1``: the first extension point of the target behavior engine.

An entity class says what a kind of target *is* to the engine: which tape class and navigation profile it records
under, which body it has (articulated walker, rigid vehicle, static street object) and from which assets, how tall and
how fast its members are, which role it plays for the activity director and which activity purposes it may take.
The engine admits and moves members of a class by these fields alone; a new class is a catalog entry, not code.

The built-in catalog (``default.json``) describes the three classes the engine always had: ``resident`` (walkers),
``local_traffic`` (parked cars) and ``street_object`` (furniture). Its spreads reproduce the legacy formulas exactly
(height 1.60 + 0.04 (j mod 6), walking speed 1.0 + 0.08 (j mod 4), car speed 3.0 + 0.2 (j mod 4)) so that recordings made
before the catalog existed are byte-identical after it.

A recipe adds or changes classes under ``classes`` (inline) or ``entity_classes`` (a catalog file), and asks for
members with ``population`` (``{"resident": 150, "runner": 8}``); the legacy ``people`` count means residents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from awr.life.common.schema import schema_id

SCHEMA = "awr.life.entity_classes.v1"
TAPE_CLASSES = ("person", "vehicle", "dog", "machine", "object", "uav")
PROFILES = ("person", "vehicle", None)
BODY_KINDS = ("articulated", "rigid", "static")
ASSET_FAMILIES = ("humans", "vehicles", "street")
DEFAULT_PATH = Path(__file__).with_name("default.json")


class EntityClassError(ValueError):
    pass


@dataclass(frozen=True)
class Spread:
    """``base + step * (j mod cycle)``: the deterministic spread of a per-member quantity over the member index."""

    base: float
    step: float = 0.0
    cycle: int = 1

    def at(self, j: int) -> float:
        return round(self.base + self.step * (j % self.cycle), 6)

    @classmethod
    def parse(cls, value, name: str) -> Spread:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return cls(float(value))
        if isinstance(value, dict) and set(value) <= {"base", "step", "cycle"} and "base" in value:
            s = cls(float(value["base"]), float(value.get("step", 0.0)), int(value.get("cycle", 1)))
            if s.cycle < 1 or s.base <= 0 or s.base + s.step * (s.cycle - 1) <= 0:
                raise EntityClassError(f"{name}: spread must stay positive")
            return s
        raise EntityClassError(f"{name}: a number or {{base, step, cycle}} required")


@dataclass(frozen=True)
class EntityClass:
    id: str
    tape_class: str
    profile: str | None
    body_kind: str
    assets: tuple[str, ...]
    role: str
    height_m: Spread | None = None
    max_speed_mps: Spread | None = None
    semantic_type: str = ""
    description: str = ""
    activities: tuple[str, ...] | None = None  # purposes this class may take; None: every purpose open to its role
    min_activity_kinds: int | None = None
    id_prefix: str = ""
    label_zh: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def moving(self) -> bool:
        return self.body_kind != "static"

    def asset(self, j: int) -> str:
        return self.assets[j % len(self.assets)]

    def height(self, j: int) -> float:
        if self.height_m is None:
            raise EntityClassError(
                f"{self.id}: no height spread (rigid and static bodies take their size from the asset)"
            )
        return self.height_m.at(j)

    def max_speed(self, j: int) -> float:
        return 0.0 if self.max_speed_mps is None else self.max_speed_mps.at(j)

    def actor_id(self, j: int) -> str:
        return f"{self.id_prefix or self.id}-{j:03d}"


def _assets(value, config: dict, name: str) -> tuple[str, ...]:
    """An explicit asset list, or a family resolved against the recipe (``human_assets`` count, ``vehicle_assets``)."""
    if value == "humans":
        return tuple(f"human-{i + 1}" for i in range(int(config.get("human_assets", 4))))
    if value == "vehicles":
        return tuple(config.get("vehicle_assets", ["sedan", "hatchback", "suv", "van", "pickup"]))
    if value == "street":
        return tuple(
            config.get("prop_assets", ["bench", "litter-bin", "hydrant", "parked-bicycle", "bollard", "bicycle-rack"])
        )
    if isinstance(value, list) and value and all(isinstance(a, str) and a for a in value):
        return tuple(value)
    raise EntityClassError(
        f"{name}.body.assets: a family ({', '.join(ASSET_FAMILIES)}) or a list of asset ids required"
    )


def _parse(cid: str, row: dict, config: dict) -> EntityClass:
    if not isinstance(row, dict):
        raise EntityClassError(f"classes.{cid}: an object required")
    known = {
        "tape_class",
        "profile",
        "body",
        "motion",
        "role",
        "semantic_type",
        "description",
        "activities",
        "min_activity_kinds",
        "id_prefix",
        "label_zh",
    }
    extra = {k: v for k, v in row.items() if k not in known}
    tape_class = row.get("tape_class")
    if tape_class not in TAPE_CLASSES:
        raise EntityClassError(f"classes.{cid}.tape_class must be one of {TAPE_CLASSES}")
    profile = row.get("profile")
    if profile not in PROFILES:
        raise EntityClassError(f"classes.{cid}.profile must be person, vehicle or null")
    body = row.get("body") or {}
    kind = body.get("kind")
    if kind not in BODY_KINDS:
        raise EntityClassError(f"classes.{cid}.body.kind must be one of {BODY_KINDS}")
    assets = _assets(body.get("assets"), config, f"classes.{cid}")
    motion = row.get("motion") or {}
    height = Spread.parse(body["height_m"], f"classes.{cid}.body.height_m") if "height_m" in body else None
    if kind == "articulated" and height is None:
        raise EntityClassError(f"classes.{cid}: an articulated body needs body.height_m")
    speed = (
        Spread.parse(motion["max_speed_mps"], f"classes.{cid}.motion.max_speed_mps")
        if "max_speed_mps" in motion
        else None
    )
    if kind != "static" and speed is None:
        raise EntityClassError(f"classes.{cid}: a moving body needs motion.max_speed_mps")
    if kind == "static" and (speed is not None and speed.base > 0):
        raise EntityClassError(f"classes.{cid}: a static body cannot have a speed")
    if kind == "articulated" and profile != "person":
        raise EntityClassError(f"classes.{cid}: articulated walkers use the person profile")
    if kind == "rigid" and profile != "vehicle":
        raise EntityClassError(f"classes.{cid}: rigid vehicles use the vehicle profile")
    role = row.get("role", cid)
    if not isinstance(role, str) or not role:
        raise EntityClassError(f"classes.{cid}.role must be a nonempty string")
    acts = row.get("activities")
    if acts is not None and (not isinstance(acts, list) or not all(isinstance(a, str) and a for a in acts)):
        raise EntityClassError(f"classes.{cid}.activities must be a list of purposes")
    mk = row.get("min_activity_kinds")
    if mk is not None and (isinstance(mk, bool) or not isinstance(mk, int) or mk < 1):
        raise EntityClassError(f"classes.{cid}.min_activity_kinds must be a positive integer")
    return EntityClass(
        id=cid,
        tape_class=tape_class,
        profile=profile,
        body_kind=kind,
        assets=assets,
        role=role,
        height_m=height,
        max_speed_mps=speed,
        semantic_type=str(row.get("semantic_type", "")),
        description=str(row.get("description", "")),
        activities=tuple(acts) if acts is not None else None,
        min_activity_kinds=mk,
        id_prefix=str(row.get("id_prefix", "")),
        label_zh=str(row.get("label_zh", "")),
        extra=extra,
    )


def default_catalog() -> dict:
    return json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))


def catalog_rows(config: dict | None = None) -> dict:
    """The merged raw catalog: built-in classes, then ``entity_classes`` (a file), then inline ``classes``."""
    config = config or {}
    rows = dict(default_catalog()["classes"])
    path = config.get("entity_classes")
    if path:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if schema_id(doc.get("schema")) != SCHEMA or not isinstance(doc.get("classes"), dict):
            raise EntityClassError(f"{path}: {SCHEMA} with a classes object required")
        rows.update(doc["classes"])
    inline = config.get("classes")
    if inline is not None:
        if not isinstance(inline, dict):
            raise EntityClassError("classes must be an object: class id -> definition")
        rows.update(inline)
    return rows


class Catalog:
    def __init__(self, classes: dict[str, EntityClass]):
        self.classes = classes

    def __getitem__(self, cid: str) -> EntityClass:
        try:
            return self.classes[cid]
        except KeyError:
            raise EntityClassError(f"unknown entity class {cid!r}; known: {sorted(self.classes)}") from None

    def __contains__(self, cid) -> bool:
        return cid in self.classes

    def of_tape_class(self, tape_class: str) -> list[EntityClass]:
        return [c for c in self.classes.values() if c.tape_class == tape_class]

    def by_role(self, role: str) -> list[EntityClass]:
        return [c for c in self.classes.values() if c.role == role]


def load(config: dict | None = None) -> Catalog:
    config = config or {}
    rows = catalog_rows(config)
    classes = {cid: _parse(cid, row, config) for cid, row in rows.items()}
    for cid in ("resident", "local_traffic", "street_object"):
        if cid not in classes:
            raise EntityClassError(f"the built-in class {cid} cannot be removed")
    prefixes: dict[str, str] = {}
    for c in classes.values():
        p = c.id_prefix or c.id
        if p in prefixes:
            raise EntityClassError(f"classes {prefixes[p]} and {c.id} share the actor id prefix {p!r}")
        prefixes[p] = c.id
    return Catalog(classes)


def population(config: dict | None, people: int, catalog: Catalog | None = None) -> list[tuple[EntityClass, int]]:
    """Ordered (class, count) of the walkers to admit. ``population`` in the recipe names classes; otherwise the legacy
    ``people`` count means residents. Residents always come first so that their indices, ids and seeds are unchanged."""
    config = config or {}
    cat = catalog or load(config)
    pop = config.get("population")
    if pop is None:
        pop = {"resident": int(config.get("people", people))}
    if not isinstance(pop, dict) or not pop:
        raise EntityClassError("population must be an object: class id -> count")
    out = []
    for cid, n in pop.items():
        c = cat[cid]
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise EntityClassError(f"population.{cid} must be a non-negative integer")
        if c.tape_class != "person" or c.body_kind != "articulated":
            raise EntityClassError(
                f"population.{cid}: only articulated person classes are admitted as walkers in this stage"
            )
        out.append((c, n))
    out.sort(key=lambda cn: cn[0].id != "resident")
    if out[0][0].id != "resident":
        out.insert(0, (cat["resident"], 0))
    return out


def describe(config: dict | None = None, people: int = 0) -> dict:
    cat = load(config)
    pop = population(config, people, cat)
    return {
        "schema": SCHEMA,
        "classes": {
            c.id: {
                "tape_class": c.tape_class,
                "profile": c.profile,
                "body": c.body_kind,
                "assets": list(c.assets),
                "role": c.role,
                "height_m": [c.height(0), c.height(c.height_m.cycle - 1)] if c.height_m else None,
                "max_speed_mps": [c.max_speed(0), c.max_speed(c.max_speed_mps.cycle - 1)] if c.max_speed_mps else None,
                "activities": list(c.activities) if c.activities is not None else "role",
                "min_activity_kinds": c.min_activity_kinds,
                "label_zh": c.label_zh,
                "description": c.description,
            }
            for c in cat.classes.values()
        },
        "population": {c.id: n for c, n in pop},
    }


__all__ = [
    "ASSET_FAMILIES",
    "BODY_KINDS",
    "SCHEMA",
    "TAPE_CLASSES",
    "Catalog",
    "EntityClass",
    "EntityClassError",
    "Spread",
    "catalog_rows",
    "default_catalog",
    "describe",
    "load",
    "population",
]
