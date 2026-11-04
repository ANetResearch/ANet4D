"""Scene principle package ``awr.life.principle.v1``: the only scene input of the target behavior engine.

A principle package is a directory with a manifest ``principle.json`` that binds, by relative path and SHA-256, every
layer the engine reads: the validated support grids, the town semantics, the lane network, the outdoor places and the
world reference the tape header cites. The mesh itself is not part of it; a producer derives the layers from whatever
scene representation it has (textured mesh, point cloud, Gaussians) and declares for each layer where it came from
(``measured``, ``derived``, ``assigned``, ``manual``) and whether a person verified it. The package also declares which
activities the scene can carry (``capabilities``); the engine refuses recipes that need more.

``check(dir)`` is the admission gate: file hashes, grid consistency, semantic references, world witness and capability
declarations. ``resolve(dir)`` hands the engine the paths. ``pack(...)`` builds a package from the legacy inputs
(``--world``, ``--nav``, ``--semantics``, ``--config``) so that existing environments keep working.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from awr.life.common.integrity import sha256_file
from awr.life.common.schema import schema_id

SCHEMA = "awr.life.principle.v1"
CHECK_SCHEMA = "awr.life.principle_check.v1"
SUPPORT_SCHEMA = "awr.life.support.v1"
SEMANTICS_SCHEMA = "awr.life.town_semantics.v1"
PLACES_SCHEMA = "awr.life.outdoor_places.v1"
SUPPORT_FILES = ("meta.json", "ground.npy", "walk_person.npy", "walk_vehicle.npy", "z_person.npy", "z_vehicle.npy")
SOURCES = ("measured", "derived", "assigned", "manual")
STATUSES = ("supported", "unsupported", "unknown", "absent")
CAPABILITIES = ("walking", "vehicles_parked", "driving", "indoor", "public_transport", "cycling", "water")
REQUIRED_LAYERS = ("support", "semantics", "world")
OPTIONAL_LAYERS = ("lane_network", "outdoor_places")


class PrincipleError(ValueError):
    pass


@dataclass
class Resolved:
    """Paths the engine consumes. ``world`` holds the tape's world reference (``world.json``, ``geometry/nav`` witnesses)."""

    root: Path
    manifest: dict
    world: Path
    support: Path
    semantics: Path
    lane_network: Path | None = None
    outdoor_places: Path | None = None
    capabilities: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.manifest["id"]

    @property
    def content_version(self) -> str:
        return self.manifest["content_version"]


def _content_version(layers: dict) -> str:
    """SHA-256 over the sorted ``layer:file:sha256`` lines of every bound file, so a changed layer changes the version."""
    lines = []
    for name, layer in layers.items():
        if "files" in layer:
            lines += [f"{name}:{f}:{h}" for f, h in layer["files"].items()]
        elif "sha256" in layer:
            lines.append(f"{name}:{layer['href']}:{layer['sha256']}")
        elif "content_version" in layer:
            lines.append(f"{name}:{layer['href']}:{layer['content_version']}")
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def load(root: Path) -> dict:
    root = Path(root)
    p = root / "principle.json"
    if not p.is_file():
        raise PrincipleError(f"{p} not found: not a principle package")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if schema_id(doc.get("schema")) != SCHEMA:
        raise PrincipleError(f"{SCHEMA} required, got {doc.get('schema')!r}")
    return doc


def resolve(root: Path) -> Resolved:
    root = Path(root)
    doc = load(root)
    L = doc["layers"]
    return Resolved(
        root=root,
        manifest=doc,
        world=root / L["world"]["href"],
        support=root / L["support"]["href"],
        semantics=root / L["semantics"]["href"],
        lane_network=root / L["lane_network"]["href"] if "lane_network" in L else None,
        outdoor_places=root / L["outdoor_places"]["href"] if "outdoor_places" in L else None,
        capabilities=dict(doc.get("capabilities", {})),
    )


def _components(mask: np.ndarray) -> tuple[int, int]:
    """Number of 4-connected components and the size (cells) of the largest."""
    try:
        from scipy.ndimage import label
    except ImportError:  # the statistic is informative only
        return -1, -1
    lab, n = label(mask)
    if n == 0:
        return 0, 0
    sizes = np.bincount(lab.ravel())[1:]
    return int(n), int(sizes.max())


def check(root: Path) -> dict:
    """Admission check. Returns ``awr.life.principle_check.v1``: ``pass``, ``errors``, ``warnings``, ``stats``."""
    root = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {}
    try:
        doc = load(root)
    except (PrincipleError, json.JSONDecodeError) as e:
        return {"schema": CHECK_SCHEMA, "pass": False, "errors": [str(e)], "warnings": [], "stats": {}}
    for key in ("id", "content_version", "frame", "layers", "capabilities", "provenance"):
        if key not in doc:
            errors.append(f"principle.json: missing {key}")
    if errors:
        return {"schema": CHECK_SCHEMA, "pass": False, "errors": errors, "warnings": warnings, "stats": stats}
    L = doc["layers"]
    for name in REQUIRED_LAYERS:
        if name not in L:
            errors.append(f"layers: missing required layer {name}")
    for name, layer in L.items():
        if name not in REQUIRED_LAYERS + OPTIONAL_LAYERS:
            warnings.append(f"layers: unknown layer {name} is ignored by the engine")
        if "href" not in layer:
            errors.append(f"layers.{name}: missing href")
        if name != "world" and layer.get("source") not in SOURCES:
            errors.append(f"layers.{name}: source must be one of {SOURCES}")
        if name != "world" and not isinstance(layer.get("verified"), bool):
            errors.append(f"layers.{name}: verified must be true or false")
    if errors:
        return {"schema": CHECK_SCHEMA, "pass": False, "errors": errors, "warnings": warnings, "stats": stats}

    # frame
    fr = doc["frame"]
    bounds = fr.get("bounds")
    if fr.get("units") != "m" or fr.get("axes") != "ENU" or fr.get("z_up") is not True:
        errors.append("frame: units m, axes ENU and z_up true are required")
    if not (
        isinstance(bounds, list)
        and len(bounds) == 4
        and all(isinstance(v, (int, float)) for v in bounds)
        and bounds[0] < bounds[2]
        and bounds[1] < bounds[3]
    ):
        errors.append("frame.bounds must be [x0, y0, x1, y1] in metres with x0 < x1 and y0 < y1")
        bounds = None

    # file hashes
    def hashed(name: str, rel: str, want: str) -> bool:
        p = root / rel
        if not p.is_file():
            errors.append(f"layers.{name}: {rel} not found")
            return False
        got = sha256_file(p)
        if got != want:
            errors.append(f"layers.{name}: {rel} sha256 {got[:12]} differs from manifest {str(want)[:12]}")
            return False
        return True

    sup = L["support"]
    sup_ok = True
    for f in SUPPORT_FILES:
        if f not in sup.get("files", {}):
            errors.append(f"layers.support.files: missing {f}")
            sup_ok = False
        else:
            sup_ok &= hashed("support", f"{sup['href']}/{f}", sup["files"][f])
    for name in ("semantics", "lane_network", "outdoor_places"):
        if name in L:
            hashed(name, L[name]["href"], L[name].get("sha256"))
    want_cv = _content_version(L)
    if doc["content_version"] != want_cv:
        errors.append(f"content_version {doc['content_version'][:12]} does not match the bound layers ({want_cv[:12]})")

    # support grids
    if sup_ok:
        sdir = root / sup["href"]
        try:
            meta = json.loads((sdir / "meta.json").read_text())
            if schema_id(meta.get("schema")) != SUPPORT_SCHEMA:
                errors.append(f"support/meta.json: schema {SUPPORT_SCHEMA} required")
            nx, ny, cell = int(meta["nx"]), int(meta["ny"]), float(meta["cell_m"])
            x0, y0 = float(meta["x0"]), float(meta["y0"])
            z = np.load(sdir / "ground.npy")
            walk = {k: np.load(sdir / f"walk_{k}.npy") for k in ("person", "vehicle")}
            zp = {k: np.load(sdir / f"z_{k}.npy") for k in ("person", "vehicle")}
            for name, a in [
                ("ground.npy", z),
                *[(f"walk_{k}.npy", v) for k, v in walk.items()],
                *[(f"z_{k}.npy", v) for k, v in zp.items()],
            ]:
                if a.shape != (ny, nx):
                    errors.append(f"support/{name}: shape {a.shape} is not (ny, nx) = ({ny}, {nx})")
            for k, v in walk.items():
                if v.dtype != np.bool_:
                    errors.append(f"support/walk_{k}.npy must be bool")
            if not (cell > 0 and np.isfinite(cell)):
                errors.append("support/meta.json: cell_m must be positive")
            if not errors:
                finite = np.isfinite(z)
                for k, v in walk.items():
                    bad = int((v & ~finite).sum())
                    if bad:
                        errors.append(f"support: {bad} walk_{k} cells have no finite ground height")
                not_person = int((walk["vehicle"] & ~walk["person"]).sum())
                if not_person:
                    warnings.append(f"support: {not_person} vehicle cells are not person cells")
                n_c, largest = _components(walk["person"])
                stats["support"] = {
                    "cell_m": cell,
                    "nx": nx,
                    "ny": ny,
                    "extent_m": [x0, y0, x0 + nx * cell, y0 + ny * cell],
                    "ground_m2": float(finite.sum() * cell * cell),
                    "person_m2": float(walk["person"].sum() * cell * cell),
                    "person_components": n_c,
                    "person_largest_m2": float(largest * cell * cell),
                    "vehicle_m2": float(walk["vehicle"].sum() * cell * cell),
                    "unknown_is_blocked": bool(meta.get("unknown_is_blocked", True)),
                }
                if bounds and (
                    x0 > bounds[0] + cell
                    or y0 > bounds[1] + cell
                    or x0 + nx * cell < bounds[2] - cell
                    or y0 + ny * cell < bounds[3] - cell
                ):
                    warnings.append("support: the grids do not cover the whole frame.bounds")
                if walk["person"].sum() == 0:
                    errors.append("support: no person-walkable cell")
        except (OSError, KeyError, ValueError) as e:
            errors.append(f"support: {e}")

    # semantics
    sem_path = root / L["semantics"]["href"]
    if sem_path.is_file():
        try:
            sem = json.loads(sem_path.read_text(encoding="utf-8"))
            if schema_id(sem.get("schema")) != SEMANTICS_SCHEMA:
                errors.append(f"semantics: schema {SEMANTICS_SCHEMA} required")
            ids: set[str] = set()

            def unique(kind: str, item: dict) -> None:
                i = item.get("id")
                if not isinstance(i, str) or not i:
                    errors.append(f"semantics.{kind}: an item has no id")
                elif i in ids:
                    errors.append(f"semantics.{kind}: duplicate id {i}")
                ids.add(i)

            def inside(pts) -> bool:
                if not bounds:
                    return True
                a = np.asarray(pts, float)
                return bool(
                    (
                        (a[:, 0] >= bounds[0])
                        & (a[:, 0] <= bounds[2])
                        & (a[:, 1] >= bounds[1])
                        & (a[:, 1] <= bounds[3])
                    ).all()
                )

            out_of_bounds = 0
            for kind, need in (
                ("roads", ("line", "width_m", "kind")),
                ("buildings", ("polygon",)),
                ("open_areas", ("polygon",)),
                ("pois", ("kind", "xy", "verified_entrance", "source")),
            ):
                items = sem.get(kind)
                if not isinstance(items, list):
                    errors.append(f"semantics: {kind} must be a list")
                    continue
                for it in items:
                    unique(kind, it)
                    for k in need:
                        if k not in it:
                            errors.append(f"semantics.{kind} {it.get('id')}: missing {k}")
                    if kind == "roads" and "line" in it:
                        if len(it["line"]) < 2 or not float(it.get("width_m", 0)) > 0:
                            errors.append(f"semantics.roads {it.get('id')}: line needs 2 points and width_m > 0")
                        elif not inside(it["line"]):
                            out_of_bounds += 1
                    if kind in ("buildings", "open_areas") and "polygon" in it:
                        if len(it["polygon"]) < 3:
                            errors.append(f"semantics.{kind} {it.get('id')}: polygon needs 3 points")
                        elif not inside(it["polygon"]):
                            out_of_bounds += 1
                    if kind == "pois":
                        if not isinstance(it.get("verified_entrance"), bool):
                            errors.append(f"semantics.pois {it.get('id')}: verified_entrance must be true or false")
                        if it.get("source") not in SOURCES:
                            errors.append(f"semantics.pois {it.get('id')}: source must be one of {SOURCES}")
                        if it.get("building") and it["building"] not in {b.get("id") for b in sem.get("buildings", [])}:
                            errors.append(
                                f"semantics.pois {it.get('id')}: building {it['building']} is not a building id"
                            )
                        if "xy" in it and not inside([it["xy"]]):
                            out_of_bounds += 1
            if out_of_bounds:
                warnings.append(f"semantics: {out_of_bounds} items lie partly outside frame.bounds")
            stats["semantics"] = {
                k: len(sem.get(k, []))
                for k in ("roads", "buildings", "open_areas", "pois", "crossings", "verified_parking_regions")
            }
            stats["semantics"]["road_km"] = round(
                sum(
                    float(np.linalg.norm(np.diff(np.asarray(r["line"], float), axis=0), axis=1).sum())
                    for r in sem.get("roads", [])
                    if isinstance(r.get("line"), list) and len(r["line"]) >= 2
                )
                / 1000.0,
                3,
            )
            stats["semantics"]["pois_verified_entrance"] = sum(
                1 for p in sem.get("pois", []) if p.get("verified_entrance") is True
            )
        except (OSError, ValueError, TypeError) as e:
            errors.append(f"semantics: {e}")

    # outdoor places
    if "outdoor_places" in L and (root / L["outdoor_places"]["href"]).is_file():
        try:
            pl = json.loads((root / L["outdoor_places"]["href"]).read_text(encoding="utf-8"))
            if schema_id(pl.get("schema")) != PLACES_SCHEMA:
                errors.append(f"outdoor_places: schema {PLACES_SCHEMA} required")
            stats["outdoor_places"] = {"places": len(pl.get("places", []))}
        except (OSError, ValueError) as e:
            errors.append(f"outdoor_places: {e}")

    # world reference: what the tape header cites
    wdir = root / L["world"]["href"]
    wj = wdir / "world.json"
    if not wj.is_file():
        errors.append(f"layers.world: {wj.relative_to(root)} not found")
    else:
        try:
            wdoc = json.loads(wj.read_text(encoding="utf-8"))
            if wdoc.get("id") != doc["id"]:
                errors.append(f"world.json id {wdoc.get('id')!r} differs from principle id {doc['id']!r}")
            if wdoc.get("contentVersion") != L["world"].get("content_version"):
                errors.append("world.json contentVersion differs from layers.world.content_version")
            from awr.sim.events.compile import raycast_sha256

            rdir = wdir / "geometry/mesh/raycast"
            if not rdir.is_dir():
                errors.append("layers.world: geometry/mesh/raycast is missing (the tape header needs its hash)")
            elif raycast_sha256(wdir) != L["world"].get("raycast_sha256"):
                errors.append("layers.world.raycast_sha256 differs from geometry/mesh/raycast")
            nav = wdir / "geometry/nav"
            for prof, f in (("person", "walk_person.npy"), ("vehicle", "walk_vehicle.npy")):
                pj = nav / f"{prof}.json"
                if pj.is_file():
                    digest = json.loads(pj.read_text()).get("bin", {}).get("sha256")
                    if sup_ok and digest != sup["files"].get(f):
                        errors.append(f"world geometry/nav/{prof}.json witness differs from support/{f}")
                elif prof == "person":
                    errors.append("layers.world: geometry/nav/person.json witness is missing")
        except (OSError, ValueError) as e:
            errors.append(f"world: {e}")

    # capabilities
    caps = doc["capabilities"]
    if not isinstance(caps, dict):
        errors.append("capabilities must be an object")
    else:
        for name, c in caps.items():
            if name not in CAPABILITIES:
                warnings.append(f"capabilities: unknown capability {name}")
            if not isinstance(c, dict) or c.get("status") not in STATUSES:
                errors.append(f"capabilities.{name}: status must be one of {STATUSES}")
            elif c["status"] == "unsupported" and not c.get("reason"):
                errors.append(f"capabilities.{name}: unsupported needs a reason")
        for name in ("walking", "driving"):
            if name not in caps:
                errors.append(f"capabilities: {name} must be declared")
        s = stats.get("support")
        if s and caps.get("walking", {}).get("status") == "supported" and s["person_m2"] == 0:
            errors.append("capabilities.walking is supported but there is no person-walkable cell")
        if s and caps.get("vehicles_parked", {}).get("status") == "supported" and s["vehicle_m2"] == 0:
            errors.append("capabilities.vehicles_parked is supported but there is no vehicle cell")
        if caps.get("driving", {}).get("status") == "supported" and "lane_network" not in L:
            errors.append("capabilities.driving is supported but no lane_network layer is bound")
    return {
        "schema": CHECK_SCHEMA,
        "pass": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
        "id": doc.get("id"),
        "content_version": doc.get("content_version"),
    }


def demand_errors(capabilities: dict, *, people: int, vehicles: int, config: dict | None = None) -> list[str]:
    """Why a recipe (``people`` walkers, ``vehicles`` cars, config ``life`` section) exceeds the declared capabilities."""
    cfg = config or {}
    life = cfg.get("life") or {}
    pop = (life.get("vehicles") or {}).get("population") or {}
    trips = int(pop.get("curb_trip", 0)) + int(pop.get("lot_trip", 0))
    out = []

    def status(name: str) -> str:
        return (capabilities.get(name) or {}).get("status", "unknown")

    if people > 0 and status("walking") != "supported":
        out.append(f"{people} walkers requested but capabilities.walking is {status('walking')}")
    if vehicles > 0 and status("vehicles_parked") != "supported" and status("driving") != "supported":
        out.append(f"{vehicles} vehicles requested but neither vehicles_parked nor driving is supported")
    if (trips > 0 or life.get("driving")) and status("driving") != "supported":
        out.append(f"vehicle trips requested but capabilities.driving is {status('driving')}")
    if cfg.get("delivery") and status("driving") != "supported":
        out.append(f"a delivery round is requested but capabilities.driving is {status('driving')}")
    return out


def admit(root: Path, config: dict | None = None, *, people: int = 0, vehicles: int = 0) -> tuple[Resolved, dict]:
    """Gate of the engine: ``check`` must pass and the recipe must stay within the declared capabilities.

    With a principle the scene files come from the package: a config that names its own ``lane_network`` or
    ``outdoor_places`` is rejected unless it names the package's file, and the package's paths are written into
    ``config`` so that the rest of the engine reads one source."""
    report = check(root)
    if not report["pass"]:
        raise PrincipleError("principle rejected: " + "; ".join(report["errors"]))
    pkg = resolve(root)
    cfg = config if config is not None else {}
    for key, path in (("lane_network", pkg.lane_network), ("outdoor_places", pkg.outdoor_places)):
        named = cfg.get(key)
        if named and (path is None or Path(named).resolve() != path.resolve()):
            raise PrincipleError(f"config.{key} names {named}; with a principle the scene files come from the package")
        if path is not None:
            cfg[key] = str(path)
        else:
            cfg.pop(key, None)
    errors = demand_errors(pkg.capabilities, people=people, vehicles=vehicles, config=cfg)
    if errors:
        raise PrincipleError("recipe exceeds the principle's capabilities: " + "; ".join(errors))
    return pkg, report


def _copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return sha256_file(dst)


def pack(
    out: Path,
    *,
    world: Path,
    nav: Path,
    semantics: Path,
    lane_network: Path | None = None,
    outdoor_places: Path | None = None,
    config: Path | None = None,
    capabilities: dict | None = None,
    sources: dict | None = None,
    name: str | None = None,
    name_zh: str | None = None,
    producer: str = "awr.life.principle.pack",
    notes: str | None = None,
) -> dict:
    """Build a principle package from the legacy inputs. ``capabilities`` overrides the defaults derived from the grids and
    from ``config.life.unavailable``; ``sources`` maps layer name to {source, verified, method}."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise PrincipleError(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    world, nav, semantics = Path(world), Path(nav), Path(semantics)
    sources = sources or {}
    wdoc = json.loads((world / "world.json").read_text(encoding="utf-8"))
    layers: dict = {}
    files = {f: _copy(nav / f, out / "support" / f) for f in SUPPORT_FILES}
    meta = json.loads((nav / "meta.json").read_text())
    src = sources.get("support", {})
    layers["support"] = {
        "href": "support",
        "schema": SUPPORT_SCHEMA,
        "files": files,
        "source": src.get("source", "derived"),
        "verified": bool(src.get("verified", False)),
        "method": src.get("method", meta.get("method", "")),
    }
    src = sources.get("semantics", {})
    layers["semantics"] = {
        "href": "semantics.json",
        "schema": SEMANTICS_SCHEMA,
        "sha256": _copy(semantics, out / "semantics.json"),
        "source": src.get("source", "derived"),
        "verified": bool(src.get("verified", False)),
        "note": src.get("method", json.loads(semantics.read_text(encoding="utf-8")).get("note", "")),
    }
    cfg = json.loads(Path(config).read_text(encoding="utf-8")) if config else {}
    lane_network = (
        Path(lane_network) if lane_network else (Path(cfg["lane_network"]) if cfg.get("lane_network") else None)
    )
    outdoor_places = (
        Path(outdoor_places) if outdoor_places else (Path(cfg["outdoor_places"]) if cfg.get("outdoor_places") else None)
    )
    if lane_network:
        src = sources.get("lane_network", {})
        layers["lane_network"] = {
            "href": "lanes/net.net.xml",
            "format": "sumo-net",
            "sha256": _copy(lane_network, out / "lanes/net.net.xml"),
            "source": src.get("source", "derived"),
            "verified": bool(src.get("verified", False)),
        }
    if outdoor_places:
        src = sources.get("outdoor_places", {})
        layers["outdoor_places"] = {
            "href": "places.json",
            "schema": PLACES_SCHEMA,
            "sha256": _copy(outdoor_places, out / "places.json"),
            "source": src.get("source", "derived"),
            "verified": bool(src.get("verified", False)),
        }
    # world reference: only the files the tape header and the provenance snapshot read
    from awr.sim.events.compile import raycast_sha256

    for p in sorted(world.rglob("*")):
        if p.is_file() and (
            p.name == "world.json" or "geometry/nav" in p.as_posix() or "geometry/mesh/raycast" in p.as_posix()
        ):
            _copy(p, out / "world" / p.relative_to(world))
    layers["world"] = {
        "href": "world",
        "content_version": wdoc["contentVersion"],
        "raycast_sha256": raycast_sha256(out / "world"),
    }
    person = np.load(out / "support/walk_person.npy")
    vehicle = np.load(out / "support/walk_vehicle.npy")
    caps = {
        "walking": {
            "status": "supported" if person.any() else "unsupported",
            "evidence": "person-walkable cells of the support grid",
        },
        "vehicles_parked": {
            "status": "supported" if vehicle.any() else "unsupported",
            "evidence": "vehicle cells of the support grid",
        },
        "driving": {"status": "unknown"},
        "indoor": {"status": "unknown"},
        "public_transport": {"status": "unknown"},
        "cycling": {"status": "unknown"},
        "water": {"status": "absent"},
    }
    if not person.any():
        caps["walking"]["reason"] = "no person-walkable cell"
    if not vehicle.any():
        caps["vehicles_parked"]["reason"] = "no vehicle cell"
    alias = {
        "car_journeys": "driving",
        "indoor_stays": "indoor",
        "public_transport": "public_transport",
        "cycling": "cycling",
    }
    for u in (cfg.get("life") or {}).get("unavailable", []) or []:
        k = alias.get(u.get("activity"))
        if k:
            caps[k] = {"status": "unsupported", "reason": u.get("reason", "")}
    caps.update(capabilities or {})
    b = wdoc.get("bounds", {})
    bounds = [
        *b.get("min", [meta["x0"], meta["y0"]]),
        *b.get("max", [meta["x0"] + meta["nx"] * meta["cell_m"], meta["y0"] + meta["ny"] * meta["cell_m"]]),
    ]
    doc = {
        "schema": SCHEMA,
        "id": wdoc["id"],
        "name": name or wdoc.get("name", wdoc["id"]),
        "name_zh": name_zh or wdoc.get("nameZh", ""),
        "content_version": _content_version(layers),
        "frame": {
            "units": "m",
            "axes": "ENU",
            "z_up": True,
            "bounds": [float(v) for v in bounds],
            "georeferenced": bool(wdoc.get("georeferenced", False)),
        },
        "layers": layers,
        "capabilities": caps,
        "provenance": {
            "producer": producer,
            "inputs": {
                "world": str(world),
                "support": str(nav),
                "semantics": str(semantics),
                "lane_network": str(lane_network) if lane_network else None,
                "outdoor_places": str(outdoor_places) if outdoor_places else None,
            },
            "notes": notes or "",
        },
    }
    (out / "principle.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


__all__ = [
    "CAPABILITIES",
    "CHECK_SCHEMA",
    "SCHEMA",
    "SOURCES",
    "STATUSES",
    "PrincipleError",
    "Resolved",
    "admit",
    "check",
    "demand_errors",
    "load",
    "pack",
    "resolve",
]
