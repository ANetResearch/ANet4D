"""Validation of an environment package against the environment feature principles v1.

    python -m awr.life.envpack.validate configs/environments/oryahovo-crop1.json \
        --queries configs/queries/ale-demo-drafts-v1.json --out docs/audit/envpack/oryahovo-crop1-validation-v1

Each check yields a finding (feature, check, status, value, note):

- pass / fail: the layer was read and the condition holds or does not;
- missing: the package does not provide what the principles require;
- unknown: the requirement cannot be decided from the layers present (for example a layer kept only on the build
  host, or a property no layer describes);
- info: a measured value without a pass condition (sizes, shares, distributions).

The thresholds in TOLERANCES are alignment tolerances, each with its reason; none is a behaviour requirement.

Capabilities are then evaluated from the findings and the package's evidence records. The evidence level of each
number is stated: candidate (mapped objects before any admission test), static (objects admitted by the adapter's
geometric rules), dynamic (measured in a recorded run bound to the same layers) and upper_bound (an area bound,
never a capacity). A capability is supported (necessary features verified and a dynamic measurement exists),
limited (supported within a stated scope), static_only (features verified, no run measures it), unsupported (a
necessary feature is absent or measured empty) or unknown (a necessary feature cannot be decided).

The validator reads and measures; it never edits a layer, fills ground or relabels a region.

Integration scope: static diagnostics only, never runtime or query admission. Historical runs are
reported separately; their successful gates do not certify the current layers or implementation.
Source: Claude working snapshot 2026-10-25 (see docs/audit/envpack-integration-v1/README.md).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from awr.life.common.schema import schema_id

MANIFEST_SCHEMAS = ("awr.life.life.environment.v1", "awr.life.environment.v2")
REPORT_SCHEMA = "awr.life.environment_validation.v2"
FEATURES = ("frame", "geometry", "regions", "passage", "entities", "resources", "versions")
CAPABILITIES = ("walk_activity", "vehicle_traffic", "parking", "recurrent_trips", "driver_chain", "boundary_entry_exit")
# Quality raster bits of the world package (awr world package, quality_2m).
QUALITY_BITS = ("nodata", "hole", "stretched", "low_texture", "multilayer", "water", "fragment", "synthetic_nav")
NO_RECONSTRUCTION = ("nodata", "hole", "water")
LOW_CONFIDENCE = ("stretched", "low_texture", "multilayer", "fragment")

TOLERANCES = {
    "height_median_abs_m": (0.5, "support heights and the reconstruction surface share one vertical frame"),
    "outside_share": (0.01, "mapped objects lie inside the measured support grid"),
    "fabricated_support_share": (0.001, "support over cells without reconstruction is ground the scan does not hold"),
    "road_supported_share": (0.9, "an edge counts as drivable when 90% of its centre-line samples are supported"),
    "road_shift_gain": (0.1, "a lateral shift that raises road support by 10 points indicates misregistration"),
    "building_walkable_share": (
        0.5,
        "a footprint that is mostly walkable ground is misplaced or missing from the scan",
    ),
    "region_supported_share": (0.2, "an open area or parking region has measured support for its use"),
    "vehicle_side_margin_m": (0.3, "clearance on each side of a vehicle body in a lane"),
    "registration_shift_m": (1.5, "mapped features within 1.5 m of their best fit to the reconstruction"),
    "registration_gain": (
        0.05,
        "a translation that puts 5 points more road samples on open ground indicates misregistration",
    ),
}


def tol(name):
    return TOLERANCES[name][0]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    with gzip.open(path, "rt") if path.suffix == ".gz" else open(path) as f:
        return json.load(f)


class Package:
    """An environment manifest and lazy access to its layers (paths relative to `root`)."""

    def __init__(self, manifest, root="."):
        self.manifest_path, self.root = Path(manifest), Path(root)
        self.m = json.loads(self.manifest_path.read_text())
        if schema_id(self.m.get("schema")) not in MANIFEST_SCHEMAS:
            raise ValueError(f"{manifest}: schema {self.m.get('schema')} is not an environment manifest")
        self.layers = self.m.get("layers", {})
        self.features = self.m.get("features", {})
        self._json = {}

    def path(self, layer, rel=None):
        entry = self.layers.get(layer)
        if not entry:
            return None
        p = self.root / entry["path"]
        return p / rel if rel else p

    def has(self, layer, rel=None):
        p = self.path(layer, rel)
        return p is not None and p.exists()

    def json(self, layer, rel=None):
        key = (layer, rel)
        if key not in self._json:
            self._json[key] = read_json(self.path(layer, rel))
        return self._json[key]


class Support:
    """The measured support grids (awr.life.support.v1): ground heights and walkable cells per profile."""

    def __init__(self, root):
        root = Path(root)
        self.meta = json.loads((root / "meta.json").read_text())
        self.x0, self.y0 = float(self.meta["x0"]), float(self.meta["y0"])
        self.nx, self.ny, self.cell = int(self.meta["nx"]), int(self.meta["ny"]), float(self.meta["cell_m"])
        self.ground = np.load(root / "ground.npy", mmap_mode="r")
        self.walk = {k: np.load(root / f"walk_{k}.npy", mmap_mode="r") for k in ("person", "vehicle")}

    @property
    def extent(self):
        return self.x0, self.y0, self.x0 + self.nx * self.cell, self.y0 + self.ny * self.cell

    def index(self, xy):
        p = np.asarray(xy, float).reshape(-1, 2)
        ix = np.floor((p[:, 0] - self.x0) / self.cell).astype(int)
        iy = np.floor((p[:, 1] - self.y0) / self.cell).astype(int)
        inside = (ix >= 0) & (iy >= 0) & (ix < self.nx) & (iy < self.ny)
        return np.clip(ix, 0, self.nx - 1), np.clip(iy, 0, self.ny - 1), inside

    def at(self, xy, profile):
        ix, iy, inside = self.index(xy)
        return inside & np.asarray(self.walk[profile])[iy, ix]

    def inside(self, xy):
        return self.index(xy)[2]

    def cell_centres(self, iy, ix):
        return np.stack([self.x0 + (ix + 0.5) * self.cell, self.y0 + (iy + 0.5) * self.cell], 1)


class Raster:
    """A world-package terrain raster (rows south to north)."""

    def __init__(self, world, name):
        meta = json.loads((Path(world) / "geometry/terrain" / f"{name}.json").read_text())
        if meta.get("rowOrder", "south-to-north") != "south-to-north":
            raise ValueError(f"{name}: unsupported row order {meta['rowOrder']}")
        self.meta = meta
        self.data = np.fromfile(Path(world) / "geometry/terrain" / meta["href"], dtype=meta["dtype"]).reshape(
            meta["height"], meta["width"]
        )
        self.ox, self.oy = map(float, meta["originXY"])
        self.cell = float(meta["cellM"])

    def at(self, xy, fill=0):
        p = np.asarray(xy, float).reshape(-1, 2)
        col = np.floor((p[:, 0] - self.ox) / self.cell).astype(int)
        row = np.floor((p[:, 1] - self.oy) / self.cell).astype(int)
        inside = (col >= 0) & (row >= 0) & (col < self.data.shape[1]) & (row < self.data.shape[0])
        out = np.full(len(p), fill, dtype=self.data.dtype)
        out[inside] = self.data[row[inside], col[inside]]
        return out


def quality_names(bits):
    return [name for k, name in enumerate(QUALITY_BITS) if int(bits) >> k & 1]


def polygon_lattice(polygon, step=1.0):
    """Points of a regular lattice inside a shapely polygon (its representative point when it is smaller)."""
    import shapely

    x0, y0, x1, y1 = polygon.bounds
    xs, ys = np.arange(x0 + step / 2, x1, step), np.arange(y0 + step / 2, y1, step)
    if not len(xs) or not len(ys):
        return np.asarray(polygon.representative_point().coords)
    gx, gy = np.meshgrid(xs, ys)
    keep = shapely.contains_xy(polygon, gx.ravel(), gy.ravel())
    pts = np.stack([gx.ravel()[keep], gy.ravel()[keep]], 1)
    return pts if len(pts) else np.asarray(polygon.representative_point().coords)


def line_samples(coords, step):
    """Samples every `step` metres along a polyline and the unit left normal at each."""
    pts, normals = [], []
    for a, b in zip(coords[:-1], coords[1:], strict=False):
        a, b = np.asarray(a[:2], float), np.asarray(b[:2], float)
        d = b - a
        length = float(np.hypot(*d))
        if length == 0:
            continue
        n = max(1, int(round(length / step)))
        t = (np.arange(n) + 0.5) / n
        pts.append(a + t[:, None] * d)
        normals.append(np.repeat([[-d[1] / length, d[0] / length]], n, 0))
    if not pts:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.vstack(pts), np.vstack(normals)


class Report:
    def __init__(self):
        self.findings = []

    def add(self, feature, check, status, value=None, note=""):
        self.findings.append(dict(feature=feature, check=check, status=status, value=value, note=note))
        return status

    def get(self, feature, check):
        for f in self.findings:
            if f["feature"] == feature and f["check"] == check:
                return f
        return None

    def status(self, feature, check):
        f = self.get(feature, check)
        return f["status"] if f else "missing"


# Base features ---------------------------------------------------------------------------------------------------


def check_frame(pkg, sup, sem, places, rep):
    if pkg.has("world", "coordinate.json"):
        coord = pkg.json("world", "coordinate.json")
        frame = coord.get("frame", {})
        ok = (
            frame.get("type") == "ENU"
            and frame.get("units") == "m"
            and frame.get("upAxis") == "+Z"
            and frame.get("handedness") == "right"
        )
        rep.add("frame", "world_axes", "pass" if ok else "fail", frame, "world frame: ENU, metres, +Z up, right-handed")
        anchor = coord.get("anchor", {})
        rep.add(
            "frame",
            "geo_anchor",
            "info",
            dict(
                georeferenced=anchor.get("georeferenced"),
                kind=anchor.get("kind"),
                uncertainty_m=anchor.get("uncertaintyM"),
            ),
            anchor.get("label", ""),
        )
    else:
        rep.add(
            "frame",
            "world_axes",
            "unknown" if "world" in pkg.layers else "missing",
            None,
            "coordinate.json of the world package is not available here",
        )
    if sup is None:
        rep.add(
            "frame",
            "layers_in_support_extent",
            "unknown" if "support" in pkg.layers else "missing",
            None,
            "measured support grids are not available here",
        )
    if sup is not None and sem is not None:
        x0, y0, x1, y1 = sup.extent
        pts = [r["line"] for r in sem.get("roads", [])] + [b["polygon"] for b in sem.get("buildings", [])]
        xy = np.asarray([q[:2] for line in pts for q in line], float).reshape(-1, 2)
        if places:
            xy = np.vstack([xy, np.asarray([p["xy"] for p in places], float)])
        outside = float((~sup.inside(xy)).mean()) if len(xy) else 0.0
        sb = sem.get("bounds")
        rep.add(
            "frame",
            "layers_in_support_extent",
            "pass" if outside <= tol("outside_share") else "fail",
            dict(
                outside_share=round(outside, 4),
                points=int(len(xy)),
                support_extent=[x0, y0, x1, y1],
                semantics_bounds=sb,
            ),
            "roads, footprints and places inside the support grid",
        )
    if sup is not None and pkg.has("world", "geometry/terrain/dsm_2m.json"):
        dsm = Raster(pkg.path("world"), "dsm_2m")
        iy, ix = np.nonzero(np.asarray(sup.walk["person"]))
        take = np.linspace(0, len(iy) - 1, min(len(iy), 20000)).astype(int) if len(iy) else np.zeros(0, int)
        iy, ix = iy[take], ix[take]
        z = np.asarray(sup.ground)[iy, ix]
        d = dsm.at(sup.cell_centres(iy, ix), fill=np.nan).astype(float) - z
        d = d[np.isfinite(d)]
        if len(d):
            med = float(np.median(np.abs(d)))
            rep.add(
                "frame",
                "height_datum",
                "pass" if med <= tol("height_median_abs_m") else "fail",
                dict(
                    samples=int(len(d)),
                    median_abs_m=round(med, 3),
                    p95_abs_m=round(float(np.percentile(np.abs(d), 95)), 3),
                    share_within_0_5_m=round(float((np.abs(d) <= 0.5).mean()), 4),
                ),
                "walkable support heights against the reconstruction DSM (2 m cells, highest mesh hit)",
            )
    if sem is not None and pkg.has("world", "geometry/terrain/dsm_2m.json"):
        from .registration import estimate

        est = estimate(pkg.path("world"), sem, search_m=6.0, step_m=1.0)["overall"]
        shift = float(np.hypot(*est["shift_m"]))
        gain = (est["open_road_share"] or 0.0) - (est["at_zero"]["open_road_share"] or 0.0)
        observed = est["open_road_share"] is not None or est["building_lift_m"] is not None
        ok = shift <= tol("registration_shift_m") or gain < tol("registration_gain")
        applied = (pkg.json("semantics").get("registration") or {}).get("transform")
        rep.add(
            "frame",
            "feature_registration",
            ("pass" if ok else "fail") if observed else "unknown",
            dict(
                best_shift_m=est["shift_m"],
                open_road_share=est["open_road_share"],
                at_zero=est["at_zero"],
                gain=round(gain, 4),
                applied_registration=applied,
            ),
            "mapped buildings and roads against the reconstruction DSM (awr.life.envpack.registration), 1 m steps",
        )
    tick = (pkg.features.get("frame") or {}).get("tick_hz")
    if tick is None:
        rep.add(
            "frame", "time_convention", "missing", None, "manifest features.frame.tick_hz and clock are not declared"
        )
    else:
        from awr.life.motion.crowd import DT

        ok = abs(1 / DT - float(tick)) < 1e-9
        rep.add(
            "frame",
            "time_convention",
            "pass" if ok else "fail",
            dict(declared_tick_hz=tick, engine_tick_hz=round(1 / DT, 6), clock=pkg.features["frame"].get("clock")),
        )


def check_geometry(pkg, sup, rep):
    if sup is None:
        rep.add(
            "geometry",
            "support",
            "unknown" if "support" in pkg.layers else "missing",
            None,
            "measured support grids are not available here",
        )
        return
    meta = sup.meta
    ok = (
        schema_id(meta.get("schema")) == "awr.life.support.v1"
        and meta.get("unknown_is_blocked") is True
        and meta.get("interpolated_heights_accepted_without_mesh", False) is False
    )
    rep.add(
        "geometry",
        "support_contract",
        "pass" if ok else "fail",
        dict(
            schema=meta.get("schema"),
            unknown_is_blocked=meta.get("unknown_is_blocked"),
            interpolated_heights_accepted_without_mesh=meta.get("interpolated_heights_accepted_without_mesh"),
            method=meta.get("method"),
        ),
        "unknown ground is blocked; no height without a mesh hit",
    )
    shapes = {k: list(np.shape(v)) for k, v in [("ground", sup.ground)] + list(sup.walk.items())}
    counts = {k: int(np.count_nonzero(np.asarray(v))) for k, v in sup.walk.items()}
    declared = {k: meta.get(f"{k}_cells") for k in sup.walk}
    ok = all(s == [sup.ny, sup.nx] for s in shapes.values()) and all(
        declared[k] is None or declared[k] == counts[k] for k in counts
    )
    rep.add(
        "geometry",
        "support_grids",
        "pass" if ok else "fail",
        dict(
            shape=[sup.ny, sup.nx],
            cell_m=sup.cell,
            cells=counts,
            declared_cells=declared,
            area_m2={k: round(c * sup.cell**2, 1) for k, c in counts.items()},
        ),
    )
    stale, absent = [], []
    for rel, digest in (meta.get("source_hashes") or {}).items():
        p = Path(rel) if Path(rel).is_absolute() else pkg.root / rel
        if not p.exists():
            absent.append(rel)
        elif sha256(p) != digest.split("-")[0][:64]:
            stale.append(rel)
    rep.add(
        "geometry",
        "support_sources_current",
        "fail" if stale else ("unknown" if absent else "pass"),
        dict(stale=stale, absent=absent, recorded=len(meta.get("source_hashes") or {})),
        "source files recorded by the support build against their current contents",
    )
    if pkg.has("world", "geometry/terrain/quality_2m.json"):
        q = Raster(pkg.path("world"), "quality_2m")
        value = {}
        worst = 0.0
        for profile in ("person", "vehicle"):
            iy, ix = np.nonzero(np.asarray(sup.walk[profile]))
            bits = q.at(sup.cell_centres(iy, ix), fill=1)
            shares = {
                name: round(float(((bits >> k) & 1).mean()), 5) if len(bits) else 0.0
                for k, name in enumerate(QUALITY_BITS)
            }
            fabricated = (
                float(np.any([(bits >> QUALITY_BITS.index(n)) & 1 for n in NO_RECONSTRUCTION], 0).mean())
                if len(bits)
                else 0.0
            )
            worst = max(worst, fabricated)
            value[profile] = dict(
                cells=int(len(bits)), quality_bit_share=shares, no_reconstruction_share=round(fabricated, 5)
            )
        rep.add(
            "geometry",
            "support_over_reconstruction",
            "pass" if worst <= tol("fabricated_support_share") else "fail",
            value,
            "share of supported cells on cells the scan marks as no data, hole or water",
        )
        areas = {}
        if pkg.has("world", "semantic/quality.geojson"):
            for f in pkg.json("world", "semantic/quality.geojson").get("features", []):
                areas[f["properties"].get("kind")] = f["properties"].get("area_m2")
        rep.add(
            "geometry", "unknown_regions", "info", areas, "reconstruction quality regions of the world package (m2)"
        )
    else:
        rep.add("geometry", "unknown_regions", "unknown", None, "world quality raster not available here")
    rep.add(
        "geometry",
        "resolution",
        "info",
        dict(cell_m=sup.cell, horizontal_quantisation_m=sup.cell / 2),
        "support is decided per cell; footprints are tested cell by cell",
    )


def check_regions(pkg, sup, sem, places, rep):
    from shapely.geometry import Polygon

    if sem is None:
        rep.add("regions", "semantics", "missing", None, "no region semantics layer")
        return
    lists = {
        k: sem.get(k, [])
        for k in (
            "roads",
            "buildings",
            "pois",
            "open_areas",
            "paths",
            "crossings",
            "verified_parking_regions",
            "parking_lots",
            "curb_slots",
            "bus_stops",
        )
    }
    dup, shared = [], defaultdict(set)
    for key, objects in lists.items():
        ids = Counter(o["id"] for o in objects if "id" in o)
        dup += [f"{key}:{i}" for i, n in ids.items() if n > 1]
        for i in ids:
            shared[i].add(key)
    shared = {i: sorted(k) for i, k in shared.items() if len(k) > 1}
    rep.add(
        "regions",
        "stable_ids",
        "pass" if not dup else "fail",
        dict(
            objects=sum(len(v) for v in lists.values()),
            duplicates=dup[:20],
            counts={k: len(v) for k, v in lists.items()},
            cross_references=len(shared),
            cross_reference_examples=dict(list(shared.items())[:6]),
        ),
        "unique within each list; one source object may appear as a point of interest and as its footprint or region",
    )
    invalid, no_kind = [], []
    for key in ("buildings", "open_areas", "verified_parking_regions", "parking_lots"):
        for o in lists[key]:
            if "polygon" in o and not Polygon(o["polygon"]).is_valid:
                invalid.append(o["id"])
            if key != "verified_parking_regions" and not o.get("kind"):
                no_kind.append(o["id"])
    rep.add(
        "regions",
        "boundaries_and_usage",
        "pass" if not invalid and not no_kind else "fail",
        dict(invalid_polygons=invalid[:20], without_usage=no_kind[:20]),
    )
    from awr.life.civil.itineraries import LIFE_FACILITY_PURPOSES

    poi_kinds = Counter(p["kind"] for p in lists["pois"]) + Counter(a["kind"] for a in lists["open_areas"])
    place_kinds = Counter(p["kind"] for p in places or [])
    mapped = sorted(set(LIFE_FACILITY_PURPOSES))
    rep.add(
        "regions",
        "activity_usage_table",
        "info",
        dict(
            table="awr.life.civil.itineraries.LIFE_FACILITY_PURPOSES",
            mapped_kinds=mapped,
            semantic_kinds=dict(poi_kinds),
            place_kinds=dict(place_kinds),
            semantic_kinds_without_activity=sorted(set(poi_kinds) - set(mapped)),
            mapped_kinds_without_place=sorted(set(mapped) - set(place_kinds)),
        ),
        "usage of a region decides which activities it admits; kinds without an entry admit none",
    )
    if sup is not None:
        rows = []
        for area in lists["open_areas"]:
            pts = polygon_lattice(Polygon(area["polygon"]))
            share = float(sup.at(pts, "person").mean())
            rows.append(
                dict(
                    id=area["id"],
                    kind=area["kind"],
                    area_m2=round(Polygon(area["polygon"]).area, 1),
                    person_supported_share=round(share, 4),
                )
            )
        bad = [r["id"] for r in rows if r["person_supported_share"] < tol("region_supported_share")]
        rep.add(
            "regions",
            "open_areas_supported",
            "pass" if not bad else "fail",
            dict(rows=rows, below=bad),
            "a mapped open area offers measured walkable ground",
        )
        rows = []
        for region in lists["verified_parking_regions"] + lists["parking_lots"]:
            poly = Polygon(region["polygon"])
            pts = polygon_lattice(poly)
            rows.append(
                dict(
                    id=region["id"],
                    area_m2=round(poly.area, 1),
                    vehicle_supported_share=round(float(sup.at(pts, "vehicle").mean()), 4),
                    verified=region.get("verified"),
                )
            )
        bad = [r["id"] for r in rows if r["vehicle_supported_share"] < tol("region_supported_share")]
        rep.add(
            "regions",
            "parking_regions_supported",
            ("pass" if not bad else "fail") if rows else "missing",
            dict(rows=rows, below=bad),
            "a parking region offers measured drivable ground",
        )
        shares = []
        for b in lists["buildings"]:
            pts = polygon_lattice(Polygon(b["polygon"]))
            shares.append((float(sup.at(pts, "person").mean()), b["id"]))
        flagged = sorted(((round(s, 3), i) for s, i in shares if s > tol("building_walkable_share")), reverse=True)
        rep.add(
            "regions",
            "building_footprints",
            ("pass" if not flagged else "fail") if shares else "missing",
            dict(
                buildings=len(shares),
                median_walkable_share=round(float(np.median([s for s, _ in shares])), 4) if shares else None,
                mostly_walkable=flagged[:30],
                mostly_walkable_count=len(flagged),
            ),
            "a building footprint should not be walkable ground",
        )
    parks = [Polygon(a["polygon"]) for a in lists["open_areas"] if a.get("kind") == "park"]
    lots = [(r["id"], Polygon(r["polygon"])) for r in lists["verified_parking_regions"] + lists["parking_lots"]]
    overlap = [
        dict(parking=i, overlap_m2=round(sum(p.intersection(poly).area for p in parks), 1))
        for i, poly in lots
        if any(p.intersects(poly) for p in parks)
    ]
    rep.add(
        "regions",
        "park_not_parking",
        "pass" if not any(o["overlap_m2"] > 1 for o in overlap) else "fail",
        overlap,
        "a park is not a parking area",
    )
    named_roads = sum(1 for r in lists["roads"] if r.get("name"))
    districts = []
    if pkg.has("world", "semantic/zones.geojson"):
        districts = [
            f["properties"].get("zone_id")
            for f in pkg.json("world", "semantic/zones.geojson").get("features", [])
            if f["properties"].get("kind") not in ("border",)
        ]
    declared = (pkg.features.get("regions") or {}).get("public_regions", [])
    status, definition, error = ("pass" if declared else "missing"), None, None
    if "public_regions" in pkg.layers:
        # The separately bound layer is what the runtime consumes. Do not fall
        # back to legacy declarations if that layer is missing or invalid.
        declared = []
        if not (pkg.has("public_regions") and pkg.has("world", "world.json") and pkg.has("semantics")):
            status, error = "unknown", "Public regions or their selected world/semantics are unavailable"
        else:
            from awr.life.civil.public_regions import load

            try:
                compiled = load(pkg.path("public_regions"), world=pkg.path("world"), semantics=pkg.path("semantics"))
                declared, definition = compiled["anchors"], compiled["definition"]
                status = "pass"
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                status, error = "fail", str(exc)
    rep.add(
        "regions",
        "public_regions",
        status,
        dict(
            named_roads=named_roads,
            roads=len(lists["roads"]),
            world_zones=districts,
            declared=declared,
            definition=definition,
            error=error,
        ),
        "Public query definitions checked with the runtime loader when a layer is supplied. "
        "Definitions do not certify land use, visible inventory, camera feasibility or task admission.",
    )


def check_passage(pkg, sup, sem, places_doc, rep, vehicle_width_m):
    places = (places_doc or {}).get("places", [])
    if places_doc:
        net = places_doc.get("network", {})
        by_comp = Counter(p.get("component") for p in places)
        top = by_comp.most_common(1)[0] if by_comp else (None, 0)
        rep.add(
            "passage",
            "person_network",
            "info",
            dict(
                free_area_m2=net.get("free_area_m2"),
                components=net.get("components"),
                largest_component_m2=net.get("largest_component_m2"),
                places=len(places),
                places_in_largest_place_component=top[1],
                place_components=len(by_comp),
            ),
            net.get("semantics", ""),
        )
        verified = sum(bool(p.get("verified_entrance")) for p in places)
        rep.add(
            "passage",
            "facility_entrances",
            "info",
            dict(
                places=len(places),
                verified_entrances=verified,
                functions=dict(Counter(p.get("function") for p in places)),
            ),
            "approach points beside mapped facilities; outdoor stays need no indoor model, doors are not claimed",
        )
    elif "places" in pkg.layers:
        rep.add("passage", "person_network", "unknown", None, "places layer not available here")
    else:
        rep.add("passage", "person_network", "missing", None, "no places layer")
    if sup is not None:
        from scipy.ndimage import label

        lab, n = label(np.asarray(sup.walk["person"]))
        sizes = np.bincount(lab.ravel())[1:] * sup.cell**2 if n else np.zeros(0)
        rep.add(
            "passage",
            "person_support_components",
            "info",
            dict(
                components=int(n),
                over_100_m2=int((sizes >= 100).sum()),
                largest_m2=round(float(sizes.max()), 1) if n else 0.0,
                largest_share=round(float(sizes.max() / sizes.sum()), 4) if n else 0.0,
            ),
            "connected walkable support before body clearance; standing alone does not imply reachability",
        )
    # Registered semantic graphs are separate derived layers; never mutate the world package.
    graph = (
        (pkg.json("road_graph") if pkg.has("road_graph") else None)
        if "road_graph" in pkg.layers
        else (
            pkg.json("world", "geometry/nav/graph.geojson") if pkg.has("world", "geometry/nav/graph.geojson") else None
        )
    )
    rep.add(
        "passage",
        "road_graph_source",
        "info",
        str(pkg.path("road_graph"))
        if "road_graph" in pkg.layers
        else str(pkg.path("world", "geometry/nav/graph.geojson")),
    )
    edges = [
        f
        for f in (graph or {}).get("features", [])
        if f.get("properties", {}).get("feature") == "edge" and "vehicle" in (f["properties"].get("agents") or [])
    ]
    rep.add(
        "passage",
        "vehicle_navmesh",
        "pass" if pkg.has("world", "geometry/nav/vehicle.json") else "missing",
        None,
        "Recast vehicle navigation mesh of the world package",
    )
    if not edges:
        status = "missing" if graph is not None or "world" not in pkg.layers else "unknown"
        rep.add("passage", "road_network", status, None, "no vehicle edges in the world navigation graph")
        rep.add("passage", "road_support", status, None, "no road network to measure")
        return None
    widths = Counter(round(float(e["properties"].get("width_m", 0)), 2) for e in edges)
    oneway = Counter(bool(e["properties"].get("oneway")) for e in edges)
    rep.add(
        "passage",
        "road_attributes",
        "info",
        dict(
            edges=len(edges),
            length_m=round(sum(float(e["properties"].get("length_m", 0)) for e in edges), 1),
            width_values=dict(widths.most_common(6)),
            oneway=dict((str(k), v) for k, v in oneway.items()),
            highway=dict(Counter(e["properties"].get("highway") for e in edges).most_common(8)),
            junction_rules="none mapped (no signals or priority signs)",
        ),
        "widths repeated across most edges are default annotations, not measured widths",
    )
    if sup is None:
        rep.add("passage", "road_support", "unknown", None, "support grids not available here")
        return None
    q = Raster(pkg.path("world"), "quality_2m") if pkg.has("world", "geometry/terrain/quality_2m.json") else None
    shifts = np.arange(-3, 3.5, 1.0)
    rows, reasons = [], Counter()
    all_shift = np.zeros(len(shifts))
    total = 0
    probe = np.arange(-6.0, 6.001, 0.25)
    widths_measured = []
    for e in edges:
        coords = e["geometry"]["coordinates"]
        pts, nrm = line_samples(coords, 1.0)
        if not len(pts):
            continue
        shares = np.array([sup.at(pts + s * nrm, "vehicle").mean() for s in shifts])
        centre = sup.at(pts, "vehicle")
        all_shift += shares * len(pts)
        total += len(pts)
        miss = pts[~centre]
        if len(miss):
            person = sup.at(miss, "person")
            bits = q.at(miss, fill=1) if q is not None else np.zeros(len(miss), np.uint8)
            for b, pe in zip(bits, person, strict=False):
                names = quality_names(b)
                if any(n in NO_RECONSTRUCTION for n in names):
                    reasons["no_reconstruction"] += 1
                elif any(n in LOW_CONFIDENCE for n in names):
                    reasons["low_confidence_reconstruction"] += 1
                elif pe:
                    reasons["walkable_not_drivable"] += 1
                else:
                    reasons["no_measured_ground"] += 1
        sub = slice(None, None, 5)
        across = pts[sub, None, :] + probe[None, :, None] * nrm[sub, None, :]
        ok = sup.at(across.reshape(-1, 2), "vehicle").reshape(across.shape[:2])
        mid = len(probe) // 2
        for row in ok:
            if not row[mid]:
                widths_measured.append(0.0)
                continue
            lo = mid
            while lo > 0 and row[lo - 1]:
                lo -= 1
            hi = mid
            while hi < len(row) - 1 and row[hi + 1]:
                hi += 1
            widths_measured.append((hi - lo + 1) * 0.25)
        p = e["properties"]
        rows.append(
            dict(
                id=p.get("edge_id", e.get("id")),
                from_node=p.get("from_node"),
                to_node=p.get("to_node"),
                length_m=round(float(p.get("length_m", 0)), 1),
                centre_supported=round(float(centre.mean()), 3),
                best_shift_m=float(shifts[int(np.argmax(shares))]),
                best_share=round(float(shares.max()), 3),
            )
        )
    shift_share = all_shift / max(total, 1)
    centre_share = float(shift_share[list(shifts).index(0.0)])
    gain = float(shift_share.max() - centre_share)
    length = sum(r["length_m"] for r in rows)
    drivable = [r for r in rows if r["centre_supported"] >= tol("road_supported_share")]
    lane = vehicle_width_m + 2 * tol("vehicle_side_margin_m")
    w = np.asarray(widths_measured)
    status = "pass" if centre_share >= tol("road_supported_share") else "fail"
    rep.add(
        "passage",
        "road_support",
        status,
        dict(
            samples=total,
            centre_supported_share=round(centre_share, 4),
            drivable_edges=len(drivable),
            edges=len(rows),
            drivable_length_m=round(sum(r["length_m"] for r in drivable), 1),
            length_m=round(length, 1),
            unsupported_reasons=dict(reasons),
        ),
        "mapped road centre lines against measured drivable support (1 m samples)",
    )
    rep.add(
        "passage",
        "road_registration",
        "pass" if gain < tol("road_shift_gain") else "fail",
        dict(shift_m=shifts.tolist(), supported_share=[round(float(s), 4) for s in shift_share], gain=round(gain, 4)),
        "support share of the centre lines shifted sideways; a large gain away from zero means misregistration",
    )
    rep.add(
        "passage",
        "road_width_measured",
        "info",
        dict(
            samples=int(len(w)),
            vehicle_width_m=round(vehicle_width_m, 3),
            single_lane_m=round(lane, 3),
            two_way_m=round(2 * lane, 3),
            median_m=round(float(np.median(w)), 2) if len(w) else None,
            share_single_lane=round(float((w >= lane).mean()), 4) if len(w) else None,
            share_two_way=round(float((w >= 2 * lane).mean()), 4) if len(w) else None,
        ),
        "contiguous drivable width across the centre line (probe +-6 m, 0.25 m); passing space needs two lanes",
    )
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def components(subset):
        parent.clear()
        for r in subset:
            parent[find(r["from_node"])] = find(r["to_node"])
        size = defaultdict(float)
        for r in subset:
            size[find(r["from_node"])] += r["length_m"]
        return sorted(size.values(), reverse=True)

    mapped, supported = components(rows), components(drivable)
    rep.add(
        "passage",
        "road_connectivity",
        "info",
        dict(
            mapped_components=len(mapped),
            mapped_largest_m=round(mapped[0], 1) if mapped else 0.0,
            drivable_components=len(supported),
            drivable_largest_m=round(supported[0], 1) if supported else 0.0,
            drivable_component_lengths_m=[round(x, 1) for x in supported[:10]],
        ),
        "undirected graph-length diagnostic, NOT a traversable route: 90% sample support can hide gaps, "
        "and this connectivity ignores direction, junction clearance and whole-body sweeps",
    )
    return dict(rows=rows, drivable=drivable)


def check_entities(pkg, runs, rep):
    catalog = read_json(pkg.root / "assets/catalog.json") if (pkg.root / "assets/catalog.json").exists() else None
    if catalog is None:
        rep.add("entities", "asset_catalog", "missing", None, "assets/catalog.json not found")
        return None
    by_kind = defaultdict(list)
    for a in catalog["assets"]:
        by_kind[a.get("kind")].append(a)
    incomplete = [a["id"] for a in catalog["assets"] if not a.get("dimensions_m") or not a.get("npz_sha256")]
    rep.add(
        "entities",
        "asset_catalog",
        "pass" if not incomplete else "fail",
        dict(
            schema=catalog.get("schema"),
            assets={k: [a["id"] for a in v] for k, v in by_kind.items()},
            incomplete=incomplete,
        ),
        "every asset has dimensions and a content hash",
    )
    declared = (pkg.features.get("entities") or {}).get("classes")
    rep.add(
        "entities",
        "motion_modes",
        "pass" if declared else "missing",
        declared,
        "per entity class: motion modes and the backend that owns its pose",
    )
    width = max((float(a["dimensions_m"][1]) for a in by_kind.get("vehicle", []) if a.get("dimensions_m")), default=2.0)
    for run in runs:
        ents = run.get("entities")
        if ents is None:
            continue
        people = [e for e in ents if e.get("class") == "person"]
        cars = [e for e in ents if e.get("class") == "vehicle"]
        plates = Counter((e.get("appearance") or {}).get("plate_text") for e in cars)
        attrs = Counter(k for e in people for k in (e.get("appearance") or {}) if k.endswith("_name"))
        dup_ids = [i for i, n in Counter(e["id"] for e in ents).items() if n > 1]
        rep.add(
            "entities",
            f"identity[{run['id']}]",
            "pass" if not dup_ids and all(n == 1 for n in plates.values()) else "fail",
            dict(
                entities=len(ents),
                people=len(people),
                vehicles=len(cars),
                duplicate_ids=dup_ids,
                duplicate_plates=[p for p, n in plates.items() if n > 1][:10],
                synthetic_plates=sum(bool((e.get("appearance") or {}).get("plate_synthetic")) for e in cars),
                person_attributes=dict(attrs),
                vehicle_types=dict(Counter(e.get("vtype") for e in cars)),
                vehicle_colours=dict(Counter((e.get("appearance") or {}).get("colour_name") for e in cars)),
            ),
            "stable identities; plates unique; appearance attributes recorded for identification",
        )
    return width


def check_resources(pkg, sup, sem, places, runs, rep, vehicle_size):
    from shapely.geometry import Polygon

    kinds = Counter(p["kind"] for p in places or [])
    rep.add(
        "resources",
        "facility_candidates",
        "info" if places else "missing",
        dict(kinds),
        "level candidate: mapped places",
    )
    for run in runs:
        slots = run.get("places")
        if slots is not None:
            per = Counter(s["kind"] for s in slots)
            groups = defaultdict(set)
            for s in slots:
                groups[s["kind"]].add(s.get("group"))
            rep.add(
                "resources",
                f"facility_slots[{run['id']}]",
                "info",
                dict(slots=dict(per), sites={k: len(v) for k, v in groups.items()}),
                "level static: standing slots admitted by the adapter (support, clearance, spacing)",
            )
        visits = run.get("visits")
        if visits is not None:
            rep.add(
                "resources",
                f"facility_use[{run['id']}]",
                "info",
                visits,
                "level dynamic: completed stays and peak concurrent occupancy per kind in the recorded run",
            )
    if sem is not None:
        rows = []
        foot = float(vehicle_size[0] * vehicle_size[1]) if vehicle_size is not None else 10.0
        for region in sem.get("verified_parking_regions", []) + sem.get("parking_lots", []):
            poly = Polygon(region["polygon"])
            pts = polygon_lattice(poly, 0.5)
            drivable = float(sup.at(pts, "vehicle").mean()) * poly.area if sup is not None else None
            rows.append(
                dict(
                    id=region["id"],
                    area_m2=round(poly.area, 1),
                    drivable_m2=round(drivable, 1) if drivable is not None else None,
                    berths_upper_bound=int(drivable // (2 * foot)) if drivable is not None else None,
                )
            )
        rep.add(
            "resources",
            "parking_regions",
            "info" if rows else "missing",
            dict(rows=rows, curb_slots=len(sem.get("curb_slots", []))),
            "level upper_bound: drivable area / (2 x body footprint), half the area kept for aisles; not a capacity",
        )
    if "driving_map" in pkg.layers and pkg.has("driving_map"):
        dm = pkg.json("driving_map")
        comps = [dict(nodes=c.get("nodes"), berths=len(c.get("berths", []))) for c in dm.get("components", [])]
        berths = [n for n in dm.get("nodes", {}) if str(n).startswith("berth:")]
        rep.add(
            "resources",
            "berths",
            "info",
            dict(
                schema=dm.get("schema"),
                berths=len(berths),
                components_with_berths=comps,
                release_evidence=dm.get("release_evidence"),
            ),
            "level candidate: berths of the driving map and their returnable road components",
        )
    if "activity_access" in pkg.layers and pkg.has("activity_access"):
        aa = pkg.json("activity_access")
        gs = aa.get("geometry_screen", {})
        rep.add(
            "resources",
            "home_berth_access",
            "info",
            dict(
                home_berths=gs.get("home_berths"),
                qualifying_berths=gs.get("qualifying_berths"),
                distinct_home_berth_matching=gs.get("distinct_home_berth_matching"),
                side_choices=len(aa.get("side_choices", [])),
            ),
            "level static: berths whose car side reaches homes and two facilities by walking (driver chains)",
        )
    for run in runs:
        ents = run.get("entities")
        if not ents or sup is None:
            continue
        cars = [e for e in ents if e.get("class") == "vehicle"]
        xy = np.asarray([e.get("spawn_xy_m", [np.nan, np.nan]) for e in cars], float)
        on = sup.at(xy, "vehicle") if len(xy) else np.zeros(0, bool)
        rep.add(
            "resources",
            f"parked_vehicles[{run['id']}]",
            "pass" if on.all() else "fail",
            dict(
                vehicles=len(cars),
                allocation=dict(Counter(e.get("allocation_mode") for e in cars)),
                centre_on_drivable=int(on.sum()),
            ),
            "historical allocation: centre-point support only; does not verify body footprint, access or joint capacity",
        )


def check_versions(pkg, rep):
    rows, bad, unhashed, absent_files = [], [], [], []
    for name, layer in pkg.layers.items():
        p = pkg.path(name)
        row = dict(layer=name, path=layer["path"], present=p.exists())
        if layer.get("sha256"):
            if not p.is_file():
                absent_files.append(name)
            if p.exists() and p.is_file():
                row["sha256_match"] = sha256(p) == layer["sha256"]
                if not row["sha256_match"]:
                    bad.append(name)
        for rel, digest in (layer.get("array_sha256") or {}).items():
            if (p / rel).exists():
                ok = sha256(p / rel) == digest
                row.setdefault("arrays_match", {})[rel] = ok
                if not ok:
                    bad.append(f"{name}/{rel}")
            else:
                absent_files.append(f"{name}/{rel}")
        if name == "world" and (p / "world.json").exists():
            row["content_version"] = json.loads((p / "world.json").read_text()).get("contentVersion")
        if not layer.get("sha256") and not layer.get("array_sha256") and "content_version" not in row:
            unhashed.append(name)
        rows.append(row)
    absent = sorted(set([r["layer"] for r in rows if not r["present"]] + absent_files))
    status = "fail" if bad else ("unknown" if absent else ("missing" if unhashed else "pass"))
    rep.add(
        "versions",
        "layer_digests",
        status,
        dict(layers=rows, mismatched=bad, absent=absent, unhashed=unhashed),
        "every layer bound by a digest or content version; absent layers cannot be checked here",
    )
    derived = []
    support_meta = pkg.path("support", "meta.json")
    if support_meta is not None and support_meta.exists() and "world" in pkg.layers:
        built_on = json.loads(support_meta.read_text()).get("world")
        derived.append(
            dict(
                layer="support",
                built_on=built_on,
                declared=pkg.layers["world"]["path"],
                match=built_on in (None, pkg.layers["world"]["path"]),
            )
        )
    if pkg.has("places"):
        built_on = pkg.json("places").get("nav")
        if "support" in pkg.layers:
            derived.append(
                dict(
                    layer="places",
                    built_on=built_on,
                    declared=pkg.layers["support"]["path"],
                    match=built_on in (None, pkg.layers["support"]["path"]),
                )
            )
    from .dependencies import declared_sources

    derived.extend(declared_sources(pkg.layers, pkg.root))
    stale = [d for d in derived if not d["match"]]
    rep.add(
        "versions",
        "derived_layer_sources",
        "fail" if stale else "pass",
        derived,
        "a derived layer names the layer it was computed from; a different source version makes its stored "
        "components and admissions stale (the adapter re-admits places on the declared support at run time)",
    )
    caps = pkg.m.get("capability_status") or {}
    undeclared = [c for c in CAPABILITIES if c not in caps]
    rep.add(
        "versions",
        "capability_declaration",
        "pass" if not undeclared else "missing",
        dict(declared=sorted(caps), undeclared=undeclared),
        "each standard capability declared supported, limited, unsupported or unknown with its reason",
    )


# Evidence ------------------------------------------------------------------------------------------------------


def load_run(pkg, entry):
    """A recorded engine run (output directory of awr.life.civil.grounded) bound to this package's layers."""
    d = pkg.root / entry["path"]
    run = dict(id=entry["id"], path=entry["path"], present=d.exists())
    if not d.exists():
        return run
    audit = d / "life-audit.json"
    if audit.exists():
        a = json.loads(audit.read_text())
        gates = (a.get("gates") or {}).get("gates", {})
        run["audit"] = dict(
            duration_s=a.get("duration_s"),
            counts=a.get("counts"),
            person_state_share=(a.get("person") or {}).get("state_share"),
            longest_stall_s=max((s["duration_s"] for s in a.get("stalls", [])), default=0.0),
            gates_failed=sorted(k for k, g in gates.items() if not g.get("pass_")),
            gates_passed=sorted(k for k, g in gates.items() if g.get("pass_")),
            tape_content_sha256=a.get("tape_content_sha256"),
        )
    if (d / "entities.json").exists():
        run["entities"] = json.loads((d / "entities.json").read_text()).get("entities", [])
    if (d / "activity-places.json").exists():
        run["places"] = json.loads((d / "activity-places.json").read_text())
    if (d / "activity-events.jsonl").exists() and run.get("places") is not None:
        kind = {p["id"]: p["kind"] for p in run["places"]}
        group = {p["id"]: p.get("group") for p in run["places"]}
        done, live, peak = Counter(), Counter(), defaultdict(int)
        for line in (d / "activity-events.jsonl").read_text().splitlines():
            ev = json.loads(line)
            place = ev.get("place")
            if place not in kind:
                continue
            if ev.get("kind") == "visit_started":
                live[group[place]] += 1
                peak[kind[place]] = max(peak[kind[place]], live[group[place]])
            elif ev.get("kind") in ("visit_completed", "visit_interrupted"):
                live[group[place]] -= 1
                if ev["kind"] == "visit_completed":
                    done[kind[place]] += 1
        run["visits"] = dict(completed=dict(done), peak_concurrent_per_site=dict(peak))
    inputs = d.parent / (d.name.split(".")[0] + ".initial-inputs.json")
    if inputs.exists():
        prov = json.loads(inputs.read_text())
        text = json.dumps(prov)
        run["bound_layers"] = {
            name: (layer.get("sha256") in text) for name, layer in pkg.layers.items() if layer.get("sha256")
        }
    return run


def load_documents(pkg):
    docs = []
    for entry in pkg.m.get("evidence", []):
        if entry.get("kind") == "run":
            continue
        p = pkg.root / entry["path"]
        row = dict(
            id=entry["id"], kind=entry.get("kind"), path=entry["path"], claim=entry.get("claim"), present=p.exists()
        )
        if p.exists():
            row["sha256_match"] = (sha256(p) == entry["sha256"]) if entry.get("sha256") else None
            doc = read_json(p)
            if isinstance(doc, dict):
                row["schema"] = doc.get("schema")
                for key in (
                    "completed_legs",
                    "expected_legs",
                    "persistent_pass",
                    "natural_completed_chains",
                    "completed_chains",
                    "simulated_s",
                    "release_evidence",
                    "admitted",
                ):
                    if key in doc:
                        row[key] = doc[key]
        docs.append(row)
    return docs


# Capabilities --------------------------------------------------------------------------------------------------


def evaluate(pkg, rep, runs, docs, road):
    s = rep.status
    out = {}

    def done(name, status, requirements, evidence, reasons):
        out[name] = dict(
            status=status,
            admitted=False,
            assessment_scope="static_diagnostic",
            requirements=requirements,
            evidence=evidence,
            reasons=reasons,
            declared=(pkg.m.get("capability_status") or {}).get(name),
        )

    # Runtime evidence admission is deliberately not implemented in this diagnostic importer.
    runs, docs = [], []
    walked = []
    req = dict(
        walking_space=s("passage", "person_network"),
        destinations=s("resources", "facility_candidates"),
        support=s("geometry", "support_contract"),
        frame=s("frame", "layers_in_support_extent"),
        registration=s("frame", "feature_registration"),
        grids=s("geometry", "support_grids"),
        reconstruction=s("geometry", "support_over_reconstruction"),
    )
    dynamic = [
        dict(
            run=r["id"],
            people=r["audit"]["counts"].get("person"),
            duration_s=r["audit"]["duration_s"],
            longest_stall_s=r["audit"]["longest_stall_s"],
            person_gates_failed=[g for g in r["audit"]["gates_failed"] if g.startswith("person")],
        )
        for r in walked
    ]
    if any(v in ("missing", "fail") for v in req.values()):
        status = "unsupported"
    elif dynamic:
        status = "supported"
    elif any(v == "unknown" for v in req.values()):
        status = "unknown"
    else:
        status = "static_only"
    slots = next(
        (
            rep.get("resources", f"facility_slots[{r['id']}]")
            for r in runs
            if rep.get("resources", f"facility_slots[{r['id']}]")
        ),
        None,
    )
    done(
        "walk_activity",
        status,
        req,
        dict(
            candidate=(rep.get("resources", "facility_candidates") or {}).get("value"),
            static=slots["value"] if slots else None,
            dynamic=dynamic,
        ),
        [] if status == "supported" else [f"{k}: {v}" for k, v in req.items() if v != "pass" and v != "info"],
    )
    road_status = s("passage", "road_support")
    reg = s("passage", "road_registration")
    conn = (rep.get("passage", "road_connectivity") or {}).get("value") or {}
    width = (rep.get("passage", "road_width_measured") or {}).get("value") or {}
    req = dict(
        road_network=s("passage", "road_attributes"),
        road_support=road_status,
        registration=reg,
        junction_rules="missing",
        vehicle_navmesh=s("passage", "vehicle_navmesh"),
    )
    vdocs = [d for d in docs if d.get("kind") in ("vehicle_journeys", "vehicle_cycles")]
    if road is None:
        status, reasons = (
            ("unknown" if road_status == "unknown" else "unsupported"),
            ["no vehicle road network measured here"],
        )
    else:
        drivable = conn.get("drivable_largest_m", 0.0)
        reasons = [
            f"drivable centre-line share {(rep.get('passage', 'road_support') or {})['value']['centre_supported_share']}",
            f"largest drivable connected road {drivable} m",
            "no junction rules mapped: the engine must yield at junctions",
            f"two-way width share {width.get('share_two_way')}",
        ]
        status = "limited" if drivable > 0 and vdocs else ("static_only" if drivable > 0 else "unsupported")
    done("vehicle_traffic", status, req, dict(static=conn, dynamic=vdocs), reasons)
    park = rep.get("regions", "parking_regions_supported")
    parked = [rep.get("resources", f"parked_vehicles[{r['id']}]") for r in runs]
    parked = [p for p in parked if p]
    req = dict(parking_regions=s("regions", "parking_regions_supported"), berths=s("resources", "berths"))
    regions_ok = bool(park) and park["status"] == "pass"
    measured = bool(parked) and all(p["status"] == "pass" for p in parked)
    if measured:
        status = "supported" if regions_ok else "limited"
        reasons = [] if regions_ok else ["curb parking only: no verified parking region"]
    elif regions_ok:
        status, reasons = "static_only", ["parked vehicles not measured in a run"]
    else:
        status = "unknown" if park and park["status"] == "unknown" else "unsupported"
        reasons = ["no verified parking region and no measured parked vehicles"]
    done(
        "parking",
        status,
        req,
        dict(
            upper_bound=(rep.get("resources", "parking_regions") or {}).get("value"),
            candidate=(rep.get("resources", "berths") or {}).get("value"),
            dynamic=[p["value"] for p in parked],
        ),
        reasons,
    )
    berths = (rep.get("resources", "berths") or {}).get("value")
    cycles = [d for d in docs if d.get("kind") == "vehicle_cycles"]
    status = (
        "limited"
        if cycles and any(d.get("persistent_pass") for d in cycles)
        else "static_only"
        if berths and any(c["berths"] >= 2 for c in berths["components_with_berths"])
        else "unsupported"
    )
    done(
        "recurrent_trips",
        status,
        dict(berths=s("resources", "berths"), road_support=road_status),
        dict(candidate=berths, dynamic=cycles),
        [
            "returnable berth components exist; recurrent trips measured only for the vehicles listed"
            if status == "limited"
            else "no driving map with returnable berth components"
            if status == "unsupported"
            else "not measured in a run"
        ],
    )
    access = (rep.get("resources", "home_berth_access") or {}).get("value")
    chains = [d for d in docs if d.get("kind") == "driver_chain"]
    natural = sum(int(d.get("natural_completed_chains") or 0) for d in chains)
    status = (
        "limited"
        if natural
        else "unsupported"
        if access is not None and not access.get("qualifying_berths")
        else "unknown"
        if access is None and "activity_access" in pkg.layers
        else "unsupported"
    )
    done(
        "driver_chain",
        status,
        dict(home_berth_access=s("resources", "home_berth_access")),
        dict(static=access, dynamic=chains),
        [
            f"{natural} natural chains measured; static home-berth matching {access and access.get('distinct_home_berth_matching')}"
        ]
        if access is not None or chains
        else ["no activity access compiled"],
    )
    gates = (pkg.features.get("boundary") or {}).get("gates")
    done(
        "boundary_entry_exit",
        "static_only" if gates else "unsupported",
        dict(gates="pass" if gates else "missing"),
        dict(candidate=gates),
        [] if gates else ["no legal entry/exit gates or lifecycle rules mapped"],
    )
    return out


def evaluate_queries(queries, caps, rep, pkg, places, runs):
    rows = []
    place_kinds = Counter(p["kind"] for p in places or [])
    slot_kinds = Counter()
    for run in runs:
        for s in run.get("places") or []:
            slot_kinds[s["kind"]] += 1
    catalog = (rep.get("entities", "asset_catalog") or {}).get("value") or {}
    vehicle_assets = set((catalog.get("assets") or {}).get("vehicle", []))
    attrs = Counter()
    for run in runs:
        for e in run.get("entities") or []:
            attrs.update(k for k in (e.get("appearance") or {}))
    regions = rep.get("regions", "public_regions") or {}
    public = (
        {r.get("kind") for r in regions.get("value", {}).get("declared", [])}
        if regions.get("status") == "pass"
        else set()
    )
    for q in queries:
        need = q.get("requires", {})
        items = []
        for c in need.get("capabilities", []):
            st = caps.get(c, {}).get("status", "unknown")
            items.append(
                dict(
                    item=f"capability:{c}",
                    status=st,
                    ok=st in ("supported", "limited", "static_only") if c != "walk_activity" else st == "supported",
                )
            )
        for k in need.get("place_kinds", []):
            items.append(
                dict(
                    item=f"place:{k}",
                    status=f"{place_kinds.get(k, 0)} candidates, {slot_kinds.get(k, 0)} slots",
                    ok=slot_kinds.get(k, 0) > 0 or (not slot_kinds and place_kinds.get(k, 0) > 0),
                )
            )
        for r in need.get("public_regions", []):
            items.append(
                dict(
                    item=f"region:{r}",
                    status="declared" if r in public else "overlay_required",
                    ok=r in public,
                    overlay=r not in public,
                )
            )
        for a in need.get("entity_attributes", []):
            items.append(
                dict(
                    item=f"attribute:{a}",
                    status="recorded" if attrs.get(a) else "not_recorded",
                    ok=bool(attrs.get(a)),
                    overlay=not attrs.get(a),
                )
            )
        for v in need.get("vehicle_assets", []):
            items.append(
                dict(
                    item=f"asset:{v}", status="in_catalog" if v in vehicle_assets else "absent", ok=v in vehicle_assets
                )
            )
        blocked = [i for i in items if not i["ok"] and not i.get("overlay")]
        overlay = [i for i in items if not i["ok"] and i.get("overlay")]
        rows.append(
            dict(
                id=q["id"],
                type=q["type"],
                text=q["text"],
                items=items,
                status="blocked" if blocked else "overlay_required" if overlay else "feature_candidates_present",
                admitted=False,
                witness_required=True,
            )
        )
    return rows


# Report --------------------------------------------------------------------------------------------------------


def source_inventory(pkg):
    """Hash inspected static inputs; avoid recursively hashing large rendering assets.

    This is a diagnostic scope, not a claim that a complete world/asset release was verified.
    """
    paths = {pkg.manifest_path, pkg.root / "assets/catalog.json"}
    for name in pkg.layers:
        path = pkg.path(name)
        if path.is_file():
            paths.add(path)
        elif name == "support":
            paths.update(path / x for x in ("meta.json", "ground.npy", "walk_person.npy", "walk_vehicle.npy"))
        elif name == "world":
            paths.update(
                path / x
                for x in (
                    "world.json",
                    "coordinate.json",
                    "geometry/nav/graph.geojson",
                    "geometry/nav/vehicle.json",
                    "semantic/quality.geojson",
                    "semantic/zones.geojson",
                )
            )
            for name in ("dsm_2m", "quality_2m"):
                meta = path / "geometry/terrain" / f"{name}.json"
                paths.add(meta)
                if meta.is_file():
                    paths.add(meta.parent / read_json(meta)["href"])
        else:
            paths.add(path)
    return {str(p): sha256(p) if p.is_file() else None for p in sorted(paths)}


def validate(manifest, root=".", queries=None):
    pkg = Package(manifest, root)
    inputs_before = source_inventory(pkg)
    rep = Report()
    sup = Support(pkg.path("support")) if pkg.has("support", "meta.json") else None
    sem = pkg.json("semantics") if pkg.has("semantics") else None
    places_doc = pkg.json("places") if pkg.has("places") else None
    places = (places_doc or {}).get("places")
    runs = [load_run(pkg, e) for e in pkg.m.get("evidence", []) if e.get("kind") == "run"]
    docs = load_documents(pkg)
    check_frame(pkg, sup, sem, places, rep)
    check_geometry(pkg, sup, rep)
    check_regions(pkg, sup, sem, places, rep)
    width = check_entities(pkg, runs, rep)
    vehicle_size = None
    catalog_path = pkg.root / "assets/catalog.json"
    if catalog_path.exists():
        dims = [
            a["dimensions_m"]
            for a in read_json(catalog_path)["assets"]
            if a.get("kind") == "vehicle" and a.get("dimensions_m")
        ]
        vehicle_size = np.median(np.asarray(dims, float), 0) if dims else None
    road = check_passage(pkg, sup, sem, places_doc, rep, width or 2.0)
    check_resources(pkg, sup, sem, places, runs, rep, vehicle_size)
    check_versions(pkg, rep)
    caps = evaluate(pkg, rep, [], [], road)
    inconsistent = [
        name
        for name in ("layer_digests", "derived_layer_sources")
        if (rep.get("versions", name) or {}).get("status") == "fail"
    ]
    if inconsistent:
        for cap in caps.values():
            cap["status"] = "unknown"
            cap["reasons"].append("inconsistent package inputs: " + ", ".join(inconsistent))
    query_rows = evaluate_queries(read_json(queries)["queries"], caps, rep, pkg, places, []) if queries else None
    if inconsistent:
        for row in query_rows or []:
            row["status"] = "blocked"
            row["items"].append(
                dict(item="environment_input_consistency", status="fail: " + ", ".join(inconsistent), ok=False)
            )
    for run in runs:
        run.pop("entities", None)
        run.pop("places", None)
    inputs_after = source_inventory(pkg)
    changed = sorted(
        k for k in inputs_before.keys() | inputs_after.keys() if inputs_before.get(k) != inputs_after.get(k)
    )
    rep.add("versions", "inputs_stable_during_measurement", "fail" if changed else "pass", changed)
    if changed:
        for cap in caps.values():
            cap["status"] = "unknown"
            cap["reasons"].append("source changed during measurement; rerun on immutable inputs")
        for row in query_rows or []:
            row["status"] = "blocked"
    summary = {f: dict(Counter(x["status"] for x in rep.findings if x["feature"] == f)) for f in FEATURES}
    report = dict(
        schema=REPORT_SCHEMA,
        admitted=False,
        assessment_scope="static_diagnostic",
        source_inventory=inputs_before,
        source_inventory_after=inputs_after if changed else None,
        historical_evidence_used_for_capabilities=False,
        environment=pkg.m.get("id"),
        manifest=str(manifest),
        manifest_sha256=sha256(pkg.manifest_path),
        tolerances={k: dict(value=v, reason=r) for k, (v, r) in TOLERANCES.items()},
        summary=summary,
        findings=rep.findings,
        capabilities=caps,
        evidence=dict(runs=runs, documents=docs),
    )
    if query_rows is not None:
        report["queries"] = query_rows
    return report


STATUS_ZH = {
    "pass": "通过",
    "fail": "不符合",
    "missing": "缺失",
    "unknown": "无法判定",
    "info": "记录",
    "supported": "支持",
    "limited": "有限支持",
    "static_only": "仅静态核验",
    "unsupported": "不支持",
    "feature_candidates_present": "存在候选条件，需任务见证",
    "overlay_required": "需场景覆盖层定义",
    "blocked": "受阻",
}
FEATURE_ZH = {
    "frame": "坐标与时间",
    "geometry": "几何与支撑",
    "regions": "区域语义",
    "passage": "通行能力",
    "entities": "实体能力",
    "resources": "资源与容量",
    "versions": "版本与能力声明",
}


def markdown(report):
    def short(v, n=160):
        s = json.dumps(v, ensure_ascii=False)
        return s if len(s) <= n else s[: n - 3] + "..."

    lines = [
        f"# 环境 feature 核验报告：{report['environment']}",
        "",
        f"清单：`{report['manifest']}`（sha256 `{report['manifest_sha256'][:12]}`）。由 `awr.life.envpack.validate` 生成，"
        "依据《环境 feature 接入原则 v1》。只做静态诊断，不授予运行或 query 准入；旧运行记录不用于当前能力认证。",
        "",
        "## 基础 feature",
        "",
        "| Feature | 检查 | 结论 | 测量值 |",
        "| --- | --- | --- | --- |",
    ]
    for f in report["findings"]:
        lines.append(
            f"| {FEATURE_ZH[f['feature']]} | `{f['check']}` | {STATUS_ZH.get(f['status'], f['status'])} | {short(f['value'])} |"
        )
    lines += ["", "## 能力评估", "", "| 能力 | 结论 | 必要条件 | 说明 |", "| --- | --- | --- | --- |"]
    for name, c in report["capabilities"].items():
        req = ", ".join(f"{k}={v}" for k, v in c["requirements"].items())
        lines.append(f"| `{name}` | {STATUS_ZH.get(c['status'], c['status'])} | {req} | {'；'.join(c['reasons'])} |")
    if report.get("queries"):
        lines += ["", "## 任务意图条件", "", "| 任务 | 类型 | 结论 | 未满足项 |", "| --- | --- | --- | --- |"]
        for q in report["queries"]:
            open_items = ", ".join(f"{i['item']}（{i['status']}）" for i in q["items"] if not i["ok"]) or "无"
            lines.append(f"| {q['id']} | {q['type']} | {STATUS_ZH.get(q['status'], q['status'])} | {open_items} |")
    lines += ["", "## 容差", "", "| 名称 | 值 | 理由 |", "| --- | --- | --- |"]
    for k, v in report["tolerances"].items():
        lines.append(f"| `{k}` | {v['value']} | {v['reason']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--queries", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="output prefix: writes <out>.json and <out>.md")
    a = ap.parse_args(argv)
    outputs = [Path(str(a.out) + ext) for ext in (".json", ".md")]
    if any(p.exists() for p in outputs):
        ap.error("output exists; choose a fresh prefix to preserve prior evidence")
    report = validate(a.manifest, a.root, a.queries)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(a.out) + ".json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float) + "\n")
    Path(str(a.out) + ".md").write_text(markdown(report))
    for f in FEATURES:
        print(f, report["summary"][f])
    for name, c in report["capabilities"].items():
        print(name, c["status"])
    for q in report.get("queries", []):
        print(q["id"], q["status"])


if __name__ == "__main__":
    main()
