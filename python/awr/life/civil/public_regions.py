"""Freeze public query areas independently of actor tracks and recording length.

This binds definitions, not land-use correctness, capacity or camera feasibility.
Semantic polygons retain their source geometry; authored districts are explicit
query boundaries and cannot fabricate a park, facility or legal parking area.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from awr.life.common.schema import schema_id

SCHEMA = "awr.life.public_regions.v1"
COLLECTIONS = {"open_areas", "parking_lots", "verified_parking_regions", "buildings"}


def _polygon(points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("Public region needs a finite 2D polygon")
    geometry = Polygon(p)
    if not geometry.is_valid or geometry.is_empty or geometry.area <= 0:
        raise ValueError("Public region polygon is invalid or empty")
    return p.tolist()


def load(path, *, world, semantics):
    """Compile a small author specification against the exact scene files."""
    region_bytes = Path(path).read_bytes()
    doc = json.loads(region_bytes)
    world_bytes = (Path(world) / "world.json").read_bytes()
    semantic_bytes = Path(semantics).read_bytes()
    if schema_id(doc.get("schema")) != SCHEMA:
        raise ValueError(f"{SCHEMA} required")
    for field, data in (("world_json_sha256", world_bytes), ("semantics_sha256", semantic_bytes)):
        if doc.get(field) != hashlib.sha256(data).hexdigest():
            raise ValueError(f"Public regions {field} differs from the selected scene")
    sem, wdoc = json.loads(semantic_bytes), json.loads(world_bytes)
    regions = doc.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("Explicit nonempty public regions required")
    anchors, sources, ids, labels = [], [], set(), set()
    for region in regions:
        ident, label = region.get("id"), region.get("label_zh")
        if (
            not isinstance(ident, str)
            or not ident.strip()
            or ident in ids
            or not isinstance(label, str)
            or not label.strip()
            or label in labels
        ):
            raise ValueError("Public region IDs and labels must be nonempty and unique")
        ids.add(ident)
        labels.add(label)
        source = region.get("source", {})
        if source.get("kind") == "semantic_polygon":
            collection = source.get("collection")
            if collection not in COLLECTIONS or "polygon" in region:
                raise ValueError("Select a semantic polygon without overriding its geometry")
            matches = [r for r in sem.get(collection, []) if r.get("id") == source.get("id")]
            if len(matches) != 1:
                raise ValueError("Public region source feature missing or ambiguous")
            feature = matches[0]
            polygon = _polygon(feature.get("polygon"))
            # Land use comes from the source, never the chosen display label.
            kind = (
                feature.get("kind")
                or {
                    "parking_lots": "parking",
                    "verified_parking_regions": "parking",
                    "buildings": "building",
                    "open_areas": "open_area",
                }[collection]
            )
            if "kind" in region and region["kind"] != kind:
                raise ValueError("Public region cannot relabel source land use")
        elif source.get("kind") == "authored_region":
            if not isinstance(source.get("rationale"), str) or not source["rationale"].strip():
                raise ValueError("Authored query boundary needs its rationale")
            if region.get("kind", "zone") != "zone":
                raise ValueError("Authored query boundary cannot assert facility or parking use")
            polygon, kind = _polygon(region.get("polygon")), "zone"
        else:
            raise ValueError("Public region needs an explicit source")
        anchors.append(dict(id=ident, label_zh=label, kind=kind, polygon=polygon, category="manual", profile="person"))
        sources.append(dict(id=ident, source=source))
    return dict(
        schema="awr.anchors.v1",
        world_id=wdoc["id"],
        anchors=sorted(anchors, key=lambda a: a["id"]),
        definition=dict(
            schema=SCHEMA,
            source_sha256=hashlib.sha256(region_bytes).hexdigest(),
            world_json_sha256=doc["world_json_sha256"],
            semantics_sha256=doc["semantics_sha256"],
            sources=sorted(sources, key=lambda a: a["id"]),
            admitted=False,
            scope="Fixed public definitions; semantic correctness, vehicle inventory and task visibility still require review.",
        ),
    )
