"""Read-only feature inventory for an explicitly bound environment snapshot.

Checks structural consistency and provenance, not physical or benchmark admission.
It deliberately does not infer capacity from counts or reject a scene for having
fewer cars than a previous scenario recipe requested.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from awr.life.common.schema import schema_id

SCHEMA = "awr.life.environment_feature_inventory.v1"
REQUIRED = (
    "world",
    "coordinate",
    "semantics",
    "support_meta",
    "ground",
    "walk_person",
    "z_person",
    "walk_vehicle",
    "z_vehicle",
    "places",
    "initial",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(bundle, root):
    """Inventory a local adapter's explicit layers; never mutate source artifacts."""
    root, bundle = Path(root), Path(bundle)
    cfg = json.loads(bundle.read_text())
    if schema_id(cfg.get("schema")) != SCHEMA or not isinstance(cfg.get("layers"), dict):
        raise ValueError("Explicit feature inventory schema/layers required")
    rows, facts, bound, data = [], {}, {str(bundle): digest(bundle)}, {}

    def record(code, status, detail, affected):
        rows.append(dict(code=code, status=status, detail=detail, affected=affected))

    for name in REQUIRED:
        if name not in cfg["layers"]:
            record("missing:" + name, "missing", "Required inventory layer absent", ["snapshot_consistency"])
    for name, layer in sorted(cfg["layers"].items()):
        p = root / layer["path"]
        if not p.is_file():
            record("missing:" + name, "missing", layer["path"], ["snapshot_consistency"])
            continue
        actual = digest(p)
        bound[layer["path"]] = actual
        if actual != layer.get("sha256"):
            record("digest:" + name, "error", "Missing or mismatching frozen input digest", ["snapshot_consistency"])
            continue
        data[name] = np.load(p, mmap_mode="r", allow_pickle=False) if p.suffix == ".npy" else json.loads(p.read_text())
    if any(name not in data for name in REQUIRED):
        return _report(cfg, rows, facts, bound)

    world, coord, sem, meta = (data[n] for n in ("world", "coordinate", "semantics", "support_meta"))
    frame = coord.get("frame", {})
    local = frame.get("units") == "m" and frame.get("type") == "ENU" and frame.get("upAxis") == "+Z"
    identity = cfg.get("id") == world.get("id") == coord.get("worldId")
    record(
        "local_frame",
        "consistent" if local and identity else "error",
        dict(frame=frame, time=coord.get("conventions", {}).get("time"), identities_match=identity),
        ["all_motion"],
    )
    facts["geographic_anchor"] = coord.get("anchor", {})
    if not coord.get("anchor", {}).get("georeferenced", False):
        record(
            "geographic_anchor",
            "unverified",
            "Local coordinates are not a verified geodetic survey",
            ["geographic_reporting"],
        )
    diagnostic = world.get("geometryOnlyDiagnostic", False)
    record(
        "render_geometry_identity",
        "unverified",
        dict(geometry_only=diagnostic, explanation="No native/web asset parity check in this inventory"),
        ["rendering", "query_evidence"],
    )

    shape = (int(meta["ny"]), int(meta["nx"]))
    facts["support"] = dict(shape=list(shape), cell_m=meta["cell_m"], method=meta.get("method"))
    shape_ok = all(data[k].shape == shape for k in ("ground", "walk_person", "z_person", "walk_vehicle", "z_vehicle"))
    record("support_shapes", "consistent" if shape_ok else "error", list(shape), ["all_motion"])
    record(
        "unknown_support",
        "consistent" if meta.get("unknown_is_blocked") is True else "error",
        "Unknown support must be blocked",
        ["all_motion"],
    )
    world_sha = cfg["layers"]["world"]["sha256"]
    record(
        "support_world_binding",
        "consistent" if world_sha in meta.get("source_hashes", {}).values() else "unverified",
        "Support provenance must include this world descriptor; mesh identity needs separate evidence",
        ["all_motion"],
    )
    if shape_ok:
        for kind in ("person", "vehicle"):
            mask, z = data["walk_" + kind], data["z_" + kind]
            boolean = mask.dtype == np.bool_
            bad, count = 0, 0
            if boolean:
                for lo in range(0, shape[0], 128):
                    m = mask[lo : lo + 128]
                    count += int(m.sum())
                    bad += int(np.count_nonzero(m & ~np.isfinite(z[lo : lo + 128])))
            facts["support"][kind + "_cells"] = count if boolean else None
            record(
                "support_mask:" + kind,
                "consistent" if boolean and bad == 0 else "error",
                dict(boolean_mask=boolean, allowed_cells_without_height=bad),
                [kind + "_motion"],
            )

    roads, pois, crossings = sem.get("roads", []), sem.get("pois", []), sem.get("crossings", [])
    facts["semantics"] = dict(
        roads=len(roads),
        road_width_histogram=dict(Counter(str(r.get("width_m")) for r in roads)),
        pois=len(pois),
        verified_poi_entrances=sum(p.get("verified_entrance") is True for p in pois),
        explicit_doors=len(sem.get("doors", [])),
        crossings=len(crossings),
        crossing_source=sem.get("crossing_source"),
        parking_regions=len(sem.get("verified_parking_regions", [])),
    )
    record(
        "road_capacity",
        "unverified",
        "Road widths/counts do not establish physical lane or fleet capacity",
        ["vehicle_motion"],
    )
    record(
        "facility_entrances", "unverified", facts["semantics"]["verified_poi_entrances"], ["indoor_facility_activity"]
    )
    record(
        "crossing_markings", "unverified", sem.get("crossing_source", "No survey evidence"), ["marked_crossing_rules"]
    )
    support_sem = data.get("support_semantics")
    if support_sem is not None:
        changed = sorted(k for k in set(sem) | set(support_sem) if sem.get(k) != support_sem.get(k))
        record("support_semantics_delta", "consistent" if not changed else "unverified", changed, ["support_scope"])

    places, initial = data["places"], data["initial"]
    ids = [p["id"] for p in places]
    duplicate = sorted(k for k, n in Counter(ids).items() if n > 1)
    record("place_ids", "consistent" if not duplicate else "error", duplicate, ["activities"])
    bad_xy = [p["id"] for p in places if np.shape(p.get("xy")) != (2,) or not np.isfinite(p["xy"]).all()]
    record("place_coordinates", "consistent" if not bad_xy else "error", bad_xy, ["activities"])
    place_ids = set(ids)
    missing_parking = sorted({b for p in places for b in p.get("parking", []) if b not in place_ids})
    record(
        "place_parking_references",
        "consistent" if not missing_parking else "error",
        missing_parking,
        ["driver_activity"],
    )
    facts["places"] = dict(
        count=len(places),
        kinds=dict(Counter(p["kind"] for p in places)),
        distinct_facility_groups=len({p.get("group", p["id"]) for p in places if p["kind"] != "home"}),
    )
    actors = initial.get("actors", [])
    aid = [a["id"] for a in actors]
    facts["initial_recipe"] = dict(count=len(actors), classes=dict(Counter(a["class"] for a in actors)))
    record(
        "actor_identity_alignment",
        "consistent" if len(aid) == len(set(aid)) == len(initial.get("positions", [])) else "error",
        "Unique IDs and one initial position per actor; does not certify collision-free placement",
        ["population"],
    )
    record(
        "joint_capacity",
        "unverified",
        "No dynamic capacity or behavioral acceptance inferred from this inventory",
        ["population", "vehicle_motion"],
    )
    old = data.get("legacy_manifest")
    if old:
        drift = {
            k: dict(declared=v.get("path"), inspected=cfg["layers"][k]["path"])
            for k, v in old.get("layers", {}).items()
            if k in cfg["layers"] and v.get("path") != cfg["layers"][k]["path"]
        }
        facts["legacy_declarations"] = old.get("capabilities")
        record("legacy_manifest_drift", "unverified" if drift else "consistent", drift, ["environment_binding"])
    # Hash again to reject a snapshot changed while being inspected.
    for name, expected in bound.items():
        p = bundle if name == str(bundle) else root / name
        if digest(p) != expected:
            raise ValueError("Inputs changed during feature audit: " + name)
    return _report(cfg, rows, facts, bound)


def _report(cfg, rows, facts, bound):
    return dict(
        schema="awr.life.environment_feature_audit.v1",
        environment=cfg.get("id"),
        admitted=False,
        release_evidence=False,
        structurally_consistent=not any(r["status"] in ("error", "missing") for r in rows),
        checks=rows,
        facts=facts,
        input_sha256=bound,
        limitations=[
            "Inventory and structural checks only; no automatic semantic truth, physics or query certification.",
            "Unavailable optional evidence remains unverified, not evidence that a capability is physically impossible.",
            "Counts describe inspected inputs, not accepted runtime capacity.",
        ],
    )
