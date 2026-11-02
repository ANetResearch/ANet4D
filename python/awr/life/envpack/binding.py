"""Bind build inputs to the declared environment; identity is not physical admission."""

import json
from pathlib import Path

from awr.life.common.schema import schema_id

from .dependencies import declared_sources
from .validate import sha256


def bind_inputs(manifest, *, world, support, semantics, config, root="."):
    """Reject mixed versions before placement. Paths may be symlink aliases to one source.

    v2 validation candidates require all consumed named layers; v1 keeps undeclared
    optional layers explicitly unbound for historical compatibility. No pass here
    authorizes simulation, publication or an unmeasured capacity.
    """
    root = Path(root)
    actual = {"world": world, "support": support, "semantics": semantics}
    for name, key in (
        ("places", "outdoor_places"),
        ("lane_network", "lane_network"),
        ("public_regions", "public_regions"),
    ):
        if config.get(key):
            actual[name] = config[key]
    driving = (config.get("life") or {}).get("driving") or {}
    for name, key in (("driving_map", "map"), ("road_graph", "source_graph")):
        if driving.get(key):
            actual[name] = driving[key]
    layers, rows, unbound = manifest["layers"], [], []
    dependencies = declared_sources(layers, root)
    stale = [row for row in dependencies if not row["match"]]
    if stale:
        names = ", ".join(f"{r['layer']} <- {r['source']}" for r in stale)
        raise ValueError(f"Environment derived layer source mismatch: {names}")
    strict = schema_id(manifest["schema"]) == "awr.life.environment.v2"
    for name, value in actual.items():
        used = Path(value).resolve()
        if not used.exists():
            raise ValueError(f"Environment input {name} is unavailable: {used}")
        layer = layers.get(name)
        if layer is None:
            if strict or name in ("world", "support", "semantics"):
                raise ValueError(f"Environment does not bind consumed layer {name}")
            unbound.append(name)
            continue
        expected = (root / layer["path"]).resolve()
        if used != expected:
            raise ValueError(f"Environment input {name} mismatch: declared {expected}, used {used}")
        rows.append({"layer": name, "path": str(used)})
    digests = {}
    # Validate every declared digest, including remote-labelled files when used locally.
    for name, layer in layers.items():
        path = root / layer["path"]
        targets = {path: layer["sha256"]} if layer.get("sha256") else {}
        targets.update({path / rel: digest for rel, digest in layer.get("array_sha256", {}).items()})
        for p, expected in targets.items():
            if not p.is_file():
                raise ValueError(f"Environment digest input missing: {name}: {p}")
            found = sha256(p)
            if found != expected:
                raise ValueError(f"Environment digest mismatch: {name}: {p}")
            digests[str(p.resolve())] = found
    # This evidence binds exactly the files the builder selected, including unversioned legacy inputs.
    for name, value in actual.items():
        p = Path(value)
        if p.is_file():
            digests[str(p.resolve())] = sha256(p)
    walking_policy = None
    if "places" in actual:
        place_doc = json.loads(Path(actual["places"]).read_text())
        policy = place_doc.get("walking_policy")
        if policy is not None:
            if not isinstance(policy, dict):
                raise ValueError("Environment places walking policy must be a mapping")
            if policy.get("shared_streets") != config.get("shared_streets") or policy.get("step_m") != float(
                config.get("walking_step_m", 0.5)
            ):
                raise ValueError(
                    "Environment places walking policy differs from scene configuration; recompile candidates"
                )
            walking_policy = policy
    return dict(
        schema="awr.life.environment_build_binding.v1",
        environment=manifest["id"],
        admitted=False,
        identity_bound=not unbound,
        unbound_optional_layers=unbound,
        inputs=rows,
        file_sha256=digests,
        declared_sources=dependencies,
        places_walking_policy=walking_policy,
    )
