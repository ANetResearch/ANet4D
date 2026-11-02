"""Check declared derived-layer sources without granting physical admission."""

from pathlib import Path


def declared_sources(layers, root):
    """Resolve manifest ``built_on`` paths in the same root as layer paths.

    A stale, malformed or dangling dependency is evidence against mixing these
    layers. Matching paths alone are not proof of a cache's content provenance;
    compiler input digests and runtime geometry checks remain necessary.
    """
    root = Path(root)
    rows = []
    for name, layer in layers.items():
        dependencies = layer.get("built_on", {})
        if not isinstance(dependencies, dict):
            rows.append(
                dict(
                    layer=name,
                    source=None,
                    built_on=dependencies,
                    declared=None,
                    match=False,
                    reason="built_on must be a mapping",
                )
            )
            continue
        for source, path in dependencies.items():
            declared = (layers.get(source) or {}).get("path")
            valid = isinstance(path, str) and bool(path) and isinstance(declared, str) and bool(declared)
            match = valid and (root / path).resolve() == (root / declared).resolve()
            rows.append(
                dict(
                    layer=name,
                    source=source,
                    built_on=path,
                    declared=declared,
                    match=bool(match),
                    reason=None if match else "missing or mismatched source layer",
                )
            )
    return rows
