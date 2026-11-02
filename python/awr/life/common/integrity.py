"""Seal an intentionally derived world; never bless unrelated changed inputs.

The content-version algorithm is the upstream awr.world.package.manifest
algorithm. Inherited overlay files may be symlinks, but their bytes must still
match the source declaration. This is a build operation, not download repair.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_version(files):
    rows = sorted(files, key=lambda row: row["path"].encode("utf-8"))
    data = "".join(f"{row['path']}:{row['sha256']}\n" for row in rows)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:12]


def validate_content_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Invalid content path")
    p = PurePosixPath(value)
    if (
        p.is_absolute()
        or str(p) != value
        or any(part in (".", "..") for part in value.split("/"))
        or value == "world.json"
    ):
        raise ValueError(f"Unsafe content path: {value}")
    return value


def seal_derived_world(root, source_bytes, metadata, *, changed, removed=()):
    """Publish only after all unchanged files and explicit outputs are verified.

    source_bytes is the original manifest, captured before the build. changed
    and removed must enumerate intentional build outputs, not discovered hash
    mismatches. The caller owns an isolated build directory.
    """
    root = Path(root)
    target = root / "world.json"
    if target.is_symlink():
        raise ValueError("Refusing to replace symlink world.json")
    source = json.loads(source_bytes)
    changed = {validate_content_path(p) for p in changed}
    removed = {validate_content_path(p) for p in removed}
    if changed & removed:
        raise ValueError("Changed and removed paths overlap")
    rows = {}
    for row in source["files"]:
        path = validate_content_path(row["path"])
        if path in rows:
            raise ValueError(f"Duplicate source path: {path}")
        rows[path] = row
    if not removed <= rows.keys():
        raise ValueError("Removed path absent from source manifest")
    # Removed physical files would remain addressable in the published package.
    if any((root / p).exists() or (root / p).is_symlink() for p in removed):
        raise ValueError("Removed content still present")
    files = []
    for path in sorted((rows.keys() | changed) - removed, key=lambda p: p.encode("utf-8")):
        p = root / path
        actual = dict(path=path, bytes=p.stat().st_size, sha256=sha256_file(p))
        if path not in changed:
            expected = rows[path]
            if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
                raise ValueError(f"Undeclared content change: {path}")
        files.append({**rows.get(path, {}), **actual})
    by_path = {r["path"]: r for r in files}
    appearance = "visual/mesh/appearance/index.json"
    if appearance in by_path:
        index = json.loads((root / appearance).read_bytes())
        glb = index["glb"]
        rel = validate_content_path(str(PurePosixPath(appearance).parent / validate_content_path(glb["href"])))
        row = by_path.get(rel)
        if row is None or any(row[k] != glb[k] for k in ("bytes", "sha256")):
            raise ValueError("Appearance index does not match its GLB content")
    world = copy.deepcopy(metadata)
    if world["id"] != source["id"]:
        raise ValueError("Derived world identity changed")
    world["files"] = files
    world["contentVersion"] = content_version(files)
    world["derivedFrom"] = dict(
        contentVersion=source.get("contentVersion"),
        manifest_sha256=hashlib.sha256(source_bytes).hexdigest(),
        changed_paths=sorted(changed),
        removed_paths=sorted(removed),
    )
    for layer in world.get("layers", []):
        if layer.get("id") == "mesh.appearance" and appearance in by_path:
            layer["bytes"] = index["glb"]["bytes"]
    # Write after every check; replacing metadata must not follow a source link.
    with tempfile.NamedTemporaryFile(dir=root, prefix=".world-", suffix=".json", delete=False) as f:
        temp = Path(f.name)
        f.write((json.dumps(world, ensure_ascii=False, indent=1) + "\n").encode())
    os.chmod(temp, 0o644)  # NamedTemporaryFile creates 0600; runtimes read the manifest as another user
    try:
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return world
