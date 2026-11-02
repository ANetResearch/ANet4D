"""Content-addressed build witnesses, independent of file modification times."""

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def snapshot(root, world, nav, extra=()):
    root = Path(root).resolve()
    files = []
    pkg = root / "python/awr/life"
    for directory in (
        pkg / "civil",
        pkg / "envpack",
        pkg / "motion",
        pkg / "appearance",
        pkg / "ambient",
        pkg / "common",
    ):
        files += sorted(directory.rglob("*.py"))
    files += sorted((pkg / "appearance/catalog").glob("*.json"))
    # The engine and the tape, sensor and contract code it depends on live in the same checkout (ANet4D).
    upstream = root
    for rel in (
        "python/awr/sim/events",
        "python/awr/sensorsvc/camera",
        "python/awr/agentsdk/env",
        "packages/contracts/event",
    ):
        files += sorted((upstream / rel).glob("*.py"))
        files += sorted((upstream / rel).glob("*.json"))
    files += [upstream / "python/awr/sensorsvc/capture.py"]
    files += sorted(Path(world).rglob("*")) + sorted(Path(nav).rglob("*")) + list(extra)
    hashes = {}
    for p in files:
        p = Path(p).resolve()
        if not p.is_file():
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        key = str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
        hashes[key] = h.hexdigest()
    packages = {}
    for package in (
        "numpy",
        "scipy",
        "shapely",
        "trimesh",
        "moderngl",
        "Pillow",
        "jsonschema",
        "imageio-ffmpeg",
        "numba",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "schema": "awr.life.build_inputs.v1",
        "content_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "files": hashes,
        "python": platform.python_version(),
        "packages": packages,
    }
