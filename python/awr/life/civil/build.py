"""One-shot build of a civil scenario from its config: SUMO net → semantics → population → schedule → simulate →
QA → assemble.

    python -m awr.life.civil.build scenarios/crop-city-002/civil.yaml --world <world dir> --nav <navgrid dir>
        --work <work dir> --out <event dir> [--world-ref <tape.json>] [--ground grid|mesh] [--duration S] [--skip-sim]

Stage outputs are cached in the work dir and reused when present (delete to rebuild).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def sh(args: list[str]) -> None:
    print("+", " ".join(str(a) for a in args), flush=True)
    r = subprocess.run([str(a) for a in args])
    if r.returncode != 0:
        raise SystemExit(f"step failed: {args[1] if len(args) > 1 else args}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--world", required=True, type=Path)
    ap.add_argument("--nav", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--world-ref",
        type=Path,
        default=None,
        help="tape.json of a compiled event for the world header (default: derive from world)",
    )
    ap.add_argument("--ground", choices=["grid", "mesh"], default="grid")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument(
        "--terrain", type=Path, default=None, help="terrain raster dir when not under <world>/geometry/terrain"
    )
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    repo = Path(__file__).resolve().parents[4]
    pkg = Path(__file__).resolve().parents[1]
    W = a.work
    if a.out.exists():
        raise SystemExit(f"Refusing to replace an existing event: {a.out}; select a new version directory")
    W.mkdir(parents=True, exist_ok=True)
    from awr.life.civil.surface import SurfaceMap

    SurfaceMap(a.nav)  # reject old DSM grids before spending time simulating
    sources = sorted((pkg / "civil").glob("*.py"))
    sources += [a.config, *sorted(a.nav.glob("*.npy")), a.nav / "meta.json"]
    sources += sorted(p for p in a.world.rglob("*") if p.is_file())
    upstream = repo / "python/awr/sim/events/tape.py"
    sources.append(upstream)
    osm_source = Path(cfg["osm"])
    sources.append(osm_source if osm_source.is_absolute() else repo / osm_source)
    if a.world_ref:
        sources.append(a.world_ref)
    if a.terrain:
        sources += sorted(p for p in a.terrain.rglob("*") if p.is_file())
    from importlib.metadata import version

    dependency_versions = {name: version(name) for name in ("numpy", "scipy", "shapely")}
    sources += sorted((pkg / "appearance/catalog").glob("*.json"))
    digest = hashlib.sha256(
        json.dumps(
            {"duration": a.duration, "ground": a.ground, "dependencies": dependency_versions}, sort_keys=True
        ).encode()
    )
    for source in sources:
        digest.update(str(source.resolve()).encode())
        with source.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    stamp = {"schema": "awr.life.build_inputs.v1", "sha256": digest.hexdigest()}
    cache_stamp = W / "build-inputs.json"
    if cache_stamp.exists():
        if json.loads(cache_stamp.read_text()) != stamp:
            raise SystemExit("Build inputs changed. Select a new --work directory; stale outputs will not be reused.")
    elif any((W / name).exists() for name in ("sim", "sumo", "plans.json", "population.json", "semantics.json")):
        raise SystemExit("Unversioned legacy cache. Select a new --work directory.")
    cache_stamp.write_text(json.dumps(stamp, indent=2) + "\n")
    py = sys.executable
    osm = repo / cfg["osm"] if not Path(cfg["osm"]).is_absolute() else Path(cfg["osm"])
    seed = int(cfg["seed"])
    (W / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    if not (W / "sumo" / "net.net.xml").exists():
        sh([py, "-m", "awr.life.civil.sumo_net", a.world, "--out", W / "sumo"])
    if not (W / "semantics.json").exists():
        sh(
            [
                py,
                "-m",
                "awr.life.civil.semantics",
                "--world",
                a.world,
                "--osm",
                osm,
                "--nav",
                a.nav,
                "--sumo",
                W / "sumo",
                "--stage",
                *[str(v) for v in cfg["stage"]],
                "--out",
                W,
            ]
            + (["--terrain", a.terrain] if a.terrain else [])
        )
    if not (W / "population.json").exists():
        sh(
            [
                py,
                "-m",
                "awr.life.civil.population",
                "--semantics",
                W / "semantics.json",
                "--config",
                a.config,
                "--seed",
                seed,
                "--out",
                W / "population.json",
            ]
        )
    if not (W / "plans.json").exists():
        sh(
            [
                py,
                "-m",
                "awr.life.civil.schedule",
                "--semantics",
                W / "semantics.json",
                "--population",
                W / "population.json",
                "--sumo",
                W / "sumo",
                "--config",
                a.config,
                "--seed",
                seed,
                "--out",
                W / "plans.json",
            ]
        )
    if not a.skip_sim and not (W / "sim" / "tracks.npz").exists():
        args = [
            py,
            "-m",
            "awr.life.civil.simulate",
            "--semantics",
            W / "semantics.json",
            "--population",
            W / "population.json",
            "--plans",
            W / "plans.json",
            "--nav",
            a.nav,
            "--sumo",
            W / "sumo",
            "--out",
            W / "sim",
            "--seed",
            seed,
        ]
        if a.duration:
            args += ["--duration", a.duration]
        sh(args)
    sh(
        [
            py,
            "-m",
            "awr.life.civil.qa",
            "--sim",
            W / "sim",
            "--semantics",
            W / "semantics.json",
            "--nav",
            a.nav,
            "--out",
            W / "sim" / "qa.json",
        ]
    )
    env = W / "env.json"
    env.write_text(json.dumps(cfg.get("environment"), ensure_ascii=False), encoding="utf-8")
    ref = a.world_ref
    if ref is None:
        ref = W / "world_ref.json"
        if not ref.exists():
            from awr.sim.events.compile import raycast_sha256

            wdoc = json.loads((a.world / "world.json").read_text(encoding="utf-8"))
            nav_p = json.loads((a.world / "geometry" / "nav" / "person.json").read_text()).get("bin", {}).get("sha256")
            nav_v = json.loads((a.world / "geometry" / "nav" / "vehicle.json").read_text()).get("bin", {}).get("sha256")
            ref.write_text(
                json.dumps(
                    {
                        "header": {
                            "world": {
                                "id": wdoc["id"],
                                "content_version": wdoc["contentVersion"],
                                "raycast_sha256": raycast_sha256(a.world),
                                "navmesh_sha256": {"person": nav_p, "vehicle": nav_v},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=a.out.name + ".pending-", dir=a.out.parent))
    args = [
        py,
        "-m",
        "awr.life.civil.assemble",
        "--sim",
        W / "sim",
        "--semantics",
        W / "semantics.json",
        "--plans",
        W / "plans.json",
        "--population",
        W / "population.json",
        "--world-ref",
        ref,
        "--event-id",
        cfg["event_id"],
        "--version",
        cfg.get("version", 1),
        "--out",
        staging,
        "--ground",
        a.ground,
        "--nav",
        a.nav,
        "--world",
        a.world,
        "--env",
        env,
        "--time-origin",
        cfg.get("time_origin_utc", "2026-06-15T08:30:00Z"),
    ]
    sh(args)
    import shutil

    sh([py, "-m", "awr.life.civil.validate_tape", "--event", staging, "--nav", a.nav])
    shutil.copy(W / "sim" / "qa.json", staging / "qa.json")
    shutil.copy(W / "config.json", staging / "civil.json")
    shutil.copy(cache_stamp, staging / "build-inputs.json")
    # Failed builds remain clearly labelled .pending-* for diagnosis. Only a
    # fully validated directory is published at the requested event path.
    os.rename(staging, a.out)
    print("built", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
