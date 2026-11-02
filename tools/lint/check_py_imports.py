#!/usr/bin/env python3
"""Python import boundaries and bus key literals (AWR-03 §4.2; AWR-11 §3.4, §7.4; AWR-17 §10.5 item 11; AWR-18 §13.1).

Rules (reported as imports/boundary unless noted):
  - awr.api must not import open3d, small_gicp, scipy.optimize, numba, mcap or zstandard (MCAP decompression);
  - `import zenoh` only in awr/runtime/bus.py;
  - open3d only in awr/world/mesh (D3 MeshWorldQuery, .venv-d3; AWR-11 §3.4 keeps it out of D1/D2);
  - numba not in the supervisor (awr/runtime), api, recorder or agent processes;
  - fastapi only in awr/api and awr/bench/api (bench-api); mcap only in awr/recorder and awr/bench/recorder (D3 run writer, M27 §7.5);
  - awr.swarm and awr.environment.weather must not import awr.sim;
  - awr.* must not import tools;
  - package boundaries of AWR-10 §3.3 rule 2 (the expansion of AWR-03 §4.2 rule 1, AD-08): awr.contracts imports no
    other awr.*; awr.runtime only awr.contracts; awr.api only awr.runtime, awr.contracts, awr.api, awr.world.package and
    the pure-function whitelist in tools/lint/py-imports.toml; awr.recorder only awr.runtime and awr.contracts;
    awr.sim.* not awr.api, awr.recorder or awr.agent; the M08 core (sim/{runtime,fleet,core,backends}) not the plugin
    packages; awr.world.*, awr.swarm and the pure awr.environment packages not awr.sim or awr.api; awr.agent not awr.sim;
    awr.reconstruction and awr.jobs not awr.sim or awr.api (relative imports are resolved);
  - D3 import direction (AWR-05 §4.8, ADR-129; D3-AC-01 item 5): no D2 file imports a D3 package (awr.world.glb, awr.world.mesh,
    awr.sim.events, awr.phys, awr.sensorsvc, awr.agentsdk, awr.bench, awr.api.bench_main); D3 code is loaded only through
    D3_PLUGINS, the d3 runtime profile and importlib strings (M04 extension point 7);
  - wall clock in simulation paths (AWR-10 §3.3 ADR-049 invariant 5, ADR-045): awr.sim.{fleet,core,safety,mission,
    sensors} and awr.environment must not call time.time, time.monotonic, time.perf_counter (or their _ns forms) or
    datetime.now;
  - KEY-01: zenoh key string literals (patterns from packages/contracts/bus/keys.json) only in the generated
    awr/contracts/bus_keys.py; WS topic names (rt/topics.json) that look alike are allowed.

Output `<file>:<line>:<col> <RULE> <message>`; exit 0 pass, 1 violations, 2 tool error; summary in runs/lint/.
Usage: python tools/lint/check_py_imports.py [paths...]
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "python" / "awr"


@dataclass(frozen=True)
class Rule:
    module: str  # imported top-level module or dotted prefix
    allowed: tuple[str, ...] = ()  # path prefixes (relative to python/awr) where the import is allowed
    denied: tuple[str, ...] = ()  # path prefixes where the import is forbidden (empty = everywhere except allowed)
    why: str = ""


RULES = [
    *(Rule(m, denied=("api/",), why="awr.api event loop must stay free of heavy work (AWR-03 §4.2)")
      for m in ("open3d", "small_gicp", "scipy.optimize", "numba", "mcap", "zstandard")),
    Rule("zenoh", allowed=("runtime/bus.py",), why="zenoh only through awr.runtime.bus (AWR-17 §10.5 item 11)"),
    Rule("open3d", allowed=("world/mesh/", "sensorsvc/lidar/"),
         why="Open3D is not a D1/D2 dependency (AWR-11 §3.4); D3 uses it only for MeshWorldQuery ray casting (AWR-05 §5.1.6) and the "
             "MeshLidar CPU path (M25 §6.8.3), .venv-d3"),
    *(Rule("numba", denied=(p,), why="numba is not allowed in this process (AWR-11 §3.4)") for p in ("runtime/", "recorder/", "agent/")),
    Rule("fastapi", allowed=("api/", "bench/api/"), why="fastapi only in the api and bench-api processes (AWR-11 §3.4; AWR-17 §18.4)"),
    Rule("mcap", allowed=("recorder/", "bench/recorder/"),
         why="MCAP only in recorder and replay-worker (AWR-11 §3.4) and the D3 run writer (M27 §7.5)"),
    Rule("tools", why="awr.* must not import tools (AWR-03 §4.2)"),
]


@dataclass(frozen=True)
class PkgRule:
    where: tuple[str, ...]  # file path prefixes relative to python/awr
    only: tuple[str, ...] | None = None  # if set, the only awr.* modules these files may import
    deny: tuple[str, ...] = ()  # awr.* module prefixes these files must not import
    unless: tuple[str, ...] = ()  # exceptions to deny
    why: str = ""
    skip: tuple[str, ...] = ()  # file path prefixes exempt from the rule


def _api_whitelist() -> tuple[str, ...]:
    p = ROOT / "tools" / "lint" / "py-imports.toml"
    if not p.exists():
        return ()
    return tuple(tomllib.loads(p.read_text(encoding="utf-8")).get("api", {}).get("whitelist", []))


_B = "AWR-10 §3.3 rule 2"
# D3 packages and their paths (AWR-05 §4.8, §12.2)
D3_PACKAGES = ("awr.world.glb", "awr.world.mesh", "awr.sim.events", "awr.phys", "awr.sensorsvc", "awr.agentsdk", "awr.bench",
               "awr.api.bench_main", "awr.life")
_D3_PATHS = ("world/glb/", "world/mesh/", "sim/events/", "phys/", "sensorsvc/", "agentsdk/", "bench/", "api/bench_main.py", "life/")
_M08_CORE = ("sim/runtime/", "sim/fleet/", "sim/core/", "sim/backends/")
_PURE = ("environment/conventions/", "environment/weather/", "environment/wind/", "environment/atmosphere/", "environment/io/",
         "swarm/", "world/")
PKG_RULES = [
    PkgRule(("contracts/",), only=("awr.contracts",), why=f"awr.contracts is generated and imports no other awr.* ({_B})"),
    PkgRule(("runtime/",), only=("awr.runtime", "awr.contracts"), why=f"awr.runtime depends only on awr.contracts ({_B})"),
    PkgRule(("api/",), only=("awr.api", "awr.runtime", "awr.contracts", "awr.world.package"),
            why="awr.api depends only on awr.runtime, awr.contracts, awr.api, awr.world.package and whitelisted pure functions "
                "(AWR-03 §4.2 rule 1; tools/lint/py-imports.toml)", skip=("api/bench_main.py",)),
    PkgRule(("recorder/",), only=("awr.recorder", "awr.runtime", "awr.contracts"), why=f"awr.recorder depends only on awr.runtime and awr.contracts ({_B})"),
    PkgRule(("sim/",), deny=("awr.api", "awr.recorder", "awr.agent"), why=f"awr.sim must not depend on api, recorder or agent ({_B})"),
    PkgRule(_M08_CORE, deny=("awr.sim.safety", "awr.sim.mission", "awr.sim.planning", "awr.sim.sensors", "awr.environment.field",
                             "awr.environment.stage"),
            why="the M08 core loads M07, M09, M10 and M13 through the composition root, not by import (AWR-10 §3.3 rules 1-2)"),
    PkgRule(_PURE, deny=("awr.sim", "awr.api"), why=f"pure algorithm and world packages must not depend on awr.sim or awr.api ({_B}; AWR-03 §4.2)"),
    PkgRule(("environment/field/", "environment/stage/"), deny=("awr.sim", "awr.api"),
            unless=("awr.sim.backends.base", "awr.sim.fleet.stages.registry"),
            why=f"EnvironmentService may use only the M08 interfaces and registries ({_B})"),
    PkgRule(("agent/",), deny=("awr.sim",), why=f"awr.agent talks to sim-core only over the Bus and a StateRing reader ({_B})"),
    PkgRule(("reconstruction/", "jobs/"), deny=("awr.sim", "awr.api"), why=f"awr.reconstruction and awr.jobs must not depend on awr.sim or awr.api ({_B})"),
    PkgRule(("",), deny=D3_PACKAGES, skip=_D3_PATHS,
            why="D2 code must not import D3 packages; load them through D3_PLUGINS, the d3 profile or importlib strings (AWR-05 §4.8)"),
]
WALL_CLOCK_PATHS = ("sim/fleet/", "sim/core/", "sim/safety/", "sim/mission/", "sim/sensors/", "environment/")
WALL_CLOCK_CALLS = {"time.time", "time.time_ns", "time.monotonic", "time.monotonic_ns", "time.perf_counter", "time.perf_counter_ns",
                    "datetime.now", "datetime.datetime.now", "datetime.utcnow", "datetime.datetime.utcnow"}


def _resolve_relative(rel_awr: str, node: ast.ImportFrom) -> str | None:
    parts = ["awr", *rel_awr.removesuffix(".py").split("/")]
    base = parts[:-1]  # package of the module (for __init__.py the package itself)
    if node.level > 1:
        base = base[: len(base) - (node.level - 1)]
    if not base:
        return None
    return ".".join([*base, node.module] if node.module else base)


def _pkg_violations(rel_awr: str, mod: str) -> list[str]:
    out = []
    if not (mod == "awr" or mod.startswith("awr.")) or mod == "awr":
        return out
    for r in PKG_RULES:
        if not any(rel_awr.startswith(w) for w in r.where) or any(rel_awr.startswith(s) for s in r.skip):
            continue
        if r.only is not None:
            allowed = r.only + (_api_whitelist() if r.where == ("api/",) else ())
            if not any(_matches(mod, a) for a in allowed):
                out.append(r.why)
        if any(_matches(mod, d) for d in r.deny) and not any(_matches(mod, u) for u in r.unless):
            out.append(r.why)
    return out


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def _matches(mod: str, rule_mod: str) -> bool:
    return mod == rule_mod or mod.startswith(rule_mod + ".")


def _key_patterns() -> tuple[list[re.Pattern[str]], list[re.Pattern[str]]]:
    def rx(p: str) -> re.Pattern[str]:
        return re.compile("^" + re.sub(r"\\\{[^}]+\\\}", "[^/]+", re.escape(p)) + "$")

    keys = json.loads((ROOT / "packages/contracts/bus/keys.json").read_text(encoding="utf-8"))["keys"]
    topics = json.loads((ROOT / "packages/contracts/rt/topics.json").read_text(encoding="utf-8"))["topics"]
    return [rx(k["key"]) for k in keys], [rx(t["pattern"]) for t in topics]


def check_file(path: Path, violations: list[tuple[str, int, int, str, str]], key_rx, topic_rx) -> None:
    rel_awr = path.relative_to(PKG).as_posix()
    rel = path.relative_to(ROOT).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
    except SyntaxError as e:
        violations.append((rel, e.lineno or 1, (e.offset or 0) + 1, "imports/boundary", f"syntax error: {e.msg}"))
        return
    for node in ast.walk(tree):
        mods: list[str] = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        for mod in mods:
            for r in RULES:
                if not _matches(mod, r.module):
                    continue
                if r.allowed and any(rel_awr == a or rel_awr.startswith(a) for a in r.allowed):
                    continue
                if r.denied and not any(rel_awr.startswith(d) for d in r.denied):
                    continue
                violations.append((rel, node.lineno, node.col_offset + 1, "imports/boundary", f"{mod}: {r.why}"))
        # package boundaries (AWR-10 §3.3 rule 2; the swarm and environment.weather rule of AWR-03 §4.2 is part of it):
        # `import a.b`, `from a.b import c` (a.b and a.b.c) and relative imports
        pmods: list[str] = []
        if isinstance(node, ast.Import):
            pmods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module if node.level == 0 else _resolve_relative(rel_awr, node)
            if base:
                pmods = [base] + [f"{base}.{a.name}" for a in node.names if not (base == "awr" and a.name.startswith("__"))]
        violations.extend((rel, node.lineno, node.col_offset + 1, "imports/boundary", f"{mod}: {why}")
                          for mod in pmods for why in _pkg_violations(rel_awr, mod))
        if isinstance(node, ast.Call) and any(rel_awr.startswith(w) for w in WALL_CLOCK_PATHS):
            name = _dotted(node.func)
            if name in WALL_CLOCK_CALLS:
                violations.append((rel, node.lineno, node.col_offset + 1, "imports/boundary",
                                   f"{name}(): wall clock in a simulation path; use the injected SimClock (ADR-045, ADR-049)"))
        if isinstance(node, ast.ImportFrom) and node.module in ("time", "datetime") and any(rel_awr.startswith(w) for w in WALL_CLOCK_PATHS):
            violations.extend((rel, node.lineno, node.col_offset + 1, "imports/boundary",
                               f"{node.module}.{a.name}: wall clock in a simulation path; use the injected SimClock (ADR-045, ADR-049)")
                              for a in node.names if f"{node.module}.{a.name}" in WALL_CLOCK_CALLS)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and rel_awr != "contracts/bus_keys.py":
            s = node.value
            if "/" in s and len(s) < 120 and any(k.match(s) for k in key_rx) and not any(t.match(s) for t in topic_rx):
                violations.append((rel, node.lineno, node.col_offset + 1, "KEY-01", f"zenoh key literal {s!r}; use awr.contracts.bus_keys"))


def main(argv: list[str] | None = None) -> int:
    t0 = time.time()
    args = [a for a in (argv if argv is not None else sys.argv[1:]) if not a.startswith("--")]
    try:
        key_rx, topic_rx = _key_patterns()
        files: list[Path] = []
        for a in args or [str(PKG)]:
            p = Path(a).resolve()
            files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
        files = [f for f in files if "__pycache__" not in f.parts and f.is_relative_to(PKG)]
        violations: list[tuple[str, int, int, str, str]] = []
        for f in files:
            check_file(f, violations, key_rx, topic_rx)
    except Exception as e:  # tool error, exit 2
        print(f"check_py_imports: tool error: {e!r}", file=sys.stderr)
        return 2
    violations = sorted(set(violations))
    for v in violations:
        print(f"{v[0]}:{v[1]}:{v[2]} {v[3]} {v[4]}")
    summary = {"tool": "check_py_imports", "ok": not violations, "violations": len(violations), "files": len(files),
               "elapsed_ms": round((time.time() - t0) * 1000), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "items": [dict(zip(("file", "line", "col", "rule", "msg"), v, strict=True)) for v in violations[:2000]]}
    try:
        out = Path(os.environ.get("AWR_RUNS_DIR", ROOT / "runs")) / "lint"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-check_py_imports.json").write_text(json.dumps(summary, indent=1) + "\n")
    except OSError:
        pass
    print(f"check_py_imports: {'ok' if not violations else f'{len(violations)} violation(s)'} ({summary['elapsed_ms']} ms, {len(files)} files)", file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
