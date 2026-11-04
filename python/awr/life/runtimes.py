"""Motion backend registry: the third extension point of the target behavior engine.

A motion backend advances the admitted world tick by tick and writes the recorded arrays; admission, the activity
director, tape assembly, final QA, the life audit and publishing stay with the engine. ``civil.grounded.build`` picks
the backend by the recipe's ``runtime`` name:

- ``grounded_v1`` (default): the builder's own compiled loop inside ``civil.grounded``;
- ``life_v1``: ``awr.life.motion.runtime.run`` (``LifeEngine``: decisions, routes, standoffs, rerouting, crowd motion);
- any name registered here, or named in the recipe under ``runtimes`` as ``"name": "module:callable"``.

A backend is a callable ``run(ctx)`` where ``ctx`` is the dict the builder assembles (``civil.grounded``: ``cfg``,
``jobs``, ``actors``, ``n``, ``out``, ``warm``, ``nt``, ``tracks``, ``vel``, ``yaw_track``, ``state``,
``container_track``, ``events``, ``occupancy``, ``rest_r``, ``surface``, ``walking``, ``ext``, ``xy``, ``yaw``,
``activity_runtime``, ``driving_binding``, ``walk_radius``, ``dynamic_validation`` …). It must fill ``tracks``,
``vel``, ``yaw_track``, ``state`` and ``container_track`` for every tick ``0 … nt-1`` and return the dict the builder
stores as the life context (``life_runtime_profile`` and anything the audit wants). It never changes the roster.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

BUILTIN = "grounded_v1"
REGISTRY: dict[str, str | Callable] = {BUILTIN: BUILTIN, "life_v1": "awr.life.motion.runtime:run"}


class RuntimeError_(ValueError):
    pass


def register(name: str, target: str | Callable) -> None:
    """Register a backend under ``name``: a ``module:callable`` string or the callable itself."""
    if not isinstance(name, str) or not name or name == BUILTIN:
        raise RuntimeError_(f"invalid runtime name {name!r}")
    if not (callable(target) or (isinstance(target, str) and ":" in target)):
        raise RuntimeError_(f"runtime {name}: a callable or 'module:callable' required")
    REGISTRY[name] = target


def names(config: dict | None = None) -> list[str]:
    out = dict(REGISTRY)
    out.update((config or {}).get("runtimes") or {})
    return sorted(out)


def resolve(config: dict | None = None) -> tuple[str, Callable | None]:
    """The backend the recipe selects: ``(name, run)``; ``run`` is None for the builtin loop."""
    config = config or {}
    name = config.get("runtime") or BUILTIN
    table = dict(REGISTRY)
    extra = config.get("runtimes") or {}
    if not isinstance(extra, dict):
        raise RuntimeError_("runtimes must be an object: name -> 'module:callable'")
    for k, v in extra.items():
        if not (isinstance(v, str) and ":" in v):
            raise RuntimeError_(f"runtimes.{k}: 'module:callable' required")
        table[k] = v
    if name not in table:
        raise RuntimeError_(f"Unknown runtime {name!r}; known: {sorted(table)}")
    target = table[name]
    if target == BUILTIN:
        return name, None
    if callable(target):
        return name, target
    mod, _, attr = target.partition(":")
    fn = getattr(importlib.import_module(mod), attr, None)
    if not callable(fn):
        raise RuntimeError_(f"runtime {name}: {target} is not a callable")
    return name, fn


__all__ = ["BUILTIN", "REGISTRY", "RuntimeError_", "names", "register", "resolve"]
