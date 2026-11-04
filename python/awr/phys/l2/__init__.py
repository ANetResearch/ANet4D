"""Unified L2 physics engine (``awr.phys.l2``): one rigid-body core, one tier, one domain model per device family.

The L2 fidelity tier of the device ladder (``awr.sim.device``) is "realistic dynamics": rotor- or actuator-level actuation,
a rate loop, control allocation, a rigid body with inertia and the device's aerodynamic or hydrodynamic forces. Before
this package the only L2 was gz-sim (``awr.phys.gz``) and it covered one vehicle model. This package gives the tier an
engine of our own so that every device the ladder lists can run at L2 on the same CPU host as Fast, with gz kept as a
cross-check where it has a model:

- ``rigid``: the shared 6-DoF rigid body and quaternion helpers (ENU / FLU, semi-implicit Euler, deterministic);
- ``multirotor``: per-rotor dynamics, PX4-gain cascade, allocation with desaturation, composite aerodynamics;
- ``tier``: ``L2Tier``, the PhysicsTier (same commands, reference generator, contact rules and PoseHistory as Fast and gz).

Ground vehicles, surface vessels and fixed wings are the next domain models on the same body and the same tier; until a
domain model exists the tier rejects that vehicle with 630 instead of pretending.
"""

from __future__ import annotations

from awr.phys.l2.tier import L2Tier

__all__ = ["L2Tier"]
