"""Support samples shared by route admission and predictive braking.

Use authored tyre centres when available. Legacy meshes without tyre metadata
keep an explicitly recorded proxy; they are not certified wheel-contact assets.
"""

import json
from functools import lru_cache

import numpy as np

from awr.life.appearance.assets import ROOT, load


@lru_cache(maxsize=32)
def vehicle_contacts(size):
    length, width = size[:2]
    # Existing assembly fits pitch/roll at these four terrain probes.
    probes = np.array([[length * 0.32, 0], [-length * 0.32, 0], [0, width * 0.37], [0, -width * 0.37]])
    entries = json.loads((ROOT / "catalog.json").read_text())["assets"]
    matches = [
        e for e in entries if e.get("kind") == "vehicle" and np.max(np.abs(np.asarray(e["dimensions_m"]) - size)) < 0.02
    ]
    if len(matches) != 1:
        raise ValueError("Vehicle dimensions do not identify one verified mesh")
    _, data, _ = load(matches[0]["id"])
    wheels = data.get("wheel_centers", np.empty((0, 3)))
    if wheels.size:
        return np.vstack([probes, wheels[:, :2]])
    return np.vstack([probes, np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * [length * 0.32, width * 0.37]])
