"""Bind only available navigation profiles, requiring those used by the recording."""

import json
import re
from pathlib import Path


def navigation_hashes(world, *, require_vehicle=False):
    """Upstream tape contracts require person navigation, but vehicle is optional.

    Omit an unavailable vehicle profile only for a recording without vehicles;
    never substitute a person hash or a null digest for missing vehicle data.
    Existing profiles remain bound even when a particular recording omits them.
    """
    root = Path(world) / "geometry/nav"
    result = {}
    for profile in ("person", "vehicle"):
        path = root / (profile + ".json")
        if profile == "vehicle" and not require_vehicle and not path.exists():
            continue
        doc = json.loads(path.read_text())
        digest = doc.get("bin", {}).get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Missing or invalid navigation digest: " + str(path))
        result[profile] = digest
    return result
