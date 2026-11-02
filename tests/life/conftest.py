"""Tests of the target life engine (awr.life), transplanted from the CHIMERA benchmark.

The engine needs the ``life`` extra (shapely, pillow, rsplan, py-trees); without it the whole directory is skipped
with a ``[dep]`` reason. Helper modules (``life_toytown``) are imported from this directory under importlib mode,
matching tests/activities and tests/targets.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("shapely", reason="[dep] awr.life needs the anet4d[life] extra (shapely)")

sys.path.insert(0, str(Path(__file__).resolve().parent))
