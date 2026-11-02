"""Tests of the target life engine (awr.life), transplanted from the CHIMERA benchmark.

The engine needs the ``life`` extra (shapely, pillow, rsplan, py-trees). The D1 lock and the hosted CI do not install
it, so without it the directory is not collected (a skip raised while importing a conftest would be a collection
error). Helper modules (``life_toytown``) are imported from this directory under importlib mode, matching
tests/activities and tests/targets.
"""

import importlib.util
import sys
from pathlib import Path

LIFE_EXTRA = ("shapely", "PIL", "rsplan", "py_trees")
MISSING = [name for name in LIFE_EXTRA if importlib.util.find_spec(name) is None]

if MISSING:
    collect_ignore_glob = ["*.py"]
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
