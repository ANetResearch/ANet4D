import itertools
from types import SimpleNamespace

import numpy as np

from awr.life.civil.itineraries import life_pools


def open_square(n=60, step=0.5):
    """A flat open walking grid of n x n cells (every cell a network node)."""
    rows, cols = np.mgrid[0:n, 0:n]
    nodes = np.column_stack([rows.ravel(), cols.ravel()])
    xy = (nodes[:, ::-1] + 0.5) * step
    net = SimpleNamespace(step=step, nodes=nodes, xy=xy, cell=lambda q: (int(q[1] // step), int(q[0] // step)))
    return net, np.ones((n, n), bool)


def test_slots_of_neighbouring_sites_keep_the_spacing():
    """Two approaches of one facility 1 m apart (the courthouse approaches of the town, where the life_v1 smoke run
    stalled): their standing slots keep slot_spacing_m from each other, not only within each site."""
    net, free = open_square()
    places = [
        dict(id="court-approach-3", xy=[14.0, 15.0], kind="cafe"),
        dict(id="court-approach-4", xy=[15.0, 15.0], kind="cafe"),
    ]
    pools = life_pools(net, {"buildings": []}, places, free, np.random.default_rng(1), dict(slot_spacing_m=2.0))
    slots = [np.asarray(sl["xy"]) for st in pools["sites"] for sl in st["slots"]]
    assert len(pools["sites"]) == 2 and len(slots) == 12
    assert min(np.linalg.norm(a - b) for a, b in itertools.combinations(slots, 2)) >= 2.0 - 1e-9
