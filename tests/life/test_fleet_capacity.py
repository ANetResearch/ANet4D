from shapely.geometry import box

from awr.life.civil.fleet_capacity import allocate


def test_conflict_search_replaces_a_greedy_bottleneck_without_waiving_routes():
    names = ["a", "b", "c", "d"]
    index = {n: i for i, n in enumerate(names)}
    bodies = {n: box(i * 10, 0, i * 10 + 1, 1) for i, n in enumerate(names)}

    def witnesses(home, occupied):
        present = {n for n in names if occupied & (1 << index[n])}
        # Home a occupies the throat needed by all other vehicles.
        invalid = "a" in present and len(present) > 1
        return [] if invalid else ["destination-1", "destination-2"]

    homes, audit = allocate(names, index, bodies, witnesses, ["a"], 3)
    assert homes == ["b", "c", "d"]
    assert audit["rounds"][-1]["fully_witnessed"]
    mask = sum(1 << index[n] for n in homes)
    assert all(len(witnesses(n, mask)) == 2 for n in homes)


def test_higher_order_route_conflict_is_never_treated_as_a_pairwise_pass():
    names = ["a", "b", "c"]
    index = {n: i for i, n in enumerate(names)}
    bodies = {n: box(i * 10, 0, i * 10 + 1, 1) for i, n in enumerate(names)}

    def witnesses(home, occupied):
        return [] if occupied == 7 else ["destination-1", "destination-2"]

    homes, audit = allocate(names, index, bodies, witnesses, ["a", "b"], 3)
    assert homes == ["a", "b"]
    assert ["a", "b", "c"] in audit["cuts"]
