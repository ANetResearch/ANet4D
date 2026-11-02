"""Destination-preserving rolling routes around temporary crowd occupancy.

Temporary waypoints are navigation actions, never visits or completed errands.
The planner can advance towards a far occupied goal, but may never cross a
blocked body, static cell or unsupported connector to do so.
"""

import numpy as np

from awr.life.civil.walking import polyline


def sample_route(route, distance):
    arc, points = route["arc"], route["points"]
    d = float(np.clip(distance, 0, arc[-1]))
    i = max(0, min(np.searchsorted(arc, d, side="right") - 1, len(points) - 2))
    return points[i] + (points[i + 1] - points[i]) * (d - arc[i]) / (arc[i + 1] - arc[i])


def _turnable(network, p):
    """A refuge must let the body rotate in place (it turns back there)."""
    turn = getattr(network, "turn_free", None)
    if turn is None:
        return True
    c = network.cell(p)
    return bool((c >= 0).all() and (c < np.array(turn.shape)).all() and turn[tuple(c)])


def rolling_route(network, start, goal, static_free, dynamic_free, horizon=14.0):
    """Return (legal route, temporary endpoint); retain the actual destination."""
    guide = network.plan(start, goal, static_free)
    if guide is None:
        return None, False
    complete = network.plan(start, goal, dynamic_free)
    if complete is not None:
        return complete, False
    length = guide["arc"][-1]
    # Direct local goal when close; otherwise only require the next horizon to
    # be free now. A distant stopped person is not a permanent topological wall.
    # Where the guide point itself is occupied, people head for the nearest open
    # space beside or along their path (within OPEN_SPACE_REACH_M of it).
    for distance in sorted(set(min(length, x) for x in (horizon, 9.0, 5.0, 2.0)), reverse=True):
        target = _open_target(network, guide, distance, dynamic_free)
        if target is None:
            continue
        route = network.plan(start, target, dynamic_free)
        if route is not None and route["arc"][-1] > 0.2:
            return route, np.linalg.norm(target - np.asarray(goal)) > 0.05
    return None, False


OPEN_SPACE_REACH_M = 1.5


def _open_target(network, guide, distance, dynamic_free):
    """The guide point at `distance`, or the nearest free cell within OPEN_SPACE_REACH_M of it
    (along the guide or sideways); None when there is none."""
    shape = np.array(dynamic_free.shape)

    def free(q):
        c = network.cell(q)
        return bool((c >= 0).all() and (c < shape).all() and dynamic_free[tuple(c)])

    base = sample_route(guide, distance)
    if free(base):
        return base
    tangent = sample_route(guide, distance + 0.25) - sample_route(guide, distance - 0.25)
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-6:
        return None
    tangent = tangent / norm
    normal = np.array([-tangent[1], tangent[0]])
    reach = OPEN_SPACE_REACH_M
    offsets = sorted(
        (
            (a, b)
            for a in np.arange(-reach, reach + 1e-9, 0.5)
            for b in np.arange(-reach, reach + 1e-9, 0.5)
            if 0 < np.hypot(a, b) <= reach + 1e-9
        ),
        key=lambda x: (np.hypot(*x), -x[0]),
    )
    for along, side in offsets:
        q = sample_route(guide, distance + along) + normal * side
        if free(q):
            return q
    return None


def yield_route(network, start, heading, dynamic_free, actor_index, positions, radii, eligible, max_distance=5.0):
    """Lower priority person steps aside, or backs into reachable space.

    Priority is stable for this encounter. No teleport, body displacement, goal
    substitution, or fabricated arrival is allowed. The caller records yielding
    separately and resumes the same visit afterwards.
    """
    start = np.asarray(start)
    nearby = np.flatnonzero(eligible & (np.linalg.norm(positions - start, axis=1) < radii + radii[actor_index] + 3.0))
    nearby = nearby[nearby != actor_index]
    # A single winner continues; peers yield with deterministic right-hand bias.
    if not len(nearby) or actor_index < int(nearby.min()):
        return None
    forward = np.array([np.cos(heading), np.sin(heading)])
    right = np.array([forward[1], -forward[0]])
    for offset in (
        right * 2.0,
        right * 3.5,
        -forward * 2.0 + right * 2.0,
        -forward * 3.5,
        -right * 2.0,
        -forward * max_distance,
    ):
        target = start + offset
        if not _turnable(network, target) or not network.clear_segment(start, target, dynamic_free):
            continue
        # Prefer a refuge that increases distance to the nearest participant.
        old_distance = np.linalg.norm(positions[nearby] - start, axis=1).min()
        new_distance = np.linalg.norm(positions[nearby] - target, axis=1).min()
        if new_distance < old_distance + 0.4:
            continue
        return polyline([start, target], "temporary_yield")
    return None


def distance_to_polyline(points, p):
    """Euclidean distance from p to the polyline given by points (n>=2)."""
    pp = np.asarray(points, float)
    p = np.asarray(p, float)
    a, b = pp[:-1], pp[1:]
    d = b - a
    length2 = np.maximum((d * d).sum(1), 1e-12)
    u = np.clip(((p - a) * d).sum(1) / length2, 0.0, 1.0)
    return float(np.linalg.norm(a + u[:, None] * d - p, axis=1).min())


def step_aside_route(network, start, corridor, dynamic_free, needed_clearance, max_distance=4.0):
    """Refuge for a standing person that obstructs a traveller's intended path.

    ``corridor`` is the traveller's guide route (points). The refuge must be
    reachable by one legal connector from the exact current pose and must leave
    at least ``needed_clearance`` between the stander's centre and the corridor.
    The nearest acceptable refuge wins. No teleport; the caller returns the body
    to ``start`` afterwards and only then resumes the agenda.
    """
    start = np.asarray(start, float)
    pp = np.asarray(corridor, float)
    if len(pp) < 2:
        return None
    a, b = pp[:-1], pp[1:]
    d = b - a
    length2 = np.maximum((d * d).sum(1), 1e-12)
    u = np.clip(((start - a) * d).sum(1) / length2, 0.0, 1.0)
    foot = a + u[:, None] * d
    k = int(np.argmin(np.linalg.norm(foot - start, axis=1)))
    tangent = d[k] / np.sqrt(length2[k])
    normal = np.array([-tangent[1], tangent[0]])
    side = float(np.sign(np.dot(start - foot[k], normal)) or 1.0)
    candidates = []
    for distance in np.arange(0.5, max_distance + 1e-9, 0.25):
        for s in (side, -side):
            for along in (0.0, 1.0, -1.0, 2.0, -2.0):
                candidates.append((distance, start + normal * s * distance + tangent * along))
    shape = np.array(dynamic_free.shape)
    for distance, target in sorted(candidates, key=lambda x: (x[0], float(np.linalg.norm(x[1] - start)))):
        if distance_to_polyline(pp, target) < needed_clearance:
            continue
        cell = network.cell(target)
        if not ((cell >= 0).all() and (cell < shape).all() and dynamic_free[tuple(cell)]):
            continue
        if not _turnable(network, target) or not network.clear_segment(start, target, dynamic_free):
            continue
        return polyline([start, target], "temporary_step_aside")
    return None


def retreat_route(network, start, corridor, dynamic_free, needed_clearance, max_distance=25.0, attempts=12):
    """Way back to the nearest passing place when two walkers block each other.

    Used by the lower-priority walker of a head-on encounter in a passage too
    narrow for both. The refuge is a free cell at least ``needed_clearance``
    from the other walker's intended path (``corridor``), reachable by a legal
    route through ``dynamic_free`` within ``max_distance``. No teleport; the
    caller waits there and then resumes the same visit.
    """
    start = np.asarray(start, float)
    pp = np.asarray(corridor, float)
    if len(pp) < 2:
        return None
    ids = np.asarray(network.tree.query_ball_point(start, max_distance), int)
    if not len(ids):
        return None
    cells = network.nodes[ids]
    keep = dynamic_free[cells[:, 0], cells[:, 1]]
    if getattr(network, "turn_free", None) is not None:
        keep &= network.turn_free[cells[:, 0], cells[:, 1]]
    ids = ids[keep]
    if not len(ids):
        return None
    q = network.xy[ids]
    a, b = pp[:-1], pp[1:]
    d = b - a
    length2 = np.maximum((d * d).sum(1), 1e-12)
    u = np.clip((((q[:, None, :] - a[None]) * d[None]).sum(-1)) / length2[None], 0.0, 1.0)
    gap = np.linalg.norm(a[None] + u[..., None] * d[None] - q[:, None, :], axis=-1).min(1)
    ok = np.flatnonzero(gap >= needed_clearance)
    if not len(ok):
        return None
    order = ok[np.argsort(np.linalg.norm(q[ok] - start, axis=1))]
    tried = []
    for k in order:
        if any(np.linalg.norm(q[k] - r) < 1.5 for r in tried):
            continue
        tried.append(q[k])
        route = network.plan(start, q[k], dynamic_free)
        if route is not None and route["arc"][-1] <= 2 * max_distance:
            route["road"] = "temporary_retreat"
            return route
        if len(tried) >= attempts:
            break
    return None
