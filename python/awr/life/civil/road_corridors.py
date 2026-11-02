"""Measured road candidates with source direction and explicit traffic modes.

Two-way geometry is not permission for simultaneous opposing traffic. Centre
corridors require exclusive reservations; all overlapping alternatives remain
subject to body-level traffic conflict admission before execution.
"""

import math
from collections import Counter

import numpy as np


def source_roads(semantics, source_graph):
    edges = {}
    for feature in source_graph["features"]:
        p = feature["properties"]
        if p.get("feature") == "edge":
            if p["edge_id"] in edges:
                raise ValueError("Duplicate source road ID")
            edges[p["edge_id"]] = feature
    result = []
    for road in semantics["roads"]:
        if road["id"] not in edges:
            raise ValueError("Road has no source direction metadata: " + road["id"])
        feature = edges[road["id"]]
        p = feature["properties"]
        if type(p.get("oneway")) is not bool or "vehicle" not in p.get("agents", []):
            raise ValueError("Unverified vehicle access/direction: " + road["id"])
        a, b = np.asarray(road["line"], float), np.asarray(feature["geometry"]["coordinates"], float)[:, :2]
        if a.shape != b.shape or not np.allclose(a, b, atol=0.01, rtol=0.0):
            raise ValueError("Road coordinates/orientation differ from source: " + road["id"])
        result.append(
            dict(
                road,
                oneway=p["oneway"],
                source_osm_id=p.get("osm_id"),
                source_from=p["from_node"],
                source_to=p["to_node"],
            )
        )
    return result


def free_offsets(surface, geometry, size, lateral):
    """free[i, k]: a body centred `lateral[i]` metres to the right of station k of a road (awr.life.civil.curb.road_geometry)
    and heading along it stands on vehicle support, by the footprint probes of awr.life.civil.grounded.support_mask (its
    test between stations and its curvature test are left to the validation of the chosen lane)."""
    from .grounded import footprint_support, probe_offsets

    _, ds, pp, yaw = geometry
    c, s = np.cos(yaw), np.sin(yaw)
    right = np.column_stack([s, -c])
    probes = probe_offsets(size, "vehicle")
    free = np.zeros((len(lateral), len(ds)), bool)
    for i, offset in enumerate(lateral):
        free[i] = footprint_support(surface, pp + offset * right, c, s, probes, "vehicle")
    return free


def lateral_paths(free, score, step=1):
    """Paths across a station grid that keep to free cells and move by at most `step` cells from one station to the
    next: [(k0, k1, cells)], cells[j] being the path's cell at station k0+j. A path runs as far as any such path from
    its first station reaches (the cells reachable there are propagated station by station), and among the paths
    that reach its last station it keeps, from the last station back, the cell of the highest `score` (an array
    shaped like `free`)."""
    n_lat, n = free.shape
    runs = []
    k = 0
    while k < n:
        if not free[:, k].any():
            k += 1
            continue
        k0 = k
        reach = [free[:, k].copy()]
        k += 1
        while k < n:
            grown = reach[-1].copy()
            for d in range(1, step + 1):
                grown[d:] |= reach[-1][:-d]
                grown[:-d] |= reach[-1][d:]
            here = grown & free[:, k]
            if not here.any():
                break
            reach.append(here)
            k += 1
        cells = np.empty(len(reach), int)
        idx = np.flatnonzero(reach[-1])
        cells[-1] = idx[np.argmax(score[idx, k0 + len(reach) - 1])]
        for j in range(len(reach) - 2, -1, -1):
            lo = max(cells[j + 1] - step, 0)
            idx = np.flatnonzero(reach[j][lo : cells[j + 1] + step + 1]) + lo
            cells[j] = idx[np.argmax(score[idx, k0 + j])]
        runs.append((k0, k0 + len(reach), cells))
    return runs


def carriageway_lanes(
    surface,
    road,
    geometry,
    size,
    *,
    lateral_step_m=0.05,
    margin_m=0.1,
    smoothing_m=5.0,
    passing_gap_m=0.5,
    centre_clearance_m=1.5,
    lane_clearance_m=0.5,
):
    """Lane candidates of one source road whose lateral position follows the measured carriageway.

    Within the mapped road (the body kept inside its width), the offsets at which the body stands on vehicle support
    (free_offsets, less `margin_m` either side) are searched for continuous paths (lateral_paths, at most
    `lateral_step_m` of lateral shift per .35 m station). A one-way or exclusive two-way lane keeps to the middle of
    the free offsets (the largest lateral clearance, up to `centre_clearance_m`; then the nearest to the centre line);
    the two lanes of a separated two-way road keep a quarter width either side of the centre line where they have
    `lane_clearance_m` of clearance, and are kept where their centres lie at least a body width and `passing_gap_m`
    apart. Offsets are smoothed over `smoothing_m`. Returns [(mode, k0, forward_offsets, reverse_offsets)] over
    stations k0.. (offsets to the right of the source direction); the caller validates the resulting poses."""
    from scipy.ndimage import binary_erosion, distance_transform_edt, uniform_filter1d

    ds = geometry[1]
    width = float(road["width_m"])
    body = float(size[1])
    bound = width / 2 - body / 2
    if bound < 0.0:
        return []
    lateral = np.arange(-bound, bound + 1e-9, lateral_step_m)
    free = free_offsets(surface, geometry, size, lateral)
    keep = int(round(margin_m / lateral_step_m))
    if keep:
        free = binary_erosion(free, structure=np.ones((2 * keep + 1, 1), bool), border_value=0)
    # Lateral distance (m) from each free offset to the nearest offset where the body does not fit (or the road edge).
    padded = np.pad(free, ((1, 1), (0, 0)))
    clearance = distance_transform_edt(padded, sampling=(lateral_step_m, 1e6))[1:-1]
    window = max(int(round(smoothing_m / 0.35)) | 1, 1)

    def paths(score):
        out = np.full(len(ds), np.nan)
        run = np.full(len(ds), -1)
        for number, (k0, k1, cells) in enumerate(lateral_paths(free, score)):
            out[k0:k1] = uniform_filter1d(lateral[cells], window, mode="nearest")
            run[k0:k1] = number
        return out, run

    lanes = []
    centre, run = paths(np.minimum(clearance, centre_clearance_m) - 0.01 * np.abs(lateral)[:, None])
    mode = "oneway" if road["oneway"] else "exclusive_bidirectional"
    for a, b in _pieces(np.isfinite(centre), run, run):
        lanes.append((mode, a, centre[a:b], centre[a:b]))
    if not road["oneway"] and width >= 2 * (body + passing_gap_m):
        held = np.minimum(clearance, lane_clearance_m)
        forward, fr = paths(held - 0.1 * np.abs(lateral - width / 4)[:, None])
        reverse, rr = paths(held - 0.1 * np.abs(lateral + width / 4)[:, None])
        apart = np.isfinite(forward) & np.isfinite(reverse)
        apart[apart] = forward[apart] - reverse[apart] >= body + passing_gap_m
        for a, b in _pieces(apart, fr, rr):
            lanes.append(("separated_two_way", a, forward[a:b], reverse[a:b]))
    return lanes


def _pieces(mask, *runs):
    """Maximal station ranges where `mask` holds and every run number stays the same."""
    out = []
    for a, b in _segments(mask):
        cut = [a] + [k for k in range(a + 1, b) if any(r[k] != r[k - 1] for r in runs)] + [b]
        out += [(x, y) for x, y in zip(cut[:-1], cut[1:], strict=False)]
    return out


def _segments(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return list(zip(edges[::2], edges[1::2], strict=False))


def lane_poses(geometry, k0, offsets):
    """Points and headings of a lane at the given offsets to the right of stations k0.. of a road (the heading follows
    the lane, including its lateral shifts)."""
    _, ds, pp, yaw = geometry
    k = np.arange(k0, k0 + len(offsets))
    points = pp[k] + np.asarray(offsets)[:, None] * np.column_stack([np.sin(yaw[k]), -np.cos(yaw[k])])
    if len(points) < 2:
        return points, yaw[k]
    tangent = np.gradient(points, axis=0)
    return points, np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))


def body_within(permitted, points, yaw, size):
    """Whether the body (without padding) at each pose lies inside the permitted area (a prepared Shapely geometry)."""
    import shapely

    half = np.asarray(size[:2], float) / 2
    local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    corners = np.stack(
        [points[:, :1] + local[:, 0] * c - local[:, 1] * s, points[:, 1:] + local[:, 0] * s + local[:, 1] * c], -1
    )
    return shapely.covers(permitted, shapely.polygons(corners))


def measured_fragments(
    surface,
    semantics,
    source_graph,
    size,
    *,
    min_length_m=8.0,
    independent_directions=False,
    beside_parking=False,
    follow_carriageway=False,
    permitted_area=None,
):
    """Directed lane fragments of the source roads, measured with the actual body.

    Layouts: two separated lanes at a quarter width either side of the centre line (wide two-way roads) and one
    exclusive centre lane. With `beside_parking`, also the lanes that remain beside a curb parking strip
    (awr.life.civil.curb.road_layout, the layout the curb survey parks cars with), the strip on either side of a
    two-way road and on the right of a one-way road: a car parked at the curb leaves these lanes open, where it
    blocks the centre and quarter-width lanes of a narrow street.

    With `follow_carriageway`, the separated and the exclusive (or one-way) lanes are not held at fixed offsets from
    the mapped centre line: they follow the measured carriageway (carriageway_lanes), so that a lane passes a local
    narrowing, an obstacle on one side of the road or a centre line lying off the carriageway where the body still
    fits within the mapped road. Their poses must also keep the body inside `permitted_area` when one is given (the
    area the map compiler admits travel edges in)."""
    from .curb import poses_at, road_geometry, road_layout
    from .grounded import segments, support_mask

    if not math.isfinite(min_length_m) or min_length_m < float(size[0]):
        raise ValueError("A road fragment must be at least one car length")
    permitted = None
    if permitted_area is not None:
        import shapely

        permitted = permitted_area.buffer(1e-7)
        shapely.prepare(permitted)
    result = []
    counts = Counter()
    for road in source_roads(semantics, source_graph):
        geometry = road_geometry(road)
        ds = geometry[1]
        width = float(road["width_m"])
        counts["source_roads"] += 1
        counts["source_oneway_roads"] += int(road["oneway"])
        # (mode, forward lane offset, reverse lane offset, parking side); offsets are to the right of the source
        # direction, so the reverse lane of a separated road lies at the negative offset.
        layouts = []
        if follow_carriageway:
            for mode, k0, forward_offsets, reverse_offsets in carriageway_lanes(surface, road, geometry, size):
                forward, yaw = lane_poses(geometry, k0, forward_offsets)
                backward, back_yaw = lane_poses(geometry, k0, reverse_offsets)
                ok = support_mask(surface, forward, yaw, size, "vehicle")
                if mode != "oneway":
                    ok &= support_mask(surface, backward, back_yaw + math.pi, size, "vehicle")
                if permitted is not None:
                    ok &= body_within(permitted, forward, yaw, size)
                    if mode != "oneway":
                        ok &= body_within(permitted, backward, back_yaw, size)
                available = ("forward",) if mode == "oneway" else ("forward", "reverse")
                for a, b in segments(ok):
                    fragment = _fragment(
                        road,
                        mode,
                        ds[k0:],
                        a,
                        b,
                        forward,
                        yaw,
                        backward,
                        back_yaw,
                        available,
                        min_length_m,
                        counts,
                        surface,
                        size,
                    )
                    if fragment is not None:
                        fragment["follows_carriageway"] = True
                        fragment["lateral_offset_m"] = [
                            round(float(np.min(forward_offsets[a:b])), 3),
                            round(float(np.max(forward_offsets[a:b])), 3),
                        ]
                        result.append(fragment)
                        counts["carriageway_" + mode] += 1
        else:
            if not road["oneway"] and width >= 2 * max(3.0, float(size[1]) + 0.3):
                layouts.append(("separated_two_way", width / 4, -width / 4, None))
            if width >= float(size[1]) + 0.6:
                layouts.append(("oneway" if road["oneway"] else "exclusive_bidirectional", 0.0, 0.0, None))
        strip = road_layout(width, size) if beside_parking else None
        if strip is not None:
            two = strip["lanes"] == 2
            right = (strip["travel_offset_m"], strip["opposite_offset_m"] if two else strip["travel_offset_m"])
            if road["oneway"]:
                layouts.append(("oneway", right[0], right[0], "right"))
            else:
                mode = "separated_two_way" if two else "exclusive_bidirectional"
                layouts.append((mode, right[0], right[1], "right"))
                layouts.append((mode, -right[1], -right[0], "left"))
        for mode, offset, reverse_offset, parking_side in layouts:
            forward, yaw = poses_at(geometry, ds, offset)
            backward, _ = poses_at(geometry, ds, reverse_offset)
            forward_ok = support_mask(surface, forward, yaw, size, "vehicle")
            supported = forward_ok.copy()
            reverse_ok = None
            if mode != "oneway":
                reverse_ok = support_mask(surface, backward, yaw + math.pi, size, "vehicle")
                supported &= reverse_ok
            # A broken opposite lane must not delete an otherwise usable
            # directed corridor. Preserve source driving permissions; do not
            # infer an extra lane or permit passing in an exclusive corridor.
            selections = [(("forward",) if mode == "oneway" else ("forward", "reverse"), supported)]
            if independent_directions and mode != "oneway":
                selections = [(("forward",), forward_ok), (("reverse",), reverse_ok)]
            for available, mask in selections:
                for a, b in segments(mask):
                    fragment = _fragment(
                        road,
                        mode,
                        ds,
                        a,
                        b,
                        forward,
                        yaw,
                        backward,
                        yaw,
                        available,
                        min_length_m,
                        counts,
                        surface,
                        size,
                    )
                    if fragment is None:
                        continue
                    if parking_side is not None:
                        fragment["beside_parking"] = parking_side
                    result.append(fragment)
                    counts[mode] += 1
                    if parking_side is not None:
                        counts["beside_parking"] += 1
    return result, dict(counts)


def _fragment(road, mode, ds, a, b, forward, yaw, backward, back_yaw, available, min_length_m, counts, surface, size):
    """The fragment over samples a..b-1 of a lane (forward poses; the reverse lane's poses in the source direction),
    or None when it is shorter than `min_length_m` or fails the sampled curvature recheck."""
    from .curb import valid_route

    length = float(ds[b - 1] - ds[a])
    if length < min_length_m:
        counts["short_fragments"] += 1
        return None
    fragment = dict(
        road=road["id"],
        width_m=float(road["width_m"]),
        length_m=length,
        traffic_mode=mode,
        traffic_resource="road:" + road["id"],
        source_osm_id=road["source_osm_id"],
        station_m=[float(ds[a]), float(ds[b - 1])],
        reservation_required=True,
        source_oneway=road["oneway"],
    )
    if "forward" in available:
        fragment.update(forward_xy=forward[a:b].tolist(), forward_yaw=yaw[a:b].tolist())
    if "reverse" in available:
        fragment.update(reverse_xy=backward[a:b][::-1].tolist(), reverse_yaw=(back_yaw[a:b][::-1] + math.pi).tolist())
    # Recheck actual sampled curvature, including the short final
    # interval. support_mask's legacy fixed-step check is not enough.
    for direction in directions(fragment):
        pp = np.asarray(fragment[direction + "_xy"])
        yy = np.asarray(fragment[direction + "_yaw"])
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pp, axis=0), axis=1))]
        if np.any(np.diff(arc) <= 1e-10) or not valid_route(surface, dict(points=pp, yaw=yy, arc=arc), size):
            counts["invalid_curvature"] += 1
            return None
    return fragment


def fragments(probe):
    """Read the legacy separated-lane probe or the explicit v2 road modes."""
    if "driving_fragments" in probe:
        return probe["driving_fragments"]
    return probe["two_way_fragments"]


def directions(fragment):
    available = tuple(k for k in ("forward", "reverse") if k + "_xy" in fragment)
    if not available or any(k + "_yaw" not in fragment for k in available):
        raise ValueError("Incomplete directed road geometry")
    if fragment.get("source_oneway") and available != ("forward",):
        raise ValueError("A source one-way street cannot acquire a reverse lane")
    return available
