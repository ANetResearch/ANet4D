"""Finite journeys between mapped outdoor approaches, not repeated short patrols.

A POI approach is deliberately not treated as a verified door or an indoor visit.
The destination, purpose and scheduled departure are fixed before simulation.
"""

import numpy as np

from awr.life.civil.agenda import Agenda, Visit


def plan_outdoor_visits(network, places, rng, actor, positions, sizes, yaws, radius, departure_s, static_free=None):
    free = network.obstacle_mask(positions, sizes, radius, yaws, min_separation=3.0)
    from scipy.ndimage import label

    # Other people are temporary obstacles, not permanent cuts in the street
    # graph. Plan intent through static geometry; reserve the actual spawn
    # against everyone and resolve moving-body encounters during simulation.
    planned_free = free if static_free is None else static_free
    # Long waits must happen in pockets, not in the middle of a single-file
    # passage. A body placed on an articulation point can immobilise an entire
    # neighbourhood before its scheduled departure.
    from scipy.ndimage import distance_transform_edt

    if static_free is not None and not hasattr(network, "waiting_clearance"):
        network.waiting_clearance = distance_transform_edt(static_free) * network.step
        # This crowd model does not yet implement single-file right-of-way.
        # Admit its journeys only through measured space with passing room,
        # rather than claiming that disconnected queues are realistic motion.
        network.travel_free = np.asarray(static_free) & (network.waiting_clearance >= 1.5)
        network.static_components = label(network.travel_free)[0]
    clearance = (
        network.waiting_clearance if static_free is not None else distance_transform_edt(planned_free) * network.step
    )
    dynamic_components = network.static_components if static_free is not None else label(planned_free)[0]
    if static_free is not None:
        planned_free = network.travel_free
    node_components = dynamic_components[network.nodes[:, 0], network.nodes[:, 1]]
    valid = []
    if not hasattr(network, "assigned_waiting_places"):
        network.assigned_waiting_places = set()
    if not hasattr(network, "assigned_waiting_points"):
        network.assigned_waiting_points = []
    for p in places:
        if any(np.linalg.norm(np.asarray(p["xy"]) - q) < 3.0 for q in network.assigned_waiting_points):
            continue
        cell = network.cell(p["xy"])
        if (
            p["id"] not in network.assigned_waiting_places
            and (cell >= 0).all()
            and (cell < free.shape).all()
            and free[tuple(cell)]
            and clearance[tuple(cell)] >= 1.5
        ):
            valid.append((p, int(network.labels[tuple(cell)])))
    labels = network.labels[network.free]
    for ii in rng.permutation(len(valid))[:80]:
        dest, component = valid[ii]
        goal = np.asarray(dest["xy"])
        distance = np.linalg.norm(network.xy - goal, axis=1)
        destination_component = dynamic_components[tuple(network.cell(goal))]
        if destination_component == 0:
            continue
        source_ids = np.flatnonzero(
            (labels == component)
            & (distance >= 80)
            & (distance <= 500)
            & (node_components == destination_component)
            & free[network.nodes[:, 0], network.nodes[:, 1]]
            & (clearance[network.nodes[:, 0], network.nodes[:, 1]] >= 1.5)
        )
        if len(source_ids) and network.assigned_waiting_points:
            from scipy.spatial import cKDTree

            separation = cKDTree(network.assigned_waiting_points).query(network.xy[source_ids])[0]
            source_ids = source_ids[separation >= 3.0]
        if not len(source_ids):
            continue
        for source in rng.choice(source_ids, min(6, len(source_ids)), replace=False):
            start = network.xy[source]
            route = network.plan(start, goal, planned_free)
            if route is None:
                continue
            purpose = ("rest_in_" if dest.get("function") == "public_open_space" else "wait_outside_") + dest["kind"]
            visits = [Visit(dest["id"], tuple(goal), purpose, float(rng.uniform(60, 240)))]
            # Optional second errand must be another mapped facility, reached
            # by a real path. It is not chosen after getting stuck.
            for jj in rng.permutation(len(valid)):
                other, comp = valid[jj]
                if (
                    comp != component
                    or other["poi"] == dest["poi"]
                    or np.linalg.norm(np.asarray(other["xy"]) - goal) < 45
                    or np.linalg.norm(np.asarray(other["xy"]) - start) < 3
                ):
                    continue
                if network.plan(goal, other["xy"], planned_free) is not None:
                    purpose = ("rest_in_" if other.get("function") == "public_open_space" else "wait_outside_") + other[
                        "kind"
                    ]
                    visits.append(Visit(other["id"], tuple(other["xy"]), purpose, float(rng.uniform(90, 300))))
                    break
            visits.append(Visit(actor + "-departure-point", tuple(start), "finish_outdoor_journey", 0.0))
            agenda = Agenda(actor, "finite_outdoor_errands", visits, departure_s)
            network.assigned_waiting_places.update(v.place for v in visits[:-1])
            # The departure point is also where this person finally stands for
            # the rest of the recording. Reserve it like any other waiting
            # place so that finished residents do not accumulate in one pocket
            # and close a corridor for later travellers.
            network.assigned_waiting_points.extend(np.asarray(v.xy) for v in visits)
            return route, agenda
    return None, None


# ---------------------------------------------------------------------------
# Daily-life chains (life scenarios). The finite errand planner above is kept
# unchanged so that published versions remain reproducible from their source.
# ---------------------------------------------------------------------------

# Purpose and dwell range (s) per mapped facility kind. Every stay happens at a
# measured outdoor waiting point: no building has a verified entrance, so no
# activity is represented as being inside.
LIFE_FACILITY_PURPOSES = {
    "park": ("rest_in_park", (240.0, 720.0)),
    "pitch": ("watch_at_pitch", (300.0, 900.0)),
    "place_of_worship": ("wait_outside_place_of_worship", (240.0, 600.0)),
    "school": ("wait_outside_school", (180.0, 600.0)),
    "cafe": ("wait_outside_cafe", (240.0, 720.0)),
    "post_office": ("wait_outside_post_office", (120.0, 420.0)),
    "townhall": ("wait_outside_townhall", (120.0, 420.0)),
    "courthouse": ("wait_outside_courthouse", (120.0, 420.0)),
}
LIFE_HOUSEHOLD = ("call_at_household", (180.0, 540.0))
LIFE_HOME = ("pause_at_home_street", (60.0, 240.0))


def life_pools(network, semantics, places, static_free, rng, cfg=None, facility_kinds=None):
    """Standing places for daily-life chains, computed once per walking network.

    Standing is only allowed off the carriageway (sidewalks, verges, squares,
    parks, SUMO pedestrian areas) with `stand_clearance_m` of measured room
    around the admitted body disk, so that a person waiting there leaves the
    street passable.

    - sites: every mapped outdoor approach of a known facility kind is a site
      with up to `max_slots` standing slots within `slot_radius_m`, at least
      `slot_spacing_m` apart, including slots belonging to other approaches or
      facilities. Slots are shared over time: a visitor takes a free
      slot of the site when setting off (see `bind_site_visit`). Approaches are
      not verified doors; nobody is shown entering a building.
    - street points: standing points thinned to `street_spacing_m`; a resident's
      home is one of them (the residence itself is not mapped: the semantics
      hold 90 building footprints for the whole stage). Points 2.5-8 m outside a
      mapped footprint are also used for calls at a household.
    """
    import shapely
    from scipy.ndimage import distance_transform_edt, label
    from scipy.spatial import cKDTree
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    cfg = cfg or {}
    stand_clear = float(cfg.get("stand_clearance_m", 0.5))
    facility_clear = float(cfg.get("facility_circulation_clearance_m", stand_clear))
    if not np.isfinite(facility_clear) or facility_clear < stand_clear:
        raise ValueError("Facility circulation clearance must be finite and at least standing clearance")
    travel_clear = float(cfg.get("travel_clearance_m", 1.5))
    slot_radius = cfg.get("slot_radius_m", {"park": 5.0, "pitch": 10.0, "default": 9.0})
    slot_spacing = float(cfg.get("slot_spacing_m", 2.0))
    max_slots = int(cfg.get("max_slots", 6))
    street_spacing = float(cfg.get("street_spacing_m", 3.0))
    approach_m = cfg.get("household_approach_m", [2.5, 8.0])
    static_free = np.asarray(static_free, bool)
    clear = distance_transform_edt(static_free) * network.step
    network.waiting_clearance = clear
    network.travel_free = static_free & (clear >= travel_clear)
    network.static_components = label(static_free)[0]
    comps = network.static_components
    off_road = ~network.carriageway if getattr(network, "carriageway", None) is not None else np.ones_like(static_free)
    standing = getattr(network, "standing_free", None)
    # New scene networks carry distinct permission for sustained activity.
    # Legacy/custom networks keep their explicitly supplied historical mask.
    stand = static_free & (standing if standing is not None else off_road) & (clear >= stand_clear)
    if getattr(network, "turn_free", None) is not None:
        stand &= network.turn_free  # a person arriving at or leaving a standing point turns there
    kinds = facility_kinds or LIFE_FACILITY_PURPOSES
    ids = np.flatnonzero(stand[network.nodes[:, 0], network.nodes[:, 1]])
    pts = network.xy[ids]
    tree = cKDTree(pts) if len(ids) else None

    def thin(order, spacing, limit=None):
        kept, grid = [], {}
        for k in order:
            q = pts[k]
            key = (int(q[0] // spacing), int(q[1] // spacing))
            if any(
                np.linalg.norm(q - r) < spacing
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for r in grid.get((key[0] + dx, key[1] + dy), ())
            ):
                continue
            grid.setdefault(key, []).append(q)
            kept.append(q)
            if limit and len(kept) >= limit:
                break
        return kept

    sites, slot_grid = [], {}
    if cfg.get("open_region_activities", False):
        # A public park is an area, not a collection of door approaches. The
        # old approach-radius rule placed some "park" stays outside the park
        # and concentrated visitors on its entrance/through path. Compile actual
        # supported resting capacity throughout each mapped open region instead.
        # Separate connected components remain separate access sites of one POI.
        regions = {
            a["id"]: a
            for a in semantics.get("open_areas", [])
            if a.get("kind") in kinds and len(a.get("polygon", [])) >= 3
        }
        places = [
            p
            for p in places
            if not (p.get("function") == "public_open_space" and p.get("poi", p.get("source", p["id"])) in regions)
        ]
        rest_margin = max(
            float(getattr(network, "radius", 0.0)), float(np.linalg.norm(getattr(network, "turn_body", (0.0, 0.0))) / 2)
        )
        for region_id, area in sorted(regions.items()):
            polygon = Polygon(area["polygon"])
            if not polygon.is_valid:
                raise ValueError("Invalid mapped open region: " + region_id)
            interior = polygon.buffer(-rest_margin)
            eligible = shapely.contains_xy(interior, pts[:, 0], pts[:, 1])
            eligible &= clear[network.nodes[ids, 0], network.nodes[ids, 1]] >= float(
                cfg.get("open_region_clearance_m", 1.5)
            )
            candidates = np.flatnonzero(eligible)
            components = comps[network.nodes[ids[candidates], 0], network.nodes[ids[candidates], 1]]
            for component in sorted(set(components.tolist()) - {0}):
                near = candidates[components == component]
                places.append(
                    dict(
                        id=f"{region_id}-open-component-{component}",
                        poi=region_id,
                        kind=area["kind"],
                        xy=area["xy"],
                        function="public_open_space",
                        region_candidates=near.tolist(),
                        region_polygon=area["polygon"],
                    )
                )
    for pl in places:
        # Outdoor accessibility is not a land-use category: a pitch remains a pitch.
        # Unknown usage must not manufacture park activities or park capacity.
        kind = pl.get("kind")
        if kind not in kinds or tree is None:
            continue
        radius = float(slot_radius.get(kind, slot_radius.get("default", 9.0)))
        regional = "region_candidates" in pl
        near = pl["region_candidates"] if regional else tree.query_ball_point(pl["xy"], radius)
        if not regional:
            # Long stays must not occupy a narrow fringe merely because a
            # resting body fits there. This is map-level circulation space;
            # it does not waive live occupancy or route admission.
            near = [k for k in near if clear[tuple(network.nodes[ids[k]])] >= facility_clear]
        if not near:
            continue
        near = sorted(
            near,
            key=(lambda k: (-clear[tuple(network.nodes[ids[k]])], *pts[k]))
            if regional
            else lambda k: float(np.linalg.norm(pts[k] - pl["xy"])),
        )
        limit = int(cfg.get("max_open_component_slots", 512)) if regional else max_slots
        if limit < 1:
            raise ValueError("Positive open-region slot budget required")
        slots = []
        # Different outdoor approaches can describe the same physical space.
        # A distinct slot id is not distinct capacity: reserve room globally,
        # not merely distinct grid coordinates. Rejected candidates must not
        # thin away later usable candidates at the edge of an occupied site.
        for k in near:
            q = pts[k]
            key = (int(q[0] // slot_spacing), int(q[1] // slot_spacing))
            if any(
                np.linalg.norm(q - r) < slot_spacing
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for r in slot_grid.get((key[0] + dx, key[1] + dy), ())
            ):
                continue
            if any(np.linalg.norm(q - r) < slot_spacing for r in slots):
                continue
            if slots and int(comps[tuple(network.cell(q))]) != int(comps[tuple(network.cell(slots[0]))]):
                continue
            slots.append(q)
            if len(slots) >= limit:
                break
        if not slots:
            continue
        comp = int(comps[tuple(network.cell(slots[0]))])
        site = dict(
            id=pl["id"],
            xy=np.asarray(pl["xy"], float),
            kind=kind,
            facility=pl.get("poi", pl["id"]),
            comp=comp,
            function=pl.get("function"),
            semantic_source=pl["id"],
            affordances=(("outdoor_stay",) if pl.get("function") in ("outdoor_approach", "public_open_space") else ()),
            slots=[dict(id=f"{pl['id']}#slot{j}", xy=q.copy()) for j, q in enumerate(slots)],
        )
        if regional:
            site["capacity_evidence"] = dict(
                source_region=pl["poi"],
                polygon=pl["region_polygon"],
                rest_margin_m=rest_margin,
                circulation_clearance_m=float(cfg.get("open_region_clearance_m", 1.5)),
                slot_budget=limit,
                admissible_candidate_cells=len(near),
                note="Measured resting capacity; routes and live body occupancy still require admission",
            )
        else:
            site["capacity_evidence"] = dict(
                source_approach=pl["id"],
                circulation_clearance_m=facility_clear,
                slot_budget=limit,
                admissible_candidate_cells=len(near),
                note="Measured outdoor approach capacity, not a verified door or indoor service",
            )
        for q in slots:
            key = (int(q[0] // slot_spacing), int(q[1] // slot_spacing))
            slot_grid.setdefault(key, []).append(q)
        sites.append(site)
    polys = [Polygon(b["polygon"]) for b in semantics.get("buildings", []) if len(b.get("polygon", [])) >= 3]
    union = unary_union(polys) if polys else None
    taken = cKDTree(np.array([sl["xy"] for st in sites for sl in st["slots"]])) if sites else None
    street = []
    candidates = [
        q
        for q in thin(rng.permutation(len(ids)), street_spacing)
        if taken is None or taken.query(q)[0] >= street_spacing
    ]
    distances = (
        shapely.distance(union, shapely.points(np.array(candidates)))
        if union is not None and candidates
        else [None] * len(candidates)
    )
    for q, dist_b in zip(candidates, distances, strict=False):
        dist_b = None if dist_b is None else float(dist_b)
        street.append(
            dict(
                id=f"street-point-{len(street):05d}",
                xy=q.copy(),
                kind="household" if dist_b is not None and approach_m[0] <= dist_b <= approach_m[1] else "street",
                facility=None,
                comp=int(comps[tuple(network.cell(q))]),
                building_distance_m=None if dist_b is None else round(dist_b, 1),
            )
        )
    return dict(sites=sites, street=street, standing_policy=getattr(network, "standing_policy", None))


def bind_site_visit(visit, slots, holders, position, others, other_radii, radius, margin=0.1):
    """Choose a free standing slot of a site for a visitor who is setting off.

    A slot is free when no other resident holds it and no other body currently
    stands on it (centre distance below the sum of both body radii plus
    `margin`). The nearest free slot is taken. Returns the bound Visit and slot
    id, or (None, None) when every slot is held or occupied.
    """
    best = None
    for sl in slots:
        if sl["id"] in holders:
            continue
        if len(others) and bool(np.any(np.linalg.norm(others - sl["xy"], axis=1) < other_radii + radius + margin)):
            continue
        d = float(np.linalg.norm(sl["xy"] - position))
        if best is None or d < best[0]:
            best = (d, sl)
    if best is None:
        return None, None
    sl = best[1]
    return Visit(sl["id"], (float(sl["xy"][0]), float(sl["xy"][1])), visit.purpose, visit.dwell_s), sl["id"]


def plan_daily_life(
    network, pools, rng, actor, positions, sizes, yaws, radius, departure_s, horizon_s, static_free, cfg=None
):
    """A resident's daily chain: from a home street point to mapped facilities (and,
    where a mapped footprint is near, a call at a household), back home and out again,
    repeated until well past the recording horizon (no scripted 'finished' standing).

    The home point is exclusive to its resident. Facility visits name a site
    ('site:<id>'); the standing slot is chosen when the resident sets off, so
    several residents can use the same park or church forecourt at once or in
    turn, as in a real town. Legs stay inside one measured walking component.
    """
    cfg = cfg or {}
    dwell_cfg = cfg.get("dwell_s", {})
    walk_speed = float(cfg.get("planning_speed_mps", 1.05))
    detour = float(cfg.get("planning_detour", 1.45))
    leg_m = cfg.get("leg_distance_m", [40.0, 450.0])
    n_dest = cfg.get("destinations_per_person", [2, 3])
    extra_s = float(cfg.get("chain_extra_s", 900.0))
    load_factor = float(cfg.get("site_load_per_slot", 2.0))
    free = network.obstacle_mask(positions, sizes, radius, yaws, min_separation=3.0)
    if not hasattr(network, "assigned_homes"):
        network.assigned_homes = []
        network.site_load = {}
        network.site_stays = {}  # site id -> planned (arrive_s, leave_s) stays of earlier residents
    from scipy.spatial import cKDTree

    homes_tree = cKDTree(network.assigned_homes) if network.assigned_homes else None
    fails = network.__dict__.setdefault("life_failures", {})

    def fail(reason):
        fails[reason] = fails.get(reason, 0) + 1

    by_comp = {}
    for st in pools["sites"]:
        by_comp.setdefault(st["comp"], []).append(st)
    hh_by_comp = {}
    for sp in pools["street"]:
        if sp["kind"] == "household":
            hh_by_comp.setdefault(sp["comp"], []).append(sp)
    homes = [
        h
        for h in pools["street"]
        if h["comp"] in by_comp and (homes_tree is None or homes_tree.query(h["xy"])[0] >= 3.0)
    ]
    if not homes:
        fail("no_open_home_point")
        return None, None, None

    def purpose_of(st):
        name, rng_s = LIFE_HOUSEHOLD if st["kind"] == "household" else LIFE_FACILITY_PURPOSES[st["kind"]]
        lo, hi = dwell_cfg.get(name, rng_s)
        return name, float(rng.uniform(lo, hi))

    order = rng.permutation(len(homes))
    order = sorted(order, key=lambda k: 0 if homes[k]["kind"] == "household" else 1)
    for hi in order[:200]:
        home = homes[hi]
        if not free[tuple(network.cell(home["xy"]))]:
            fail("home_within_3m_of_a_body")
            continue

        def ok(st):
            room = len(st["slots"]) * load_factor if "slots" in st else load_factor
            return (
                network.site_load.get(st["id"], 0) < room
                and leg_m[0] <= np.linalg.norm(st["xy"] - home["xy"]) <= leg_m[1]
            )

        fac = [st for st in by_comp.get(home["comp"], []) if ok(st)]
        hh = [
            sp
            for sp in hh_by_comp.get(home["comp"], [])
            if sp["id"] != home["id"] and ok(sp) and (homes_tree is None or homes_tree.query(sp["xy"])[0] >= 3.0)
        ]
        want = int(rng.integers(n_dest[0], n_dest[1] + 1))
        picks = []
        # Prefer distinct facilities; parks last so that every kind is visited.
        for k in sorted(rng.permutation(len(fac)), key=lambda k: (fac[k]["kind"] == "park", rng.random())):
            st = fac[k]
            if any(p["facility"] == st["facility"] for p in picks):
                continue
            picks.append(st)
            if len(picks) >= want:
                break
        if len(picks) < want and hh:
            picks.append(hh[int(rng.integers(len(hh)))])
        if len(picks) < n_dest[0]:
            fail("too_few_destinations_in_reach" if fac or hh else "no_destination_in_reach")
            continue
        first = picks[0]["slots"][0]["xy"] if "slots" in picks[0] else picks[0]["xy"]
        route = network.plan(home["xy"], first, static_free)
        if route is None:
            fail("first_leg_unplannable")
            continue
        visits, t, prev = [], float(departure_s), home["xy"]
        stays = []  # (site, arrive_s, leave_s) of this chain
        k = 0
        while t < horizon_s + extra_s:
            for _ in range(int(rng.integers(1, 3))):
                st = picks[k % len(picks)]
                k += 1
                if visits and visits[-1].place == "site:" + st["id"]:
                    st = picks[k % len(picks)]
                    k += 1
                name, dwell = purpose_of(st)
                visits.append(Visit("site:" + st["id"], tuple(float(v) for v in st["xy"]), name, dwell))
                # the slot is held from setting off until leaving it
                t0 = t
                t += np.linalg.norm(st["xy"] - prev) * detour / walk_speed + dwell
                stays.append((st, t0, t))
                prev = st["xy"]
            lo, hi_ = dwell_cfg.get(LIFE_HOME[0], LIFE_HOME[1])
            visits.append(
                Visit(actor + "-home", tuple(float(v) for v in home["xy"]), LIFE_HOME[0], float(rng.uniform(lo, hi_)))
            )
            t += np.linalg.norm(home["xy"] - prev) * detour / walk_speed + visits[-1].dwell_s
            prev = home["xy"]

        # Stays planned until the end of the recording must fit the standing slots
        # of each site together with the stays of residents planned before; the
        # margin grows along the chain with the drift of actual walking times.
        # Otherwise residents set off to a full site and wait at home.
        def overfull(st, a, b):
            margin = min(180.0, 30.0 + 0.15 * (a - float(departure_s)))
            cap = len(st.get("slots", [None]))
            return (
                sum(1 for a2, b2 in network.site_stays.get(st["id"], []) if a2 < b + margin and b2 > a - margin) >= cap
            )

        if any(overfull(st, a, b) for st, a, b in stays if a < horizon_s):
            fail("site_slots_taken_at_planned_time")
            continue
        agenda = Agenda(actor, "daily_life", visits, departure_s)
        network.assigned_homes.append(home["xy"])
        for st in picks:
            network.site_load[st["id"]] = network.site_load.get(st["id"], 0) + 1
        for st, a, b in stays:
            if a < horizon_s:
                network.site_stays.setdefault(st["id"], []).append((a, b))
        meta = dict(
            home=home["id"],
            home_xy=[float(v) for v in home["xy"]],
            home_kind=home["kind"],
            home_building_distance_m=home.get("building_distance_m"),
            destinations=[
                dict(site=p["id"], kind=p["kind"], facility=p["facility"], slots=len(p.get("slots", [None])))
                for p in picks
            ],
            planned_visits=len(visits),
            planned_until_s=round(t, 1),
        )
        return route, agenda, meta
    fail("person_rejected")
    return None, None, None
