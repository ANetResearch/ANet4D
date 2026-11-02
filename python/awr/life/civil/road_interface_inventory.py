"""Source-incidence audit of existing joins; no geometry or permissions added.

Road resources and source direction identify candidate interfaces. Incidence
alone does not certify a junction approach's geometry, turning restrictions or
the validity of grouping its alternative poses. Output stays diagnostic.
"""

import math
from collections import Counter, defaultdict


def inventory(document, source_graph):
    roads = {}
    for feature in source_graph["features"]:
        p = feature["properties"]
        if p.get("feature") != "edge" or "vehicle" not in p.get("agents", []):
            continue
        key = p["edge_id"]
        if key in roads or type(p.get("oneway")) is not bool:
            raise ValueError("Unique source roads with explicit direction required")
        coordinates = feature["geometry"]["coordinates"]
        roads[key] = dict(
            start=str(p["from_node"]),
            end=str(p["to_node"]),
            oneway=p["oneway"],
            xy_start=coordinates[0][:2],
            xy_end=coordinates[-1][:2],
        )
    if not roads:
        raise ValueError("No source vehicle roads")
    nodes = document["nodes"]
    counts = Counter()
    rows = []
    ports = defaultdict(set)
    for eid, edge in sorted(document["edges"].items()):
        if edge["kind"] != "join":
            continue
        a, b = (nodes[edge[k]] for k in ("source", "target"))
        resources = [n.get("traffic_resource", "") for n in (a, b)]
        ids = [r[5:] if r.startswith("road:") else None for r in resources]
        row = dict(edge=eid, source=edge["source"], target=edge["target"], roads=ids)
        if any(r not in roads for r in ids):
            status = "missing_source_road"
        elif any(n.get("direction") not in ("forward", "reverse") for n in (a, b)):
            status = "missing_source_direction"
        elif any(roads[r]["oneway"] and n["direction"] == "reverse" for r, n in zip(ids, (a, b), strict=False)):
            status = "against_source_oneway"
        elif ids[0] == ids[1]:
            status = "same_road_same_direction" if a["direction"] == b["direction"] else "same_road_direction_change"
        else:
            ra, rb = (roads[r] for r in ids)
            shared = {ra["start"], ra["end"]} & {rb["start"], rb["end"]}
            incoming = ra["end"] if a["direction"] == "forward" else ra["start"]
            outgoing = rb["start"] if b["direction"] == "forward" else rb["end"]
            if not shared:
                status = "no_shared_source_junction"
            elif incoming != outgoing:
                status = "source_direction_not_a_junction_turn"
            elif edge.get("source_junction") not in (None, incoming):
                status = "explicit_junction_conflicts_with_source"
            else:
                status = "source_incident_turn_candidate"
                row["source_junction"] = incoming
                centres = [
                    ra["xy_end"] if a["direction"] == "forward" else ra["xy_start"],
                    rb["xy_start"] if b["direction"] == "forward" else rb["xy_end"],
                ]
                if math.dist(*centres) > 1e-6:
                    status = "source_junction_coordinates_disagree"
                else:
                    row["approach_distance_m"] = [
                        math.dist(n["pose"][:2], c) for n, c in zip((a, b), centres, strict=False)
                    ]
                    for rid, side, nid, direction in zip(
                        ids,
                        ("incoming", "outgoing"),
                        (edge["source"], edge["target"]),
                        (a["direction"], b["direction"]),
                        strict=False,
                    ):
                        ports[(rid, incoming, side, direction)].add(nid)
        counts[status] += 1
        if edge.get("source_junction") is not None:
            counts["explicit_junction_metadata"] += 1
        row["status"] = status
        rows.append(row)
    catalog = [
        dict(road=r, junction=j, side=s, direction=d, nodes=sorted(ns)) for (r, j, s, d), ns in sorted(ports.items())
    ]
    # A short road can have the same candidate pose acting at opposite ends.
    # Record this; never turn it into a zero-length through connection.
    memberships = defaultdict(list)
    for i, port in enumerate(catalog):
        for n in port["nodes"]:
            memberships[n].append(i)
    ambiguous = [dict(node=n, interfaces=ix) for n, ix in sorted(memberships.items()) if len(ix) > 1]
    return dict(
        schema="awr.life.road_interface_inventory.v1",
        join_counts=dict(counts),
        source_roads=len(roads),
        candidate_interfaces=catalog,
        ambiguous_memberships=ambiguous,
        joins=rows,
        admitted=False,
        scope="Source incidence/direction only. Candidate interface poses need spatial/source review; "
        "unclassified joins are not automatically illegal. No map or parking rule changed.",
    )
