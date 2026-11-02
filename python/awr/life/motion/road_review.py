"""Apply source-bound visual road exclusions before offline fleet planning.

Photogrammetry may depict a vehicle on a physically flat surface. Support and
collision evidence cannot certify that route's visual semantics. This filter
only removes candidates; it grants no driving permission or physical approval.
"""

import re

from awr.life.civil.driving_graph import load_candidate
from awr.life.common.schema import schema_id

from .fleet_frontiers import nominate


def screen(document, roads, review, context):
    required = {"world_manifest_sha256", "appearance_sha256", "source_graph_sha256"}
    if (
        schema_id(review.get("schema")) != "awr.life.source_road_review.v1"
        or set(review.get("context", {})) != required
        or set(context) != required
        or any(not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v) for v in context.values())
        or review["context"] != context
    ):
        raise ValueError("Review must match the exact source world, appearance and road graph")
    evidence = review.get("evidence", {})
    if not evidence or any(
        not isinstance(k, str) or not k or not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v)
        for k, v in evidence.items()
    ):
        raise ValueError("Hashed review evidence required")
    rows = review.get("rejected_roads", [])
    if not rows or len(rows) > len(roads):
        raise ValueError("Explicit bounded rejected source roads required")
    rejected = {}
    for row in rows:
        key = row.get("road")
        if (
            key not in roads
            or key in rejected
            or not isinstance(row.get("reason"), str)
            or not row["reason"].strip()
            or not row.get("evidence")
            or not isinstance(row["evidence"], list)
            or any(p not in evidence for p in row["evidence"])
        ):
            raise ValueError("Unique known road with reason and bound evidence required")
        rejected[key] = row
    graph = load_candidate(document)
    blocked = {}
    unclassified = []
    for key, edge in graph.edges.items():
        names = {graph.nodes[edge[end]].get("road") for end in ("source", "target")}
        resources = [edge.get("traffic_resource"), *edge.get("traffic_resources", [])]
        names.update(x[5:] for x in resources if isinstance(x, str) and x.startswith("road:"))
        names.discard(None)
        if not names or names - roads.keys():
            unclassified.append(key)
        hits = sorted(names & rejected.keys())
        if hits:
            blocked[key] = hits
    return dict(
        schema="awr.life.source_road_review_screen.v1",
        candidate_only=True,
        release_evidence=False,
        blocked_edges=sorted(blocked),
        blocked_by_source=blocked,
        unclassified_edges=sorted(unclassified),
        rejected_roads=sorted(rejected),
        context=dict(context),
        limitations=[
            "Unlisted roads are unreviewed, never visually certified clear.",
            "Whole source-road quarantine is conservative, not an obstacle polygon.",
            "No physical geometry, actor placement or traffic permission is changed.",
        ],
    )


def nominate_reviewed(document, roads, origins, review, context, *, max_families=4):
    result = screen(document, roads, review, context)
    graph = load_candidate(document)
    # Only the temporary planning graph is filtered. Original scene actors,
    # source metadata, candidate file and previous evidence remain intact.
    for key in set(result["blocked_edges"]) | set(result["unclassified_edges"]):
        del graph.edges[key]
    eligible = {k: v for k, v in roads.items() if k not in result["rejected_roads"]}
    selection = nominate(graph, eligible, origins, max_families=max_families)
    return dict(review_screen=result, frontier_nomination=selection, candidate_only=True, release_evidence=False)
