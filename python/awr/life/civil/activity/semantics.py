"""Static semantic compatibility, separate from geometry and runtime admission.

This consumes normalized Place/Activity objects from any environment adapter.
A source reference is traceability, not proof that its content is correct.
"""

from .director import supports_activity


def coverage(places, activities):
    places, activities = list(places), list(activities)
    if len({p.id for p in places}) != len(places):
        raise ValueError("Duplicate place ID")
    if len({a.purpose for a in activities}) != len(activities):
        raise ValueError("Duplicate activity purpose")
    rows = []
    for activity in sorted(activities, key=lambda a: a.purpose):
        candidates = sorted((p for p in places if p.kind in activity.kinds), key=lambda p: p.id)
        matched = [p for p in candidates if supports_activity(p, activity)]
        rows.append(
            dict(
                purpose=activity.purpose,
                requires=sorted(activity.requires),
                status=(
                    "legacy_unconstrained"
                    if not activity.requires
                    else "no_matching_kind"
                    if not candidates
                    else "missing_affordances"
                    if not matched
                    else "semantic_match"
                ),
                candidate_places=len(candidates),
                matched_places=len(matched),
                matched_facilities=len({p.group or p.id for p in matched}),
                missing_by_feature={
                    key: sum(key not in p.affordances for p in candidates) for key in sorted(activity.requires)
                },
                matched_sources=sorted({p.semantic_source for p in matched if p.semantic_source}),
            )
        )
    return dict(
        schema="awr.life.activity_semantics.v1",
        places=len(places),
        activities=rows,
        admitted=False,
        release_evidence=False,
        limitations=[
            "Semantic matches are necessary conditions, not route, capacity or execution proofs.",
            "Source references require independent content and spatial verification.",
            "Legacy activities with empty requirements are not feature-validated.",
            "Facility counts are distinct IDs, not a verified simultaneous population capacity.",
        ],
    )
