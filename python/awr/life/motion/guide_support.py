"""Offline candidate rejection against the existing measured support mask.

Samples are body reference points, not a whole-car swept-volume certificate.
This can reject candidates early; passing still requires native physics and
continuous body/tyre checks. It never fits terrain or mutates a driving graph.
"""

import numpy as np


def check_centers(points, surface, *, max_samples=100000):
    """Bounded rejection-only check shared by route compilation and screening."""
    if not np.isfinite(surface.cell_m) or surface.cell_m <= 0 or type(max_samples) is not int or max_samples < 2:
        raise ValueError("Finite support grid and bounded sampling required")
    spacing = min(0.25, surface.cell_m / 2)
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("Finite body reference route required")
    ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
    if np.any(ds <= 0):
        raise ValueError("Nonzero route segments required")
    raw_counts = np.ceil(ds / spacing)
    if raw_counts.sum() + 1 > max_samples:
        raise ValueError("Edge exceeds explicit support prefilter sample budget")
    counts = raw_counts.astype(int)
    samples = (
        p
        if np.all(counts == 1)
        else np.vstack(
            [*[a + np.arange(n)[:, None] / n * (b - a) for a, b, n in zip(p[:-1], p[1:], counts, strict=False)], p[-1]]
        )
    )
    bad = np.flatnonzero(~surface.supported(samples, "vehicle"))
    return dict(
        samples=len(samples),
        rejected_samples=len(bad),
        first_rejected_xy=samples[bad[0]].tolist() if len(bad) else None,
    )


def screen_centers(graph, surface, *, max_samples_per_edge=100000):
    rows = {
        key: check_centers(edge["route"]["points"], surface, max_samples=max_samples_per_edge)
        for key, edge in sorted(graph.edges.items())
    }
    spacing = min(0.25, surface.cell_m / 2)
    total = sum(r["samples"] for r in rows.values())
    return dict(
        schema="awr.life.guide_support_prefilter.v1",
        candidate_only=True,
        release_evidence=False,
        spacing_m=spacing,
        total_samples=total,
        blocked_edges=[k for k, r in rows.items() if r["rejected_samples"]],
        rows=rows,
        limitations=[
            "Sampled body centers only; not complete body or continuous support proof.",
            "Coarse masks may conservatively exclude physically valid routes.",
            "A surviving candidate still requires original native physical gates.",
        ],
    )
