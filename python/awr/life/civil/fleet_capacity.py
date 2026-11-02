"""Offline berth allocation with checked route conflicts, never actor scripts.

The caller supplies an already witnessed baseline. The integer solver proposes
occupancy only; an exact route oracle must accept every new home set before it
can replace that baseline. Failure cuts are valid for
monotone occupancy: adding parked bodies cannot create a route/free service bay.
"""

import numpy as np


def allocate(candidates, index, bodies, witnesses, baseline, requested, *, max_cuts=96):
    from scipy import __version__ as scipy_version
    from scipy.optimize import Bounds, LinearConstraint, milp

    names = sorted(candidates)
    local = {n: i for i, n in enumerate(names)}
    selected = list(baseline)
    cuts = []
    rounds = []
    learned = set()
    if not names:
        return selected, dict(rounds=[], cuts=[], scipy_version=scipy_version)
    for i, a in enumerate(names):
        if len(witnesses(a, 1 << index[a])) < 2:
            cuts.append((a,))
        for b in names[i + 1 :]:
            if bodies[a].intersects(bodies[b]):
                cuts.append((a, b))
    learned.update(cuts)
    for iteration in range(max_cuts):
        matrix = np.zeros((len(cuts) + 1, len(names)))
        matrix[0] = 1.0
        upper = [requested]
        for row, cut in enumerate(cuts, 1):
            matrix[row, [local[n] for n in cut]] = 1.0
            upper.append(len(cut) - 1)
        result = milp(
            -np.ones(len(names)),
            integrality=np.ones(len(names)),
            bounds=Bounds(0.0, 1.0),
            constraints=LinearConstraint(matrix, -np.inf, upper),
            options=dict(node_limit=2000, mip_rel_gap=0.0),
        )
        row = dict(iteration=iteration, solver_status=int(result.status))
        rounds.append(row)
        # A solver limit/error is not an admission certificate. Keep the last
        # fully witnessed set and report the unfinished search explicitly.
        if result.status != 0 or result.x is None:
            break
        proposal = [n for n, value in zip(names, result.x, strict=False) if value > 0.5]
        row["proposed"] = len(proposal)
        if len(proposal) <= len(selected):
            row["no_larger_set_in_relaxation"] = True
            break
        mask = sum(1 << index[n] for n in proposal)
        failed = next((n for n in proposal if len(witnesses(n, mask)) < 2), None)
        if failed is None:
            selected = proposal
            row["fully_witnessed"] = True
            break
        # Shrink this invalid occupancy to an irreducible conflict. Retain the
        # failed home itself: without it no trip requirement exists at that bay.
        conflict = list(proposal)
        for other in proposal:
            if other == failed:
                continue
            reduced = mask & ~(1 << index[other])
            if len(witnesses(failed, reduced)) < 2:
                conflict.remove(other)
                mask = reduced
        cut = tuple(sorted(conflict))
        if cut in learned:
            raise RuntimeError("Occupancy solver repeated an already forbidden set")
        learned.add(cut)
        cuts.append(cut)
        row.update(failed_home=failed, learned_conflict=list(cut))
    return selected, dict(
        rounds=rounds,
        cuts=[list(c) for c in cuts],
        scipy_version=scipy_version,
        max_rounds=max_cuts,
        admission="every returned home has independently checked routes",
    )
