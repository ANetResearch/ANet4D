"""Invalidate a walking route after an off-route backstep without moving a body."""

import math

import numpy as np


def finish_backstep(job, xy, yaw, t, *, max_speed=0.2):
    """Replace obsolete arc coordinates with a zero-length measured-pose hold.

    A backstep advances world XY independently of the ordinary route's arc.
    Resuming/finishing an extension of that old route would snap the body back.
    Keep the destination agenda and its reservations, but require a new route
    starting at the observed body. Even a failed replanning attempt may only
    hold here, never resurrect the abandoned route geometry.
    """
    xy = np.asarray(xy, float)
    if (
        xy.shape != (2,)
        or not np.isfinite(xy).all()
        or not all(math.isfinite(x) for x in (yaw, t, max_speed, job["speed"]))
        or max_speed <= 0
        or abs(job["speed"]) > max_speed + 1e-8
    ):
        raise ValueError("Backstep handoff requires a finite observed pose at backstep speed")
    job.pop("backstep", None)
    for key in ("extending", "yielding", "_partial"):
        job.pop(key, None)
    job.update(
        route=dict(
            road="backstep_replan_hold",
            points=np.repeat(xy[None], 2, axis=0),
            arc=np.zeros(2),
            yaw=np.full(2, yaw),
            smooth_vehicle_heading=True,
        ),
        s=0.0,
        goal_s=0.0,
        direction=1,
        speed=0.0,
        needs_route=True,
        replan_after=t,
        temporary_endpoint=False,
        _mode=(round(t, 1), "backstep_complete_replan"),
    )
