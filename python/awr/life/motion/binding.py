"""Bind an admitted mixed population to the builder's recording loop.

The environment adapter supplies real places, ownership and a DrivingRoutes
backend. This binding does not assign homes, snap cars to berths, manufacture
routes or relax environment capability demands. It verifies that the backend
and the recorded bodies refer to the same fresh world before any tick is run.
"""

import math
from dataclasses import dataclass

import numpy as np

from awr.life.civil.activity.driver_execution import DriverExecution
from awr.life.civil.activity.driving import DrivingRoutes

from .agenda import LifeAgenda


@dataclass(frozen=True)
class DrivingBinding:
    routes: DrivingRoutes

    def prepare(self, ctx):
        routes = self.routes
        if not isinstance(routes, DrivingRoutes) or routes.surface is not ctx["surface"]:
            raise ValueError("Driving binding requires the recording world support surface")
        activity = ctx["activity_runtime"]
        director = activity.director
        actors, jobs = ctx["actors"], ctx["jobs"]
        ids = [a["id"] for a in actors]
        if len(ids) != len(set(ids)) or len(jobs) != len(ids):
            raise ValueError("Driving binding requires a distinct complete body roster")
        persons = {ids[i]: i for i, j in enumerate(jobs) if j["kind"] == "person"}
        vehicles = {i: ids[i] for i, j in enumerate(jobs) if j["kind"] == "vehicle"}
        rigid = {ids[i]: i for i, j in enumerate(jobs) if j["kind"] != "person"}
        if set(director.actors) != set(persons) or routes.roster != set(rigid):
            raise ValueError("Driving binding must include all residents and rigid bodies")
        if routes.active or routes.executions or routes.completed_motion or director.active:
            raise ValueError("Driving binding requires a fresh world, not a partial restart")
        for place in director.places.values():
            if place.kind != "berth":
                continue
            node = routes.graph.nodes.get(place.id)
            if (
                node is None
                or node.get("kind") != "berth"
                or place.access_xy is None
                or not np.allclose(place.xy, node["pose"][:2], rtol=0.0, atol=1e-8)
            ):
                raise ValueError(f"Driving binding has a mismatched destination berth: {place.id}")
        sizes = {}
        for aid, i in rigid.items():
            size_mm = np.asarray(actors[i].get("size_mm", ()), float)
            body = routes.occupancy[aid]
            if (
                size_mm.shape != (3,)
                or not np.isfinite(size_mm).all()
                or np.any(size_mm <= 0)
                or body is None
                or not np.array_equal(size_mm, np.rint(np.asarray(body.size) * 1000.0))
                or not np.allclose(ctx["xy"][i], body.pose[:2], rtol=0.0, atol=1e-8)
                or abs(math.remainder(ctx["yaw"][i] - body.pose[2], 2 * math.pi)) > 1e-8
                or abs(jobs[i].get("speed", 0.0)) > 1e-8
            ):
                raise ValueError(f"Driving binding differs from recorded body: {aid}")
            # Tape metadata is rounded to mm; execution retains admitted float
            # dimensions rather than silently resizing the physical body.
            sizes[i] = body.size
        for aid, i in persons.items():
            ag = jobs[i].get("agenda")
            resident = director.actors[aid]
            if (
                not isinstance(ag, LifeAgenda)
                or ag.director is not director
                or ag.actor != aid
                or ag.phase != "scheduled"
                or ag.index != 0
                or math.dist(ctx["xy"][i], director.places[resident.home].xy) > 0.05
                or abs(jobs[i].get("speed", 0.0)) > 1e-8
            ):
                raise ValueError(f"Driving binding requires a fresh resident agenda at its actual home: {aid}")
            if resident.vehicle is not None and resident.vehicle not in director.vehicles:
                raise ValueError(f"Driving binding lacks the resident vehicle: {aid}")
        for vid, vehicle in director.vehicles.items():
            if vid not in vehicles.values() or vehicle.owner not in persons:
                raise ValueError(f"Driving binding lacks the actual driver/car: {vid}")
            resident = director.actors[vehicle.owner]
            berth = director.places[vehicle.berth]
            node = routes.graph.nodes.get(vehicle.berth)
            if (
                resident.vehicle != vid
                or vehicle.in_trip
                or vehicle.occupant is not None
                or vehicle.berth not in director.places[resident.home].parking
                or berth.access_xy is None
                or node is None
                or node.get("kind") != "berth"
                or not np.allclose(berth.xy, node["pose"][:2], rtol=0.0, atol=1e-8)
                or not routes._at(routes.occupancy[vid].pose, node["pose"], 0.0)
                or not np.allclose(routes.occupancy[vid].size, routes.size, rtol=0.0, atol=1e-8)
            ):
                raise ValueError(f"Driving binding requires the owned car at its mapped home berth: {vid}")
        return dict(
            vehicle_ids=vehicles, sizes=sizes, driving=routes, driver_execution=DriverExecution(director, routes)
        )
