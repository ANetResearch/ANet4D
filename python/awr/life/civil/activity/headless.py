"""Topology-only activity integration runner, explicitly NOT a physical simulator.

Loads named places and directed walk/drive edges. Motion follows those edges;
no rendering, APIs, SUMO or world downloads are needed. This fast diagnostic
checks demand/ownership/resources across maps before expensive grounded QA.
"""

import argparse
import hashlib
import heapq
import json
import math
from dataclasses import asdict
from pathlib import Path

from .director import Activity, Actor, Director, Place, Route, Vehicle


class Graph:
    def __init__(self, nodes, edges):
        self.nodes = {k: tuple(v) for k, v in nodes.items()}
        self.ids = {v: k for k, v in self.nodes.items()}
        self.edges = {"walk": {}, "drive": {}}
        for edge in edges:
            for mode in edge["modes"]:
                a, b = edge["from"], edge["to"]
                self.edges[mode].setdefault(a, []).append((b, math.dist(self.nodes[a], self.nodes[b])))
                if edge.get("bidirectional", False):
                    self.edges[mode].setdefault(b, []).append((a, math.dist(self.nodes[a], self.nodes[b])))
        self.paths = {}

    def route(self, mode, start, end, entity):
        def node_at(xy):
            return next((node for node, p in self.nodes.items() if math.dist(p, xy) < 1e-6), None)

        source, goal = node_at(start), node_at(end)
        if source is None or goal is None:
            return None
        heap = [(0.0, source, (source,))]
        seen = set()
        while heap:
            distance, node, path = heapq.heappop(heap)
            if node in seen:
                continue
            seen.add(node)
            if node == goal:
                key = f"{mode}:{source}:{goal}"
                self.paths[key] = [self.nodes[n] for n in path]
                return Route(distance, distance / (4.0 if mode == "drive" else 1.2), key)
            for nxt, length in sorted(self.edges[mode].get(node, [])):
                heapq.heappush(heap, (distance + length, nxt, path + (nxt,)))
        return None

    def at(self, key, distance):
        path = self.paths[key]
        for a, b in zip(path, path[1:], strict=False):
            length = math.dist(a, b)
            if distance < length:
                return tuple(x + (y - x) * distance / length for x, y in zip(a, b, strict=False))
            distance -= length
        return path[-1]


def simulate(config, duration, seed):
    graph = Graph(config["nodes"], config["edges"])
    places = [
        Place(
            p["id"],
            graph.nodes[p["node"]],
            p["kind"],
            tuple(p["slots"]),
            p.get("opens_s", -math.inf),
            p.get("closes_s", math.inf),
            tuple(p.get("parking", ())),
            graph.nodes[p["access"]] if p.get("access") else None,
            p.get("group"),
            p.get("affordances", ()),
            p.get("semantic_source"),
        )
        for p in config["places"]
    ]
    activities = [
        Activity(
            a["purpose"],
            tuple(a["kinds"]),
            tuple(a["dwell_s"]),
            a["recurrence_s"],
            a.get("weight", 1.0),
            tuple(a.get("roles", ["resident"])),
            tuple(a.get("modes", ["walk"])),
            requires=a.get("requires", ()),
        )
        for a in config["activities"]
    ]
    director = Director(
        places,
        activities,
        graph.route,
        seed=seed,
        home_dwell_s=tuple(config.get("home_dwell_s", [60.0, 180.0])),
        max_wait_s=float(config.get("max_wait_s", 120.0)),
    )
    xy, vehicles, progress, distance = {}, {}, {}, {}
    for entry in sorted(config["population"], key=lambda x: x["id"]):
        actor = Actor(entry["id"], entry["home"], vehicle=entry.get("vehicle"))
        director.add_actor(actor)
        xy[actor.id] = director.places[actor.home].xy
        distance[actor.id] = 0.0
        if actor.vehicle:
            v = Vehicle(actor.vehicle, actor.id, entry["berth"])
            director.add_vehicle(v)
            vehicles[v.id] = director.places[v.berth].xy
            distance[v.id] = 0.0
    for t in range(int(duration) + 1):
        for actor_id in sorted(director.actors):
            actor = director.actors[actor_id]
            intent = director.choose(actor_id, float(t), xy[actor_id])
            if intent is None:
                continue
            step = intent.step
            speed = vspeed = 0.0
            if step.kind in ("walk", "drive"):
                key = (actor_id, actor.decisions, intent.index)
                old = progress.get(key, 0.0)
                new = min(step.route.length_m, old + (4.0 if step.kind == "drive" else 1.2))
                point = graph.at(step.route.evidence, new)
                progress[key] = new
                distance[step.entity] += new - old
                if step.kind == "walk":
                    xy[actor_id] = point
                    speed = new - old
                else:
                    vehicles[actor.vehicle] = point
                    xy[actor_id] = point  # explicit contained occupant, not a visible walker
                    vspeed = new - old
            elif step.kind == "alight":
                # Diagnostic transfer endpoint only, NOT a 3D door animation.
                xy[actor_id] = step.xy
            director.observe_departure(actor_id, xy[actor_id], vehicles.get(actor.vehicle))
            director.acknowledge(
                actor_id, float(t), xy[actor_id], speed, vehicle_xy=vehicles.get(actor.vehicle), vehicle_speed=vspeed
            )
    events = director.events
    summary = dict(
        schema="awr.life.activity_diagnostic.v1",
        seed=seed,
        duration_s=duration,
        admitted_for_benchmark=False,
        limitations=[
            "topology only; no collisions, tyre/ground checks, parking/door animation or camera validation",
            "not a SUMO or JuPedSim performance test",
        ],
        completed_activities=sum(e["kind"] == "activity_completed" for e in events),
        car_activities=sum(e["kind"] == "activity_selected" and e["mode"] == "car" for e in events),
        walk_activities=sum(e["kind"] == "activity_selected" and e["mode"] == "walk" for e in events),
        travelled_m={k: round(v, 3) for k, v in distance.items()},
        actor_states={k: asdict(v) for k, v in director.actors.items()},
        vehicle_states={k: asdict(v) for k, v in director.vehicles.items()},
        remaining_reservations=director.holders,
        event_sha256=hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest(),
    )
    return summary, events


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--duration", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    if args.duration <= 0:
        ap.error("duration must be positive")
    config = json.loads(args.config.read_text())
    summary, events = simulate(config, args.duration, args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (args.out / "events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events))
    (args.out / "inputs.json").write_text(
        json.dumps(
            dict(
                config=config,
                code_sha256={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {k: summary[k] for k in ("completed_activities", "car_activities", "walk_activities", "event_sha256")}
        )
    )


if __name__ == "__main__":
    main()
