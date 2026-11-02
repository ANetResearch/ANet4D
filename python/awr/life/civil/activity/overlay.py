"""Scenario overlay: per-query adjustments of an autonomous scene (awr.life.scenario_overlay.v1).

An overlay changes who lives where and what they wear, not how anyone moves. Every adjusted actor is still admitted
by the scene's own checks and then acts autonomously through the shared Director and the life engine; nobody gets a
scripted trajectory. Regions are the scene's frozen public regions (awr.life.civil.public_regions), so an overlay names
the same areas as the queries.

- `persons`: each entry adjusts one resident, in order (the first entry the first resident admitted, and so on):
  - `region`: the resident's home lies in this public region;
  - `places`: `"region"` restricts the resident's destinations to the activity places in `places_region` (default:
    `region`; a search or tracking target that stays in its area, or visits one public place), `"any"` keeps all
    destinations;
  - `min_activity_kinds`: the activity kinds the resident must reach among its destinations, instead of the
    scene's min_activity_kinds (an area may hold fewer kinds; the measured capability uses the same number);
  - `appearance`: `{"top": <colour name>, "bottom": <colour name>}` from awr.life.appearance.identity. The outfit is
    reserved: no other person wears the same combination, so the public description names one person. Other people
    keep outfits that share a top or a bottom colour (distractors).
- `parked_sets`: each entry freezes the parked vehicles of one public region for the whole recording (a count or
  plate query needs a stable answer set): no resident car parks there, so only the scene's parked cars, which never
  move, stand in it. Moving cars elsewhere remain. Optional `colours` maps CAR_COLOURS names to exact counts of
  generated permanent cars. This does not account for baked vehicles in the reconstructed background.
  Optional `placement: {"mode": "curb_parked"|"lot_parked", "minimum": N}` assigns N existing
  roster requests to the region through the physical allocator. Full footprints and clearance must fit;
  support, legality, access and passage checks are unchanged. No extra population, actor poses or fallback
  outside the region is inserted. Failure is an infeasible allocation, not proof of maximum capacity.

An entry that the scene cannot satisfy (no home in the region with the required activity access) fails the build;
it is never relaxed or skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import Point, Polygon

from awr.life.common.schema import schema_id

from ...render.identity import PALETTE, TROUSERS
from ..population import CAR_COLOURS, VEHICLE_TYPES

SCHEMA = "awr.life.scenario_overlay.v1"
OUTFITS = len(PALETTE) * len(TROUSERS)


def outfit_ordinal(top, bottom):
    """The awr.life.appearance.identity.clothing ordinal of a (top, bottom) colour pair."""
    tops = [name for name, _ in PALETTE]
    bottoms = [name for name, _ in TROUSERS]
    if top not in tops or bottom not in bottoms:
        raise ValueError(f"Unknown outfit colours: {top}/{bottom}")
    return tops.index(top) + len(PALETTE) * bottoms.index(bottom)


class Overlay:
    def __init__(self, doc, regions):
        """`doc`: the overlay document; `regions`: the compiled public regions (awr.life.civil.public_regions.load)."""
        if schema_id(doc.get("schema")) != SCHEMA:
            raise ValueError(f"{SCHEMA} required")
        polygons = {a["id"]: Polygon(a["polygon"]) for a in (regions or {}).get("anchors", [])}
        self.id = doc.get("id")
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Scenario overlay needs a nonempty identity")
        self.persons, self.parked_sets = [], []
        keys = set()
        for row in doc.get("persons", []):
            key, region = row.get("key"), row.get("region")
            if not isinstance(key, str) or not key or key in keys or region not in polygons:
                raise ValueError(f"Overlay person needs a unique key and a public region: {row}")
            if row.get("places", "any") not in ("region", "any"):
                raise ValueError('Overlay places must be "region" or "any"')
            places_region = row.get("places_region", region)
            if places_region not in polygons:
                raise ValueError(f"Overlay places region is not a public region: {places_region}")
            appearance = row.get("appearance")
            ordinal = None if appearance is None else outfit_ordinal(appearance.get("top"), appearance.get("bottom"))
            minimum = row.get("min_activity_kinds")
            if minimum is not None and (type(minimum) is not int or minimum < 1):
                raise ValueError("Overlay min_activity_kinds must be a positive integer")
            keys.add(key)
            self.persons.append(
                dict(
                    key=key,
                    region=region,
                    polygon=polygons[region],
                    places=row.get("places", "any"),
                    min_activity_kinds=minimum,
                    places_region=places_region,
                    places_polygon=polygons[places_region],
                    outfit=ordinal,
                    rationale=row.get("rationale"),
                )
            )
        outfits = [p["outfit"] for p in self.persons if p["outfit"] is not None]
        if len(outfits) != len(set(outfits)):
            raise ValueError("Overlay targets must wear distinct outfits")
        if len(outfits) >= OUTFITS:
            raise ValueError("Overlay reserves every outfit")
        for row in doc.get("parked_sets", []):
            key, region = row.get("key"), row.get("region")
            if not isinstance(key, str) or not key or key in keys or region not in polygons:
                raise ValueError(f"Overlay parked set needs a unique key and a public region: {row}")
            colours = row.get("colours", {})
            names = {name for _, _, name in CAR_COLOURS}
            if not isinstance(colours, dict) or any(
                c not in names or type(n) is not int or n < 0 for c, n in colours.items()
            ):
                raise ValueError("Overlay parked colours need known colour names and nonnegative integer counts")
            placement = row.get("placement")
            if placement is not None and (
                not isinstance(placement, dict)
                or placement.get("mode") not in ("curb_parked", "lot_parked")
                or type(placement.get("minimum")) is not int
                or placement["minimum"] < 1
                or set(placement) != {"mode", "minimum"}
            ):
                raise ValueError("Overlay parking placement needs a static mode and positive minimum")
            keys.add(key)
            self.parked_sets.append(
                dict(
                    key=key,
                    region=region,
                    polygon=polygons[region],
                    colours=dict(colours),
                    placement=None if placement is None else dict(placement),
                    rationale=row.get("rationale"),
                )
            )
        self.reserved = set(outfits)

    def validate_config(self, cfg, people):
        """Reject silent omission of authored targets before scene generation."""
        if len(self.persons) > people:
            raise ValueError("Scenario overlay targets exceed the requested population")
        if self.persons and (
            (cfg.get("life") or {}).get("behavior") != "activities_v1" or not cfg.get("walking_network")
        ):
            raise ValueError("Scenario overlay persons require autonomous walking activities")
        if self.reserved and cfg.get("visual_identity") is not True:
            raise ValueError("Scenario overlay outfits require visual_identity")
        if any(s["colours"] for s in self.parked_sets) and (
            cfg.get("visual_identity") is not True or not cfg.get("life")
        ):
            raise ValueError("Scenario overlay parked colours require life and visual_identity")
        if any(s["placement"] for s in self.parked_sets) and (
            ((cfg.get("life") or {}).get("vehicles") or {}).get("behavior") != "curb_v1"
        ):
            raise ValueError("Scenario parking placement requires verified curb_v1 allocation")

    def parking_requests(self, roster):
        """Assign regional minima to the declared fleet, without adding cars or inventing poses.

        Existing allocation proves each complete footprint fits its requested
        region as well as all usual terrain, parking and passage requirements.
        Unassigned cars retain ordinary allocation; a minimum is not an exact
        answer count. The actual resulting set is measured after placement.
        """
        reserved = {}
        for entry in sorted(self.parked_sets, key=lambda s: s["key"]):
            placement = entry["placement"]
            if placement is None:
                continue
            available = [r for r in roster if r.mode == placement["mode"] and r.index not in reserved]
            if len(available) < placement["minimum"]:
                raise ValueError(
                    f"Overlay parked set {entry['key']}: insufficient declared {placement['mode']} population"
                )
            for request in available[: placement["minimum"]]:
                reserved[request.index] = entry
        return reserved

    def paint(self, cars):
        """Compile region colour counts to appearance edits of admitted permanent cars.

        Adapted from Claude e7c7792. No actor IDs or trajectories in the recipe.
        Cars are (id, xy, colour_name). Conflicting overlapping requirements fail
        rather than quietly letting the last region overwrite an earlier count.
        """
        cars = list(cars)
        original = {car: name for car, _, name in cars}
        if len(original) != len(cars):
            raise ValueError("Duplicate car in overlay paint input")
        palette = {name: rgb for rgb, _, name in CAR_COLOURS}
        current = dict(original)
        regions = []
        for entry in sorted(self.parked_sets, key=lambda s: s["key"]):
            requested = entry["colours"]
            inside = sorted(car for car, xy, _ in cars if entry["polygon"].covers(Point(*xy[:2])))
            regions.append((entry, inside))
            if not requested:
                continue
            if sum(requested.values()) > len(inside):
                raise ValueError(f"Overlay parked set {entry['key']}: insufficient permanent cars")
            remaining = set(inside)
            chosen = {}
            # Preserve existing matches for every colour before allocating extras.
            for colour, n in sorted(requested.items()):
                chosen[colour] = [car for car in inside if current[car] == colour][:n]
                remaining.difference_update(chosen[colour])
            for colour, n in sorted(requested.items()):
                extra = sorted(remaining)[: n - len(chosen[colour])]
                chosen[colour].extend(extra)
                remaining.difference_update(extra)
                for car in chosen[colour]:
                    current[car] = colour
            alternatives = [name for name in palette if name not in requested]
            for car in sorted(remaining):
                if current[car] in requested:
                    if not alternatives:
                        raise ValueError(
                            f"Overlay parked set {entry['key']}: no unconstrained colour for remaining cars"
                        )
                    current[car] = alternatives[0]
        for entry, inside in regions:
            for colour, n in entry["colours"].items():
                if sum(current[car] == colour for car in inside) != n:
                    raise ValueError(f"Conflicting overlapping parked colour constraints: {entry['key']}")
        return {car: (palette[name], name) for car, name in current.items() if name != original[car]}

    def paint_actors(self, actors, positions):
        """Change rendered body colour and matching description; preserve geometry and plates."""
        edits = self.paint(
            [
                (a["id"], positions[i], a.get("appearance", {}).get("colour_name"))
                for i, a in enumerate(actors)
                if a.get("class") == "vehicle" and a.get("role") == "resident_parked"
            ]
        )
        for actor in actors:
            if actor["id"] in edits:
                rgb, name = edits[actor["id"]]
                actor["appearance"].update(body_rgb=list(rgb), colour_name=name)
                actor["description"] = f"{name}{VEHICLE_TYPES[actor['vtype']][2]}"

    def parked_bindings(self, actors, positions):
        """Validate final rendered colours after all appearance overlays; author truth only.

        This lists synthetic permanent cars, never a complete visible inventory
        of a photogrammetric background. Moving cars are not included.
        """
        palette = {name: list(rgb) for rgb, _, name in CAR_COLOURS}
        rows = []
        for entry in self.parked_sets:
            inside = [
                a
                for i, a in enumerate(actors)
                if a.get("class") == "vehicle"
                and a.get("role") == "resident_parked"
                and entry["polygon"].covers(Point(*positions[i][:2]))
            ]
            placement = entry["placement"]
            if placement is not None:
                placed = [
                    a
                    for a in inside
                    if a.get("parking_overlay") == entry["key"] and a.get("allocation_mode") == placement["mode"]
                ]
                if len(placed) < placement["minimum"]:
                    raise ValueError(f"Overlay parking placement was not satisfied: {entry['key']}")
            for colour, count in entry["colours"].items():
                if sum(a.get("appearance", {}).get("colour_name") == colour for a in inside) != count:
                    raise ValueError(f"Rendered parked set colour count differs: {entry['key']}")
            if entry["colours"]:
                for a in inside:
                    appearance = a.get("appearance", {})
                    if appearance.get("body_rgb") != palette.get(appearance.get("colour_name")):
                        raise ValueError(f"Rendered parked set RGB differs from colour label: {a['id']}")
            rows.append(
                dict(
                    key=entry["key"],
                    region=entry["region"],
                    scope="synthetic permanent cars only",
                    vehicles=[
                        dict(id=a["id"], type=a.get("vtype"), **a.get("appearance", {}))
                        for a in sorted(inside, key=lambda a: a["id"])
                    ],
                )
            )
        return rows

    def bindings(self, actors, runtime):
        """Validate and record actual admitted targets; never an agent input."""
        from ...render.identity import clothing

        by_id = {a["id"]: a for a in actors}
        rows = []
        for k, entry in enumerate(self.persons):
            aid = f"person-{k:03d}"
            if runtime is None or aid not in by_id or aid not in runtime.director.actors:
                raise ValueError(f"Scenario overlay target {entry['key']} was not admitted")
            actor = runtime.director.actors[aid]
            home = runtime.director.places[actor.home]
            if actor.vehicle is not None or not entry["polygon"].covers(Point(*home.xy)):
                raise ValueError(f"Scenario overlay target {aid} has the wrong home or transport mode")
            if entry["outfit"] is not None:
                expected = clothing(entry["outfit"])
                if any(by_id[aid].get("appearance", {}).get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Scenario overlay outfit differs from rendered actor {aid}")
                if any(
                    other["id"] != aid
                    and all(other.get("appearance", {}).get(key) == expected[key] for key in ("top_rgb", "bottom_rgb"))
                    for other in actors
                ):
                    raise ValueError(f"Scenario overlay outfit is not unique for {aid}")
            if entry["places"] == "region":
                if not actor.places or any(
                    not entry["places_polygon"].covers(Point(*runtime.director.places[p].xy)) for p in actor.places
                ):
                    raise ValueError(f"Scenario overlay destinations missing or outside region for {aid}")
            rows.append(
                dict(
                    key=entry["key"],
                    actor=aid,
                    region=entry["region"],
                    home_xy=list(home.xy),
                    description=by_id[aid].get("description"),
                    destinations=None if actor.places is None else sorted(actor.places),
                    min_activity_kinds=runtime.kind_minimum.get(aid, runtime.min_activity_kinds),
                )
            )
        return dict(
            self.describe(),
            admitted=False,
            bindings=rows,
            scope="Author-side scene bindings, not query admission or an agent input.",
        )

    @classmethod
    def load(cls, path, regions):
        return cls(json.loads(Path(path).read_text()), regions)

    def person(self, index):
        """The overlay entry of the index-th admitted resident, or None."""
        return self.persons[index] if index < len(self.persons) else None

    def outfit(self, index, ordinal):
        """The outfit ordinal the index-th person wears: its overlay outfit, else `ordinal` moved past the reserved
        outfits (deterministic; tops and bottoms of the targets still occur in other combinations)."""
        entry = self.person(index)
        if entry is not None and entry["outfit"] is not None:
            return entry["outfit"]
        o = ordinal % OUTFITS
        while o in self.reserved:
            o = (o + 1) % OUTFITS
        return o

    def frozen(self, xy):
        """Whether a position lies in a region whose parked set is frozen."""
        point = Point(float(xy[0]), float(xy[1]))
        return any(s["polygon"].covers(point) for s in self.parked_sets)

    def describe(self):
        """The overlay as recorded with the build (privileged author data: not for agents)."""
        return dict(
            schema=SCHEMA,
            id=self.id,
            persons=[{k: v for k, v in p.items() if not k.endswith("polygon")} for p in self.persons],
            parked_sets=[{k: v for k, v in s.items() if k != "polygon"} for s in self.parked_sets],
        )
