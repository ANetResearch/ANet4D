"""Explicit demo appearance authoring, separate from activity and motion policy.

Only palette-backed material colours change. Identity, geometry, ownership,
placement, routes, timing and plates are outside this overlay's authority.
"""

from copy import deepcopy

from awr.life.common.schema import schema_id

from .population import BOTTOMS, CAR_COLOURS, TOPS


def apply(actors, overlay):
    if not isinstance(overlay, dict) or set(overlay) != {"schema", "edits"}:
        raise ValueError("Visual overlay needs schema and edits only")
    if schema_id(overlay["schema"]) != "awr.life.visual_overlay.v1" or not isinstance(overlay["edits"], list):
        raise ValueError("Unknown visual overlay schema or edits")
    result = deepcopy(actors)
    index = {a["id"]: a for a in result}
    if len(index) != len(result):
        raise ValueError("Unique actor identities required")
    seen = set()
    fields = {
        "vehicle_colour": ("vehicle", "colour_name", "body_rgb", CAR_COLOURS),
        "top": ("person", "top_name", "top_rgb", TOPS),
        "bottom": ("person", "bottom_name", "bottom_rgb", BOTTOMS),
    }
    for edit in overlay["edits"]:
        if not isinstance(edit, dict) or set(edit) != {"actor", "expected_asset", "expected_appearance", "colours"}:
            raise ValueError("Visual edit requires an actor, frozen appearance and palette colours")
        ident = edit["actor"]
        if not isinstance(ident, str) or ident not in index or ident in seen:
            raise ValueError("Visual edit actor missing or repeated")
        seen.add(ident)
        actor = index[ident]
        old = actor.get("appearance", {})
        if (
            not edit["expected_asset"]
            or old.get("asset") != edit["expected_asset"]
            or old != edit["expected_appearance"]
        ):
            raise ValueError("Visual edit belongs to another actor appearance: " + ident)
        colours = edit["colours"]
        if not isinstance(colours, dict) or not colours or set(colours) - fields.keys():
            raise ValueError("Only explicit palette colour edits are allowed")
        for field, name in colours.items():
            cls, label, rgb, palette = fields[field]
            choices = {n: list(c) for c, _, n in palette}
            if actor["class"] != cls or not isinstance(name, str) or name not in choices:
                raise ValueError("Colour or actor class does not match visual edit")
            old[label], old[rgb] = name, choices[name]
        # Existing descriptions may mention the original colour. Consumers use
        # structured visible attributes; omit stale prose rather than guessing
        # a replacement role/age/type description.
        actor.pop("description", None)
    return result
