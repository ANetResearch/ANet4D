"""Bind authoring metadata to the actual rendered tape roster.

The tape intentionally omits some author fields. The civil entity registry keeps
vehicle types, while older plan.json exports do not. Never infer a type from an
actor ID, role or description, or trust a registry for a different appearance.
"""

import json
from pathlib import Path


def rendered_entities(event_dir, tape_actors, plan):
    rows = list(tape_actors)

    def index(items):
        result = {}
        for item in items:
            ident = item.get("id")
            if not isinstance(ident, str) or not ident or ident in result:
                raise ValueError("Unique entity identities required")
            result[ident] = item
        return result

    tape = index(rows)
    sources = [index(plan["actors"])] if plan.get("actors") else []
    path = Path(event_dir) / "entities.json"
    if path.exists():
        doc = json.loads(path.read_bytes())
        registry = doc if isinstance(doc, list) else doc.get("entities", doc.get("actors"))
        if not isinstance(registry, list):
            raise ValueError("Invalid entity registry")
        sources.append(index(registry))
    result = {ident: dict(row) for ident, row in tape.items()}
    for source in sources:
        for ident, rendered in tape.items():
            if ident not in source:
                raise ValueError("Entity metadata missing rendered actor: " + ident)
            meta = source[ident]
            for key in ("class", "size_mm", "appearance"):
                if key not in rendered or meta.get(key) != rendered[key]:
                    raise ValueError("Entity metadata differs from rendered " + key + ": " + ident)
            for key in ("semantic_type", "vtype", "description", "role"):
                value = meta.get(key)
                if value is None:
                    continue
                old = result[ident].get(key)
                if old is not None and old != value:
                    raise ValueError("Conflicting entity metadata " + key + ": " + ident)
                result[ident][key] = value
    # Keep the tape order/index even when the source registry contains omitted
    # actors. They must never become phantom detections or query targets.
    return result
