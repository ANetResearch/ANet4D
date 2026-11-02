"""Run candidate environments for evidence without admitting their recordings."""

import json
from pathlib import Path

MARKER = "environment-validation.json"


def mark_validation(directory, environment_id):
    record = dict(
        schema="awr.life.environment_validation.v1",
        environment=environment_id,
        admitted=False,
        published=False,
        scope="dynamic diagnostic; requires scene and task admission",
    )
    (Path(directory) / MARKER).write_text(json.dumps(record, indent=2) + "\n")
    return record


def finish_output(pending, out, *, dynamic_validation=False):
    """Even passing candidate tapes stay separate and cannot silently become releases."""
    pending, out = Path(pending), Path(out)
    marked = (pending / MARKER).exists()
    if dynamic_validation != marked:
        raise ValueError("Validation marker and output mode disagree")
    target = out.parent / (out.name + ".diagnostic") if dynamic_validation else out
    if target.exists():
        raise ValueError("Choose a fresh output version")
    pending.rename(target)
    return target
