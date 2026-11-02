"""Immutable diagnostic appearance variants of a recorded civil scene.

Only palette colours may change. Re-encode the unchanged motion records with a
new actor header; never re-run planning, alter event timing or inherit camera
admission. Scene/world repair is outside this transform's authority.
"""

import hashlib
import json
import shutil
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from awr.life.civil.visual_overlay import apply
from awr.life.common.entities import rendered_entities

# These contain recorded geometry/activity, not old task or release verdicts.
SIDECARS = (
    "anchors.json",
    "facts.jsonl",
    "activity-events.jsonl",
    "agendas.json",
    "activity-places.json",
    "life-plans.json",
    "route-plans.json",
    "public-region-definition.json",
    "asset-catalog.json",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(event_dir, overlay_path, out, *, event_id):
    from awr.sim.events.jcs import jcs
    from awr.sim.events.tape import manifest, open_tape, write_tape

    source, overlay_path, out = Path(event_dir), Path(overlay_path), Path(out)
    if out.exists():
        raise ValueError("Choose a fresh appearance variant directory")
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("A distinct event identity is required")
    required = ("tape.bin", "tape.json", "plan.json", "entities.json", "anchors.json")
    paths = [source / n for n in dict.fromkeys((*required, *SIDECARS)) if n in required or (source / n).exists()]
    paths += [overlay_path, Path(__file__), Path(__file__).with_name("visual_overlay.py")]
    hashes = {str(p): sha(p) for p in paths}
    began = time.perf_counter()
    tape = open_tape(source)
    plan = json.loads((source / "plan.json").read_bytes())
    if event_id == tape.header["event_id"]:
        raise ValueError("Variant must use a distinct event identity")
    rendered_entities(source, tape.actors, plan)  # reject mismatched author labels
    overlay = json.loads(overlay_path.read_bytes())
    actors = apply(tape.actors, overlay)
    header = deepcopy(tape.header)
    header.update(event_id=event_id, event_version=1, actors=actors)
    plan.update(event_id=event_id, version=1, actors=apply(plan["actors"], overlay))
    anchor_hash = hashlib.sha256(
        jcs(sorted(json.loads((source / "anchors.json").read_bytes())["anchors"], key=lambda a: a["id"]))
    ).hexdigest()
    if header["world"]["anchors_sha256"] != anchor_hash:
        raise ValueError("Source public anchors do not match the recorded world binding")
    header["event_sha256"] = hashlib.sha256(jcs(dict(plan=plan, anchors=anchor_hash))).hexdigest()
    registry = json.loads((source / "entities.json").read_bytes())
    if isinstance(registry, list):
        registry = apply(registry, overlay)
    else:
        field = "entities" if "entities" in registry else "actors"
        registry[field] = apply(registry[field], overlay)
    frames = np.concatenate([tape.block_frames(i) for i in range(len(tape.blocks))])
    original_body = hashlib.sha256(frames.tobytes()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="." + out.name + "-", dir=out.parent))
    try:
        h = write_tape(tmp / "tape.bin", header, frames, n_scripted=tape.n_scripted)
        if h["body_sha256"] != original_body:
            raise ValueError("Appearance transform changed recorded motion")
        for name in SIDECARS:
            if (source / name).exists():
                shutil.copyfile(source / name, tmp / name)

        def dump(name, value):
            (tmp / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

        provenance = dict(
            generator="awr.life.civil.appearance_variant",
            source_content_sha256=json.loads((source / "tape.json").read_bytes())["content_sha256"],
            overlay_sha256=hashes[str(overlay_path)],
            diagnostic_only=True,
        )
        dump(
            "tape.json",
            manifest(
                h,
                header,
                provenance=provenance,
                report=dict(
                    arrivals=[],
                    boards=[],
                    min_dist_m={},
                    warnings=["Appearance variant only; no inherited camera, task, world or publication admission."],
                ),
            ),
        )
        dump("plan.json", plan)
        dump("entities.json", registry)
        dump("appearance-overlay.json", overlay)
        reopened = open_tape(tmp)
        rendered_entities(tmp, reopened.actors, plan)
        if len(reopened.blocks) != len(tape.blocks) or any(
            not np.array_equal(tape.block_frames(i), reopened.block_frames(i)) for i in range(len(tape.blocks))
        ):
            raise ValueError("Reopened variant differs from original motion records")
        if any(sha(p) != digest for p, digest in hashes.items()):
            raise ValueError("Appearance source changed during conversion")
        changed = [old["id"] for old, new in zip(tape.actors, actors, strict=False) if old != new]
        report = dict(
            schema="awr.life.appearance_variant.v1",
            event_id=event_id,
            source_content_sha256=provenance["source_content_sha256"],
            content_sha256=h["content_sha256"],
            input_sha256=hashes,
            output_sha256={p.name: sha(p) for p in sorted(tmp.iterdir())},
            changed_actors=changed,
            motion_body_sha256=original_body,
            frames_identical=True,
            n_ticks=tape.n_ticks,
            n_actors=tape.n_actors,
            admitted=False,
            release_evidence=False,
            wall_s=time.perf_counter() - began,
            limitations=[
                "Only appearance changed; the source world binding and any existing scene limitations remain.",
                "New content requires new visual/camera/query checks; no source admission files were copied.",
            ],
        )
        dump("appearance-variant.json", report)
        if out.exists():
            raise ValueError("Variant output appeared during conversion")
        tmp.rename(out)
        return report
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("event", "overlay", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--event-id", required=True)
    args = ap.parse_args()
    result = build(args.event, args.overlay, args.out, event_id=args.event_id)
    print(
        json.dumps(
            {k: result[k] for k in ("content_sha256", "changed_actors", "frames_identical", "wall_s", "admitted")}
        )
    )


if __name__ == "__main__":
    main()
