"""Evidence-only access to complete candidate recordings; never release admission."""

import hashlib
import json
import math
from pathlib import Path

from awr.life.common.admission import AdmissionError
from awr.life.common.integrity import sha256_file
from awr.life.common.schema import schema_id

from .execution import MARKER


def candidate_witness_evidence(event, *, world, build_root):
    """Verify frozen build, full tape, physical QA and configured life gates.

    A diagnostic still cannot be displayed or used as an admitted benchmark.
    This function permits collecting the camera evidence needed to assess it,
    without deleting its marker, rewriting it, or granting scene capacity.
    """
    from awr.life.civil.life_audit import evaluate_gates
    from awr.life.civil.validate_tape import QA_CHECKS
    from awr.sim.events.tape import open_tape

    event, world, build_root = map(lambda p: Path(p).resolve(), (event, world, build_root))

    def read(name):
        return json.loads((event / name).read_text())

    def require(condition, reason):
        if not condition:
            raise AdmissionError(f"{event}: candidate witness refused: {reason}")

    marker, build = read(MARKER), read("build-inputs.json")
    require(
        schema_id(marker.get("schema")) == "awr.life.environment_validation.v1"
        and marker.get("admitted") is False
        and marker.get("published") is False,
        "missing explicit diagnostic status",
    )
    require(
        schema_id(build.get("schema")) == "awr.life.build_inputs.v1"
        and build.get("execution_mode") == "dynamic_validation",
        "not a completed diagnostic build",
    )
    binding = build.get("environment_binding") or {}
    require(
        binding.get("identity_bound") is True
        and binding.get("environment") == marker.get("environment")
        and bool(marker.get("environment")),
        "environment identity is not bound",
    )
    files = build.get("files") or {}
    require(isinstance(files, dict) and bool(files), "missing build source hashes")
    require(
        hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() == build.get("content_sha256"),
        "build manifest digest differs",
    )
    digests = dict(binding.get("file_sha256") or {})
    require(bool(digests), "missing environment input hashes")
    for name, digest in files.items():
        path = str((build_root / name).resolve())
        require(path not in digests or digests[path] == digest, "conflicting input digests")
        digests[path] = digest
    world_file = str(world / "world.json")
    require(world_file in digests, "actual camera world is not a build input")
    require(
        all(Path(p).is_file() and sha256_file(p) == h for p, h in digests.items()),
        "frozen source or environment input changed",
    )
    cfg = read("testbed-config.json")
    # The copy carried by the tape must be one of the unchanged recorded inputs.
    require(sha256_file(event / "testbed-config.json") in digests.values(), "unbound scenario config")
    tape = open_tape(event)
    manifest = read("tape.json")
    content = manifest.get("content_sha256")
    require(isinstance(content, str) and len(content) == 64, "missing tape content hash")
    tape.verify(expect_content_sha256=content)
    world_doc = json.loads((world / "world.json").read_text())
    tape.check_world(world_doc["contentVersion"])
    duration = float(cfg.get("duration_s", 0))
    require(
        math.isfinite(duration) and duration > 0 and tape.n_ticks == round(duration * float(tape.header["tick_hz"])),
        "incomplete configured duration",
    )
    qa, audit = read("final_qa.json"), read("life-audit.json")
    counts = qa.get("counts") or {}
    require(
        schema_id(qa.get("schema")) == "awr.life.final_qa.v1"
        and qa.get("pass") is True
        and qa.get("tape_content_sha256") == content
        and qa.get("ticks") == tape.n_ticks
        and set(counts) == set(QA_CHECKS)
        and all(type(v) is int and v == 0 for v in counts.values()),
        "physical QA incomplete, failed or belongs to another tape",
    )
    gates = evaluate_gates(audit, cfg.get("life_gates"))
    require(
        audit.get("tape_content_sha256") == content
        and gates is not None
        and bool(gates["gates"])
        and gates["all_pass"] is True
        and gates == audit.get("gates"),
        "configured life checks incomplete, failed or belong to another tape",
    )
    evidence_names = (
        MARKER,
        "build-inputs.json",
        "testbed-config.json",
        "tape.json",
        "tape.bin",
        "final_qa.json",
        "life-audit.json",
    )
    return dict(
        status="candidate_evidence_only",
        admitted=False,
        publication=False,
        event=str(event),
        world=str(world),
        build_root=str(build_root),
        environment=marker["environment"],
        tape_content_sha256=content,
        duration_s=duration,
        gates=len(gates["gates"]),
        evidence_sha256={name: sha256_file(event / name) for name in evidence_names},
        scope="Camera evidence collection only; no display, agent benchmark or release admission.",
    )
