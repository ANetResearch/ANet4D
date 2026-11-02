"""Candidate recordings run the engine but cannot be mistaken for admitted tapes."""

import json

import life_toytown as toy
import pytest
from test_environment_build_binding import fixture
from test_life_runtime import context

from awr.life.civil.grounded import build
from awr.life.common.admission import AdmissionError, life_admission
from awr.life.envpack.execution import finish_output, mark_validation
from awr.life.motion.runtime import run


def test_passing_diagnostic_never_creates_release_or_admits_tasks(tmp_path):
    pending = tmp_path / "pending"
    pending.mkdir()
    mark_validation(pending, "candidate")
    (pending / "tape.json").write_text(json.dumps(dict(content_sha256="aa")))
    (pending / "life-audit.json").write_text(
        json.dumps(dict(tape_content_sha256="aa", gates=dict(gates=dict(g=dict(pass_=True)))))
    )
    out = tmp_path / "v1"
    with pytest.raises(ValueError, match="mode disagree"):
        finish_output(pending, out)
    target = finish_output(pending, out, dynamic_validation=True)
    assert target.name == "v1.diagnostic" and not out.exists()
    for historical in (False, True):
        with pytest.raises(AdmissionError, match="candidate environment diagnostic"):
            life_admission(target, allow_historical=historical)


def test_validation_needs_explicit_marker_and_does_not_overwrite(tmp_path):
    pending = tmp_path / "pending"
    pending.mkdir()
    out = tmp_path / "v1"
    with pytest.raises(ValueError, match="mode disagree"):
        finish_output(pending, out, dynamic_validation=True)
    mark_validation(pending, "candidate")
    (tmp_path / "v1.diagnostic").mkdir()
    with pytest.raises(ValueError, match="fresh output"):
        finish_output(pending, out, dynamic_validation=True)
    assert pending.exists()


def test_dynamic_builder_keeps_layer_identity_guard(tmp_path):
    m, args = fixture(tmp_path)
    manifest = tmp_path / "environment.json"
    manifest.write_text(json.dumps(m))
    cfg = tmp_path / "scenario.json"
    cfg.write_text(
        json.dumps(dict(environment=str(manifest), environment_root=str(tmp_path), runtime="life_v1", **args["config"]))
    )
    wrong = tmp_path / "old-support"
    wrong.mkdir()
    with pytest.raises(ValueError, match="support mismatch"):
        build(args["world"], wrong, args["semantics"], tmp_path / "v1", config=cfg, dynamic_validation=True)
    assert not (tmp_path / "v1.initial-inputs.json").exists()


def test_dynamic_runtime_executes_candidate_without_inventing_declared_capabilities(tmp_path):
    manifest = tmp_path / "environment.json"
    manifest.write_text(json.dumps(dict(schema="awr.life.environment.v2", id="candidate", layers={})))
    w = toy.build(walkers=2, drivers=0)
    ctx = context(w, 20.0, tmp_path, dict(w.agendas))
    ctx["cfg"].update(environment=str(manifest), people=2)
    with pytest.raises(ValueError, match="validation_only"):
        run(ctx)
    ctx["dynamic_validation"] = True
    run(ctx)
    assert ctx["life_runtime_profile"]["ticks"] == 200
    assert ctx["life_runtime_profile"]["capabilities"]["environment"] == "candidate"
    assert (ctx["state"] == 1).any()
