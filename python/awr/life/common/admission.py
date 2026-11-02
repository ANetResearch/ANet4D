"""Life-audit admission of an event for display and task building.

A life-scenario event is used only when its life audit (written by the builder
before publishing, awr.life.civil.life_audit) passed every gate of its config and
belongs to the same tape. Events built before the audit became a publish
condition are named explicitly below and may only be shown as historical data;
they never admit new task packs.
"""

from __future__ import annotations

import json
from pathlib import Path

HISTORICAL_EVENTS = {
    ("town-sandbox-001", 7): "Historical evaluation tape (2026-10-20), built before the life audit became a publish "
    "condition; shown as historical data only, not admitted for new task packs.",
}


class AdmissionError(RuntimeError):
    pass


def life_admission(event: Path, *, allow_historical: bool = False) -> dict:
    event = Path(event)
    from awr.life.envpack.execution import MARKER

    if (event / MARKER).exists():
        raise AdmissionError(f"{event}: candidate environment diagnostic; not admitted for display or task packs")
    key = (event.parent.name, int(event.name.removeprefix("v")) if event.name[1:].isdigit() else None)
    tape_sha = (
        json.loads((event / "tape.json").read_text()).get("content_sha256") if (event / "tape.json").exists() else None
    )
    path = event / "life-audit.json"
    if path.exists():
        report = json.loads(path.read_text())
        if report.get("tape_content_sha256") != tape_sha:
            raise AdmissionError(
                f"{event}: life-audit.json belongs to tape {str(report.get('tape_content_sha256'))[:12]}, not {str(tape_sha)[:12]}"
            )
        gates = (report.get("gates") or {}).get("gates") or {}
        failed = sorted(k for k, g in gates.items() if g.get("pass_") is not True)
        if not gates or failed:
            raise AdmissionError(f"{event}: life audit did not pass ({', '.join(failed) or 'no gates'})")
        return dict(status="passed", event=str(event), tape_content_sha256=tape_sha, gates=len(gates))
    if key in HISTORICAL_EVENTS:
        if allow_historical:
            return dict(
                status="historical", event=str(event), tape_content_sha256=tape_sha, note=HISTORICAL_EVENTS[key]
            )
        raise AdmissionError(f"{event}: historical tape without a life audit; not admitted for new task packs")
    raise AdmissionError(
        f"{event}: no life-audit.json; build it with a life config (life_gates) or audit it before use"
    )
