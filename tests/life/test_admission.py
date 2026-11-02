import json
import tempfile
import unittest
from pathlib import Path

from awr.life.common.admission import AdmissionError, life_admission


def event(root, scenario, version, tape_sha="aa", audit=None):
    d = Path(root) / scenario / f"v{version}"
    d.mkdir(parents=True)
    (d / "tape.json").write_text(json.dumps(dict(content_sha256=tape_sha)))
    if audit is not None:
        (d / "life-audit.json").write_text(json.dumps(audit))
    return d


class Admission(unittest.TestCase):
    def test_passing_audit_of_the_same_tape_is_admitted(self):
        with tempfile.TemporaryDirectory() as t:
            d = event(
                t, "town-life-002", 13, audit=dict(tape_content_sha256="aa", gates=dict(gates=dict(g=dict(pass_=True))))
            )
            self.assertEqual(life_admission(d)["status"], "passed")

    def test_failed_foreign_or_missing_audit_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            bad = event(t, "a", 1, audit=dict(tape_content_sha256="aa", gates=dict(gates=dict(g=dict(pass_=False)))))
            other = event(t, "b", 1, audit=dict(tape_content_sha256="bb", gates=dict(gates=dict(g=dict(pass_=True)))))
            none = event(t, "c", 1)
            for d in (bad, other, none):
                with self.assertRaises(AdmissionError):
                    life_admission(d, allow_historical=True)

    def test_historical_tape_is_shown_but_never_admits_task_packs(self):
        with tempfile.TemporaryDirectory() as t:
            d = event(t, "town-sandbox-001", 7)
            self.assertEqual(life_admission(d, allow_historical=True)["status"], "historical")
            with self.assertRaises(AdmissionError):
                life_admission(d)


if __name__ == "__main__":
    unittest.main()
