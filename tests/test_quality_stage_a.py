import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_quality_stage_a import verify_audit, selected_arms
from src.quality_protocol import sha256


class QualityStageATests(unittest.TestCase):
    def test_explicit_subset_keeps_protocol_order_and_excludes_failed_arm(self):
        self.assertEqual(selected_arms(["RGB-113-SOF-GEO", "RGB-112-SOF-NVS"]),
                         ["RGB-112-SOF-NVS", "RGB-113-SOF-GEO"])
        for invalid in ([], ["RGB-111-PGSR"] * 2, ["unregistered"]):
            with self.assertRaises(ValueError):
                selected_arms(invalid)

    def test_exact_parent_result_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, audit = root / "run", root / "audit"
            run.mkdir()
            audit.mkdir()
            (run / "result.json").write_text('{"example": 1}')
            (audit / "result.json").write_text(json.dumps({"status": "passed", "run_result_sha256": sha256(run / "result.json")}))
            verify_audit(audit, run, "passed")
            (run / "result.json").write_text('{"example": 2}')
            with self.assertRaises(ValueError):
                verify_audit(audit, run, "passed")

    def test_probe_status_not_inferred_from_presence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, audit = root / "run", root / "audit"
            run.mkdir()
            audit.mkdir()
            (run / "result.json").write_text('{}')
            (audit / "result.json").write_text(json.dumps({"status": "failed", "run_result_sha256": sha256(run / "result.json")}))
            with self.assertRaises(ValueError):
                verify_audit(audit, run, "passed")
