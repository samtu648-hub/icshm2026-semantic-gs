import copy
import unittest
from src.quality_pgsr_prerequisite import validate_diagnostic, validate_smoke


class QualityPGSRPrerequisiteTests(unittest.TestCase):
    def test_diagnostic_requires_full_finite_window_and_exact_repair_source(self):
        expected = {"minimum_completed_optimizer_steps": 3000, "patch_sha256": "patch", "helper": "helper.py", "helper_sha256": "code"}
        source = {"train.py": "safe"}
        record = {"status": "completed_numeric_diagnostic_not_quality_experiment", "backward_diagnostic": {
            "status": "diagnostic_limit_reached_no_nonfinite_observed", "completed_optimizer_steps": 3000},
            "source_manifest": source, "source_delta_patch_sha256": "patch", "code_sha256": {"helper.py": "code"}}
        validate_diagnostic(record, expected, source, "code")
        for bad in ({**record, "source_manifest": {"train.py": "original"}}, {**record, "source_delta_patch_sha256": "wrong"}):
            with self.assertRaises(ValueError):
                validate_diagnostic(bad, expected, source, "code")
        bad = copy.deepcopy(record)
        bad["backward_diagnostic"]["completed_optimizer_steps"] = 20
        with self.assertRaises(ValueError):
            validate_diagnostic(bad, expected, source, "code")

    def test_full_trial_rejects_wrong_mode_nan_and_source_drift(self):
        record = {"experiment_id": "RGB115", "mode": "smoke", "status": "completed_native_training_pending_independent_reload_and_evaluation",
                  "config_sha256": "config", "protocol_sha256": "protocol", "code_sha256": {"code": "hash"},
                  "source_manifest_sha256": "source", "extra_sources": [], "numeric_inventory": {"ply": {"finite": True}}}
        expected = dict(experiment="RGB115", config_hash="config", protocol_hash="protocol", code_hashes={"code": "hash"}, source_hash="source", extra_sources=[])
        validate_smoke(record, **expected)
        for changes in ({"mode": "full"}, {"source_manifest_sha256": "old"}, {"code_sha256": {}},
                        {"numeric_inventory": {"ply": {"finite": False}}}, {"numeric_inventory": {}}):
            with self.assertRaises(ValueError):
                validate_smoke({**record, **changes}, **expected)
