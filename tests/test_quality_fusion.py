import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import numpy as np
from src.quality_fusion import apply_stage, pipeline, choose_threshold
from src.quality_semantic_audit import metrics


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.base = np.array([[0, 1, 2, 3, 4]], np.uint8)
        self.five = np.zeros((5, 1, 5), np.float32); self.five[2] = 1
        self.p = np.ones((1, 5), np.float32)

    def test_old_role_lane_and_foundation(self):
        result = pipeline(self.base, self.five, self.p, self.p,
            {"fiveclass_lane": .5, "cable": .5, "foundation": .5, "deck": .5})
        np.testing.assert_array_equal(result["deck"], [[2, 1, 2, 3, 0]])
        self.assertEqual(result["cable"][0, 4], 4)

    def test_off_is_identity(self):
        result = pipeline(self.base, self.five, self.p, self.p, {})
        for v in result.values():
            np.testing.assert_array_equal(v, self.base)

    def test_rejected_foundation_not_deck(self):
        bg = np.zeros_like(self.base)
        out = apply_stage("deck", bg, self.five, self.p, self.p, .5, self.base)
        np.testing.assert_array_equal(out, [[1, 0, 0, 0, 0]])

    def test_threshold_tie_off(self):
        m = metrics(np.eye(5, dtype=np.int64))
        row = choose_threshold([{"threshold": .9, "metrics": m}, {"threshold": None, "metrics": m}], 1)
        self.assertIsNone(row["threshold"])

    def test_missing_off_refused(self):
        with self.assertRaises(ValueError):
            choose_threshold([{"threshold": .5, "metrics": metrics(np.eye(5, dtype=np.int64))}], 2)

    def test_unknown_stage_and_invalid_probability(self):
        with self.assertRaises(ValueError):
            apply_stage("erase", self.base, self.five, self.p, self.p, None)
        with self.assertRaises(ValueError):
            pipeline(self.base, self.five * 2, self.p, self.p, {})

    def test_scoring_stage_masks_and_independent_counts(self):
        from scripts.evaluate_quality_fusion import score_views
        class Labels:
            def read(inner, name):
                if name != "001.png":
                    raise ValueError("Wrong label role")
                return self.base.copy()
        with tempfile.TemporaryDirectory() as tmp, patch("scripts.evaluate_quality_fusion.read_inputs", return_value=(self.base, self.five, self.p, self.p)):
            out = score_views({"001.png": {"partition": "oof-0"}}, ["001.png"], Labels(), {}, Path(tmp), None, True)
            self.assertEqual(out["metrics"]["mIoU_with_background"], 1.)
            self.assertEqual(out["changes"]["changed"], 0)
            self.assertTrue((Path(tmp) / "deck_evidence-modified/001.png").is_file())

    def test_incomplete_fit_batch_not_accepted(self):
        import json
        from scripts.evaluate_quality_fusion import open_parent
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "result.json"
            p.write_text(json.dumps({"status": "running_serial_first_round"}))
            with self.assertRaises(ValueError):
                open_parent(Path(tmp), "completed_four_arm_inputs_fusion_calibration_pending")


if __name__ == "__main__":
    unittest.main()
