import unittest
import numpy as np
from src.quality_diagnostics import diagnose_stages, stage_changes, deck_gap_provenance


class QualityDiagnosticsTests(unittest.TestCase):
    def test_deck_gap_origin_is_disjoint_and_last_loss_not_first(self):
        gt = np.array([[1, 1, 1, 0]], np.uint8)
        stages = {"static": np.array([[1, 0, 1, 1]], np.uint8),
                  "fusion": np.array([[0, 0, 0, 0]], np.uint8),
                  "recovery": np.array([[1, 0, 0, 0]], np.uint8),
                  "review": np.array([[0, 0, 0, 0]], np.uint8)}
        result, masks = deck_gap_provenance(stages, gt)
        self.assertEqual(result["final_true_deck_as_background"], 3)
        self.assertEqual(result["earlier_deck_lost_later"], 2)
        self.assertEqual(result["never_deck_in_deployed_sequence"], 1)
        self.assertEqual(result["last_deck_loss_stage"], {"fusion": 1, "recovery": 0, "review": 1})
        self.assertFalse((masks["earlier_deck_lost_later"] & masks["upstream_unresolved"]).any())

    def test_disjoint_correct_harmful_and_still_wrong(self):
        before = np.array([[0, 1, 2, 3]], np.uint8)
        after = np.array([[1, 0, 4, 3]], np.uint8)
        target = np.array([[1, 1, 1, 3]], np.uint8)
        counts, mask = stage_changes(before, after, target)
        self.assertEqual([counts[n] for n in ("changed_pixels", "correct_repairs", "harmful_changes", "wrong_to_different_wrong")], [3, 1, 1, 1])
        np.testing.assert_array_equal(mask, [[True, True, True, False]])

    def test_interpolation_does_not_get_truth_or_quality(self):
        a = np.zeros((2, 2), np.uint8)
        result, _ = diagnose_stages({"static": a}, interpolation_view=True)
        self.assertEqual(result["scope"], "interpolation_continuity_only")
        self.assertIsNone(result["stages"]["static"]["metrics"])
        with self.assertRaises(ValueError):
            diagnose_stages({"static": a}, target=a, interpolation_view=True)

    def test_postprocess_deletion_and_roi(self):
        target = np.ones((2, 2), np.uint8)
        after = np.zeros((2, 2), np.uint8)
        roi = np.array([[True, False], [False, False]])
        result, masks = diagnose_stages({"expert": target, "review": after}, target=target, roi=roi, has_same_camera_photo=True)
        self.assertEqual(result["changes"]["expert -> review"]["harmful_changes"], 1)
        self.assertEqual(int(masks["expert -> review"].sum()), 1)
        self.assertTrue(result["geometry_attribution"].startswith("unresolved"))

    def test_reject_invalid_labels_grid_and_probabilities(self):
        a = np.zeros((2, 2), np.uint8)
        for invalid in (np.ones((2, 2), np.uint8) * 5, -np.ones((2, 2), np.int8), np.ones((1, 2), np.uint8), np.ones((2, 2), float)):
            with self.assertRaises(ValueError):
                stage_changes(a, invalid)
        with self.assertRaises(ValueError):
            diagnose_stages({"a": a}, probabilities={"five": np.ones((5, 2, 2))})
