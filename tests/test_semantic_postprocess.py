from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_sem070_post_clean import choose_check_threshold

from src.semantic_postprocess import (
    CLASS_NAMES,
    apply_anchor_probability_transfer,
    finalize_anchor_rows,
    load_suppression_mask,
    mask_for_threshold,
    new_anchor_statistics,
    save_suppression_mask,
    semantic_quad,
    split_training_views,
    training_half,
    update_anchor_statistics,
    validate_suppression_mask,
)


class SemanticPostprocessTest(unittest.TestCase):
    def test_zero_mask_preserves_probabilities_and_labels(self) -> None:
        probabilities = np.asarray(
            [[0.10, 0.20, 0.30, 0.15, 0.25], [0.55, 0.10, 0.05, 0.20, 0.10]], dtype=np.float32
        )
        zero = np.zeros(2, dtype=np.bool_)
        transformed = apply_anchor_probability_transfer(probabilities, zero)
        np.testing.assert_array_equal(transformed, probabilities)
        np.testing.assert_array_equal(transformed.argmax(axis=1), probabilities.argmax(axis=1))

    def test_probability_transfer_preserves_simplex_and_only_moves_cable_mass(self) -> None:
        probabilities = np.asarray(
            [[0.10, 0.20, 0.30, 0.15, 0.25], [0.55, 0.10, 0.05, 0.20, 0.10]], dtype=np.float64
        )
        transformed = apply_anchor_probability_transfer(probabilities, np.asarray([True, False]))
        np.testing.assert_allclose(transformed.sum(axis=1), 1.0, atol=1e-12)
        self.assertEqual(float(transformed[0, 0]), 0.40)
        self.assertEqual(float(transformed[0, 2]), 0.0)
        np.testing.assert_array_equal(transformed[1], probabilities[1])

    @unittest.skipUnless(__import__("importlib.util").util.find_spec("torch") is not None, "Torch is optional in the lightweight test environment")
    def test_torch_offset_transfer_preserves_simplex(self) -> None:
        import torch

        from src.semantic_postprocess import apply_offset_probability_transfer

        probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.15, 0.25], [0.55, 0.1, 0.05, 0.2, 0.1]], dtype=torch.float32)
        transformed = apply_offset_probability_transfer(probabilities, torch.tensor([2, 4]), np.asarray([False, False, True, False, False, False], dtype=np.bool_))
        self.assertTrue(torch.allclose(transformed.sum(dim=1), torch.ones(2)))
        self.assertEqual(float(transformed[0, 0]), 0.4)
        self.assertEqual(float(transformed[0, 2]), 0.0)
        self.assertTrue(torch.equal(transformed[1], probabilities[1]))

    def test_mask_round_trip_is_one_boolean_vector(self) -> None:
        mask = np.asarray([True, False, True, False], dtype=np.bool_)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "postprocess_mask.npz"
            digest = save_suppression_mask(path, mask, 4)
            self.assertEqual(len(digest), 64)
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual(payload.files, ["mask"])
                self.assertEqual(payload["mask"].dtype, np.bool_)
            np.testing.assert_array_equal(load_suppression_mask(path, 4), mask)

    def test_mask_shape_and_dtype_are_strict(self) -> None:
        with self.assertRaises(ValueError):
            validate_suppression_mask(np.zeros((2, 2), dtype=np.bool_), 4)
        with self.assertRaises(ValueError):
            validate_suppression_mask(np.zeros(3, dtype=np.bool_), 4)
        with self.assertRaises(TypeError):
            validate_suppression_mask(np.zeros(4, dtype=np.uint8), 4)

    def test_hash_parity_split_is_deterministic_and_disjoint(self) -> None:
        names = [f"{index:03d}.png" for index in range(1, 220)]
        calibration, check = split_training_views(names)
        self.assertEqual(set(calibration) | set(check), set(names))
        self.assertFalse(set(calibration) & set(check))
        self.assertEqual(calibration, [name for name in names if training_half(name) == 0])
        self.assertEqual(check, [name for name in names if training_half(name) == 1])

    def test_anchor_statistics_and_threshold_use_calibration_values(self) -> None:
        statistics = new_anchor_statistics(2)
        target = np.asarray([[0, 0, 2, 0], [2, 0, 0, 2]], dtype=np.uint8)
        prediction = np.asarray([[2, 2, 2, 0], [2, 0, 2, 2]], dtype=np.uint8)
        anchors = np.asarray([[0, 0, 0, 1], [1, 1, 1, 1]], dtype=np.int64)
        update_anchor_statistics(statistics, train_half=0, anchor_indices=anchors, target=target, prediction=prediction)
        update_anchor_statistics(statistics, train_half=0, anchor_indices=anchors, target=target, prediction=prediction)
        update_anchor_statistics(statistics, train_half=1, anchor_indices=anchors, target=target, prediction=prediction)
        logits = np.asarray([[0.0, -1.0, 2.0, -1.0, -1.0], [3.0, 0.0, -2.0, 0.0, 0.0]])
        rows = finalize_anchor_rows(statistics, logits)
        self.assertTrue(rows[0]["stay_cable_leaning"])
        self.assertEqual(rows[0]["calibration_support_views"], 2)
        self.assertEqual(rows[0]["calibration_dominant_pixels"], 6)
        mask = mask_for_threshold(rows, 0.50, minimum_support_views=2, minimum_dominant_pixels=4)
        self.assertTrue(bool(mask[0]))
        self.assertFalse(bool(mask[1]))

    def test_quad_has_gt_raw_cleaned_error_panels(self) -> None:
        target = np.asarray([[0, 2], [1, 4]], dtype=np.uint8)
        raw = np.asarray([[2, 2], [1, 4]], dtype=np.uint8)
        clean = np.asarray([[0, 2], [1, 4]], dtype=np.uint8)
        quad = semantic_quad(target, raw, clean)
        self.assertEqual(quad.shape, (2, 8, 3))

    def test_contract_config_and_external_entry_are_frozen_and_no_reference_based(self) -> None:
        config = json.loads((ROOT / "experiments/runs/SEM-070-POST-CLEAN/config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["postprocess"]["class_mapping"]["stay_cable"], 2)
        self.assertEqual(config["postprocess"]["threshold_grid"], [0.5, 0.6, 0.7, 0.8, 0.9])
        self.assertTrue(config["frozen_semantic"]["checkpoint_is_not_modified"])
        external_source = (ROOT / "scripts/render_sem070_external_camera.py").read_text(encoding="utf-8")
        self.assertIn("CAMERA_FROM_WORLD", external_source)
        self.assertIn("SIMPLE_RADIAL", external_source)
        self.assertNotIn("read_label", external_source)
        self.assertNotIn("reference_rgb", external_source)

    def test_formal_class_names_keep_stay_cable_as_id_two(self) -> None:
        self.assertEqual(CLASS_NAMES[2], "stay_cable")

    def test_check_threshold_maximizes_reduction_under_protection_constraints(self) -> None:
        rows = [
            {"anchor_index": 0, "stay_cable_leaning": True, "calibration_support_views": 3, "calibration_dominant_pixels": 128, "calibration_false_positive_rate": 0.80},
            {"anchor_index": 1, "stay_cable_leaning": True, "calibration_support_views": 3, "calibration_dominant_pixels": 128, "calibration_false_positive_rate": 0.60},
        ]
        raw = {
            "background_to_stay_cable_false_positive_pixels": 100,
            "stay_cable": {"IoU": 0.1, "recall": 0.5},
            "per_class": {"deck": {"IoU": 0.8}, "tower": {"IoU": 0.8}, "foundation": {"IoU": 0.8}},
            "mIoU_with_background": 0.6,
        }

        def evaluate(mask: np.ndarray) -> dict:
            count = int(mask.sum())
            fp = 20 if count == 1 else 0
            return {
                "background_to_stay_cable_false_positive_pixels": fp,
                "stay_cable": {"IoU": 0.1, "recall": 0.5},
                "per_class": {"deck": {"IoU": 0.8}, "tower": {"IoU": 0.8}, "foundation": {"IoU": 0.8}},
                "mIoU_with_background": 0.6,
            }

        selected, threshold, candidates = choose_check_threshold(
            rows,
            [0.5, 0.7],
            minimum_support_views=3,
            minimum_dominant_pixels=128,
            raw_check_metrics=raw,
            evaluate_candidate=evaluate,
            minimum_recall_retention=0.98,
            maximum_protected_iou_drop=0.005,
        )
        self.assertEqual(threshold, 0.5)
        np.testing.assert_array_equal(selected, np.asarray([True, True]))
        self.assertTrue(all(candidate["meets_check_constraints"] for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
