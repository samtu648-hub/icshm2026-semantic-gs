from __future__ import annotations

import unittest

from src.semantic_rendered_final import (
    final_retention_gate,
    require_exact_view_set,
    validate_final_training_result,
)


class RenderedFinalContractTests(unittest.TestCase):
    def test_final_checkpoint_must_be_frozen_before_validation(self) -> None:
        result = {
            "experiment_id": "SEM-102-RENDERED-SEG-FINAL",
            "status": "completed_final_checkpoint_frozen",
            "checkpoint_reload": "passed",
            "checkpoint_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "fixed_validation_opened": False,
            "unlabeled_rgb_opened": False,
            "real_rgb_input_opened": False,
        }
        validate_final_training_result(
            result, checkpoint_sha256="a" * 64, config_sha256="b" * 64
        )
        result["fixed_validation_opened"] = True
        with self.assertRaisesRegex(ValueError, "validation"):
            validate_final_training_result(
                result, checkpoint_sha256="a" * 64, config_sha256="b" * 64
            )

    def test_retention_gate_needs_all_metric_protections(self) -> None:
        baseline = {
            "mIoU_with_background": 0.60,
            "background_to_stay_cable_pixels": 1000,
            "worst_10_percent_mIoU_without_background": 0.40,
            "challenge_140_146_mIoU_without_background": 0.50,
            "per_class": {
                "deck": {"iou": 0.80},
                "stay_cable": {"iou": 0.10, "recall": 0.30},
                "tower": {"iou": 0.70},
                "foundation": {"iou": 0.40},
            },
        }
        gates = {
            "mIoU_with_background_drop_max": 0.0,
            "stay_cable_iou_drop_max": 0.0,
            "stay_cable_recall_drop_max": 0.0,
            "protected_classes": ["deck", "tower", "foundation"],
            "protected_class_iou_drop_max": 0.005,
            "background_to_stay_cable_reduction_min_fraction": 0.01,
            "worst_10_percent_mIoU_without_background_drop_max": 0.005,
            "challenge_140_146_mIoU_without_background_drop_max": 0.005,
        }
        metrics = {
            "mIoU_with_background": 0.61,
            "background_to_stay_cable_pixels": 900,
            "worst_10_percent": {"mean_mIoU_without_background": 0.41},
            "challenge_140_146": {"mean_mIoU_without_background": 0.51},
            "per_class": {
                "deck": {"iou": 0.80},
                "stay_cable": {"iou": 0.11, "recall": 0.31},
                "tower": {"iou": 0.70},
                "foundation": {"iou": 0.40},
            },
        }
        gate = final_retention_gate(metrics, baseline, gates)
        self.assertTrue(gate["passed"])
        metrics["background_to_stay_cable_pixels"] = 995
        self.assertFalse(final_retention_gate(metrics, baseline, gates)["passed"])

    def test_view_plan_must_be_exact_and_unique(self) -> None:
        self.assertEqual(
            require_exact_view_set(
                ["001.png", "002.png"],
                ["001.png", "002.png"],
                context="test plan",
            ),
            ["001.png", "002.png"],
        )
        with self.assertRaisesRegex(ValueError, "test plan"):
            require_exact_view_set(
                ["001.png", "001.png"],
                ["001.png", "002.png"],
                context="test plan",
            )


if __name__ == "__main__":
    unittest.main()
