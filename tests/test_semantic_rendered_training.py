from __future__ import annotations

import unittest

import numpy as np

from src.semantic_rendered_training import (
    oof_qualification_gate,
    rendered_segmentation_panel,
    select_rendered_training_views,
    validate_oof_prediction_plan,
)


class RenderedTrainingContractTests(unittest.TestCase):
    def test_training_plan_rejects_validation_or_unlabelled_leakage(self) -> None:
        split = {
            "semantic_train_labeled": ["001.png", "002.png", "003.png"],
            "semantic_validation_labeled": ["004.png"],
            "excluded_unlabeled": ["005.png"],
        }
        self.assertEqual(
            select_rendered_training_views(split, expected_count=3),
            ["001.png", "002.png", "003.png"],
        )
        split["semantic_train_labeled"][2] = "004.png"
        with self.assertRaisesRegex(ValueError, "validation"):
            select_rendered_training_views(split, expected_count=3)

    def test_oof_plan_requires_exactly_one_prediction_per_training_view(self) -> None:
        split = {
            "folds": [
                {"fold": 0, "oof": ["001.png"]},
                {"fold": 1, "oof": ["002.png"]},
                {"fold": 2, "oof": ["003.png"]},
            ],
            "fixed_validation_read_only": ["004.png"],
        }
        records = [
            {"fold": 0, "names": ["001.png"]},
            {"fold": 1, "names": ["002.png"]},
            {"fold": 2, "names": ["003.png"]},
        ]
        validate_oof_prediction_plan(
            training_names=["001.png", "002.png", "003.png"],
            oof_split=split,
            fold_records=records,
        )
        records[2] = {"fold": 2, "names": ["004.png"]}
        with self.assertRaisesRegex(ValueError, "OOF"):
            validate_oof_prediction_plan(
                training_names=["001.png", "002.png", "003.png"],
                oof_split=split,
                fold_records=records,
            )

    def test_fixed_gate_and_panel_contract(self) -> None:
        metrics = {
            "mIoU_with_background": 0.72,
            "background_to_stay_cable_fraction": 0.019,
            "per_class": {
                "stay_cable": {"iou": 0.71, "precision": 0.81, "recall": 0.91}
            },
        }
        gate = oof_qualification_gate(
            metrics,
            {
                "mIoU_with_background_min": 0.70,
                "stay_cable_iou_min": 0.70,
                "stay_cable_precision_min": 0.80,
                "stay_cable_recall_min": 0.90,
                "background_to_stay_cable_fraction_max": 0.02,
                "selection_rule": "All fixed checks must pass.",
            },
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["selection_rule"], "All fixed checks must pass.")
        rendered = np.zeros((3, 4, 3), dtype=np.uint8)
        target = np.zeros((3, 4), dtype=np.uint8)
        prediction = target.copy()
        prediction[0, 0] = 2
        panel = rendered_segmentation_panel(rendered, target, prediction)
        self.assertEqual(panel.shape, (3, 16, 3))
        self.assertEqual(panel.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
