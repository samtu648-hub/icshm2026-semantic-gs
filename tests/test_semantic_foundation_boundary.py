import unittest

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.semantic_foundation_boundary import (
    foundation_boundary_cross_entropy,
    foundation_inner_boundary_mask,
    quality_gates,
)


class FoundationBoundaryContracts(unittest.TestCase):
    def test_inner_boundary_selects_only_foundation(self):
        label = np.zeros((9, 9), np.uint8)
        label[1:8, 1:8] = 4
        boundary = foundation_inner_boundary_mask(label, radius=2)
        self.assertTrue(boundary[1, 4])
        self.assertTrue(boundary[2, 4])
        self.assertFalse(boundary[4, 4])
        self.assertFalse(boundary[0, 4])
        self.assertTrue(np.all(label[boundary] == 4))

    def test_empty_foundation_and_bad_radius(self):
        label = np.zeros((3, 3), np.uint8)
        self.assertFalse(foundation_inner_boundary_mask(label, radius=1).any())
        with self.assertRaises(ValueError):
            foundation_inner_boundary_mask(label, radius=0)

    @unittest.skipIf(torch is None, "requires optional Torch runtime")
    def test_control_is_original_weighted_cross_entropy(self):
        torch.manual_seed(3)
        logits = torch.randn(1, 5, 4, 4)
        target = torch.randint(0, 5, (1, 4, 4))
        boundary = torch.zeros_like(target, dtype=torch.bool)
        weights = torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0])
        expected = torch.nn.functional.cross_entropy(logits, target, weight=weights)
        actual, _ = foundation_boundary_cross_entropy(
            logits, target, boundary, class_weights=weights, multiplier=1.0
        )
        self.assertTrue(torch.equal(actual, expected))

    @unittest.skipIf(torch is None, "requires optional Torch runtime")
    def test_candidate_emphasises_wrong_boundary_pixel(self):
        target = torch.tensor([[[4, 0], [0, 0]]])
        boundary = torch.tensor([[[True, False], [False, False]]])
        logits = torch.full((1, 5, 2, 2), -4.0)
        logits[:, 0] = 4.0
        weights = torch.ones(5)
        control, _ = foundation_boundary_cross_entropy(
            logits, target, boundary, class_weights=weights, multiplier=1.0
        )
        candidate, selected = foundation_boundary_cross_entropy(
            logits, target, boundary, class_weights=weights, multiplier=2.0
        )
        self.assertGreater(float(candidate), float(control))
        self.assertEqual(float(selected.sum()), 1.0)

    @unittest.skipIf(torch is None, "requires optional Torch runtime")
    def test_mask_cannot_include_non_foundation(self):
        target = torch.zeros((1, 2, 2), dtype=torch.long)
        with self.assertRaises(ValueError):
            foundation_boundary_cross_entropy(
                torch.zeros((1, 5, 2, 2)),
                target,
                torch.ones_like(target, dtype=torch.bool),
                class_weights=torch.ones(5),
                multiplier=2.0,
            )

    def test_quality_retains_control_gain_and_recovers_recall(self):
        def metrics(iou, precision, recall, miou):
            return {
                "per_class": {
                    "foundation": {"iou": iou, "precision": precision, "recall": recall}
                },
                "mIoU_with_background": miou,
            }

        gates = {
            "foundation_iou_gain": 0.01,
            "foundation_precision_gain": 0.01,
            "maximum_recall_drop": 0.005,
            "recall_floor": 0.95,
            "minimum_recall_gain_vs_control": 0.01,
            "maximum_iou_drop_vs_control": 0.005,
            "maximum_miou_drop_vs_control": 0.001,
        }
        checks = quality_gates(
            metrics(0.65, 0.67, 0.956, 0.86),
            metrics(0.70, 0.73, 0.953, 0.879),
            metrics(0.703, 0.74, 0.938, 0.880),
            gates,
        )
        self.assertTrue(all(checks.values()))


if __name__ == "__main__":
    unittest.main()
