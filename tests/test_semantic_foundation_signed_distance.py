import unittest

import numpy as np

from src.semantic_foundation_signed_distance import (
    foundation_signed_distance,
    five_class_signed_distance_loss,
)


try:
    import torch
    import torch.nn.functional as functional

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


class SignedDistanceTests(unittest.TestCase):
    def test_foundation_signed_distance_has_opposite_sides_and_is_clipped(self):
        label = np.zeros((9, 9), dtype=np.uint8)
        label[3:6, 3:6] = 4
        target = foundation_signed_distance(label, truncation_pixels=2)
        self.assertEqual(target.dtype, np.float32)
        self.assertGreater(target[4, 4], 0)
        self.assertLess(target[0, 0], 0)
        self.assertLessEqual(target.max(), 1.0)
        self.assertGreaterEqual(target.min(), -1.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "requires optional Torch runtime")
    def test_zero_auxiliary_weight_is_exact_weighted_cross_entropy(self):
        torch.manual_seed(7)
        logits = torch.randn(1, 5, 3, 3)
        distance = torch.randn(1, 1, 3, 3)
        labels = torch.tensor([[[0, 1, 2], [3, 4, 0], [1, 2, 3]]])
        target = torch.zeros(1, 1, 3, 3)
        weights = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0])
        actual, parts = five_class_signed_distance_loss(
            logits,
            distance,
            labels,
            target,
            class_weights=weights,
            auxiliary_weight=0.0,
            smooth_l1_beta=0.25,
        )
        self.assertTrue(
            torch.equal(actual, functional.cross_entropy(logits.float(), labels, weight=weights))
        )
        self.assertEqual(float(parts["signed_distance"]), 0.0)
