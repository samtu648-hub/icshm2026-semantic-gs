import unittest

try:
    import torch
except ImportError:  # The repository health environment keeps Torch optional.
    torch = None

from src.quality_mask2former_boundary import (
    dense_semantic_probabilities,
    soft_morphological_boundary,
    stay_cable_boundary_dice,
)


@unittest.skipIf(torch is None, "requires optional Torch runtime")
class Mask2FormerBoundaryContracts(unittest.TestCase):
    def test_dense_probability_is_normalized_and_differentiable(self):
        logits = torch.zeros((1, 2, 6), requires_grad=True)
        masks = torch.zeros((1, 2, 3, 4), requires_grad=True)
        probability = dense_semantic_probabilities(
            {"pred_logits": logits, "pred_masks": masks}, size=(6, 8)
        )
        self.assertEqual(tuple(probability.shape), (1, 5, 6, 8))
        self.assertTrue(torch.allclose(probability.sum(1), torch.ones((1, 6, 8))))
        probability[:, 2].sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(masks.grad)

    def test_boundary_and_dice_reward_matching_cable_edge(self):
        target = torch.zeros((1, 8, 8), dtype=torch.long)
        target[:, 2:6, 3:5] = 2
        probability = torch.zeros((1, 5, 8, 8), dtype=torch.float32)
        probability[:, 0] = 1.0
        probability[:, 0, 2:6, 3:5] = 0.0
        probability[:, 2, 2:6, 3:5] = 1.0
        matching, _ = stay_cable_boundary_dice(probability, target)
        missing = probability.clone()
        missing[:, 0] = 1.0
        missing[:, 2] = 0.0
        absent, _ = stay_cable_boundary_dice(missing, target)
        self.assertLess(float(matching), 1e-6)
        self.assertGreater(float(absent), float(matching))
        edge = soft_morphological_boundary(probability[:, 2:3])
        self.assertGreater(float(edge.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
