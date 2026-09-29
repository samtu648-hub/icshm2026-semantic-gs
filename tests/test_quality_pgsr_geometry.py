import unittest
try:
    import torch
except ImportError:
    torch = None
from src.quality_pgsr_geometry import accepted_cycle_noise


@unittest.skipIf(torch is None, "optional torch runtime")
class PgsrAcceptedCycleTests(unittest.TestCase):
    def cycle(self, points, pixels):
        projected = points[:, :2] / points[:, 2:3]
        return torch.linalg.vector_norm(projected - pixels, dim=-1), points[:, 2] > 0

    def test_rejected_zero_depth_does_not_poison_accepted_gradients(self):
        points = torch.tensor([[0., 0., 0.], [.2, .3, 1.]], requires_grad=True)
        pixels = torch.zeros(2, 2)
        noise, mask, weights = accepted_cycle_noise(points, pixels, self.cycle, 1.)
        (noise[mask] * weights[mask]).mean().backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertEqual(points.grad[0].abs().sum().item(), 0)
        self.assertEqual(mask.tolist(), [False, True])

    def test_finite_inliers_values_weights_and_gradients_match_native(self):
        a = torch.tensor([[.2, .3, 1.], [3., 4., 1.]], requires_grad=True)
        b = a.detach().clone().requires_grad_()
        pixels = torch.zeros(2, 2)
        original, native_mask = self.cycle(a, pixels)
        mask = native_mask & (original < 1.)
        weights = (1 / torch.exp(original)).detach()
        expected = (weights[mask] * original[mask]).mean()
        actual_noise, actual_mask, actual_weights = accepted_cycle_noise(b, pixels, self.cycle, 1.)
        actual = (actual_weights[actual_mask] * actual_noise[actual_mask]).mean()
        expected.backward(); actual.backward()
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        self.assertEqual(mask.tolist(), actual_mask.tolist())

    def test_all_invalid_abstains_without_creating_bad_graph(self):
        points = torch.zeros(2, 3, requires_grad=True)
        noise, mask, weights = accepted_cycle_noise(points, torch.zeros(2, 2), self.cycle, 1.)
        self.assertFalse(mask.any())
        self.assertTrue(torch.isfinite(noise).all())
        self.assertFalse(noise.requires_grad)
        self.assertEqual(weights.sum().item(), 0)
