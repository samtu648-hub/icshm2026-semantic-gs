"""CPU and optional CUDA tests for the isolated RGB-092 loss/protection contract."""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None or not os.environ.get("R1_SCAFFOLD_SOURCE"), "requires frozen Torch and Scaffold source")
class TransientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.environ["R1_SCAFFOLD_SOURCE"])
        from src import scaffold_transient as transient
        cls.t = transient
        torch.set_num_threads(2)

    def test_all_one_loss_and_gradient_equivalence(self):
        from utils.loss_utils import l1_loss, ssim
        torch.manual_seed(7)
        x = torch.rand(3, 37, 49, requires_grad=True)
        y = torch.rand_like(x)
        original = .8 * l1_loss(x, y) + .2 * (1 - ssim(x, y))
        original_grad, = torch.autograd.grad(original, x)
        a, b = self.t.weighted_losses(x, y, torch.ones(37, 49))
        actual = .8 * a + .2 * b
        grad, = torch.autograd.grad(actual, x)
        torch.testing.assert_close(original, actual, rtol=1e-6, atol=2e-7)
        torch.testing.assert_close(original_grad, grad, rtol=2e-5, atol=1e-8)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA device required")
    def test_cuda_native_size_loss_and_gradient_equivalence(self):
        from utils.loss_utils import l1_loss, ssim
        with self.t.isolated_rng():
            torch.manual_seed(7)
            x = torch.rand(3, 989, 1320, device="cuda", requires_grad=True)
            y = torch.rand_like(x)
            original = .8 * l1_loss(x, y) + .2 * (1 - ssim(x, y))
            original_grad, = torch.autograd.grad(original, x)
            a, b = self.t.weighted_losses(x, y, torch.ones(989, 1320, device="cuda"))
            actual = .8 * a + .2 * b
            grad, = torch.autograd.grad(actual, x)
            torch.testing.assert_close(original, actual, rtol=1e-6, atol=2e-7)
            torch.testing.assert_close(original_grad, grad, rtol=2e-5, atol=1e-8)

    def test_protection_and_warmup(self):
        labels = torch.zeros(40, 50, dtype=torch.uint8)
        labels[20, 25] = 2
        protected = self.t.protection_mask(labels)
        self.assertEqual(int(protected.sum()), 17 * 17)
        sigma = torch.full((40, 50), 2.)
        for step, expected in [(1999, 1.), (2000, 1.), (2250, .6), (2500, .2)]:
            w = self.t.observation_weights(sigma, protected, step)
            self.assertTrue(torch.all(w[protected] == 1))
            self.assertAlmostEqual(float(w[0, 0]), expected, places=6)

    def test_ssim_supervision_of_structures_is_preserved(self):
        from utils.loss_utils import ssim
        x = torch.rand(3, 41, 41, requires_grad=True)
        y = torch.rand_like(x)
        labels = torch.zeros(41, 41, dtype=torch.uint8)
        labels[20, 20] = 1
        w = self.t.observation_weights(torch.full((41, 41), 2.), self.t.protection_mask(labels), 2500)
        original, = torch.autograd.grad(1 - ssim(x, y), x)
        _, loss = self.t.weighted_losses(x, y, w)
        masked, = torch.autograd.grad(loss, x)
        torch.testing.assert_close(original[:, 20, 20], masked[:, 20, 20], rtol=2e-5, atol=1e-8)
        self.assertFalse(torch.allclose(original[:, 0, 0], masked[:, 0, 0]))

    def test_uncertainty_cannot_backpropagate_to_rendered_features(self):
        real = torch.randn(1, 4, 3, 3, requires_grad=True)
        rendered = torch.randn_like(real, requires_grad=True)
        sigma = torch.ones(1, 1, 3, 3, requires_grad=True)
        distance = self.t.feature_distance(real, rendered)
        self.t.uncertainty_loss(sigma, distance, torch.ones_like(sigma)).backward()
        self.assertIsNone(real.grad)
        self.assertIsNone(rendered.grad)
        self.assertIsNotNone(sigma.grad)

    def test_invalid_sigma_and_labels(self):
        with self.assertRaises(ValueError):
            self.t.protection_mask(torch.full((3, 3), 5))
        with self.assertRaises(ValueError):
            self.t.observation_weights(torch.zeros(3, 3), torch.zeros(3, 3, dtype=torch.bool), 3000)

    def test_unlabeled_bypasses_feature_model_and_weighting(self):
        from utils.loss_utils import l1_loss, ssim
        controller = self.t.TransientController.__new__(self.t.TransientController)
        controller.allowed = {"301.png"}
        controller.labeled = set()
        controller.spec = {"head_start": 500}
        x, y = torch.rand(3, 17, 19), torch.rand(3, 17, 19)
        a, b = controller.losses(x, y, "301", 30000)
        self.assertTrue(torch.equal(a, l1_loss(x, y)))
        self.assertTrue(torch.equal(b, 1 - ssim(x, y)))
        self.assertIsNone(controller.pending)
        with self.assertRaises(ValueError):
            controller.losses(x, y, "054", 30000)

    def test_sigma_threshold(self):
        w = self.t.observation_weights(torch.tensor([[.1, .7, .71, 10.]]), torch.zeros(1, 4, dtype=torch.bool), 30000)
        torch.testing.assert_close(w, torch.tensor([[1., 1., .2, .2]]))

    def test_rng_isolation_including_exception(self):
        import random
        random.seed(2)
        np.random.seed(2)
        torch.manual_seed(2)
        states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        with self.assertRaises(RuntimeError):
            with self.t.isolated_rng():
                random.random()
                np.random.rand()
                torch.rand(5)
                raise RuntimeError("test")
        self.assertEqual(random.getstate(), states[0])
        np.testing.assert_array_equal(np.random.get_state()[1], states[1][1])
        self.assertTrue(torch.equal(torch.get_rng_state(), states[2]))


if __name__ == "__main__":
    unittest.main()
