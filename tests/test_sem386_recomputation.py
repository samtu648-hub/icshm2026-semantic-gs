"""Numerical CPU check of the pinned native Swin checkpoint branch."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest


try:
    import torch
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Optional Torch environment absent")
class SwinRecomputationTest(unittest.TestCase):
    def test_shifted_windows_dropout_forward_gradients_and_rng(self):
        root = Path(__file__).resolve().parents[1]
        vendor = root / "outputs/quality/mask2former/SEM-340/vendor"
        source = vendor / "Mask2Former/mask2former/modeling/backbone/swin.py"
        if not source.exists():
            self.skipTest("Pinned upstream Swin runtime is not installed")
        previous_path = list(sys.path)
        previous_threads = torch.get_num_threads()
        try:
            sys.path[:0] = [str(vendor.parent / "runtime"), str(vendor / "detectron2")]
            torch.set_num_threads(2)
            spec = importlib.util.spec_from_file_location("sem386_test_swin", source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            torch.manual_seed(41)
            ordinary = module.BasicLayer(dim=16, depth=2, num_heads=2, window_size=4,
                                         drop=.1, attn_drop=.1, drop_path=.2)
            recomputed = copy.deepcopy(ordinary)
            recomputed.use_checkpoint = True
            sample = torch.randn(1, 64, 16)
            states, outputs, gradients = [], [], []
            for model in [ordinary, recomputed]:
                torch.manual_seed(67)
                x = sample.clone().requires_grad_(True)
                y = model(x, 8, 8)[0]
                y.square().mean().backward()
                outputs.append(y.detach())
                gradients.append([x.grad, *[p.grad for p in model.parameters()]])
                states.append(torch.get_rng_state())
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
            self.assertEqual(len(gradients[0]), len(gradients[1]))
            for normal, checkpointed in zip(*gradients):
                self.assertIsNotNone(normal)
                self.assertIsNotNone(checkpointed)
                torch.testing.assert_close(normal, checkpointed, rtol=1e-6, atol=1e-7)
        finally:
            sys.path[:] = previous_path
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
