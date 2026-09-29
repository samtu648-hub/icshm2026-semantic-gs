"""Optional real Mask2Former criterion tests, without model training or CUDA use."""

import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(AVAILABLE, "Optional native Torch runtime unavailable")
class NativeSamplingTests(unittest.TestCase):
    def test_disabled_loss_gradient_rng_and_enabled_negative_budget(self):
        import torch

        runtime = ROOT / "outputs/quality/mask2former/SEM-340"
        sys.path[:0] = [
            str(runtime / "runtime"),
            str(runtime / "vendor/detectron2"),
            str(runtime / "vendor/Mask2Former"),
            str(runtime / "vendor/Mask2Former/mask2former/modeling/pixel_decoder/ops"),
        ]
        from mask2former.modeling.criterion import SetCriterion
        from src.component_point_sampling import ComponentSampler

        torch.set_num_threads(2)
        criterion = SetCriterion(5, None, {}, 0.1, ["masks"], 12544, 3.0, 0.75)
        label = np.zeros((768, 768), np.uint8)
        label[50:55, 50:55] = 2
        label[100:102, 300:305] = 2
        target = {
            "labels": torch.tensor([0, 2]),
            "masks": torch.from_numpy(np.stack((label == 0, label == 2))),
        }
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        torch.manual_seed(19)
        logits = torch.randn(1, 2, 192, 192, requires_grad=True)
        state = torch.get_rng_state().clone()
        old = criterion.loss_masks({"pred_masks": logits}, [target], indices, 2)
        sum(old.values()).backward()
        gradient = logits.grad.clone()
        rng_after = torch.get_rng_state().clone()
        logits.grad = None
        sampler = ComponentSampler(enabled=False)
        sampler.set_sample(
            {"name": "fixture.png", "top": 0, "left": 0, "flip": False}, label
        )
        sampler.install()
        try:
            torch.set_rng_state(state)
            disabled = criterion.loss_masks(
                {"pred_masks": logits}, [target], indices, 2
            )
            sum(disabled.values()).backward()
            self.assertTrue(all(torch.equal(old[k], disabled[k]) for k in old))
            self.assertTrue(torch.equal(logits.grad, gradient))
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_after))
            logits.grad = None
            sampler.enabled = True
            torch.set_rng_state(state)
            candidate = criterion.loss_masks(
                {"pred_masks": logits}, [target], indices, 2
            )
            sum(candidate.values()).backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_after))
            self.assertEqual(sampler.stats["replaced_points"], 26)
        finally:
            sampler.restore()


if __name__ == "__main__":
    unittest.main()
