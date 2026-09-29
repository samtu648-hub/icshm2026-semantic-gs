import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.standard_rgb_supervision import (
        AuxLossController,
        load_structure_mask,
        masked_rgb_l1,
        multiscale_gradient_l1,
    )


@unittest.skipUnless(
    torch is not None, "torch is an optional GPU experiment dependency"
)
class StandardRgbSupervisionTest(unittest.TestCase):
    def test_mask_dilation_and_area_normalized_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            label = np.zeros((989, 1320), dtype=np.uint8)
            label[20, 30] = 2
            path = Path(directory) / "label.png"
            Image.fromarray(label).save(path)
            mask = load_structure_mask(path, dilation_px=2)
            self.assertEqual(int(mask.sum()), 25)
            photo = torch.zeros(3, 989, 1320)
            rendered = torch.zeros_like(photo)
            rendered[:, mask[0].bool()] = 0.25
            self.assertAlmostEqual(
                float(masked_rgb_l1(rendered, photo, mask)), 0.25, places=6
            )

    def test_gradient_is_computed_before_region_weighting(self):
        mask = torch.zeros(1, 16, 16)
        mask[:, :, 8:] = 1
        rendered = torch.full((3, 16, 16), 0.4, requires_grad=True)
        photo = torch.full((3, 16, 16), 0.6)
        loss = multiscale_gradient_l1(rendered, photo, mask, scales=(1, 4))
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(rendered.grad).all())

    def test_gradient_matches_full_and_coarse_scales(self):
        photo = torch.rand(3, 17, 19)
        mask = torch.ones(1, 17, 19)
        self.assertAlmostEqual(
            float(multiscale_gradient_l1(photo, photo, mask, (1, 4))), 0.0, places=7
        )
        shifted = photo.clone()
        shifted[:, :, 9:] += 0.1
        self.assertGreater(
            float(multiscale_gradient_l1(shifted, photo, mask, (1, 4))), 0.0
        )

    def test_control_is_identity_and_attempt_summary_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = AuxLossController(
                {
                    "auxiliary_supervision": {
                        "mode": "control",
                        "coefficient": 0,
                        "region": {"dilation_px": 4},
                        "gradient": {"scales": [1, 4]},
                        "label_root": directory,
                    }
                },
                Path(directory),
                "resume-test",
            )
            loss = torch.tensor(1.0, requires_grad=True)
            self.assertIs(controller.apply(1, None, None, None, loss, None), loss)
            controller.finalize()
            self.assertTrue(
                (Path(directory) / "aux_summary-resume-test.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
