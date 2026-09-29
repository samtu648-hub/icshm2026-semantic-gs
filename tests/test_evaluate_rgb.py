from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import numpy as np

    from evaluate_rgb import psnr, ssim, validation_view_names  # noqa: E402
except ModuleNotFoundError:
    HAS_METRIC_DEPS = False
else:
    HAS_METRIC_DEPS = True


@unittest.skipUnless(HAS_METRIC_DEPS, "NumPy and OpenCV are optional before R1 environment setup")
class EvaluateRgbTest(unittest.TestCase):
    def test_identical_images_have_infinite_psnr_and_unit_ssim(self) -> None:
        image = np.full((16, 16, 3), 0.25, dtype=np.float32)
        self.assertTrue(math.isinf(psnr(image, image)))
        self.assertAlmostEqual(ssim(image, image), 1.0, places=6)

    def test_maximum_difference_has_zero_db_psnr(self) -> None:
        reference = np.zeros((16, 16, 3), dtype=np.float32)
        prediction = np.ones((16, 16, 3), dtype=np.float32)
        self.assertAlmostEqual(psnr(reference, prediction), 0.0, places=6)
        self.assertLess(ssim(reference, prediction), 0.01)

    def test_reads_frozen_validation_view_list(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            split = Path(temp) / "split.json"
            split.write_text(
                json.dumps({"primary_fold": {"validation_labeled": ["001.png", "002.png"]}}),
                encoding="utf-8",
            )
            self.assertEqual(validation_view_names(split), ["001.png", "002.png"])


if __name__ == "__main__":
    unittest.main()
