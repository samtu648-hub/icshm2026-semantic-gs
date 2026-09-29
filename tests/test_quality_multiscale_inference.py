import unittest

import numpy as np

from src.quality_multiscale_inference import (
    bridge_strip_bounds,
    fixed_ensemble,
    normalize_probability,
    resize_probability,
    scaled_shape,
)


class MultiscaleInferenceTest(unittest.TestCase):
    def test_registered_shapes(self):
        self.assertEqual(scaled_shape(989, 1320, 1.5), (1484, 1980))
        self.assertEqual(scaled_shape(989, 1320, 2.0), (1978, 2640))

    def test_strip_uses_lower_bridge_support_and_clamps(self):
        prediction = np.zeros((989, 1320), np.uint8)
        prediction[300:401, 10:20] = 2
        bounds = bridge_strip_bounds(prediction, 384, (1, 2), 85, 494)
        self.assertFalse(bounds["used_fallback"])
        self.assertEqual(bounds["anchor_y"], 385)
        self.assertEqual(bounds["bottom"] - bounds["top"], 384)
        prediction.fill(0)
        fallback = bridge_strip_bounds(prediction, 384, (1, 2), 85, 494)
        self.assertTrue(fallback["used_fallback"])
        self.assertEqual(fallback["anchor_y"], 494)

    def test_resize_and_fixed_fusion_preserve_simplex(self):
        base = np.zeros((5, 4, 6), np.float32)
        base[0] = 0.75
        base[1] = 0.25
        other = base.copy()
        other[0], other[1] = 0.25, 0.75
        resized = resize_probability(base, 7, 9)
        self.assertEqual(resized.shape, (5, 7, 9))
        self.assertTrue(np.allclose(resized.sum(0), 1))
        fused = fixed_ensemble({"native": base, "other": other}, {"native": 0.5, "other": 0.5})
        self.assertTrue(np.allclose(fused[0], 0.5))
        self.assertTrue(np.allclose(fused[1], 0.5))
        self.assertTrue(np.allclose(fused.sum(0), 1))
        with self.assertRaises(ValueError):
            normalize_probability(np.zeros((5, 2, 2), np.float32))


if __name__ == "__main__":
    unittest.main()
