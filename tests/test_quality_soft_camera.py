import unittest
import numpy as np
from src.camera_adapter import SimpleRadialCamera, _redistortion_map
from src.quality_soft_camera import expert_soft, independent_linear


class SoftCameraTest(unittest.TestCase):
    def test_identity_and_log_roundtrip(self):
        p = np.random.default_rng(8).random((5, 7, 9), dtype=np.float32)
        p /= p.sum(0)
        c = SimpleRadialCamera(9, 7, 6, 4.5, 3.5, 0)
        soft, logs = expert_soft(p, c)
        np.testing.assert_allclose(soft, p, atol=2e-7, rtol=0)
        np.testing.assert_array_equal(logs.argmax(0), soft.argmax(0))
        np.testing.assert_allclose(np.exp(logs), soft, atol=2e-7, rtol=0)

    def test_independent_radial_interpolation(self):
        p = np.random.default_rng(9).random((5, 71, 93), dtype=np.float32)
        p /= p.sum(0)
        c = SimpleRadialCamera(93, 71, 65, 46.5, 35.5, .00898786)
        soft, _ = expert_soft(p, c)
        np.testing.assert_allclose(independent_linear(p, _redistortion_map(c)), soft, atol=1e-6, rtol=0)

    def test_reject_invalid_probability(self):
        with self.assertRaises(ValueError):
            expert_soft(np.zeros((5, 7, 9), np.float32), SimpleRadialCamera(9, 7, 6, 4.5, 3.5, 0))
