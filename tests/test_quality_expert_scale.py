import unittest
import numpy as np
try:
    import torch
    from src.quality_expert_scale import resize_pair
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'Optional Torch runtime absent')
class ExpertScaleTests(unittest.TestCase):
    def test_identity_preserves_native_pixels(self):
        rgb = np.arange(768*768*3, dtype=np.uint8).reshape(768, 768, 3)
        y = (rgb[:, :, 0] % 5).copy()
        x, target = resize_pair(rgb, y, 768)
        np.testing.assert_array_equal(x.numpy(), rgb.transpose(2, 0, 1))
        np.testing.assert_array_equal(target, y)

    def test_nearest_exact_pixel_center_mapping(self):
        y = (np.indices((768, 768))[1] % 5).astype(np.uint8)
        rgb = np.repeat(y[:, :, None], 3, axis=2)
        x, target = resize_pair(rgb, y)
        index = np.floor((np.arange(864)+.5)*768/864).astype(int)
        np.testing.assert_array_equal(target, y[index[:, None], index[None, :]])
        self.assertEqual(tuple(x.shape), (3, 864, 864))
        self.assertEqual(set(np.unique(target)), set(np.unique(y)))

    def test_reject_scale_search(self):
        with self.assertRaises(ValueError):
            resize_pair(np.zeros((768, 768, 3), np.uint8), np.zeros((768, 768), np.uint8), 960)
