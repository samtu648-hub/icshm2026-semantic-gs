import unittest
import random
import numpy as np
from src.component_point_sampling import (
    full_small_components,
    crop_regions,
    quotas_for_regions,
    component_hits,
)
from src.gaussian_training_dropout import GaussianDropout


class ComponentQuotaTest(unittest.TestCase):
    def test_full_large_component_cannot_become_small_by_cropping(self):
        labels = np.zeros((100, 100), np.uint8)
        labels[:50, :50] = 1
        labels[80:82, 80:82] = 2
        components = full_small_components(labels)
        self.assertEqual([c["class_id"] for c in components], [2])

    def test_quota_is_unique_capped_and_inside_regions(self):
        labels = np.zeros((768, 768), np.uint8)
        labels[10:20, 10:20] = 2
        labels[100:110, 100:110] = 2
        regions = crop_regions(
            full_small_components(labels), {"top": 0, "left": 0, "flip": True}
        )
        q = quotas_for_regions(regions, 418)[2]
        self.assertEqual(len(q), 32)
        self.assertEqual(len(np.unique(q, axis=0)), 32)
        pixels = np.floor(q * 768).astype(int)
        self.assertTrue((labels[:, ::-1][pixels[:, 1], pixels[:, 0]] == 2).all())
        limited = quotas_for_regions(regions, 418, point_count=100, fraction=0.05)[2]
        self.assertEqual(len(limited), 5)

    def test_private_quota_rng_does_not_change_global_stream(self):
        labels = np.zeros((768, 768), np.uint8)
        labels[10:15, 10:15] = 4
        regions = crop_regions(
            full_small_components(labels), {"top": 0, "left": 0, "flip": False}
        )
        random.seed(8)
        np.random.seed(8)
        py = random.getstate()
        state = np.random.get_state()
        a = quotas_for_regions(regions, 418)
        b = quotas_for_regions(regions, 418)
        self.assertTrue(np.array_equal(a[4], b[4]))
        self.assertEqual(py, random.getstate())
        self.assertTrue(np.array_equal(state[1], np.random.get_state()[1]))

    def test_bilinear_hit_distinguishes_tiny_component(self):
        region = {"yx": np.array([[10, 20]])}
        coords = np.array([[20.5 / 768, 10.5 / 768], [100.5 / 768, 100.5 / 768]])
        hits, mass = component_hits(coords, region)
        self.assertEqual(hits, 1)
        self.assertAlmostEqual(mass, 1)

    def test_dropout_schedule_rejects_invalid_probability(self):
        with self.assertRaises(ValueError):
            GaussianDropout(1.0)
        self.assertEqual(GaussianDropout(0).maximum, 0)


if __name__ == "__main__":
    unittest.main()
