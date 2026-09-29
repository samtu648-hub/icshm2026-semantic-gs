import unittest
try:
    import numpy as np
    from src.semantic_second_source_veto import second_source, review_second_source
    from src.semantic_dense_geometry import normalized_grid
    from src.geometry_prior.camera_io import ColmapCamera, ColmapImage
except ImportError:
    np = None


@unittest.skipIf(np is None, "requires optional native geometry runtime")
class SecondSourceTests(unittest.TestCase):
    def setUp(self):
        self.camera = ColmapCamera(1, "SIMPLE_RADIAL", 32, 24, (30., 16., 12., 0.))
        self.a = ColmapImage(1, "new-target", 1, np.array([1., 0, 0, 0]), np.zeros(3), ())
        self.b = ColmapImage(2, "source-bank", 1, self.a.qvec, np.array([1., 0, 0]), ())
        grid = normalized_grid(8, 8)
        self.maps = [(grid, np.ones((8, 8))), (grid, np.ones((8, 8)))]
        self.filters = {"overlap_each_direction_minimum": .5, "cycle_maximum_native_pixels": 2., "epipolar_maximum_native_pixels": 2.}
        self.base = np.full((24, 32), 4, np.uint8); self.dense = np.zeros_like(self.base)

    def run_review(self, labels, **kwargs):
        return review_second_source(self.base, self.dense, labels, self.maps, self.a, self.b,
                                    self.camera, self.camera, self.filters, **kwargs)

    def test_order_is_first_following_pass_not_label_selection(self):
        self.assertEqual(second_source({"accessible_sources": ["z", "c", "a"], "selected_source": "z"}), "c")
        self.assertIsNone(second_source({"accessible_sources": ["z"], "selected_source": "z"}))
        self.assertIsNone(second_source({"accessible_sources": [], "selected_source": None}))
        with self.assertRaises(ValueError): second_source({"accessible_sources": ["a", "b"], "selected_source": "b"})

    def test_no_second_source_and_disabled_keep_dense(self):
        result, _ = review_second_source(self.base, self.dense)
        np.testing.assert_array_equal(result, self.dense)
        disabled, stats = self.run_review(np.full_like(self.base, 4), enabled=False)
        np.testing.assert_array_equal(disabled, self.dense); self.assertEqual(stats["foundation_veto_pixels"], 0)

    def test_only_foundation_label_can_veto_with_known_support(self):
        for k in (0, 1, 2, 3):
            result, stats = self.run_review(np.full_like(self.base, k))
            np.testing.assert_array_equal(result, self.dense); self.assertEqual(stats["foundation_veto_pixels"], 0)
        result, stats = self.run_review(np.full_like(self.base, 4))
        self.assertEqual(stats["foundation_veto_pixels"], 28 * 22)
        self.assertTrue((result[1:-1, 2:-2] == 4).all())
        self.assertTrue((result[0] == 0).all())

    def test_unreliable_evidence_retains_dense(self):
        self.maps[0] = (self.maps[0][0], np.full((8, 8), .49))
        result, stats = self.run_review(np.full_like(self.base, 4))
        np.testing.assert_array_equal(result, self.dense); self.assertEqual(stats["reliable_old_deletions"], 0)

    def test_independent_distorted_and_protected_masks(self):
        try: import torch
        except ImportError: self.skipTest("optional torch unavailable")
        self.camera = ColmapCamera(1, "SIMPLE_RADIAL", 32, 24, (30., 16., 12., .01))
        self.base = (np.arange(24 * 32).reshape(24, 32) % 5).astype(np.uint8)
        self.dense = np.where(self.base == 4, 0, self.base).astype(np.uint8)
        labels = np.full_like(self.base, 4)
        primary, a = self.run_review(labels, target_is_distorted=True)
        independent, b = self.run_review(labels, independent=True, target_is_distorted=True)
        np.testing.assert_array_equal(primary, independent); self.assertEqual(a, b)
        self.assertGreater(a["foundation_veto_pixels"], 0)
        for k in (1, 2, 3): np.testing.assert_array_equal(primary == k, self.base == k)

    def test_invalid_transition_rejected(self):
        with self.assertRaises(ValueError): review_second_source(np.array([[1]]), np.array([[0]]))


if __name__ == "__main__": unittest.main()
