import unittest
try:
    import numpy as np
    from src.semantic_dense_review import review_foundation, native_prediction, native_support, quality_gates
    from src.semantic_dense_geometry import normalized_grid
    from src.geometry_prior.camera_io import ColmapCamera, ColmapImage
except ImportError:
    np = None


@unittest.skipIf(np is None, "optional native geometry runtime unavailable")
class NativeReviewTests(unittest.TestCase):
    def setUp(self):
        self.camera = ColmapCamera(1, "SIMPLE_RADIAL", 32, 24, (30., 16., 12., 0.))
        self.a = ColmapImage(1, "a", 1, np.array([1., 0, 0, 0]), np.zeros(3), ())
        self.b = ColmapImage(2, "b", 1, np.array([1., 0, 0, 0]), np.array([1., 0, 0]), ())
        self.grid = normalized_grid(8, 8)
        self.maps = [(self.grid, np.ones((8, 8))), (self.grid, np.ones((8, 8)))]
        self.filters = {"overlap_each_direction_minimum": .5, "cycle_maximum_native_pixels": 2., "epipolar_maximum_native_pixels": 2.}

    def test_exact_protection_and_abstention(self):
        old = np.array([[0, 1, 2, 3, 4, 4]], np.uint8)
        labels = np.array([[0, 0, 0, 0, 0, 1]], np.uint8)
        np.testing.assert_array_equal(review_foundation(old, labels, np.zeros_like(old, bool)), old)
        np.testing.assert_array_equal(review_foundation(old, labels, np.ones_like(old, bool)), [[0, 1, 2, 3, 0, 4]])

    def test_native_identity_and_strict_field_boundary(self):
        old = np.full((24, 32), 4, np.uint8)
        result, stats = native_prediction(old, np.zeros_like(old), self.maps, self.a, self.b, self.camera, self.camera, self.filters)
        self.assertEqual(stats["changed_pixels"], 28 * 22)
        self.assertTrue((result[1:-1, 2:-2] == 0).all())
        self.assertTrue((result[:, :2] == 4).all())
        self.assertTrue((result[0] == 4).all())

    def test_overlap_and_cycle_rejection(self):
        p = np.array([[[16.5, 12.5]]])
        low = [(self.grid, np.full((8, 8), .49)), self.maps[1]]
        _, keep = native_support(p, low, self.a, self.b, self.camera, self.camera, self.filters)
        self.assertFalse(keep.any())
        bad = [self.maps[0], (self.grid + [.3, 0], np.ones((8, 8)))]
        _, keep = native_support(p, bad, self.a, self.b, self.camera, self.camera, self.filters)
        self.assertFalse(keep.any())

    def test_cycle_consistent_wrong_epipolar_match_rejected(self):
        points = np.array([[[16.5, 12.5]]])
        maps = [(self.grid + [0, .3], np.ones((8, 8))),
                (self.grid - [0, .3], np.ones((8, 8)))]
        _, keep = native_support(points, maps, self.a, self.b, self.camera, self.camera, self.filters)
        self.assertFalse(keep.any())

    def test_official_world_translation_and_chunking_invariant(self):
        old = np.full((24, 32), 4, np.uint8)
        source = np.zeros_like(old)
        reference, counts = native_prediction(old, source, self.maps, self.a, self.b,
                                               self.camera, self.camera, self.filters)
        offset = np.array([12., -7., 30.])
        a = ColmapImage(90, "novel-a", 1, self.a.qvec, self.a.tvec - offset, ())
        b = ColmapImage(91, "novel-b", 1, self.b.qvec, self.b.tvec - offset, ())
        result, actual_counts = native_prediction(old, source, self.maps, a, b,
                                                  self.camera, self.camera, self.filters, rows_per_chunk=3)
        np.testing.assert_array_equal(reference, result)
        self.assertEqual(counts, actual_counts)

    def test_primary_independent_radial_and_pose(self):
        try: import torch
        except ImportError: self.skipTest("optional torch unavailable")
        camera = ColmapCamera(1, "SIMPLE_RADIAL", 32, 24, (30., 16., 12., .01))
        y, x = np.mgrid[2:22, 3:29]
        points = np.stack((x + .5, y + .5), -1)
        a, va = native_support(points, self.maps, self.a, self.b, camera, camera, self.filters)
        b, vb = native_support(points, self.maps, self.a, self.b, camera, camera, self.filters, True)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-8)
        np.testing.assert_array_equal(va, vb)

    def test_native_shape_rejected(self):
        with self.assertRaises(ValueError):
            native_prediction(np.zeros((8, 8), np.uint8), np.zeros((24, 32), np.uint8), self.maps,
                              self.a, self.b, self.camera, self.camera, self.filters)

    def test_fixed_quality_gates_do_not_ignore_recall(self):
        gates = {"minimum_miou_delta_pp": .3, "minimum_foundation_iou_delta_pp": 1.5,
                 "maximum_foundation_recall_loss_pp": 2., "minimum_foundation_recall": .9,
                 "minimum_improving_foundation_folds": 2, "minimum_fold_foundation_iou_delta_pp": -1.,
                 "minimum_fold_miou_delta_pp": -.3}
        summary = {"delta_miou_pp": .4, "delta_foundation_iou_pp": 2., "delta_foundation_recall_pp": -1.9,
                   "candidate": {"per_class": {"foundation": {"recall": .93}}}}
        folds = [{"delta_foundation_iou_pp": 2., "delta_miou_pp": .4}] * 3
        self.assertTrue(all(quality_gates(summary, folds, gates).values()))
        summary["delta_foundation_recall_pp"] = -2.01
        self.assertFalse(quality_gates(summary, folds, gates)["foundation_recall_loss"])

    def test_check_inference_and_scoring_functions_unchanged(self):
        import ast
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        def functions(name):
            tree = ast.parse((root / "scripts" / name).read_text())
            return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
        a = functions("run_sem325_native_review.py")
        b = functions("run_sem325_check_review.py")
        for name in ("predict", "score", "metric_summary", "verify_prediction_seal"):
            self.assertEqual(a[name], b[name])


if __name__ == "__main__": unittest.main()
