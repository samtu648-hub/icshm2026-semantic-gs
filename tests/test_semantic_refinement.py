import unittest
import numpy as np
from src.semantic_refinement import mutable_lane, fuse_residual, class_boundary_distance


class RefinementContracts(unittest.TestCase):
    def test_zero_and_bidirectional_and_protection(self):
        base = np.array([[0, 1, 2, 3, 4, 0]], np.uint8)
        static = np.array([[0, 1, 0, 3, 4, 4]], np.uint8)
        np.testing.assert_array_equal(fuse_residual(base, static, np.zeros_like(base)), base)
        residual = np.array([[3, -3, 99, 99, 99, 99]], np.float32)
        np.testing.assert_array_equal(fuse_residual(base, static, residual), [[1, 0, 2, 3, 4, 0]])
        np.testing.assert_array_equal(mutable_lane(base, static), [[True, True, False, False, False, False]])

    def test_neighbours_are_not_protected(self):
        base = np.array([[0, 2, 1]], np.uint8)
        self.assertTrue(mutable_lane(base, base)[0, 0])
        self.assertTrue(mutable_lane(base, base)[0, 2])

    def test_bad_residual(self):
        a = np.zeros((2, 2), np.uint8)
        with self.assertRaises(ValueError):
            fuse_residual(a, a, np.full((2, 2), np.nan))

    def test_absent_class_distance(self):
        self.assertTrue(np.isinf(class_boundary_distance(np.zeros((5, 5), np.uint8), 1)).all())

    def test_zero_head_and_matched_control(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable')
        from src.semantic_refinement import point_head, mask_fine_features
        head = point_head()
        x = torch.randn(16, 75)
        self.assertTrue(torch.equal(head(x), torch.zeros(16, 1)))
        control = mask_fine_features(x, 'coarse_control')
        self.assertTrue(torch.equal(control[:, :67], torch.zeros_like(control[:, :67])))
        self.assertTrue(torch.equal(control[:, 67:], x[:, 67:]))
        self.assertFalse(torch.equal(x[:, :67], control[:, :67]))


if __name__ == '__main__':
    unittest.main()
