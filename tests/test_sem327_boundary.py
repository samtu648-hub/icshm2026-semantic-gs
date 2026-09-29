import unittest
import numpy as np
try:
    from src.semantic_source_boundary import foundation_distances, sample_boundary_distance, boundary_histogram, protect_boundary
except ImportError:
    foundation_distances = None


@unittest.skipIf(foundation_distances is None, "requires optional scipy runtime")
class SourceBoundaryTests(unittest.TestCase):
    def test_known_euclidean_distances_and_independent_tree(self):
        labels = np.zeros((8, 9), np.uint8); labels[2, 3] = 4
        field = foundation_distances(labels)
        self.assertEqual(field[5, 7], 5)
        points = np.array([[[7.8, 5.2], [3.5, 2.5], [2.9, 2.5]]])
        a = sample_boundary_distance(points, labels, field)
        b = sample_boundary_distance(points, labels, field, True)
        for x, y in zip(a, b): np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(a[1], [[5, 0, 1]])

    def test_empty_foundation_and_invalid_coordinates_abstain(self):
        labels = np.zeros((2, 3), np.uint8); field = foundation_distances(labels)
        points = np.array([[[0.5, 0.5], [-.01, 1], [3., 1], [np.nan, 1]]])
        for independent in (False, True):
            _, d, valid = sample_boundary_distance(points, labels, field, independent)
            self.assertTrue(np.all(np.isinf(d)))
            np.testing.assert_array_equal(valid, [[True, False, False, False]])

    def test_bin_boundaries_and_overflow(self):
        h = boundary_histogram(np.array([1., 2., 2.01, 4., np.inf]), np.array([0, 4, 4, 0, 4]), [1, 2, 4])
        np.testing.assert_array_equal(h[:, 0], [1, 0, 1, 0])
        np.testing.assert_array_equal(h[:, 4], [0, 1, 1, 1])

    def test_fixed_boundary_protection_keeps_all_other_pixels(self):
        base = np.array([[4, 4, 4, 1, 2, 3, 0]], np.uint8)
        dense = np.array([[0, 0, 0, 1, 2, 3, 0]], np.uint8)
        result, restored = protect_boundary(base, dense, np.array([1., 2., 2.01]), 2)
        np.testing.assert_array_equal(result, [[4, 4, 0, 1, 2, 3, 0]])
        self.assertEqual(restored.sum(), 2)
        disabled, _ = protect_boundary(base, dense, np.array([1., 2., np.inf]), 0)
        np.testing.assert_array_equal(disabled, dense)

    def test_illegal_transition_and_negative_radius_rejected(self):
        with self.assertRaises(ValueError): protect_boundary(np.array([[1]]), np.array([[0]]), np.array([1.]), 2)
        with self.assertRaises(ValueError): protect_boundary(np.array([[4]]), np.array([[0]]), np.array([1.]), -1)


if __name__ == "__main__": unittest.main()
