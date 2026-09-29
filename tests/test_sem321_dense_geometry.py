import unittest
try:
    import numpy as np
    from src.semantic_dense_geometry import (normalized_grid, bilinear_sample, cycle_support,
                                             fundamental_from_w2c, symmetric_epipolar_pixels, screen_pair)
except ImportError:
    np = None


@unittest.skipIf(np is None, "optional numpy unavailable")
class DenseGeometryTests(unittest.TestCase):
    def test_identity_sampler_and_native_cycle(self):
        grid = normalized_grid(8, 8)
        sampled, valid = bilinear_sample(grid, grid)
        np.testing.assert_allclose(sampled, grid, atol=1e-14)
        self.assertTrue(valid.all())
        support = cycle_support(grid, np.ones((8, 8)), grid, np.ones((8, 8)), (1320, 989), .5, 2)
        self.assertTrue(support["cycle_kept"].all())
        shifted = grid.copy(); shifted[..., 0] += 6 / 1320
        support = cycle_support(grid, np.ones((8, 8)), shifted, np.ones((8, 8)), (1320, 989), .5, 2)
        np.testing.assert_allclose(support["cycle"], 3, atol=1e-10)
        self.assertFalse(support["cycle_kept"].any())

    def test_out_of_bounds_not_accepted(self):
        grid = normalized_grid(4, 4); grid[0, 0] = [-2, 2]
        sampled, valid = bilinear_sample(np.ones((4, 4)), grid)
        self.assertFalse(valid[0, 0]); self.assertTrue(np.isnan(sampled[0, 0]))

    def test_official_w2c_epipolar_geometry(self):
        k = np.array([[900, 0, 660], [0, 900, 494.5], [0, 0, 1]], float)
        ra = np.eye(3); rb = np.array([[.8, 0, .6], [0, 1, 0], [-.6, 0, .8]])
        ta = np.array([.1, -.2, 0]); tb = np.array([-1, .3, .2])
        xyz = np.array([[0, 0, 10], [1, .5, 12], [-.5, 2, 15]], float)
        def project(r, t):
            points = (xyz @ r.T + t) @ k.T
            return points[:, :2] / points[:, 2:]
        a, b = project(ra, ta), project(rb, tb)
        f = fundamental_from_w2c(ra, ta, rb, tb, k, k)
        np.testing.assert_allclose(symmetric_epipolar_pixels(a, b, f), 0, atol=1e-10)
        self.assertTrue((symmetric_epipolar_pixels(a, b + [0, 20], f) > 2).all())

    def test_empty_evidence_fails_screen(self):
        gates = {"minimum_reliable_grid_fraction_each_pair": .01, "minimum_epipolar_inlier_fraction_after_cycle_each_pair": .8}
        self.assertFalse(any(screen_pair(0, None, None, None, gates).values()))


if __name__ == "__main__": unittest.main()
