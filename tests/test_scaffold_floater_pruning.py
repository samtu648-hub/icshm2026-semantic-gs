import unittest

import numpy as np

from src.scaffold_floater_pruning import residual_support_prune_mask


class ScaffoldFloaterPruningTests(unittest.TestCase):
    def test_registered_rule_is_conjunctive_and_boundary_inclusive(self) -> None:
        arrays = {
            "dominant_pixels": np.array([256, 255, 300, 300, 300]),
            "dominant_views": np.array([4, 1, 5, 1, 1]),
            "mean_RGB_MAE": np.array([0.15, 0.50, 0.50, 0.149, 0.50]),
            "high_error_pixel_fraction": np.array([0.50, 1.0, 1.0, 1.0, 0.49]),
        }
        result = residual_support_prune_mask(
            arrays,
            minimum_dominant_pixels=256,
            maximum_dominant_views=4,
            minimum_mean_rgb_mae=0.15,
            minimum_high_error_fraction=0.50,
        )
        self.assertTrue(np.array_equal(result, [True, False, False, False, False]))

    def test_missing_array_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            residual_support_prune_mask(
                {},
                minimum_dominant_pixels=256,
                maximum_dominant_views=4,
                minimum_mean_rgb_mae=0.15,
                minimum_high_error_fraction=0.50,
            )


if __name__ == "__main__":
    unittest.main()

