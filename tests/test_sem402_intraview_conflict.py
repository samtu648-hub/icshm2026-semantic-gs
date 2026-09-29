import unittest

import numpy as np

from src.semantic_intraview_conflict import mask_signal, select_conflicted_parents


class IntraviewConflictTest(unittest.TestCase):
    def test_mask_signal_distinguishes_balanced_and_one_sided_mass(self):
        actual = mask_signal(np.array([2.0, 4.0, 0.0]), np.array([2.0, 0.0, 3.0]))
        np.testing.assert_allclose(actual, [0.0, 1.0, 1.0], atol=1e-8)

    def test_selection_requires_coverage_and_ranks_low_signal_stably(self):
        selected = select_conflicted_parents(
            signal_sum=np.array([0.6, 0.6, 0.4, 0.4]),
            visible_views=np.array([3, 3, 2, 4]),
            ambiguous_views=np.array([3, 3, 2, 2]),
            eligible=np.array([True, True, True, True]),
            signal_threshold=0.5,
            minimum_visible_views=3,
            minimum_ambiguous_views=3,
        )
        np.testing.assert_array_equal(selected.qualified, [True, True, False, False])
        np.testing.assert_array_equal(selected.ranked_eligible, [3, 0, 1, 2])

    def test_rejects_negative_contribution(self):
        with self.assertRaises(ValueError):
            mask_signal(np.array([-1.0]), np.array([1.0]))


if __name__ == "__main__":
    unittest.main()
