import unittest

import numpy as np

from scripts.prepare_rgb139_reliable_initialization import allocate, percentile_score


class ReliableInitializationTests(unittest.TestCase):
    def test_largest_remainder_allocation_is_exact_and_proportional(self):
        strata = {"low_angle": list(range(7)), "high_angle": list(range(7, 10))}
        counts = allocate(strata, 8)
        self.assertEqual(sum(counts.values()), 8)
        self.assertEqual(counts, {"low_angle": 6, "high_angle": 2})

    def test_percentile_score_direction_and_missing_neutrality(self):
        lower = percentile_score([3.0, 1.0, np.nan], higher_is_better=False, missing=.5)
        higher = percentile_score([3.0, 1.0, np.nan], higher_is_better=True, missing=.5)
        self.assertGreater(lower[1], lower[0])
        self.assertGreater(higher[0], higher[1])
        self.assertEqual(lower[2], .5)
        self.assertEqual(higher[2], .5)


if __name__ == "__main__":
    unittest.main()
