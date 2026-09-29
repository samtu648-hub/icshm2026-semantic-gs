import unittest

from src.bridge_responsibility_clone import summarize_population, summarize_responsibility


class BridgeResponsibilityCloneTest(unittest.TestCase):
    def test_stability_requires_two_sided_median_and_small_iqr(self):
        stable = summarize_responsibility(
            [0.35, 0.40, 0.45], 0.4,
            minimum_bridge_fraction=0.15, maximum_bridge_fraction=0.85, maximum_iqr=0.30,
        )
        one_sided = summarize_responsibility(
            [0.02, 0.04, 0.06], 0.4,
            minimum_bridge_fraction=0.15, maximum_bridge_fraction=0.85, maximum_iqr=0.30,
        )
        self.assertTrue(stable["stable_two_responsibility"])
        self.assertFalse(one_sided["stable_two_responsibility"])
        self.assertLess(stable["alpha_conservation_abs_error"], 1e-12)

    def test_population_counts_stable_rows(self):
        rows = [
            summarize_responsibility(
                values, 0.3, minimum_bridge_fraction=0.15,
                maximum_bridge_fraction=0.85, maximum_iqr=0.30,
            )
            for values in ([0.3, 0.4, 0.5], [0.01, 0.02, 0.03])
        ]
        summary = summarize_population(rows)
        self.assertEqual(summary["gaussians"], 2)
        self.assertEqual(summary["stable_two_responsibility"], 1)


if __name__ == "__main__":
    unittest.main()
