import unittest
import numpy as np
from src.quality_review import SourceEvidence, recover_deck


class QualityReviewTests(unittest.TestCase):
    def setUp(self):
        self.base = np.array([[0, 0, 1, 2, 3, 4]], np.uint8)
        self.p = np.full(self.base.shape, .99, np.float32)
        self.sources = [SourceEvidence(n, np.ones_like(self.base), np.ones_like(self.base, bool), self.p.copy()) for n in ("001.png", "002.png")]

    def run_review(self, **kwargs):
        defaults = dict(threshold=.9, target_name="003.png", allowed_sources={"001.png", "002.png"})
        defaults.update(kwargs)
        return recover_deck(self.base, self.p, self.sources, **defaults)

    def test_only_background_changes_and_trace_is_complete(self):
        result, trace = self.run_review()
        np.testing.assert_array_equal(result, [[1, 1, 1, 2, 3, 4]])
        np.testing.assert_array_equal(trace["x"], [0, 1])
        self.assertEqual(len(trace["sources"]), 2)
        np.testing.assert_array_equal(self.base, [[0, 0, 1, 2, 3, 4]])

    def test_one_source_or_conflict_abstains(self):
        self.sources[1].reliable[0, 0] = False
        self.sources[1].labels[0, 1] = 0
        result, _ = self.run_review()
        np.testing.assert_array_equal(result, self.base)

    def test_disable_is_exact(self):
        result, trace = self.run_review(threshold=None)
        np.testing.assert_array_equal(result, self.base)
        self.assertEqual(trace["x"].size, 0)

    def test_probability_gate(self):
        self.p[0, :2] = .1
        result, _ = self.run_review()
        np.testing.assert_array_equal(result, self.base)

    def test_duplicate_self_and_forbidden_sources_rejected(self):
        with self.assertRaises(ValueError):
            self.run_review(target_name="001.png")
        with self.assertRaises(ValueError):
            self.run_review(allowed_sources={"001.png"})
        self.sources.append(self.sources[0])
        with self.assertRaises(ValueError):
            self.run_review()

    def test_nan_and_non_native_grid_rejected(self):
        self.p[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.run_review()
        self.p[0, 0] = .99
        self.sources[0] = SourceEvidence("001.png", np.ones((2, 2), np.uint8), np.ones((2, 2), bool), np.ones((2, 2)))
        with self.assertRaises(ValueError):
            self.run_review()


if __name__ == "__main__":
    unittest.main()
