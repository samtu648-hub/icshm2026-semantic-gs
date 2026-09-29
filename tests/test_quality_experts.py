import unittest
import numpy as np
from src.quality_experts import expert_counts_weights, target_for_expert, validate_probability
from src.quality_coverage import coverage_counts, retained_conditioning
from src.quality_deck_evidence import add_deck_evidence, change_counts


class QualityExpertTests(unittest.TestCase):
    def test_binary_counts_use_selected_training_counts(self):
        counts, weights = expert_counts_weights([100, 20, 10, 5, 1], 1)
        np.testing.assert_array_equal(counts, [116, 20])
        np.testing.assert_allclose(weights, [68/116, 68/20])

    def test_targets(self):
        a = np.arange(5, dtype=np.uint8)[None]
        np.testing.assert_array_equal(target_for_expert(a, 2), [[0, 0, 1, 0, 0]])
        np.testing.assert_array_equal(target_for_expert(a), a)
        with self.assertRaises(ValueError):
            expert_counts_weights([1, 2, 3, 4, 5], 4)

    def test_bad_probability(self):
        p = np.ones((5, 2, 3), np.float32) / 5
        validate_probability(p, 5, (2, 3))
        for q in (p * 2, p.astype(np.float16), np.full_like(p, np.nan)):
            with self.assertRaises(ValueError):
                validate_probability(q, 5, (2, 3))

    def test_residual_background_ceiling(self):
        a = np.array([[0., .25, .49, .5, .9, 1.]], np.float32)
        p = np.zeros((5, 1, 6), np.float32)
        p[0], p[1] = 1-a, a
        r = coverage_counts(a, p, np.ones_like(a, dtype=np.uint8))
        self.assertEqual(r["classes"]["1"]["low_coverage_errors"], 3)
        self.assertEqual(sum(b["pixels"] for b in r["bins"]), 6)
        with self.assertRaises(ValueError):
            coverage_counts(a, np.roll(p, 1, axis=0), np.ones_like(a, dtype=np.uint8))

    def test_initialization_intersection_not_all_raw_tracks(self):
        data = {"point_id": np.array([1, 2, 3]), "error_px": np.array([6., 7., 1.]), "source_max_ray_angle_deg": np.array([.1, 2., .1])}
        r, keep = retained_conditioning([2, 3, 4], data, 3)
        self.assertEqual(r["large_error_retained"], 1)
        self.assertEqual(r["weak_large_retained"], 0)
        self.assertEqual(r["initial_without_successful_cross_prediction"], 1)

    def test_deck_protects_foreground_and_history(self):
        before = np.array([[0, 1, 2, 3, 4, 0]], np.uint8)
        history = before.copy(); history[0, -1] = 4
        p = np.ones_like(before, np.float32)
        out, evidence = add_deck_evidence(before, history, before, p, p, .5)
        np.testing.assert_array_equal(out, [[1, 1, 2, 3, 4, 0]])
        self.assertEqual(evidence["history_excluded"].sum(), 1)
        off, _ = add_deck_evidence(before, history, before, p, p, None)
        np.testing.assert_array_equal(off, before)

    def test_deck_requires_both_experts(self):
        bg = np.zeros((1, 2), np.uint8)
        out, _ = add_deck_evidence(bg, bg, bg, np.array([[.9, .2]]), np.array([[.2, .9]]), .5)
        np.testing.assert_array_equal(out, bg)

    def test_modification_accounting(self):
        r = change_counts(np.array([[0, 0, 0]]), np.array([[1, 1, 1]]), np.array([[1, 0, 4]]))
        self.assertEqual(r, {"changed": 3, "corrected": 1, "harmed": 1, "wrong_to_wrong": 1})


if __name__ == "__main__":
    unittest.main()
