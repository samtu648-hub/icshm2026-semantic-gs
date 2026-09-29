from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_sem394_seed_ensemble import (  # type: ignore[import-not-found]  # noqa: E402
    disagreement_diagnostics,
    ensemble_probability,
    evaluate_gates,
    hard_oracle_prediction,
    paired_bootstrap,
    smallest_class_argmax,
    tie_pixel_count,
)


def stack(rows):
    return np.asarray(rows, dtype=np.float32)


def confusion(tp_a, fp_b):
    matrix = np.zeros((5, 5), dtype=np.int64)
    for target, prediction, count in zip(tp_a, fp_b, [0] * 0 + [1] * len(tp_a)):
        matrix[target, prediction] += count
    return matrix


def make_confusion(diagonal):
    matrix = np.zeros((5, 5), dtype=np.int64)
    for index, value in enumerate(diagonal):
        matrix[index, index] = value
    return matrix


class EnsembleArithmeticTest(unittest.TestCase):
    def test_float64_average_of_float32_stacks(self):
        a = stack(
            [[[0.2, 0.6]], [[0.8, 0.3]], [[0.0, 0.05]], [[0.0, 0.05]], [[0.0, 0.0]]]
        )
        b = stack(
            [[[0.1, 0.5]], [[0.9, 0.4]], [[0.0, 0.05]], [[0.0, 0.05]], [[0.0, 0.0]]]
        )
        merged = ensemble_probability(a, b)
        self.assertEqual(merged.dtype, np.float64)
        expected0 = 0.5 * np.float64(a[0, 0, 0]) + 0.5 * np.float64(b[0, 0, 0])
        self.assertEqual(merged[0, 0, 0], expected0)

    def test_rejects_mismatched_shapes(self):
        a = np.zeros((5, 2, 2), np.float64)
        b = np.zeros((5, 3, 3), np.float64)
        with self.assertRaises(ValueError):
            ensemble_probability(a, b)

    def test_smallest_class_wins_exact_ties(self):
        a = np.zeros((5, 1, 1), np.float64)
        a[1, 0, 0] = 0.5
        a[3, 0, 0] = 0.5
        self.assertEqual(smallest_class_argmax(a)[0, 0], 1)
        self.assertEqual(tie_pixel_count(a), 1)

    def test_tie_count_excludes_strict_maxima(self):
        a = np.zeros((5, 1, 2), np.float64)
        a[2, 0, 0] = 1.0
        a[1, 0, 1] = 0.5
        a[4, 0, 1] = 0.5
        self.assertEqual(tie_pixel_count(a), 1)


class DiagnosticsTest(unittest.TestCase):
    def test_disagreement_and_oracle_on_small_example(self):
        gt = np.asarray([[0, 1, 2], [2, 1, 0]], np.uint8)
        a = np.asarray([[0, 1, 1], [2, 1, 3]], np.uint8)
        b = np.asarray([[0, 2, 2], [0, 1, 0]], np.uint8)
        stats = disagreement_diagnostics(gt, a, b)
        self.assertEqual(stats["total_pixels"], 6)
        self.assertEqual(stats["disagreement_pixels"], 4)
        self.assertEqual(stats["only_a_correct_pixels"], 2)
        self.assertEqual(stats["only_b_correct_pixels"], 2)
        self.assertEqual(stats["both_wrong_disagreement_pixels"], 0)
        oracle = hard_oracle_prediction(gt, a, b)
        self.assertTrue(np.array_equal(oracle, gt))

    def test_shares_are_relative_to_total_pixels(self):
        gt = np.zeros((10, 10), np.uint8)
        a = np.zeros((10, 10), np.uint8)
        b = np.ones((10, 10), np.uint8)
        stats = disagreement_diagnostics(gt, a, b)
        self.assertAlmostEqual(
            stats["disagreement_pixels"] / stats["total_pixels"], 1.0
        )


class BootstrapTest(unittest.TestCase):
    def records(self):
        rows = []
        for index in range(30):
            baseline = make_confusion([100 + index, 50, 40, 30, 20])
            candidate = make_confusion([100 + index, 50, 40, 30, 20])
            baseline[0, 1] += 10
            candidate[0, 1] += 5
            rows.append(
                (baseline, candidate, np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]))
            )
        return rows

    def test_deterministic_and_positive(self):
        first = paired_bootstrap(self.records(), seed=394, draws=200)
        second = paired_bootstrap(self.records(), seed=394, draws=200)
        self.assertEqual(
            first["metrics"]["five_class_miou_pp"]["median_pp"],
            second["metrics"]["five_class_miou_pp"]["median_pp"],
        )
        stats = first["metrics"]["five_class_miou_pp"]
        self.assertGreaterEqual(stats["probability_strictly_positive"], 0.0)
        self.assertLessEqual(stats["probability_strictly_positive"], 1.0)
        self.assertGreater(stats["median_pp"], 0.0)

    def test_requires_thirty_views(self):
        with self.assertRaises(ValueError):
            paired_bootstrap(self.records()[:29], seed=394, draws=10)


class GateTest(unittest.TestCase):
    def holdout(
        self,
        *,
        baseline,
        ensemble,
        groups=None,
        class_iou=None,
        ensemble_class_iou=None,
    ):
        return {
            "baseline": {
                "five_class_miou": baseline,
                "foreground_miou": baseline,
                "per_class_iou": class_iou
                or {
                    "background": 0.99,
                    "deck": 0.97,
                    "stay_cable": 0.91,
                    "tower": 0.93,
                    "foundation": 0.87,
                },
                "boundary_f1": 0.80,
            },
            "ensemble": {
                "five_class_miou": ensemble,
                "foreground_miou": ensemble,
                "per_class_iou": ensemble_class_iou
                or class_iou
                or {
                    "background": 0.99,
                    "deck": 0.97,
                    "stay_cable": 0.91,
                    "tower": 0.93,
                    "foundation": 0.87,
                },
                "boundary_f1": 0.80,
            },
            "groups_five_class_miou": groups
            or {
                "regular-local": {"baseline": baseline, "ensemble": ensemble},
                "oblique-local": {"baseline": baseline, "ensemble": ensemble},
                "distant-local": {"baseline": baseline, "ensemble": ensemble},
            },
        }

    def bootstrap(self, median, probability):
        return {
            "metrics": {
                "five_class_miou_pp": {
                    "median_pp": median,
                    "probability_strictly_positive": probability,
                }
            }
        }

    def test_all_gates_pass_on_clear_improvement(self):
        holdout = self.holdout(baseline=0.90, ensemble=0.905)
        gates = evaluate_gates(holdout, self.bootstrap(0.5, 0.99))
        self.assertTrue(gates["all_passed"])
        self.assertEqual(
            gates["outcome"],
            "worth_training_rgb140_second_seed_for_full_ensemble_validation",
        )

    def test_g1_fails_below_threshold(self):
        holdout = self.holdout(baseline=0.90, ensemble=0.9005)
        gates = evaluate_gates(holdout, self.bootstrap(0.05, 0.99))
        self.assertFalse(gates["all_passed"])
        self.assertFalse(
            gates["checks"]["g1_holdout_five_class_miou_vs_sem392"]["passed"]
        )

    def test_g6_fails_without_bootstrap_support(self):
        holdout = self.holdout(baseline=0.90, ensemble=0.905)
        gates = evaluate_gates(holdout, self.bootstrap(0.5, 0.90))
        self.assertFalse(
            gates["checks"]["g6_paired_bootstrap_median_and_probability"]["passed"]
        )

    def test_g3_fails_on_class_drop(self):
        iou = {
            "background": 0.99,
            "deck": 0.97,
            "stay_cable": 0.91,
            "tower": 0.93,
            "foundation": 0.87,
        }
        ensemble_iou = {
            "background": 0.99,
            "deck": 0.97,
            "stay_cable": 0.91,
            "tower": 0.93,
            "foundation": 0.86,
        }
        holdout = self.holdout(
            baseline=0.90,
            ensemble=0.905,
            class_iou=iou,
            ensemble_class_iou=ensemble_iou,
        )
        gates = evaluate_gates(holdout, self.bootstrap(0.5, 0.99))
        self.assertFalse(
            gates["checks"]["g3_holdout_class_iou_drop_vs_sem392"]["passed"]
        )

    def test_g4_fails_on_group_drop(self):
        groups = {
            "regular-local": {"baseline": 0.90, "ensemble": 0.906},
            "oblique-local": {"baseline": 0.90, "ensemble": 0.905},
            "distant-local": {"baseline": 0.90, "ensemble": 0.88},
        }
        holdout = self.holdout(baseline=0.90, ensemble=0.905, groups=groups)
        gates = evaluate_gates(holdout, self.bootstrap(0.5, 0.99))
        self.assertFalse(
            gates["checks"]["g4_holdout_group_five_class_miou_drop_vs_sem392"]["passed"]
        )


if __name__ == "__main__":
    unittest.main()
