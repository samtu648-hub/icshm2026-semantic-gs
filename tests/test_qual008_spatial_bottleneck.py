import unittest

from scripts.audit_qual008_spatial_bottleneck import classify, spearman


class Qual008Test(unittest.TestCase):
    def test_spearman_tracks_monotone_order(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 6, 9]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [9, 6, 3]), -1.0)

    def test_mixed_classification_requires_residual_and_coupling(self):
        better = {
            "boundary": {"f1": 0.80},
            "semantic": {"mIoU_with_background": 0.93},
        }
        correlations = {
            "PSNR_dB": {"mIoU_with_background": 0.75, "boundary_f1": 0.2},
            "LPIPS_Alex": {"mIoU_with_background": -0.1, "boundary_f1": -0.2},
        }
        analysis = {
            "semantic_residual_rule": {
                "better_half_boundary_f1_below": 0.85,
                "better_half_views_with_stay_cable_recall_below_0_90_at_least": 3,
                "or_better_half_five_class_miou_below": 0.92,
            },
            "strong_rgb_coupling_absolute_spearman_at_least": 0.70,
        }
        actual = classify(better, 3, correlations, analysis)
        self.assertEqual(actual["outcome"], "mixed_rgb_and_semantic_bottleneck")
        self.assertTrue(actual["boundary_candidate_supported"])


if __name__ == "__main__":
    unittest.main()
