import unittest

import numpy as np

from src.gaussian_ray_responsibility import (
    PixelTrace,
    replay_pixel,
    responsibility_prediction,
    split_depth_layers,
    split_depth_modes,
    trace_statistics,
)


class GaussianRayResponsibilityTest(unittest.TestCase):
    def test_replay_matches_front_to_back_compositing(self):
        trace = replay_pixel(
            0,
            0,
            1,
            1,
            means2d=np.zeros((2, 2), dtype=np.float32),
            conic_opacity=np.array([[1.0, 0.0, 1.0, 0.5], [1.0, 0.0, 1.0, 0.5]], dtype=np.float32),
            depths=np.array([1.0, 2.0], dtype=np.float32),
            point_list=np.array([0, 1], dtype=np.int32),
            ranges=np.array([[0, 2]], dtype=np.int32),
            n_contrib=np.array([2], dtype=np.int32),
        )
        np.testing.assert_allclose(trace.weights, [0.5, 0.25])
        self.assertAlmostEqual(trace.final_transmittance, 0.25)
        np.testing.assert_array_equal(trace.gaussian_ids, [0, 1])

    def test_alpha_cutoff_and_early_stop_match_native_rules(self):
        trace = replay_pixel(
            0,
            0,
            1,
            1,
            means2d=np.zeros((4, 2), dtype=np.float32),
            conic_opacity=np.array(
                [
                    [1.0, 0.0, 1.0, 0.001],
                    [1.0, 0.0, 1.0, 0.99],
                    [1.0, 0.0, 1.0, 0.98],
                    [1.0, 0.0, 1.0, 0.60],
                ],
                dtype=np.float32,
            ),
            depths=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            point_list=np.array([0, 1, 2, 3], dtype=np.int32),
            ranges=np.array([[0, 4]], dtype=np.int32),
            n_contrib=np.array([4], dtype=np.int32),
        )
        np.testing.assert_array_equal(trace.gaussian_ids, [1, 2])
        self.assertAlmostEqual(trace.final_transmittance, 0.0002, places=7)

    def test_statistics_report_concentration_and_depth_spread(self):
        stats = trace_statistics(
            PixelTrace(
                gaussian_ids=np.array([4, 5]),
                depths=np.array([1.0, 3.0]),
                weights=np.array([0.6, 0.2]),
                final_transmittance=0.2,
                checked_entries=2,
            )
        )
        self.assertAlmostEqual(stats["top_share"], 0.75)
        self.assertAlmostEqual(stats["effective_contributors"], 1.6)
        self.assertEqual(stats["prefix80_contributors"], 2)
        self.assertAlmostEqual(stats["depth_mean"], 1.5)

    def test_surface_prefix_is_front_to_back_and_standard_keeps_background(self):
        trace = PixelTrace(
            gaussian_ids=np.array([0, 1, 2]),
            depths=np.array([1.0, 2.0, 3.0]),
            weights=np.array([0.4, 0.3, 0.1]),
            final_transmittance=0.2,
            checked_entries=3,
        )
        probability = np.array(
            [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float64
        )
        result = responsibility_prediction(trace, probability, surface_mass_fraction=0.8)
        np.testing.assert_allclose(result.standard_probability, [0.6, 0.4])
        np.testing.assert_allclose(result.surface_probability, [3 / 7, 4 / 7])
        np.testing.assert_allclose(result.tail_probability, [1.0, 0.0])
        self.assertEqual(result.surface_contributors, 2)

    def test_depth_layer_split_uses_largest_mass_admissible_gap(self):
        trace = PixelTrace(
            gaussian_ids=np.array([0, 1, 2, 3]),
            depths=np.array([1.0, 1.1, 3.0, 6.0]),
            weights=np.array([0.25, 0.25, 0.30, 0.20]),
            final_transmittance=0.0,
            checked_entries=4,
        )
        probability = np.array(
            [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.0, 1.0]], dtype=np.float64
        )
        result = split_depth_layers(
            trace,
            probability,
            minimum_side_mass_fraction=0.2,
            minimum_relative_depth_gap=0.5,
        )
        self.assertTrue(result.eligible)
        self.assertTrue(result.clear)
        self.assertEqual(result.split_index, 3)
        self.assertAlmostEqual(result.front_mass_fraction, 0.8)
        self.assertAlmostEqual(result.back_mass_fraction, 0.2)
        self.assertAlmostEqual(result.depth_gap, 3.0)
        np.testing.assert_allclose(result.front_probability, [0.56875, 0.43125])
        np.testing.assert_allclose(result.back_probability, [0.0, 1.0])

    def test_depth_layer_split_rejects_no_mass_admissible_boundary(self):
        trace = PixelTrace(
            gaussian_ids=np.array([0, 1]),
            depths=np.array([1.0, 5.0]),
            weights=np.array([0.95, 0.05]),
            final_transmittance=0.0,
            checked_entries=2,
        )
        result = split_depth_layers(trace, np.eye(2), minimum_side_mass_fraction=0.2)
        self.assertFalse(result.eligible)
        self.assertFalse(result.clear)
        self.assertIsNone(result.split_index)

    def test_depth_mode_split_maximizes_weighted_between_variance(self):
        trace = PixelTrace(
            gaussian_ids=np.array([0, 1, 2, 3]),
            depths=np.array([1.0, 1.2, 2.0, 5.0]),
            weights=np.array([0.25, 0.25, 0.30, 0.20]),
            final_transmittance=0.0,
            checked_entries=4,
        )
        probability = np.array(
            [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]], dtype=np.float64
        )
        result = split_depth_modes(
            trace,
            probability,
            minimum_side_mass_fraction=0.2,
            minimum_explained_depth_variance=0.5,
        )
        self.assertTrue(result.eligible)
        self.assertTrue(result.clear)
        self.assertEqual(result.split_index, 3)
        self.assertGreater(result.explained_depth_variance, 0.5)
        self.assertLess(result.front_depth_mean, result.back_depth_mean)
        np.testing.assert_allclose(result.back_probability, [0.0, 1.0])

    def test_depth_mode_split_marks_zero_variance_as_unclear(self):
        trace = PixelTrace(
            gaussian_ids=np.array([0, 1]),
            depths=np.array([2.0, 2.0]),
            weights=np.array([0.5, 0.5]),
            final_transmittance=0.0,
            checked_entries=2,
        )
        result = split_depth_modes(trace, np.eye(2))
        self.assertTrue(result.eligible)
        self.assertFalse(result.clear)
        self.assertEqual(result.explained_depth_variance, 0.0)


if __name__ == "__main__":
    unittest.main()
