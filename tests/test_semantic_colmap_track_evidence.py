import unittest

import numpy as np

from src.semantic_colmap_track_evidence import (
    binary_source_posteriors,
    deck_confusion,
    exact_oof_source_contract,
    observation_index,
    oracle_deck_iou_gain,
    target_majority_map,
)


class _Image:
    def __init__(self, points2d):
        self.points2d = points2d


class ColmapTrackEvidenceTests(unittest.TestCase):
    def test_source_posterior_never_reads_target_label(self):
        observations = {7: (("source.png", 1, 1), ("target.png", 1, 1), ("other.png", 1, 1))}
        calls = []
        labels = {
            "source.png": np.array([[0, 0], [0, 1]], dtype=np.uint8),
            "other.png": np.array([[0, 0], [0, 1]], dtype=np.uint8),
        }
        posterior, support = binary_source_posteriors(
            observations,
            {"source.png", "other.png"},
            lambda name: calls.append(name) or labels[name],
            minimum_source_observations=2,
        )
        self.assertEqual(calls, ["source.png", "other.png"])
        self.assertEqual(support, {7: 2})
        self.assertAlmostEqual(posterior[7], 0.75)
        evidence = target_majority_map((3, 3), observations, posterior, target_name="target.png")
        self.assertTrue(evidence[1, 1])

    def test_observation_filter_and_oracle_upper_bound(self):
        images = {"a.png": _Image(((0.2, 1.6, 4), (8.0, 0.0, -1))), "b.png": _Image(((2.0, 2.0, 4),))}
        indexed = observation_index(images, {"a.png", "b.png"}, width=3, height=3, permitted_track_ids={4})
        self.assertEqual(indexed, {4: (("a.png", 0, 2), ("b.png", 2, 2))})
        baseline = deck_confusion(np.array([[1, 0], [0, 0]], np.uint8), np.array([[1, 1], [0, 1]], np.uint8))
        self.assertAlmostEqual(oracle_deck_iou_gain(baseline, 2), 2 / 3)
        with self.assertRaises(ValueError):
            exact_oof_source_contract("a.png", {"a.png"})
