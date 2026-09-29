from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_crossview_consensus import (
    add_view_support,
    apply_retained_recovery,
    dominant_anchor_map,
    other_view_support_map,
    retain_recovery_by_crossview_support,
    view_anchor_cable_evidence,
)


class CrossViewConsensusTests(unittest.TestCase):
    def test_dominant_offsets_map_to_frozen_anchor_ids(self) -> None:
        dominant = np.asarray([[0, 1, -1], [2, 3, 8]], dtype=np.int32)
        offsets = np.asarray([5, 2, 5, 7], dtype=np.int64)
        actual = dominant_anchor_map(dominant, offsets, anchor_count=8)
        expected = np.asarray([[5, 2, -1], [5, 7, -1]], dtype=np.int64)
        self.assertTrue(np.array_equal(actual, expected))

    def test_per_view_evidence_uses_mean_and_dominant_count(self) -> None:
        anchors = np.asarray([[0, 0, 1, 1], [0, 2, 2, -1]], dtype=np.int64)
        probability = np.asarray(
            [[0.40, 0.20, 0.90, 0.10], [0.30, 0.35, 0.25, 0.0]],
            dtype=np.float32,
        )
        actual = view_anchor_cable_evidence(
            anchors,
            probability,
            anchor_count=3,
            minimum_dominant_pixels=2,
            minimum_mean_cable_probability=0.30,
        )
        # Anchor 0: three pixels, mean 0.30.  Anchor 1: mean 0.50.
        # Anchor 2 has too few dominant pixels.
        self.assertTrue(np.array_equal(actual, np.asarray([0, 1], dtype=np.int64)))

    def test_current_camera_is_excluded_from_source_count(self) -> None:
        counts = np.zeros((3, 5), dtype=np.uint16)
        add_view_support(counts, fold=1, supported_anchor_ids=np.asarray([1, 3]))
        add_view_support(counts, fold=1, supported_anchor_ids=np.asarray([1, 2]))
        add_view_support(counts, fold=1, supported_anchor_ids=np.asarray([1, 4]))
        anchors = np.asarray([[1, 2, 3], [4, -1, 1]], dtype=np.int64)
        # The current camera is the first contribution: it supported 1 and 3.
        actual = other_view_support_map(anchors, counts[1], np.asarray([1, 3]))
        expected = np.asarray([[2, 1, 0], [1, 0, 2]], dtype=np.int32)
        self.assertTrue(np.array_equal(actual, expected))

    def test_consensus_only_filters_proposed_additions(self) -> None:
        hard = np.asarray([[2, 0, 1], [0, 3, 0]], dtype=np.uint8)
        additions = np.asarray([[False, True, False], [True, False, True]], dtype=np.bool_)
        support = np.asarray([[0, 2, 9], [1, 8, 0]], dtype=np.int32)
        retained = retain_recovery_by_crossview_support(
            additions, support, minimum_source_views=2
        )
        actual = apply_retained_recovery(hard, additions, retained)
        expected = np.asarray([[2, 2, 1], [0, 3, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))
        self.assertEqual(actual[0, 0], hard[0, 0])
        self.assertEqual(actual[0, 2], hard[0, 2])
        self.assertEqual(actual[1, 1], hard[1, 1])

    def test_invalid_anchor_pixels_cannot_gain_crossview_support(self) -> None:
        counts = np.asarray([0, 3], dtype=np.uint16)
        anchors = np.asarray([[-1, 1]], dtype=np.int64)
        support = other_view_support_map(anchors, counts, np.asarray([1], dtype=np.int64))
        additions = np.asarray([[True, True]], dtype=np.bool_)
        retained = retain_recovery_by_crossview_support(
            additions, support, minimum_source_views=1
        )
        self.assertTrue(np.array_equal(retained, np.asarray([[False, True]])))


if __name__ == "__main__":
    unittest.main()
