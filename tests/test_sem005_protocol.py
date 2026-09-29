import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.create_sem005_oof_split import deterministic_clusters
from src.semantic2d import CLASS_NAMES, confusion_metrics, tile_starts, update_confusion


class Sem005ProtocolTests(unittest.TestCase):
    def test_spatial_clusters_are_deterministic_and_complete(self):
        points = np.asarray([[float(index), 0.0, 0.0] for index in range(12)])
        first, centres_first, seeds_first = deterministic_clusters(points, 20260805)
        second, centres_second, seeds_second = deterministic_clusters(points, 20260805)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(centres_first, centres_second)
        self.assertEqual(seeds_first, seeds_second)
        self.assertEqual(set(first.tolist()), {0, 1, 2})

    def test_sliding_grid_covers_full_resolution(self):
        ys = tile_starts(989, 768, 512)
        xs = tile_starts(1320, 768, 512)
        coverage = np.zeros((989, 1320), dtype=np.uint8)
        for top in ys:
            for left in xs:
                coverage[top : top + 768, left : left + 768] += 1
        self.assertTrue(np.all(coverage > 0))
        self.assertEqual(ys[-1], 221)
        self.assertEqual(xs[-1], 552)

    def test_confusion_uses_official_class_order(self):
        self.assertEqual(CLASS_NAMES[2], "stay_cable")
        target = np.asarray([[0, 2, 3, 4, 1]], dtype=np.uint8)
        prediction = np.asarray([[2, 2, 3, 0, 1]], dtype=np.uint8)
        confusion = np.zeros((5, 5), dtype=np.int64)
        update_confusion(confusion, target, prediction)
        result = confusion_metrics(confusion)
        self.assertEqual(result["background_to_stay_cable_pixels"], 1)
        self.assertEqual(result["per_class"]["stay_cable"]["tp"], 1)


if __name__ == "__main__":
    unittest.main()
