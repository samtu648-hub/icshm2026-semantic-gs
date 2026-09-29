from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_dev30_full_chain_review import colorize, crop_coverage  # type: ignore[import-not-found]  # noqa: E402


class Dev30ReviewTest(unittest.TestCase):
    def test_crop_coverage_preserves_native_grid_coordinates(self) -> None:
        coverage = crop_coverage([{"top": 1, "left": 2}], crop=3)
        self.assertEqual(coverage.shape, (989, 1320))
        self.assertEqual(int(coverage.sum()), 9)
        self.assertTrue(np.all(coverage[1:4, 2:5] == 1))
        self.assertEqual(int(coverage[0, 0]), 0)

    def test_colorize_requires_native_class_ids(self) -> None:
        labels = np.asarray([[0, 1, 2, 3, 4]], dtype=np.uint8)
        colored = colorize(labels)
        self.assertEqual(colored.shape, (1, 5, 3))
        with self.assertRaises(ValueError):
            colorize(np.asarray([[5]], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
