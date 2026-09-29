from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from create_dev30_composition_split import (  # type: ignore[import-not-found]  # noqa: E402
    classify_stratum,
    deck_axis_angle_deg,
    vector_angle_deg,
)


class Dev30CompositionSplitTest(unittest.TestCase):
    def test_stratum_boundaries_are_explicit(self) -> None:
        self.assertEqual(classify_stratum(0.2, 20.0, 0.1, 50.0), "regular")
        self.assertEqual(classify_stratum(0.1, 20.0, 0.1, 50.0), "distant")
        self.assertEqual(classify_stratum(0.2, 50.0, 0.1, 50.0), "oblique")
        self.assertEqual(classify_stratum(0.1, 50.0, 0.1, 50.0), "hybrid")

    def test_deck_axis_is_an_undirected_line(self) -> None:
        horizontal = np.zeros((20, 20), dtype=np.uint8)
        horizontal[9:11, 2:18] = 1
        vertical = np.zeros((20, 20), dtype=np.uint8)
        vertical[2:18, 9:11] = 1
        self.assertAlmostEqual(deck_axis_angle_deg(horizontal), 0.0)
        self.assertAlmostEqual(deck_axis_angle_deg(vertical), 90.0)

    def test_vector_angle_clips_numeric_roundoff(self) -> None:
        axis = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(vector_angle_deg(axis, axis), 0.0)
        self.assertAlmostEqual(vector_angle_deg(axis, -axis), 180.0)


if __name__ == "__main__":
    unittest.main()
