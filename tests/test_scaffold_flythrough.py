from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from create_scaffold_flythrough import catmull_rom, quaternion_to_rotation, slerp  # noqa: E402


class ScaffoldFlythroughTest(unittest.TestCase):
    def test_slerp_remains_normalized_and_uses_short_arc(self) -> None:
        q0 = np.asarray([1.0, 0.0, 0.0, 0.0])
        q1 = -np.asarray([0.0, 0.0, 0.0, 1.0])
        middle = slerp(q0, q1, 0.5)
        self.assertAlmostEqual(float(np.linalg.norm(middle)), 1.0, places=12)
        rotation = quaternion_to_rotation(middle)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)

    def test_catmull_rom_hits_segment_endpoints(self) -> None:
        p0 = np.asarray([-1.0, 0.0, 0.0])
        p1 = np.asarray([0.0, 1.0, 0.0])
        p2 = np.asarray([2.0, 1.0, 0.0])
        p3 = np.asarray([3.0, 0.0, 0.0])
        np.testing.assert_allclose(catmull_rom(p0, p1, p2, p3, 0.0), p1)
        np.testing.assert_allclose(catmull_rom(p0, p1, p2, p3, 1.0), p2)


if __name__ == "__main__":
    unittest.main()
