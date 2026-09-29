from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_sem384_dev270 import sampling_plan  # type: ignore[import-not-found]  # noqa: E402


class Sem384SamplingPlanTest(unittest.TestCase):
    def test_plan_is_deterministic_and_role_limited(self) -> None:
        names = ["001.png", "002.png", "003.png"]
        first = sampling_plan(names, seed=7, steps=20)
        second = sampling_plan(names, seed=7, steps=20)
        self.assertEqual(first, second)
        self.assertEqual([row["step"] for row in first], list(range(1, 21)))
        self.assertTrue({row["name"] for row in first} <= set(names))
        self.assertTrue(all(0 <= row["top"] <= 221 for row in first))
        self.assertTrue(all(0 <= row["left"] <= 552 for row in first))

    def test_plan_rejects_unstable_name_order(self) -> None:
        with self.assertRaises(ValueError):
            sampling_plan(["002.png", "001.png"], seed=7, steps=1)


if __name__ == "__main__":
    unittest.main()
