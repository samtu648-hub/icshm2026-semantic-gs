from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_development import development_score, select_development_candidate


class SemanticDevelopmentTests(unittest.TestCase):
    def _metrics(self, miou: float, iou: float, recall: float, fp: int) -> dict:
        return {
            "mIoU_with_background": miou,
            "background_to_stay_cable_pixels": fp,
            "per_class": {"stay_cable": {"iou": iou, "recall": recall}},
        }

    def test_score_weights_global_cable_overlap_and_recall(self) -> None:
        score = development_score(self._metrics(0.8, 0.7, 0.6, 10))
        self.assertAlmostEqual(score, 0.74)

    def test_selects_highest_development_score_then_lower_fp(self) -> None:
        metrics = self._metrics(0.8, 0.7, 0.6, 10)
        selected = select_development_candidate(
            [
                {"stay_cable_threshold": 0.7, "metrics": metrics, "developer_score": 0.74},
                {
                    "stay_cable_threshold": 0.6,
                    "metrics": {**metrics, "background_to_stay_cable_pixels": 9},
                    "developer_score": 0.74,
                },
            ]
        )
        self.assertEqual(selected["stay_cable_threshold"], 0.6)
