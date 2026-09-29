from __future__ import annotations

import unittest

import numpy as np

from src.semantic_render_domain import (
    domain_metric_delta,
    render_domain_panel,
    rgb_psnr_db,
    select_preflight_oof_views,
    validate_probability_simplex,
)


class RenderDomainTests(unittest.TestCase):
    def test_training_only_oof_plan_is_complete_and_disjoint(self) -> None:
        split = {
            "fixed_validation_read_only": ["054.png"],
            "folds": [
                {"fold": 0, "oof": ["001.png", "002.png"]},
                {"fold": 1, "oof": ["003.png"]},
                {"fold": 2, "oof": ["004.png"]},
            ],
        }
        plan = select_preflight_oof_views(
            split,
            {"0": ["001.png", "002.png"], "1": ["003.png"], "2": ["004.png"]},
            expected_count=4,
        )
        self.assertEqual([row["fold"] for row in plan], [0, 0, 1, 2])
        with self.assertRaises(ValueError):
            select_preflight_oof_views(
                split,
                {"0": ["054.png"], "1": ["003.png"], "2": ["004.png"]},
                expected_count=3,
            )

    def test_probability_contract_rejects_non_simplex(self) -> None:
        valid = np.full((5, 3, 4), 0.2, dtype=np.float32)
        validate_probability_simplex(valid)
        invalid = valid.copy()
        invalid[0, 0, 0] = 0.7
        with self.assertRaises(ValueError):
            validate_probability_simplex(invalid)

    def test_panel_preserves_layout_and_domain_delta(self) -> None:
        rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        target = np.zeros((3, 4), dtype=np.uint8)
        panel = render_domain_panel(rgb, rgb, target, target, target)
        self.assertEqual(panel.shape, (3, 20, 3))
        self.assertIsNone(rgb_psnr_db(rgb, rgb))
        base = {
            "mIoU_with_background": 0.6,
            "mIoU_without_background": 0.5,
            "background_to_stay_cable_pixels": 10,
            "background_to_stay_cable_fraction": 0.1,
            "per_class": {"stay_cable": {"iou": 0.2, "precision": 0.3, "recall": 0.4}},
        }
        changed = {
            **base,
            "mIoU_with_background": 0.5,
            "background_to_stay_cable_pixels": 13,
            "per_class": {"stay_cable": {"iou": 0.1, "precision": 0.2, "recall": 0.3}},
        }
        self.assertEqual(
            domain_metric_delta(base, changed)["background_to_stay_cable_pixels_delta"],
            3,
        )
