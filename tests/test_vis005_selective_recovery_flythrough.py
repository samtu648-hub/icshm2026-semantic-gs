from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Vis005ContractTests(unittest.TestCase):
    def test_registered_columns_are_exact_sem123_records(self) -> None:
        config = json.loads(
            (ROOT / "experiments/runs/VIS-005-SELECTIVE-RECOVERY-FLYTHROUGH/config.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["experiment_id"], "VIS-005-SELECTIVE-RECOVERY-FLYTHROUGH")
        self.assertEqual(config["status"], "completed_developer_visualization")
        columns = config["comparison"]["columns"]
        self.assertEqual([column["id"] for column in columns], ["hard030", "sem123_leader", "sem123_conservative"])
        self.assertEqual(columns[0]["rule"]["stay_cable_threshold"], 0.30)
        self.assertEqual(columns[1]["rule"], {
            "kind": "selective_band_recovery", "lower_threshold": 0.20, "upper_threshold": 0.30,
            "probability_minimum": 0.20, "maximum_seed_distance": 2.0, "prior_mode": "any",
        })
        self.assertEqual(columns[2]["rule"], {
            "kind": "selective_band_recovery", "lower_threshold": 0.20, "upper_threshold": 0.30,
            "probability_minimum": 0.20, "maximum_seed_distance": 4.0, "prior_mode": "sem001_stay_cable",
        })

    def test_visualisation_forbids_labels_and_display_cleanup(self) -> None:
        config = json.loads(
            (ROOT / "experiments/runs/VIS-005-SELECTIVE-RECOVERY-FLYTHROUGH/config.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(config["trajectory"]["uses_reference_rgb_or_labels"] is False)
        self.assertTrue(config["contract"]["no_reference_rgb_or_labels"])
        self.assertTrue(config["contract"]["no_extra_display_filter"])
        self.assertTrue(config["contract"]["no_video_driven_rule_change"])
