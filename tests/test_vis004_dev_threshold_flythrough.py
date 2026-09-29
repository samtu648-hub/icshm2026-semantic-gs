from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Vis004ContractTests(unittest.TestCase):
    def test_registered_comparison_is_the_sem120_leader_and_pareto_pair(self) -> None:
        config = json.loads(
            (ROOT / "experiments/runs/VIS-004-DEV-THRESHOLD-FLYTHROUGH/config.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["experiment_id"], "VIS-004-DEV-THRESHOLD-FLYTHROUGH")
        self.assertEqual(config["status"], "completed_developer_visualization")
        self.assertEqual(
            [
                config["comparison"]["sem111_historical_threshold"],
                config["comparison"]["sem120_developer_leader_threshold"],
                config["comparison"]["sem120_pareto_threshold"],
            ],
            [0.8, 0.2, 0.3],
        )
        self.assertTrue(config["trajectory"]["uses_reference_rgb_or_labels"] is False)
        self.assertTrue(config["contract"]["no_extra_display_filter"])
