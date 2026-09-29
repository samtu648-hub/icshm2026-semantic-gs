from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.semantic_crossview_consensus import oof_fold_by_name


class Sem130RegistrationTests(unittest.TestCase):
    def test_registered_crossview_rule_is_fixed_and_training_only(self) -> None:
        config = json.loads(
            (ROOT / "experiments/runs/SEM-130-ANCHOR-CROSSVIEW-CONSENSUS-OOF/config.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["experiment_id"], "SEM-130-ANCHOR-CROSSVIEW-CONSENSUS-OOF")
        self.assertEqual(config["status"], "completed_calibration_gate_failed")
        rule = config["crossview_consensus"]
        self.assertEqual(rule["minimum_source_view_grid"], [1, 2, 3, 4, 6, 8])
        self.assertEqual(rule["minimum_dominant_pixels_per_source_view"], 8)
        self.assertEqual(rule["minimum_mean_oof_stay_cable_probability"], 0.30)
        self.assertIn("deleting_hard_0.30_stay_cable", config["forbidden"])
        self.assertIn("fixed_74_view_access", config["forbidden"])

    def test_oof_source_contract_excludes_target_from_training(self) -> None:
        split = json.loads((ROOT / "experiments/splits/sem_001_scaffold_static_v1.json").read_text(encoding="utf-8"))
        oof = json.loads((ROOT / "experiments/splits/sem_005_oof_spatial_v1.json").read_text(encoding="utf-8"))
        names = split["semantic_train_labeled"]
        mapping, evidence = oof_fold_by_name(oof, names)
        self.assertEqual(set(mapping), set(names))
        self.assertTrue(evidence["source_views_exclude_current_camera"])
        self.assertTrue(evidence["target_absent_from_its_source_model_training"])


if __name__ == "__main__":
    unittest.main()
