import copy
import json
from pathlib import Path
import unittest

from src.spatial_seed_replication import (
    PAIR_CONFIG,
    confusion_metrics,
    direction_gates,
    validate_pair_configs,
    validate_pair_endpoints,
)

ROOT = Path(__file__).resolve().parents[1]


class SpatialSeedReplicationTests(unittest.TestCase):
    def test_registered_pair_and_rejection_of_wrong_seed_reference(self):
        pair = json.loads((ROOT / PAIR_CONFIG).read_text())
        arms = {
            k: json.loads((ROOT / v["config"]).read_text())
            for k, v in pair["arms"].items()
        }
        split = json.loads((ROOT / pair["split_path"]).read_text())
        validate_pair_configs(arms, split)
        for arm in arms.values():
            arm["seed_aware_initialization"]["reference_seed"] = 20260805
        with self.assertRaisesRegex(ValueError, "current seed"):
            validate_pair_configs(arms, split)

    def test_training_label_leak_rejected_before_confirmation(self):
        good = {
            "status": "completed_sealed_6000",
            "trained_steps": 6000,
            "seed": 20260919,
            "seed_aware_initialization_validated": True,
            "holdout_labels_accessed_during_fit": [],
            "sampling_plan_exact": True,
            "frozen_rgb_hashes_exact": True,
            "initial_model_state_sha256": "same",
            "pretrained_load": {},
            "split_sha256": "same",
            "access": {},
        }
        bad = copy.deepcopy(good)
        bad["holdout_labels_accessed_during_fit"] = ["054.png"]
        with self.assertRaisesRegex(ValueError, "Confirmation labels"):
            validate_pair_endpoints(good, bad)
        bad["holdout_labels_accessed_during_fit"] = []
        bad["trained_steps"] = 3000
        with self.assertRaisesRegex(ValueError, "6000-step"):
            validate_pair_endpoints(good, bad)

    def test_positive_mean_cannot_hide_second_seed_failure(self):
        d = {
            "20260805": {
                "foreground_miou": 0.03,
                "stay_cable_iou": 0.02,
                "boundary_f1": 0.05,
            },
            "20260919": {
                "foreground_miou": -0.01,
                "stay_cable_iou": 0.01,
                "boundary_f1": 0.02,
            },
        }
        self.assertFalse(direction_gates(d)["all_direction_gates_pass"])
        d["20260919"]["foreground_miou"] = 0.001
        self.assertTrue(direction_gates(d)["all_direction_gates_pass"])
        d["20260919"]["boundary_f1"] = 0
        self.assertFalse(direction_gates(d)["all_direction_gates_pass"])

    def test_foreground_does_not_reward_true_background(self):
        matrix = [[100, 2, 0, 0, 0], [3, 5, 0, 0, 0], [0] * 5, [0] * 5, [0] * 5]
        a = confusion_metrics(matrix)
        self.assertEqual(a["mIoU_without_background"], 0.5)
        matrix[0][0] = 10000
        b = confusion_metrics(matrix)
        self.assertEqual(a["mIoU_without_background"], b["mIoU_without_background"])
        self.assertNotEqual(a["mIoU_with_background"], b["mIoU_with_background"])
        self.assertEqual(a["per_class"]["deck"]["fp"], 2)
        self.assertEqual(a["per_class"]["deck"]["fn"], 3)


if __name__ == "__main__":
    unittest.main()
