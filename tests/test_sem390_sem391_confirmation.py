from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from scripts.build_sem386_comparison import aggregate_scores
from scripts.run_sem384_dev270 import sampling_plan, training_recipe


ROOT = Path(__file__).resolve().parents[1]
CONTROL = (
    ROOT / "experiments/runs/SEM-390-RGB150-DEV270-SEED20260919-MEMORY/config.yaml"
)
CANDIDATE = (
    ROOT
    / "experiments/runs/SEM-391-RGB150-DEV270-CONTEXT50-SEED20260919-MEMORY/config.yaml"
)


class CorrectedConfirmationProtocolTest(unittest.TestCase):
    def test_aggregate_scores_uses_pooled_confusion(self) -> None:
        matrix = np.diag([10, 9, 8, 7, 6])
        scores = aggregate_scores(matrix)
        self.assertEqual(scores["five_class_miou"], 1.0)
        self.assertEqual(scores["foreground_miou"], 1.0)
        self.assertEqual(scores["stay_cable_iou"], 1.0)

    def test_registered_seed_is_threaded_into_training_recipe(self) -> None:
        recipe = training_recipe({"seed": 20260919}, [])
        self.assertEqual(recipe["seed"], 20260919)

    def test_second_seed_native_plan_starts_from_registered_seed(self) -> None:
        split = json.loads(
            (ROOT / "experiments/splits/dev30_local_composition_v1.json").read_text()
        )
        names = sorted(split["primary_fold"]["semantic_train_labeled"])
        self.assertEqual(
            sampling_plan(names, 20260919, steps=1)[0],
            {
                "step": 1,
                "name": "051.png",
                "top": 206,
                "left": 359,
                "flip": True,
            },
        )

    def test_control_seed_aware_contract(self) -> None:
        config = json.loads(CONTROL.read_text(encoding="utf-8"))
        self.assertEqual(config["seed"], 20260919)
        self.assertEqual(config["context_fraction"], 0.0)
        self.assertEqual(config["preflight_mode_counts"], {"native_crop": 8})
        initialization = config["seed_aware_initialization"]
        self.assertEqual(initialization["reference_seed"], 20260805)
        self.assertEqual(len(initialization["allowed_cross_seed_tensors"]), 2)

    def test_candidate_is_exactly_paired_when_bound(self) -> None:
        if not CANDIDATE.exists():
            self.skipTest("Candidate config is bound after SEM390 preparation")
        control = json.loads(CONTROL.read_text(encoding="utf-8"))
        candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
        for key in (
            "seed",
            "split_path",
            "rgb_run",
            "rgb_evaluation",
            "label_root",
            "label_manifest",
            "iterations",
            "crop",
            "memory_policy",
        ):
            self.assertEqual(control[key], candidate[key])
        self.assertEqual(candidate["context_fraction"], 0.5)
        self.assertEqual(
            candidate["preflight_mode_counts"],
            {"native_crop": 4, "full_frame_context": 4},
        )
        self.assertIn(
            "paired_control_preflight", candidate["seed_aware_initialization"]
        )


if __name__ == "__main__":
    unittest.main()
