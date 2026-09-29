from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts.evaluate_sem389_confirmation_gate import confirmation_gate
from src.semantic_context_sampling import context_plan


ROOT = Path(__file__).resolve().parents[1]
CONTROL = (
    ROOT / "experiments/runs/SEM-388-RGB150-DEV270-SEED20260919-MEMORY/config.yaml"
)
CANDIDATE = (
    ROOT
    / "experiments/runs/SEM-389-RGB150-DEV270-CONTEXT50-SEED20260919-MEMORY/config.yaml"
)


class ConfirmationProtocolTest(unittest.TestCase):
    def test_control_contract(self) -> None:
        config = json.loads(CONTROL.read_text(encoding="utf-8"))
        self.assertEqual(config["seed"], 20260919)
        self.assertEqual(config["context_fraction"], 0.0)
        self.assertEqual(config["preflight_mode_counts"], {"native_crop": 8})
        self.assertTrue(config["memory_policy"]["swin_activation_recomputation"])

    def test_candidate_contract_and_pairing(self) -> None:
        if not CANDIDATE.exists():
            self.skipTest(
                "Candidate config is bound after the control plan is prepared"
            )
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
        ):
            self.assertEqual(control[key], candidate[key])
        self.assertEqual(candidate["context_fraction"], 0.5)
        self.assertEqual(
            candidate["preflight_mode_counts"],
            {"native_crop": 4, "full_frame_context": 4},
        )
        self.assertEqual(control["memory_policy"], candidate["memory_policy"])

    def test_context_plan_keeps_control_trace(self) -> None:
        control = [
            {
                "step": step,
                "name": f"{step:03d}.png",
                "top": step,
                "left": step + 1,
                "flip": step % 3 == 0,
            }
            for step in range(1, 9)
        ]
        candidate = context_plan(control)
        for source, derived in zip(control, candidate):
            self.assertEqual(source, {key: derived[key] for key in source})
        self.assertEqual(
            [row["mode"] for row in candidate],
            ["native_crop", "full_frame_context"] * 4,
        )

    def test_gate_uses_overall_classes_and_group_five_class(self) -> None:
        def result(seed: int) -> dict:
            holdout = {
                "five_class_delta_pp": 0.04,
                "foreground_delta_pp": 0.05,
                "boundary_f1_delta_pp": 0.01,
                "per_class_delta_pp": {
                    "background": 0.0,
                    "deck": 0.1,
                    "stay_cable": 0.0,
                    "tower": -0.19,
                    "foundation": 0.1,
                },
            }
            group = {"five_class_delta_pp": -0.19}
            return {
                "seed": seed,
                "comparisons": {
                    "holdout": holdout,
                    **{
                        name: group
                        for name in (
                            "regular-local",
                            "oblique-local",
                            "distant-local",
                        )
                    },
                },
            }

        gate = confirmation_gate([result(20260805), result(20260919)])
        self.assertTrue(gate["quantitative_passed"])
        failed = result(20260919)
        failed["comparisons"]["holdout"]["per_class_delta_pp"]["tower"] = -0.21
        self.assertFalse(
            confirmation_gate([result(20260805), failed])["quantitative_passed"]
        )


if __name__ == "__main__":
    unittest.main()
