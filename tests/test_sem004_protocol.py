"""Dependency-free protocol checks for the diagnostic-only SEM-004 run."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestSEM004Protocol(unittest.TestCase):
    def test_config_is_diagnostic_only_and_keeps_sem001_boundaries(self) -> None:
        config = json.loads((ROOT / "experiments/runs/SEM-004-DIAG/config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["experiment_id"], "SEM-004-DIAG")
        self.assertEqual(config["status"], "diagnostic_only_not_a_semantic_candidate")
        for key in ("trains_new_parameters", "updates_RGB", "updates_semantic_parameters", "uses_unlabeled_RGB", "uses_pseudo_labels"):
            self.assertFalse(config[key], key)
        self.assertEqual(config["data"]["training_images"], 219)
        self.assertEqual(config["data"]["validation_images"], 74)
        self.assertEqual(config["frozen_rgb"]["pose_convention"], "CAMERA_FROM_WORLD")
        self.assertEqual(config["frozen_semantic"]["class_names"][2], "stay_cable")

    def test_adapter_exposes_diagnostic_only_dominant_contributor_output(self) -> None:
        source = (ROOT / "src/scaffold_visibility_supervision.py").read_text(encoding="utf-8")
        cuda = (ROOT / "adapters/scaffold_gs/visibility_rasterizer/visibility_rasterizer.cu").read_text(encoding="utf-8")
        self.assertIn("render_frozen_contribution_diagnostics", source)
        self.assertIn("diagnostic_stats", source)
        self.assertIn("dominant_offset_index", source)
        self.assertIn("dominant_point_index", cuda)
        self.assertIn("visibility_diagnostic_stats_cuda", cuda)

    def test_s3_configs_freeze_one_sparse_direction_variable(self) -> None:
        configs = [
            json.loads((ROOT / f"experiments/runs/{experiment}/config.yaml").read_text(encoding="utf-8"))
            for experiment in ("SEM-040-EQ", "SEM-041-PF", "SEM-004")
        ]
        for config in configs:
            self.assertEqual(config["residual"]["enabled_anchor_count"], 7069)
            self.assertEqual(config["residual"]["trainable_coefficients_per_anchor"], 3)
            self.assertEqual(config["residual"]["view_direction"], "normalize(anchor_position - official_camera_center)")
            self.assertEqual(config["residual"]["directly_unchanged_logits"], ["deck", "tower", "foundation"])
            self.assertEqual(config["training"]["trainable_parameter_allowlist"], ["direction_coefficients[7069,3]"])
            self.assertEqual(config["frozen_rgb"]["pose_convention"], "CAMERA_FROM_WORLD")
            self.assertFalse(config["data"]["uses_unlabeled_rgb"])
            self.assertFalse(config["data"]["uses_validation_for_hyperparameter_or_mask_selection"])
        self.assertEqual(configs[-1]["full"]["iterations"], 10000)

    def test_s3_implementation_protects_static_semantics_and_rgb(self) -> None:
        source = (ROOT / "src/scaffold_directional_residual.py").read_text(encoding="utf-8")
        runner = (ROOT / "scripts/run_sem004_residual.py").read_text(encoding="utf-8")
        self.assertIn("anchor_positions[global_indices] - camera_center", source)
        self.assertIn("base[:, BACKGROUND] - delta", source)
        self.assertIn("base[:, STAY_CABLE] + delta", source)
        self.assertIn("direction_coefficients", runner)
        self.assertIn("Frozen SEM-001 static logits changed", runner)
        self.assertIn("RGB_isolation_evidence", runner)


if __name__ == "__main__":
    unittest.main()
