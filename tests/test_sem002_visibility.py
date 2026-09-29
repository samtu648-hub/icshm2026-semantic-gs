import json
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Keep the repository's lightweight health check dependency-free.
    torch = None


ROOT = Path(__file__).resolve().parents[1]


class Sem002VisibilityTest(unittest.TestCase):
    def test_configs_freeze_training_only_gate_protocol(self):
        diagnostic = json.loads((ROOT / "experiments/runs/SEM-020-DIAG/config.yaml").read_text(encoding="utf-8"))
        cumulative = json.loads((ROOT / "experiments/runs/SEM-021/config.yaml").read_text(encoding="utf-8"))
        relative = json.loads((ROOT / "experiments/runs/SEM-022/config.yaml").read_text(encoding="utf-8"))
        entries = {item["name"]: item["threshold"] for item in diagnostic["training_side_threshold_diagnostic"]["gates"]}
        self.assertEqual(entries, {"cumulative": 0.95, "relative_max": 0.05})
        self.assertEqual(cumulative["visibility_gate"]["threshold"], entries["cumulative"])
        self.assertEqual(relative["visibility_gate"]["threshold"], entries["relative_max"])
        for config in (cumulative, relative):
            self.assertEqual(config["data"]["split_id"], "sem_001_scaffold_static_v1")
            self.assertFalse(config["data"]["unlabeled_rgb_in_semantic_supervision"])
            self.assertFalse(config["data"]["pseudo_labels"])
            self.assertEqual(config["frozen_rgb"]["pose_convention"], "CAMERA_FROM_WORLD")

    @unittest.skipIf(torch is None, "torch is intentionally absent from the lightweight repository-health interpreter")
    def test_pinned_geometry_workspace_layout_decodes_typed_views(self):
        from src.scaffold_visibility_supervision import _geometry_offsets, _typed_workspace_view

        # The opaque CUDA buffer layout is address-aligned, not simply packed.
        # A CPU tensor exercises the byte offsets without a GPU test runner.
        buffer = torch.empty(4096, dtype=torch.uint8)
        offsets = _geometry_offsets(buffer, point_count=7)
        means = _typed_workspace_view(buffer, offsets["means2d"], (7, 2), torch.float32)
        conic = _typed_workspace_view(buffer, offsets["conic_opacity"], (7, 4), torch.float32)
        self.assertEqual(means.shape, (7, 2))
        self.assertEqual(conic.shape, (7, 4))
        self.assertEqual(means.data_ptr() % torch.empty((), dtype=torch.float32).element_size(), 0)
        self.assertEqual(conic.data_ptr() % torch.empty((), dtype=torch.float32).element_size(), 0)

    def test_visibility_adapter_is_isolated_from_rgb_renderer_source(self):
        source = (ROOT / "adapters/scaffold_gs/visibility_rasterizer/visibility_rasterizer.cu").read_text(encoding="utf-8")
        self.assertIn("contribution(float2 center", source)
        self.assertIn("gate_mode == 1 ? prefix < cumulative_limit", source)
        self.assertIn("r >= threshold * peak", source)
        self.assertIn("grad_probabilities", source)
        self.assertNotIn("means3D", source)
