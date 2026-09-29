import importlib
import inspect
import json
from pathlib import Path
import unittest

from scripts.run_sem387_feature_splatting import (
    EID,
    make_sampling_plan,
    normalized_parent_samples,
    prepare,
)
from src.quality_feature_splatting import render_signed_features


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"


class Sem387ProtocolTests(unittest.TestCase):
    def torch(self):
        try:
            return importlib.import_module("torch")
        except ImportError:
            self.skipTest("optional torch runtime unavailable")

    def test_registered_recipe_is_fixed(self):
        config = json.loads(CONFIG.read_text())
        self.assertEqual(config["arms"], ["label_only", "dinov3_distilled"])
        self.assertEqual(config["iterations"], 6000)
        self.assertEqual(config["analysis_checkpoint"], 3000)
        self.assertEqual(config["primary_checkpoint"], 6000)
        self.assertEqual(config["feature_splatting"]["latent_dim"], 32)
        self.assertEqual(config["feature_splatting"]["latent_learning_rate"], 0.0025)
        self.assertEqual(config["feature_splatting"]["decoder_learning_rate"], 0.001)
        self.assertEqual(config["loss"]["distillation_weight"], 0.01)
        self.assertEqual(config["dinov3"]["selected_output"].split()[0], "normalized")
        self.assertEqual(
            config["dinov3"]["source_manifest_sha256"],
            "cbd11fe8edd8083efad8979ff4151baabd341631ed222fa5bd1ea64edf3bc2e3",
        )
        self.assertEqual(config["rgb_source"], "outputs/quality/rgb140/source-r1")
        self.assertEqual(config["preflight"]["successful_updates_per_arm"], 8)

    def test_registry_contains_one_completed_entry(self):
        registry = json.loads((ROOT / "experiments/registry.yaml").read_text())
        rows = [row for row in registry["experiments"] if row["experiment_id"] == EID]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "completed_dev30_not_promoted")
        self.assertEqual(
            rows[0]["holdout_status"],
            "opened_only_after_both_arm_6000_checkpoints_were_sealed",
        )
        self.assertEqual(
            rows[0]["metrics"]["holdout"]["dinov3_distilled_five_class_miou"],
            0.8750121165492395,
        )

    def test_unlabeled_plan_is_deterministic_and_complete(self):
        names = [f"{index:03}.png" for index in range(301, 401)]
        first = make_sampling_plan(names, 20260806, 6000, 768)
        self.assertEqual(first, make_sampling_plan(names, 20260806, 6000, 768))
        self.assertEqual({row["name"] for row in first}, set(names))
        self.assertTrue(all(row["crop"] == 768 for row in first))

    @unittest.skipUnless((ROOT / "outputs/quality/SEM-384/r1/sampling-plan.json").is_file(), "optional saved run artifact is not distributed")
    def test_parent_plan_normalization_does_not_change_trace(self):
        parent = json.loads(
            (ROOT / "outputs/quality/SEM-384/r1/sampling-plan.json").read_text()
        )
        normalized = normalized_parent_samples(parent, 768)
        self.assertEqual(len(normalized), 6000)
        for source, actual in zip(parent["samples"], normalized):
            self.assertEqual(
                {key: actual[key] for key in ("step", "name", "top", "left", "flip")},
                source,
            )

    def test_prepare_only_hashes_training_rgb_and_labels(self):
        source = inspect.getsource(prepare)
        self.assertIn(
            'verify_hashes(rgb_evaluation / "images", training_rgb_hashes)', source
        )
        self.assertIn("label_hashes = {name: sha256(label_root / name)", source)
        self.assertNotIn(
            'verify_hashes(rgb_evaluation / "images", rgb_eval_result["image_hashes"])',
            source,
        )
        self.assertIn('"holdout_rgb_hashes_read": 0', source)
        self.assertIn('"holdout_label_hashes_read": 0', source)

    def test_signed_chunk_render_reassembles_exact_weighted_sum(self):
        torch = self.torch()
        weights = torch.tensor([0.1, 0.2, 0.3])
        values = torch.tensor(
            [[-2.0, 0.0, 2.0, 1.0], [1.0, -1.0, 0.5, 3.0], [0.0, 2.0, -0.5, -4.0]],
            requires_grad=True,
        )

        class Model:
            get_xyz = torch.zeros((3, 3))

        def native_render(view, model, pipe, background, override_color):
            image = torch.einsum("n,nc->c", weights, override_color)[:, None, None]
            return {"render": image.clamp(0, 1)}

        rendered, alpha = render_signed_features(
            native_render, None, Model(), None, values, torch.zeros(3)
        )
        expected = torch.einsum("n,nc->c", weights, values)[:, None, None]
        torch.testing.assert_close(rendered, expected, atol=2e-7, rtol=1e-6)
        self.assertAlmostEqual(float(alpha), float(weights.sum()), places=6)
        rendered.sum().backward()
        torch.testing.assert_close(values.grad, weights[:, None].expand_as(values))


if __name__ == "__main__":
    unittest.main()
