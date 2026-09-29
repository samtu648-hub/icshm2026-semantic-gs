import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from scripts.run_quality_static import verify_smoke
from scripts.queue_quality_static import require_sof_success
from src.quality_protocol import CLASSES, sha256, write_new_json
from src.quality_static_training import TrainingLabels, assert_frozen_renderer, fit_static, median_frequency_weights, semantic_access

ROOT = Path(__file__).resolve().parents[1]


class QualityStaticAccessTests(unittest.TestCase):
    def setUp(self):
        self.split = json.loads((ROOT / "experiments/splits/sem_001_scaffold_static_v1.json").read_text())
        self.oof = json.loads((ROOT / "experiments/splits/sem_005_oof_spatial_v1.json").read_text())
        rgb = json.loads((ROOT / "experiments/splits/rgb_r1_spatial_v1.json").read_text())["primary_fold"]
        self.parent = {"train": rgb["train_rgb_400"], "evaluation": rgb["validation_labeled"]}

    def test_real_partitions_and_honest_shared_rgb_scope(self):
        for partition, count, target in (("development", 219, 74), ("oof-0", 114, 95), ("oof-1", 128, 84), ("oof-2", 177, 40)):
            access = semantic_access(self.split, self.oof, self.parent, partition)
            self.assertEqual((len(access["train"]), len(access["prediction"])), (count, target))
            self.assertFalse(set(access["train"]) & (set(access["prediction"]) | set(access["guard"])))
            self.assertFalse(access["whole_chain_spatial_validation"])
            self.assertEqual(len(access["rgb_training_overlap_with_prediction"]), 0 if partition == "development" else target)

    def test_target_leak_missing_role_and_unregistered_partition_rejected(self):
        bad = copy.deepcopy(self.oof)
        bad["folds"][0]["train"].append(bad["folds"][0]["oof"][0])
        with self.assertRaises(ValueError):
            semantic_access(self.split, bad, self.parent, "oof-0")
        parent = copy.deepcopy(self.parent)
        parent["train"].remove(self.split["semantic_train_labeled"][0])
        with self.assertRaises(ValueError):
            semantic_access(self.split, self.oof, parent, "development")
        with self.assertRaises(ValueError):
            semantic_access(self.split, self.oof, self.parent, "outer-0")

    def test_loss_weights_match_registered_recipe_and_fail_on_missing_class(self):
        np.testing.assert_allclose(median_frequency_weights([1000, 10, 20, 2, 1]), [.05, 1, .5, 5, 10])
        for invalid in ([1, 2, 3, 4, 0], [1, 2, 3, 4, np.nan], [1, 2]):
            with self.assertRaises(ValueError):
                median_frequency_weights(invalid)

    def test_reader_never_uses_global_or_target_counts_and_checks_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label = np.array([[0, 1, 2, 3, 4]], np.uint8)
            Image.fromarray(label).save(root / "train.png")
            manifest = {"class_mapping": {n: i for i, n in enumerate(CLASSES)}, "pixel_counts_train": {"0": 999999},
                        "records": [{"image": "train.png", "label_png_sha256": sha256(root / "train.png"),
                                     "pixel_counts": {str(i): 1 for i in range(5)}}]}
            reader = TrainingLabels(root, ["train.png"], manifest, [5, 1])
            np.testing.assert_array_equal(reader.audit_counts(), np.ones(5))
            for forbidden in ("target.png", "../train.png", "guard.png"):
                with self.assertRaises(ValueError):
                    reader.read(forbidden)
            self.assertEqual(reader.accessed, {"train.png"})
            Image.fromarray(label + 1).save(root / "train.png")
            with self.assertRaises(ValueError):
                reader.read("train.png")

    def test_smoke_is_bound_to_exact_parent_partition_code_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            binding, access, code = {"parent": "scaffold", "seed": 0}, {"train": ["a.png"]}, {"x.py": "hash"}
            (path / "field.pt").write_bytes(b"test checkpoint")
            write_new_json(path / "result.json", {"status": "passed_static_semantic_smoke", "binding": binding,
                "access": access, "code_sha256": code, "artifacts": {"field.pt": sha256(path / "field.pt")}})
            self.assertEqual(verify_smoke(path, binding, access, code), sha256(path / "result.json"))
            for b, a, c in (({"parent": "sof"}, access, code), (binding, {"train": ["b.png"]}, code), (binding, access, {})):
                with self.assertRaises(ValueError):
                    verify_smoke(path, b, a, c)
            (path / "field.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_smoke(path, binding, access, code)

    def test_followup_requires_both_sof_acceptance_events_not_just_exit_status(self):
        good = {"status": "completed_selected_arms_other_arms_not_certified",
                "selected_arms": ["RGB-112-SOF-NVS", "RGB-113-SOF-GEO"],
                "events": [{"name": "arm_completed_not_promoted", "experiment_id": n}
                           for n in ("RGB-112-SOF-NVS", "RGB-113-SOF-GEO")]}
        require_sof_success(good)
        for bad in ({**good, "status": "stopped_stage_a_queue_not_method_rejection"},
                    {**good, "events": good["events"][:1]}, {**good, "selected_arms": ["RGB-111-PGSR"]}):
            with self.assertRaises(ValueError):
                require_sof_success(bad)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch runtime")
class QualityStaticFitTests(unittest.TestCase):
    def setUp(self):
        import torch
        from src.quality_backbone import ModelIdentity
        from src.quality_semantic_fields import PrimitiveSemanticField
        self.torch = torch
        self.renderer = SimpleNamespace(identity=ModelIdentity("pgsr", "a" * 64, "b" * 64),
            field_positions=torch.zeros(5, 3), model=SimpleNamespace(rgb=torch.zeros(5, 3)))
        self.renderer.render_channels = lambda camera, values: values.T[:, None, :]
        self.field = PrimitiveSemanticField(self.renderer)
        self.labels = SimpleNamespace(allowed={"train.png"}, read=lambda n: np.array([[0, 1, 2, 3, 4]], np.uint8))

    def fit(self, steps=12):
        return fit_static(self.renderer, self.field, {"train.png": None}, self.labels, np.ones(5),
                          steps=steps, seed=0, learning_rate=.05, weight_decay=0)

    def test_real_optimizer_reduces_loss_and_preserves_frozen_model(self):
        rgb = self.renderer.model.rgb.clone()
        result = self.fit()
        self.assertEqual(result["steps"], 12)
        self.assertLess(result["last_loss"], result["first_loss"])
        self.assertTrue(self.torch.equal(rgb, self.renderer.model.rgb))
        self.assertIsNone(self.renderer.model.rgb.grad)
        self.assertTrue(self.torch.isfinite(self.field.logits).all())

    def test_nested_rgb_mlp_cannot_be_trainable_or_hold_gradient(self):
        self.renderer.model.mlp = self.torch.nn.Linear(3, 3)
        with self.assertRaises(ValueError):
            self.fit(1)
        self.renderer.model.mlp.requires_grad_(False)
        assert_frozen_renderer(self.renderer)
        self.renderer.model.mlp.weight.grad = self.torch.ones_like(self.renderer.model.mlp.weight)
        with self.assertRaises(ValueError):
            self.fit(1)

    def test_nonfinite_gradient_refuses_optimizer_update(self):
        before = self.field.logits.detach().clone()
        handle = self.field.logits.register_hook(lambda gradient: gradient * float("nan"))
        with self.assertRaisesRegex(ValueError, "gradients"):
            self.fit(1)
        handle.remove()
        self.assertTrue(self.torch.equal(before, self.field.logits))

    def test_wrong_training_camera_role_rejected(self):
        with self.assertRaises(ValueError):
            fit_static(self.renderer, self.field, {"target.png": None}, self.labels, np.ones(5),
                       steps=1, seed=0, learning_rate=.05, weight_decay=0)
