import copy
import json
from pathlib import Path
import random
import unittest
import numpy as np
from src.quality_mask2former_training import CONFIG, poly_factor, sampled_crop, stability_passes, validate_training_config
from src.semantic_cable_binary import random_native_pair

ROOT = Path(__file__).resolve().parents[1]


class Mask2FormerTrainingContracts(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / CONFIG).read_text())

    def test_registered_recipe(self):
        validate_training_config(self.cfg)

    def test_no_capacity_budget_or_resolution_reduction(self):
        for key, value in [("steps", 3000), ("crop", 512), ("batch_size", 2), ("activation_checkpoint", True)]:
            cfg = copy.deepcopy(self.cfg); cfg["training"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_training_config(cfg)

    def test_no_hidden_lr_or_checkpoint_selection(self):
        for key, value in [("learning_rate", .001), ("poly_power", .5), ("stage_checkpoints", [1000, 3000])]:
            cfg = copy.deepcopy(self.cfg); cfg["training"][key] = value
            with self.assertRaises(ValueError):
                validate_training_config(cfg)

    def test_poly_uses_full_budget_and_successful_steps(self):
        self.assertEqual(poly_factor(0), 1.)
        self.assertEqual(poly_factor(6000), 0.)
        self.assertAlmostEqual(poly_factor(3000), .5**.9)
        self.assertGreater(poly_factor(128), .98)
        for v in [-1, 6001, .5]:
            with self.assertRaises(ValueError):
                poly_factor(v)

    def test_sampling_sequence_exact_b2_order(self):
        names = ["a.png", "b.png"]
        image = np.arange(989*1320*3, dtype=np.uint8).reshape(989, 1320, 3)
        target = np.zeros((989, 1320), np.uint8)
        images, targets = {n: image for n in names}, {n: target for n in names}
        a, b = random.Random(20260805), random.Random(20260805)
        for _ in range(4):
            x, y, trace = sampled_crop(a, names, images, targets)
            name = names[b.randrange(len(names))]
            xr, yr = random_native_pair(images[name], targets[name], 768, b)
            self.assertEqual(trace["name"], name)
            np.testing.assert_array_equal(x, xr); np.testing.assert_array_equal(y, yr)
            self.assertEqual(a.getstate(), b.getstate())

    def valid_stability(self):
        return {"status": "completed_mask2former_stability_pending_replay", "trained_steps": 128,
                "partition": "oof-0", "unique_sampled_views": 64, "sampled_crop_class_counts": [1]*5,
                "amp_skips": 1}

    def test_stability_is_not_two_step_or_low_coverage_pass(self):
        stability_passes(self.valid_stability(), self.cfg)
        for key, value in [("trained_steps", 2), ("unique_sampled_views", 1), ("partition", "oof-1"),
                           ("sampled_crop_class_counts", [1, 1, 1, 1, 0]), ("amp_skips", 33)]:
            r = self.valid_stability(); r[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                stability_passes(r, self.cfg)

    def test_stability_bound_is_fixed(self):
        r = self.valid_stability(); r["amp_skips"] = 32
        stability_passes(r, self.cfg)
        self.assertFalse(self.cfg["stability"]["quality_gate"])


if __name__ == "__main__":
    unittest.main()
