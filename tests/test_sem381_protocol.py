import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EID = "SEM-381-DINOV3-VITB-SFP-ALL300"


class Sem381ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (ROOT / "experiments/runs" / EID / "config.yaml").read_text()
        )

    def test_registered_single_architecture_package(self):
        self.assertEqual(self.config["experiment_id"], EID)
        self.assertEqual(self.config["seed"], 20260805)
        self.assertEqual(self.config["iterations"], 6000)
        self.assertEqual(self.config["crop"], 768)
        self.assertEqual(self.config["stride"], 512)
        self.assertEqual(self.config["sampler"], "original uniform SEM378 sampler")
        self.assertFalse(self.config["resize_augmentation"])
        self.assertEqual(self.config["rgb_run"], "outputs/quality/rgb140/full-r2")
        self.assertEqual(self.config["baseline_semantic"], "outputs/quality/SEM-378/r1")

    def test_dinov3_source_and_frozen_feature_contract(self):
        dino = self.config["dinov3"]
        self.assertEqual(dino["model_dir_env"], "DINOV3_VITB16_DIR")
        self.assertEqual(dino["selected_hidden_states"], [3, 6, 9, 12])
        self.assertEqual(dino["weight_bytes"], 342662192)
        self.assertTrue(dino["frozen"])
        self.assertEqual(self.config["sfp"]["strides"], [4, 8, 16, 32])
        self.assertEqual(self.config["sfp"]["channels"], [192, 384, 768, 1536])
        self.assertEqual(self.config["optimizer"]["dinov3_learning_rate"], 0.0)

    def test_runner_has_no_forbidden_sampler_or_network_loader(self):
        source = (ROOT / "scripts/run_sem381.py").read_text()
        self.assertNotIn("semantic_cable_sampling", source)
        self.assertNotIn("CableRegionSampler", source)
        self.assertNotIn('from_pretrained("facebook/', source)
        self.assertIn(
            "local_files_only=True",
            (ROOT / "src/quality_dinov3_mask2former.py").read_text(),
        )


if __name__ == "__main__":
    unittest.main()
