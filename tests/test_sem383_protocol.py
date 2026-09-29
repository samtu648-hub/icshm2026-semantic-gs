import json
import random
import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts.replay_verify_sem383 import reset_model_rng


ROOT = Path(__file__).resolve().parents[1]
EID = "SEM-383-DINOV3-VITB-SPATIAL-SFP-ALL300"


class Sem383ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (ROOT / "experiments/runs" / EID / "config.yaml").read_text()
        )

    def test_single_spatial_adapter_variable(self):
        self.assertEqual(self.config["experiment_id"], EID)
        self.assertEqual(
            self.config["parent_experiment"],
            "SEM-382-DINOV3-VITB-SFP-LR1E4-ALL300",
        )
        self.assertEqual(self.config["optimizer"]["sfp_learning_rate"], 1e-4)
        self.assertEqual(
            self.config["optimizer"]["spatial_adapter_learning_rate"], 1e-4
        )
        self.assertEqual(self.config["optimizer"]["decoder_learning_rate"], 1e-4)
        self.assertEqual(self.config["optimizer"]["dinov3_learning_rate"], 0.0)

    def test_shapes_counts_sampling_and_endpoints_are_fixed(self):
        adapter = self.config["spatial_adapter"]
        self.assertEqual(adapter["output_strides"], [4, 8, 16, 32])
        self.assertEqual(adapter["output_channels"], [192, 384, 768, 1536])
        counts = self.config["expected_parameter_counts"]
        self.assertEqual(counts["spatial_adapter"], 514048)
        self.assertEqual(counts["sfp"], 4356544)
        self.assertEqual(counts["decoder"], 20250310)
        self.assertEqual(counts["frozen_dinov3"], 85660416)
        self.assertEqual(
            self.config["expected_initial_state"]["sampling_plan_sha256"],
            "732cf8aabe69e809dbb7846588b5593daa7c1abf980ca8e0e672228d0548c8d0",
        )
        self.assertEqual(self.config["analysis_checkpoint"], 3000)
        self.assertEqual(self.config["primary_checkpoint"], 6000)

    def test_honest_scope_and_forbidden_changes(self):
        scope = self.config["spatial_adapter"]["honest_scope"]
        self.assertIn("not Meta DINOv3 Adapter", scope)
        self.assertIn("not ViT-Adapter", scope)
        source = (ROOT / "src/quality_dinov3_spatial_sfp.py").read_text()
        self.assertIn("OFFICIAL_ADAPTER_COMMIT", source)
        self.assertNotIn("MSDeformAttn(", source)
        self.assertIn("self.dinov3.eval()", source)

    def test_input_diagnostic_is_no_training(self):
        diagnostic = self.config["input_diagnostic"]
        self.assertEqual(diagnostic["views"], ["311.png", "350.png", "356.png"])
        self.assertFalse(diagnostic["training"])
        self.assertIn("no GT", diagnostic["interpretation"])

    def test_verifier_replay_resets_all_model_rngs(self):
        numpy_module = Mock()
        torch_module = Mock()
        torch_module.cuda.is_available.return_value = True
        seed = self.config["seed"]

        previous_state = random.getstate()
        try:
            reset_model_rng(seed, numpy_module, torch_module)
            python_value = random.random()
        finally:
            random.setstate(previous_state)

        self.assertEqual(python_value, random.Random(seed).random())
        numpy_module.random.seed.assert_called_once_with(seed)
        torch_module.manual_seed.assert_called_once_with(seed)
        torch_module.cuda.manual_seed_all.assert_called_once_with(seed)


if __name__ == "__main__":
    unittest.main()
