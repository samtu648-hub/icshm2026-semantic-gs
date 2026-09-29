import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EID = "SEM-382-DINOV3-VITB-SFP-LR1E4-ALL300"


class Sem382ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (ROOT / "experiments/runs" / EID / "config.yaml").read_text()
        )

    def test_single_learning_rate_variable(self):
        self.assertEqual(self.config["experiment_id"], EID)
        self.assertEqual(
            self.config["parent_experiment"],
            "SEM-381-DINOV3-VITB-SFP-ALL300",
        )
        self.assertEqual(self.config["optimizer"]["sfp_learning_rate"], 1e-4)
        self.assertEqual(self.config["optimizer"]["decoder_learning_rate"], 1e-4)
        self.assertEqual(self.config["optimizer"]["dinov3_learning_rate"], 0.0)
        self.assertTrue(self.config["sfp"]["include_normalization_parameters"])

    def test_initialization_sampling_and_endpoints_are_fixed(self):
        expected = self.config["expected_initial_state"]
        self.assertEqual(
            expected["trainable_sha256"],
            "cc677af8686be7df215b0d8b1a3f82efe6ff114d23d9c6b5fd543d538b86b995",
        )
        self.assertEqual(
            expected["sampling_plan_sha256"],
            "732cf8aabe69e809dbb7846588b5593daa7c1abf980ca8e0e672228d0548c8d0",
        )
        self.assertEqual(self.config["stage_checkpoints"], [3000, 6000])
        self.assertEqual(self.config["analysis_checkpoint"], 3000)
        self.assertEqual(self.config["primary_checkpoint"], 6000)

    def test_parameter_contract_and_forbidden_changes(self):
        counts = self.config["expected_parameter_counts"]
        self.assertEqual(counts["sfp"], 3842496)
        self.assertEqual(counts["decoder"], 20250310)
        self.assertEqual(counts["frozen_dinov3"], 85660416)
        self.assertEqual(counts["total"], 109753222)
        source = (ROOT / "scripts/run_sem382.py").read_text()
        self.assertIn("BACKBONE_MULTIPLIER = 1.0", source)
        self.assertIn("sfp_norm", source)
        self.assertNotIn("CableRegionSampler", source)
        self.assertNotIn('from_pretrained("facebook/', source)


if __name__ == "__main__":
    unittest.main()
