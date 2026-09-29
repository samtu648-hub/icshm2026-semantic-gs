import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EID = "SEM-380-RGB148-GT-REGION-RGB-ALL300"


class Sem380ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (ROOT / "experiments/runs" / EID / "config.yaml").read_text()
        )

    def test_registered_single_variable(self):
        self.assertEqual(self.config["experiment_id"], EID)
        self.assertEqual(self.config["seed"], 20260805)
        self.assertEqual(self.config["iterations"], 6000)
        self.assertEqual(self.config["crop"], 768)
        self.assertEqual(self.config["stride"], 512)
        self.assertEqual(self.config["sampler"], "original uniform SEM378 sampler")
        self.assertFalse(self.config["resize_augmentation"])

    def test_candidate_is_bound_to_rgb148(self):
        self.assertEqual(
            self.config["rgb_run"],
            "outputs/quality/rgb147/region-rgb-r1",
        )
        self.assertEqual(
            self.config["baseline_semantic"],
            "outputs/quality/SEM-378/r1",
        )

    def test_training_runner_does_not_import_cable_sampler(self):
        source = (ROOT / "scripts/run_sem380.py").read_text()
        self.assertNotIn("semantic_cable_sampling", source)
        self.assertNotIn("CableRegionSampler", source)
        self.assertNotIn("training.sampled_crop =", source)


if __name__ == "__main__":
    unittest.main()
