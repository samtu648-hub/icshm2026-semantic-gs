import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from summarize_rgb_transient import gate
from finalize_rgb_transient_audit import visual_gate


class PipelineTests(unittest.TestCase):
    def test_visual_gate_requires_coverage_and_no_erosion(self):
        names = ["054.png", "140.png", "151.png"]
        rows = [{"name": name, "candidate_vs_control": "improved",
                 "candidate_vs_historical": "unchanged_or_uncertain",
                 "structure_erosion": False, "observations": "Native paired observation"}
                for name in names]
        self.assertTrue(visual_gate({"views": rows}, names)["passes"])
        rows[0]["structure_erosion"] = True
        self.assertFalse(visual_gate({"views": rows}, names)["passes"])
        with self.assertRaises(ValueError):
            visual_gate({"views": rows[:2]}, names)

    def test_retention_is_required_even_with_primary_gain(self):
        config = json.loads((REPO / "experiments/runs/RGB-092-TRANSIENT/config.yaml").read_text())["evaluation"]
        old = dict(PSNR=20., SSIM=.85, LPIPS=.3, worst10_PSNR=17., challenge_PSNR=21.)
        candidate = {**old, "PSNR": 20.1, "SSIM": .848}
        self.assertFalse(gate(candidate, old, config, True)["passes"])
        candidate["SSIM"] = .85
        self.assertTrue(gate(candidate, old, config, True)["passes"])
        self.assertFalse(gate(old, old, config, True)["passes"])
        self.assertTrue(gate(old, old, config, False)["passes"])

    def test_control_candidate_have_exact_single_switch(self):
        configs = [json.loads((REPO / f"experiments/runs/{name}/config.yaml").read_text())
                   for name in ["RGB-091-CTRL", "RGB-092-TRANSIENT"]]
        for cfg in configs:
            cfg.pop("experiment_id")
            cfg.pop("title")
        self.assertFalse(configs[0]["transient"]["enabled"])
        configs[0]["transient"]["enabled"] = True
        self.assertEqual(configs[0], configs[1])

    def test_sem180_does_not_open_models_or_labels_when_rgb_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audit = root / "rgb_audit.json"
            audit.write_text(json.dumps({"numeric_gate_passes": True, "visual_gate_passes": True,
                                         "SEM180_allowed": False}))
            result = subprocess.run([sys.executable, str(REPO / "scripts/run_sem180_rgb_input_audit.py"),
                      "--output-root", str(root), "--rgb-audit", str(audit)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = json.loads((root / "r2/sem_180/input_audit/full-20260906-r1/result.json").read_text())
            self.assertEqual(saved["semantic_labels_opened"], 0)
            self.assertEqual(saved["new_semantic_predictions"], 0)


if __name__ == "__main__":
    unittest.main()
