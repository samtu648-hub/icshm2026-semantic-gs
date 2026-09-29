"""Protect the matched experiment and weight-only scientific change."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts.run_quality_selector_weight_ablation import matched_parent
ROOT=Path(__file__).resolve().parents[1]
class WeightAblationTests(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((ROOT/'experiments/runs/SEM-356-SELECTOR-WEIGHT-ABLATION/config.yaml').read_text())
    def test_matched_roles_and_policy(self):
        # Artifact hashing is covered by actual preflight; keep unit test portable.
        import scripts.run_quality_selector_weight_ablation as runner
        real=runner.read
        def local(p):
            return {'files':{}} if str(p).endswith('predictions_sealed.json') else real(p)
        with patch.object(runner,'read',side_effect=local),patch.object(runner,'verify_hashes'):
            matched_parent(self.c)
    def test_reject_extra_scientific_changes(self):
        for key in ('seed','guard_radius'):
            c=copy.deepcopy(self.c);c[key]+=1
            with self.assertRaisesRegex(ValueError,'scientific'):
                matched_parent(c)
        c=copy.deepcopy(self.c);c['learner']['ridge']=.1
        with self.assertRaisesRegex(ValueError,'learner'):
            matched_parent(c)
    def test_reject_wrong_weight_policy(self):
        self.c['learner']['class_weight']='inverse frequency'
        with self.assertRaisesRegex(ValueError,'weight policy'):
            matched_parent(self.c)
