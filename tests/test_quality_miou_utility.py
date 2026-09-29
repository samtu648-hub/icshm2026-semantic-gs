import copy
import json
from pathlib import Path
import unittest
import numpy as np
from src.quality_miou_utility import marginal_utility_table,reference_utility_table
from scripts.run_quality_miou_utility_selector import matched_parent
ROOT=Path(__file__).resolve().parents[1]
class MiouUtilityTests(unittest.TestCase):
    def test_all_transitions_match_rational_matrix_edits(self):
        for m in [np.array([[100,2,7,3,4],[8,40,1,1,2],[9,5,80,3,1],[2,2,1,30,1],[4,1,2,3,10]],dtype=np.int64),np.full((5,5),10000000,dtype=np.int64)]:
            np.testing.assert_allclose(marginal_utility_table(m),reference_utility_table(m),atol=1e-12,rtol=1e-12)
    def test_both_wrong_can_have_nonzero_metric_value(self):
        m=np.diag([100,50,30,20,10]);m[0,1]=4;m[0,4]=2
        table=marginal_utility_table(m)
        self.assertNotEqual(table[0,1,4],0)
        self.assertTrue(np.isnan(table[1,0,2]))
        self.assertEqual(table[0,1,1],0)
    def test_absent_class_requires_registered_fallback(self):
        m=np.eye(5,dtype=np.int64);m[4,4]=0
        with self.assertRaisesRegex(ValueError,'support'):marginal_utility_table(m)
    def test_changes_outside_target_are_rejected(self):
        c=json.loads((ROOT/'experiments/runs/SEM-357-MIOU-UTILITY-SELECTOR/config.yaml').read_text());c['learner']['ridge']=.1
        with self.assertRaisesRegex(ValueError,'learner'):matched_parent(c)
    def test_complete_same_recipe_and_fixed_new_target(self):
        from unittest.mock import patch
        import scripts.run_quality_miou_utility_selector as runner
        c=json.loads((ROOT/'experiments/runs/SEM-357-MIOU-UTILITY-SELECTOR/config.yaml').read_text());original=runner.read
        def local(path):return {'files':{}} if str(path).endswith('predictions_sealed.json') else original(path)
        with patch.object(runner,'read',side_effect=local),patch.object(runner,'verify_hashes'):matched_parent(c)
