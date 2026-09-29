import copy
import json
from pathlib import Path
import unittest
import tempfile
from src.quality_expert_duration import validate_recipe, check_binding, compare_states

ROOT = Path(__file__).resolve().parents[1]


class DurationContractTests(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((ROOT/'experiments/runs/SEM-331-QUALITY-EXPERTS/config.yaml').read_text())
        self.config = json.loads((ROOT/'experiments/runs/SEM-335-SCAFFOLD-EXPERT-DURATION/config.yaml').read_text())

    def test_only_registered_duration(self):
        validate_recipe(self.config, self.base)

    def test_hyperparameter_drift_refused(self):
        for key in ('learning_rate', 'weight_decay', 'steps'):
            bad = copy.deepcopy(self.config); bad['training'][key] *= 2
            with self.assertRaises(ValueError):
                validate_recipe(bad, self.base)

    def test_resize_seed_and_architecture_drift_refused(self):
        for key, value in (('resolution',[660,495]), ('seed',0), ('pixel_space','official'), ('model',{})):
            bad = copy.deepcopy(self.config); bad[key] = value
            with self.assertRaises(ValueError):
                validate_recipe(bad, self.base)

    def test_development_and_backbone_expansion_refused(self):
        for key, value in (('partitions',['development']), ('allowed_rgb_parents',['RGB-112-SOF-NVS'])):
            bad = copy.deepcopy(self.config); bad[key] = value
            with self.assertRaises(ValueError):
                validate_recipe(bad, self.base)

    def test_binary_and_automatic_promotion_refused(self):
        for key, value in (('expert','deck'), ('automatic_promotion',True), ('automatic_development_fit',True)):
            bad = copy.deepcopy(self.config); bad['study'][key] = value
            with self.assertRaises(ValueError):
                validate_recipe(bad, self.base)

    def test_exact_binding_except_registered_config(self):
        a = {k: [] for k in ('access','class_counts','class_weights','train_label_hashes','runtime')}
        a['binding'] = {'config_sha256':'old', 'seed':2}
        b = copy.deepcopy(a); b['binding']['config_sha256'] = 'new'
        check_binding(b, a)
        b['binding']['seed'] = 3
        with self.assertRaises(ValueError):
            check_binding(b, a)

    def test_runtime_and_train_labels_must_match(self):
        a = {k: [] for k in ('access','class_counts','class_weights','train_label_hashes','runtime')}
        a['binding'] = {'config_sha256':'old'}
        for key in ('runtime','train_label_hashes','class_weights','access'):
            b = copy.deepcopy(a); b[key] = ['changed']
            with self.assertRaises(ValueError):
                check_binding(b, a)

    def test_control_tensor_exactness(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Optional Torch unavailable')
        a = {'weight': torch.tensor([1., 2.])}
        self.assertTrue(compare_states(a, {'weight': a['weight'].clone()})['exact'])
        with self.assertRaises(ValueError):
            compare_states(a, {'weight': torch.tensor([1., 2.00001])})
        with self.assertRaises(ValueError):
            compare_states(a, {'weight': a['weight'].double()})

    def test_fusion_protects_three_other_classes_for_any_fiveclass(self):
        import numpy as np
        from src.quality_fusion import pipeline
        base = np.array([[0,1,2,3,4,0,4]], np.uint8)
        five_a = np.zeros((5,1,7), np.float32); five_a[4] = 1
        five_b = np.zeros_like(five_a); five_b[2] = 1
        p = np.array([[.9,.1,.1,.1,.1,.1,.1]], np.float32)
        thresholds = {'fiveclass_lane': .8, 'cable': .75, 'foundation': .8, 'deck': .98}
        a = pipeline(base, five_a, p, 1-p, thresholds)['deck']
        b = pipeline(base, five_b, p, 1-p, thresholds)['deck']
        for k in (1,2,3):
            np.testing.assert_array_equal(a == k, b == k)
        self.assertTrue(np.any(a != b))

    def test_independent_fusion_reconstruction(self):
        import numpy as np
        from scripts.evaluate_quality_expert_duration import check_fixed_foundation_transition
        static = np.array([[0,1,2,3,4,4]],np.uint8)
        old = np.array([[2,1,0,3,0,4]],np.uint8)
        new = np.array([[2,1,0,3,4,0]],np.uint8)
        p4 = np.array([[0,0,0,0,.8,.1]],np.float32)
        t = {'fiveclass_lane':.8,'cable':.75,'foundation':.8,'deck':.98}
        check_fixed_foundation_transition(static,old,new,p4,t)
        bad = new.copy(); bad[0,1] = 0
        with self.assertRaises(ValueError):
            check_fixed_foundation_transition(static,old,bad,p4,t)
        with self.assertRaises(ValueError):
            check_fixed_foundation_transition(static,old,new,p4,{**t,'cable':None})

    def test_queue_acceptance_rejects_incomplete_or_wrong_prefix(self):
        from scripts.queue_quality_expert_duration import accepted_first
        row = {'status':'completed_expert_fit_pending_independent_audit','trained_steps':6000,
            'prefix_3000':{'exact':True},'binding':{'config_sha256':'cfg'},'access':{'partition':'oof-0'}}
        accepted_first(row,'cfg')
        for field,value in (('trained_steps',3000),('prefix_3000',{'exact':False}),('access',{'partition':'oof-1'})):
            with self.assertRaises(ValueError):
                accepted_first({**row,field:value},'cfg')

    def test_candidate_requires_two_matching_fresh_audits(self):
        from src.quality_protocol import sha256
        from scripts.evaluate_quality_expert_duration import accepted_candidate, CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = root/CONFIG; c.parent.mkdir(parents=True); c.write_text(json.dumps(self.config))
            run,audit = root/'batch/full-oof-0',root/'batch/audit-oof-0'
            run.mkdir(parents=True); audit.mkdir()
            control = {'access':{'partition':'oof-0'},'prediction_views':['001.png']}
            record = {'experiment_id':self.config['experiment_id'],'status':'completed_expert_fit_pending_independent_audit',
                'trained_steps':6000,'prefix_3000':{'exact':True},'binding':{'config_sha256':sha256(c)},
                'control_result_sha256':self.config['study']['controls']['oof-0']['sha256'],
                'access':control['access'],'artifacts':{},'code_sha256':{}}
            (run/'result.json').write_text(json.dumps(record))
            ar = {'status':'completed_independent_expert_replay','binding':record['binding'],'run_result_sha256':sha256(run/'result.json'),
                'fresh_reloads':[{'pid':i,'probabilities_exact':True,'views':['001.png']} for i in (1,2)]}
            (audit/'result.json').write_text(json.dumps(ar))
            accepted_candidate(root,root/'batch','oof-0',self.config,control)
            ar['fresh_reloads'][1]['pid'] = 1
            (audit/'result.json').write_text(json.dumps(ar))
            with self.assertRaises(ValueError):
                accepted_candidate(root,root/'batch','oof-0',self.config,control)

    def test_repeatability_audit_counts_numeric_changes(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Optional Torch unavailable')
        from scripts.audit_quality_expert_reproducibility import compare_repeats
        a = {'x':torch.tensor([1.,2.]),'n':torch.tensor(8)}
        b = {'x':torch.tensor([1.,2.01]),'n':torch.tensor(8)}
        row = compare_repeats(a,b)
        self.assertEqual(row['different_tensors'],1)
        self.assertEqual(row['different_elements'],1)
        self.assertEqual(compare_repeats(a,a)['different_tensors'],0)
        with self.assertRaises(ValueError):
            compare_repeats(a,{'x':torch.tensor([1.,float('nan')]),'n':torch.tensor(8)})


if __name__ == '__main__':
    unittest.main()
