import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.quality_expert_pair import validate_recipe, save_endpoint
from scripts.queue_quality_expert_pair import accepted_first

ROOT = Path(__file__).resolve().parents[1]


class PairedDurationTests(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((ROOT/'experiments/runs/SEM-331-QUALITY-EXPERTS/config.yaml').read_text())
        self.cfg = json.loads((ROOT/'experiments/runs/SEM-336-SCAFFOLD-PAIRED-DURATION/config.yaml').read_text())

    def test_fixed_paired_recipe(self):
        validate_recipe(self.cfg,self.base)
        for endpoints in ([3000],[6000],[3000,12000]):
            cfg=copy.deepcopy(self.cfg); cfg['study']['endpoints']=endpoints
            with self.assertRaises(ValueError): validate_recipe(cfg,self.base)

    def test_no_extra_training_changes(self):
        for key,value in (('learning_rate',.001),('steps',12000),('amp','fp32')):
            cfg=copy.deepcopy(self.cfg); cfg['training'][key]=value
            with self.assertRaises(ValueError): validate_recipe(cfg,self.base)

    def test_pair_admission_requires_both_endpoints_and_rng_guard(self):
        row={'status':'completed_expert_fit_pending_independent_audit','trained_steps':6000,
            'paired_control_3000':{'rng_preserved':True},'endpoints':[3000,6000],
            'binding':{'config_sha256':'cfg'},'access':{'partition':'oof-0'}}
        accepted_first(row,'cfg')
        for key,value in (('endpoints',[6000]),('paired_control_3000',{'rng_preserved':False}),('trained_steps',3000)):
            with self.assertRaises(ValueError): accepted_first({**row,key:value},'cfg')

    def test_saving_endpoint_preserves_rng_and_detaches_checkpoint(self):
        try: import torch
        except ImportError: self.skipTest('Optional Torch unavailable')
        with tempfile.TemporaryDirectory() as tmp, patch('torch.cuda.get_rng_state_all',return_value=[]):
            model=torch.nn.Linear(2,2)
            rng=torch.get_rng_state().clone()
            expected={k:v.detach().clone() for k,v in model.state_dict().items()}
            path=Path(tmp)/'expert-3000.pt'
            proof=save_endpoint(model,{'config':'paired'},{'partition':'oof-0'},3000,path)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(proof['rng_preserved'])
            with torch.no_grad(): model.weight.add_(1)
            payload=torch.load(path,map_location='cpu',weights_only=True)
            self.assertEqual(payload['steps'],3000)
            self.assertTrue(all(torch.equal(payload['state_dict'][k],v) for k,v in expected.items()))

    def test_paired_independent_fusion_guard(self):
        import numpy as np
        from scripts.evaluate_quality_expert_pair import check_fixed_foundation_transition
        static=np.array([[0,1,2,3,4]],np.uint8)
        old=np.array([[2,1,0,3,0]],np.uint8)
        new=np.array([[2,1,0,3,4]],np.uint8)
        check_fixed_foundation_transition(static,old,new,np.ones((1,5),np.float32),
            {'fiveclass_lane':.8,'cable':.75,'foundation':.8,'deck':.98})


    def test_candidate_requires_two_fresh_audits_per_endpoint(self):
        from src.quality_protocol import sha256
        from scripts.evaluate_quality_expert_pair import accepted_candidate, CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = root/CONFIG; c.parent.mkdir(parents=True); c.write_text(json.dumps(self.cfg))
            run,audit = root/'batch/full-oof-0',root/'batch/audit-oof-0'
            run.mkdir(parents=True); audit.mkdir()
            control = {'access':{'partition':'oof-0'},'prediction_views':['001.png']}
            record = {'experiment_id':self.cfg['experiment_id'],'status':'completed_expert_fit_pending_independent_audit',
                'trained_steps':6000,'paired_control_3000':{'rng_preserved':True},'endpoints':[3000,6000],'binding':{'config_sha256':sha256(c)},
                'historical_reference_result_sha256':self.cfg['study']['controls']['oof-0']['sha256'],
                'access':control['access'],'artifacts':{},'code_sha256':{}}
            (run/'result.json').write_text(json.dumps(record))
            ar = {'status':'completed_independent_expert_replay','binding':record['binding'],'run_result_sha256':sha256(run/'result.json'),
                'fresh_reloads':[{'pid':i+1,'steps':step,'probabilities_exact':True,'views':['001.png']} for i,step in enumerate((3000,3000,6000,6000))]}
            (audit/'result.json').write_text(json.dumps(ar))
            accepted_candidate(root,root/'batch','oof-0',self.cfg,control)
            ar['fresh_reloads'][1]['pid'] = 1
            (audit/'result.json').write_text(json.dumps(ar))
            with self.assertRaises(ValueError):
                accepted_candidate(root,root/'batch','oof-0',self.cfg,control)


if __name__=='__main__': unittest.main()
