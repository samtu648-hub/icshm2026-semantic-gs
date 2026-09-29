import json
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.quality_mask2former_final import sampling_plan, enforce_budget, require_approval
from src.quality_mask2former_training import sampled_crop
from src.quality_static_training import semantic_access


class FinalFitTest(unittest.TestCase):
    def test_sampler_matches_existing_native_crop_rng(self):
        names=[f'{i:03d}.png' for i in range(219)]
        rgb=np.zeros((989,1320,3),np.uint8);label=np.zeros((989,1320),np.uint8)
        rng=random.Random(20260805)
        plan=sampling_plan(names,20260805,steps=5)
        for expected in plan:
            _,_,actual=sampled_crop(rng,names,{n:rgb for n in names},{n:label for n in names})
            self.assertEqual(actual,expected)

    def test_final_role_cannot_train_on_development(self):
        split={'semantic_train_labeled':['001.png'],'semantic_validation_labeled':['002.png']}
        parent={'train':['001.png'],'evaluation':['002.png']}
        a=semantic_access(split,{},parent,'development')
        self.assertEqual(a['labels_allowed_during_fit'],['001.png'])
        self.assertFalse(set(a['train'])&set(a['prediction']))
        with self.assertRaises(ValueError):
            semantic_access(split,{},dict(parent,train=['001.png','002.png']),'development')

    def test_budget_includes_other_stages_and_pending_write(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'fit').mkdir();(root/'fit/model').write_bytes(b'x'*60)
            (root/'fresh-1').mkdir();(root/'fresh-1/extra').write_bytes(b'x'*20)
            self.assertEqual(enforce_budget(root,1000,pending_bytes=20,maximum_bytes=100,reserve_bytes=900),80)
            with self.assertRaises(OSError):enforce_budget(root,1000,pending_bytes=21,maximum_bytes=100,reserve_bytes=900)
            with self.assertRaises(OSError):enforce_budget(root,919,pending_bytes=20,maximum_bytes=100,reserve_bytes=900)

    def test_missing_approval_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'approval.json';p.write_text(json.dumps({'decision':'not_approved'}))
            with self.assertRaises(ValueError):require_approval(p,None,None)
