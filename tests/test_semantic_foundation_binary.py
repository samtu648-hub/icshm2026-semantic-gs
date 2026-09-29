import unittest
import numpy as np
from src.semantic_foundation_binary import (fuse_foundation, target_labels,
    training_plan, validate_fold, quality_gates)


class FoundationContracts(unittest.TestCase):
    def test_disabled_and_both_directions(self):
        base = np.array([[0,4,1,2,3,0]], np.uint8)
        p = np.array([[1,0,0,1,1,.5]], np.float32)
        np.testing.assert_array_equal(fuse_foundation(base,p,strength=0),base)
        np.testing.assert_array_equal(fuse_foundation(base,p),[[4,0,1,2,3,4]])
        np.testing.assert_array_equal(base,[[0,4,1,2,3,0]])

    def test_no_protection_dilation_or_old_rejection(self):
        base = np.array([[3,0,4]], np.uint8)
        np.testing.assert_array_equal(fuse_foundation(base,np.ones((1,3))),[[3,4,4]])

    def test_target_class4_only(self):
        labels=np.arange(5,dtype=np.uint8)[None]
        np.testing.assert_array_equal(target_labels(labels,'binary_candidate'),[[0,0,0,0,1]])
        np.testing.assert_array_equal(target_labels(labels,'five_control'),labels)

    def test_probability_and_strength_validation(self):
        base=np.zeros((2,2),np.uint8)
        for p in [np.zeros((1,2)),np.full((2,2),np.nan),np.full((2,2),1.01)]:
            with self.assertRaises(ValueError): fuse_foundation(base,p)
        with self.assertRaises(ValueError): fuse_foundation(base,np.zeros((2,2)),strength=.5)

    def test_shared_plan_deterministic(self):
        a=training_plan(['a','b'],12,0,30,960)
        self.assertEqual(a,training_plan(['a','b'],12,0,30,960))
        self.assertTrue(all(0<=r['top']<=29 and 0<=r['left']<=360 for r in a))
        with self.assertRaises(ValueError): training_plan(['a','a'],0,0,2,960)

    def test_leak_and_guard_rejection(self):
        center={'a':[0,0,0],'b':[2,0,0],'c':[2.1,0,0]}
        f={'train':['a'],'oof':['b'],'guard':['c']}
        validate_fold(f,list(center),center,1.)
        with self.assertRaises(ValueError): validate_fold({**f,'train':['a','b']},list(center),center,1.)
        with self.assertRaises(ValueError): validate_fold({**f,'train':['c'],'guard':['a']},list(center),center,1.)

    def test_quality_cannot_trade_away_recall(self):
        def m(i,p,r): return {'per_class':{'foundation':{'iou':i,'precision':p,'recall':r}},'mIoU_with_background':.8}
        g={'foundation_iou_gain':.01,'foundation_precision_gain':.01,'maximum_recall_drop':.005,'recall_floor':.95,'developer_recall_floor':.90}
        checks=quality_gates(m(.65,.68,.96),m(.7,.75,.94),m(.69,.73,.95),g)
        self.assertTrue(checks['foundation_iou_gain'])
        self.assertFalse(checks['foundation_recall_retention'])


if __name__=='__main__': unittest.main()
