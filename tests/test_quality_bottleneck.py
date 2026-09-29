import unittest
import numpy as np
from src.quality_bottleneck import distribution,role_shift,core_counts,window_stats
class BottleneckTests(unittest.TestCase):
    def test_distribution_invariant_to_image_count_scaling(self):
        cm=np.diag([90,40,30,20,10]);cm[0,2]=10
        d=role_shift(cm,cm*3)
        self.assertEqual(d['gt_total_variation'],0)
        np.testing.assert_allclose(d['union_fraction_evaluation_over_fit'],1)
    def test_core_recovery_deduplicates_sources(self):
        cm=np.eye(5,dtype=np.int64)*10;cm[0,2]=4;cm[2,0]=3
        repair=np.zeros((5,5),np.int64);repair[0,2]=4
        row={'raw':{'confusion':cm.tolist(),'scopes':{'core':{'recovered_confusion':repair.tolist()}},'wrong_correctness_joint':[[0,2,3,4],[2,0,32,3]]}}
        d=core_counts([row]);self.assertEqual(sum(d['core_recovered']),4);self.assertEqual(sum(d['core_common_errors']),3)
        row['raw']['wrong_correctness_joint'][0][3]=5
        with self.assertRaises(ValueError):core_counts([row])
    def test_window_means_and_skips(self):
        rows=[dict(loss=v,grad_norm=v,lr_factor=.1,scale=64,losses=dict(loss_ce=v,loss_mask=0,loss_dice=0),skips=[] if v==1 else [{}],sample=dict(name=n)) for v,n in [(1,'a'),(3,'b')]]
        d=window_stats(rows);self.assertEqual(d['loss']['mean'],2);self.assertEqual(d['amp_skips'],1);self.assertEqual(d['views'],{'a':1,'b':1})
