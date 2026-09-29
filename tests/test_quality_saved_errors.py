import unittest
import numpy as np
from scripts.diagnose_quality_saved_errors import compare


class SavedErrorDiagnosisTest(unittest.TestCase):
    def test_one_absent_class_false_positive_changes_active_denominator(self):
        t=np.zeros((5,5,5),np.int64)
        for cl in range(4):t[cl,cl,cl]=100
        t[0,0,0]-=1;t[0,0,4]=1
        r=compare(t.sum(2),t.sum(1),t)
        self.assertEqual(len(r['active_old']),4)
        self.assertEqual(len(r['active_new']),5)
        self.assertAlmostEqual(r['delta_miou_pp'],-20.2)
        self.assertAlmostEqual(r['iou_sum_change_at_old_denominator_pp'],-.25)
        self.assertAlmostEqual(r['denominator_change_at_new_iou_sum_pp'],-19.95)
        self.assertEqual(r['harmed'],1)
        self.assertIsNone(r['classes']['foundation']['delta_iou_pp'])

    def test_removed_yellow_can_remain_wrong_and_cable_recall_churn_is_separate(self):
        t=np.zeros((5,5,5),np.int64)
        t[0,2,1]=7;t[2,2,0]=3;t[2,0,2]=2
        r=compare(t.sum(2),t.sum(1),t)
        self.assertEqual(r['background_cable_churn']['removed'],7)
        self.assertEqual(r['background_cable_churn']['removed_to_true_background'],0)
        self.assertEqual(r['background_cable_churn']['removed_to_other_wrong_class'],7)
        self.assertEqual(r['cable_recall_churn']['recovered'],2)
        self.assertEqual(r['cable_recall_churn']['newly_missed'],3)
        self.assertEqual((r['corrected'],r['harmed'],r['wrong_to_wrong']),(2,3,7))

    def test_transition_counts_cannot_disagree_with_marginals(self):
        t=np.zeros((5,5,5),np.int64);t[0,0,0]=1
        wrong=t.sum(1);wrong[0,0]+=1
        with self.assertRaises(ValueError):compare(t.sum(2),wrong,t)
