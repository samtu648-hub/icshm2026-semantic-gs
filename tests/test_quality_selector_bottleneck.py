import unittest
import numpy as np
from src.quality_selector_bottleneck import source_counts, decision_counts


class DiagnosticPartitionTests(unittest.TestCase):
    def test_all_seven_decision_cases_and_missing_binary_class(self):
        gt=np.array([[0,1,2,3,4,0,1]],np.uint8)
        raw=np.array([[1,0,0,0,0,0,1]],np.uint8)
        pred=np.array([[0,0,1,0,2,0,2]],np.uint8)
        maps=np.full((5,1,7),5,np.uint8);maps[0,0,:3]=gt[0,:3]
        expected=np.zeros((5,7),np.int64)
        for i,t in enumerate(gt.ravel()):expected[t,i]=1
        for reference in (False,True):
            hist,oracle,recover=source_counts(gt,raw,maps,reference)
            self.assertEqual(int(hist.sum()),5)
            self.assertEqual(int(hist[:,0].sum()),2)
            self.assertEqual(int(np.trace(oracle)),5)
            np.testing.assert_array_equal(decision_counts(gt,raw,pred,recover,reference),expected)

    def test_correct_prediction_cannot_exceed_source_oracle(self):
        gt=np.array([2],np.uint8);raw=np.array([0],np.uint8)
        for reference in (False,True):
            with self.assertRaises(ValueError):
                decision_counts(gt,raw,gt,np.array([False]),reference)


if __name__=='__main__':unittest.main()
