import unittest
import numpy as np
from src.quality_tower_gate import apply_tower_gate


class TowerGateTest(unittest.TestCase):
    def test_proposal_uses_original_raw_and_does_not_cascade(self):
        raw=np.array([[2,2,3,0,4]],np.uint8);base=np.array([[2,2,4,0,4]],np.uint8);static=np.array([[3,4,3,3,3]],np.uint8)
        p=np.zeros((5,1,5),np.float32);p[2]=.51;p[3]=.49
        actual=apply_tower_gate(raw,base,static,p,.05)
        np.testing.assert_array_equal(actual,[[3,2,4,0,4]])
        np.testing.assert_array_equal(base,[[2,2,4,0,4]])
        np.testing.assert_array_equal(apply_tower_gate(raw,base,static,p,None),base)

    def test_margin_is_inclusive_and_only_changes_eligible_cable(self):
        raw=np.array([[2,2,2,1]],np.uint8);static=np.array([[3,3,3,3]],np.uint8);p=np.zeros((5,1,4),np.float32)
        p[2]=[.625,.75,.5,.625];p[3]=[.375,.25,.5,.375]
        np.testing.assert_array_equal(apply_tower_gate(raw,raw,static,p,.25),[[3,2,3,1]])

    def test_unregistered_baseline_and_invalid_margin_rejected(self):
        raw=np.array([[2]],np.uint8);p=np.ones((5,1,1),np.float32)/5
        with self.assertRaises(ValueError):apply_tower_gate(raw,np.array([[4]],np.uint8),raw,p,.05)
        for margin in [-.1,1.01,float('nan')]:
            with self.assertRaises(ValueError):apply_tower_gate(raw,raw,raw,p,margin)
