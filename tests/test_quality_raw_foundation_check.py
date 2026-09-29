import unittest
import numpy as np
from src.quality_raw_foundation_check import components,evidence_masks,validate_selection


class RawFoundationCheckTests(unittest.TestCase):
    def test_diagonal_connectivity_and_native_bounds(self):
        mask=np.zeros((5,6),bool);mask[0,0]=True;mask[1,1]=True;mask[4,5]=True
        self.assertEqual(components(mask),[{'pixels':2,'bbox_xyxy':[0,0,2,2]},{'pixels':1,'bbox_xyxy':[5,4,6,5]}])
        self.assertEqual(components(np.zeros((2,3),bool)),[])

    def test_exact_gain_harm_and_other_errors(self):
        raw=np.array([[3,3,3,4]],np.uint8);candidate=np.array([[4,4,4,4]],np.uint8)
        encoded,local=evidence_masks(raw,candidate,np.array([[4,3,0,4]],np.uint8))
        self.assertEqual(encoded.tolist(),[[1,2,3,0]])
        self.assertEqual([sum(c['pixels'] for c in v) for v in local.values()],[1,1,1])
        with self.assertRaises(ValueError):evidence_masks(raw,np.array([[0,4,4,4]],np.uint8),raw)

    def test_reject_unapproved_margin(self):
        s={'experiment_id':'SEM-343-RAW-FOUNDATION-GATE','selected_margin':0.1}
        with self.assertRaises(ValueError):validate_selection(s,{}, {})


if __name__=='__main__':unittest.main()
