import unittest
import numpy as np
from src.quality_raw_foundation_gate import apply_gate


class RawFoundationGateTests(unittest.TestCase):
    def test_off_exact_and_only_registered_direction(self):
        raw = np.array([[0,1,2,3,3,4]], np.uint8)
        fused = np.array([[4,4,4,4,0,0]], np.uint8)
        five = np.zeros((5,1,6),np.float32); five[3]=0.6; five[4]=0.4
        self.assertTrue(np.array_equal(apply_gate(raw,fused,five,None),raw))
        self.assertEqual(apply_gate(raw,fused,five,0.4).tolist(),[[0,1,2,4,3,4]])
        self.assertEqual(apply_gate(raw,fused,five,0.1).tolist(),raw.tolist())

    def test_equal_margin_included_and_invalid_rejected(self):
        raw = np.array([[3]],np.uint8); fused = np.array([[4]],np.uint8)
        five = np.zeros((5,1,1),np.float32); five[3]=0.625; five[4]=0.375
        self.assertEqual(apply_gate(raw,fused,five,0.25)[0,0],4)
        for t in (-0.1,1.1,float('nan')):
            with self.assertRaises(ValueError): apply_gate(raw,fused,five,t)


if __name__ == '__main__': unittest.main()
