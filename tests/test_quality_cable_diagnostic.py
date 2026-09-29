import unittest
import numpy as np
try:
 from src.quality_cable_diagnostic import detached_cable,evidence_masks,joint_evidence,truth_counts
 AVAILABLE=True
except ImportError:AVAILABLE=False
@unittest.skipUnless(AVAILABLE,'Optional scipy unavailable')
class CableDiagnosticTests(unittest.TestCase):
 def test_diagonal_connection_and_detached_island(self):
  raw=np.array([[1,0,0,0],[0,2,0,0],[0,0,0,2]],np.uint8)
  np.testing.assert_array_equal(detached_cable(raw),[[False]*4,[False]*4,[False,False,False,True]])
 def test_joint_preserves_truth_without_making_predictions(self):
  raw=np.full((2,3),2,np.uint8);static=np.array([[0,1,0],[2,3,4]],np.uint8);binary=np.array([[0,1,1],[0,1,0]],np.uint8);target=np.array([[0,2,1],[3,4,2]],np.uint8);prob=np.array([[.5,.8,.95],[1.,.4,.3]],np.float32)
  before=raw.copy();j=joint_evidence(raw,static,binary,prob,target,[0,.5,.8,.95,1.000001]);m=evidence_masks(raw,static,binary)
  np.testing.assert_array_equal(j.sum(axis=(0,1,2,3)),truth_counts(raw==2,target))
  np.testing.assert_array_equal(j[0,0,1].sum(0),truth_counts(m['detached_both_background'],target));np.testing.assert_array_equal(raw,before)
