import unittest
try:
    import numpy as np
    from src.semantic_dense_label_audit import sample_labels,count_evidence,aggregate
except ImportError:
    np=None


@unittest.skipIf(np is None,"optional numpy unavailable")
class LabelAuditTests(unittest.TestCase):
    def test_pixel_center_and_invalid_positions(self):
        labels=np.array([[0,1,2],[3,4,0]],np.uint8)
        points=np.array([[[.5,.5],[1.5,1.5],[2.49,.1],[-1,1],[3.,1.]]])
        actual,valid=sample_labels(labels,points)
        self.assertEqual(actual[0,:3].tolist(),[0,4,2])
        self.assertEqual(valid.tolist(),[[True,True,True,False,False]])

    def test_counts_corrections_harms_and_abstentions(self):
        gt=np.array([[1,2,3,4,0]],np.uint8); old=np.array([[0,2,0,4,1]],np.uint8); new=np.array([[1,0,2,4,0]],np.uint8)
        support=np.array([[True,True,True,True,False]])
        a=count_evidence(gt,old,new,support); b=count_evidence(gt,old,new,support,True)
        self.assertEqual(a,b)
        self.assertEqual((a["corrected"],a["harmed"],a["both_wrong"],a["both_correct"]),(1,1,1,1))
        self.assertEqual(a["baseline_errors_total"],3)
        blank=count_evidence(gt,old,new,np.zeros_like(support))
        result=aggregate([a,blank])
        self.assertEqual(result["total_grid_points"],10)
        self.assertEqual(result["baseline_error_coverage"],2/6)

    def test_torch_sampler_matches_non_boundary_coordinates(self):
        try: import torch
        except ImportError: self.skipTest("optional torch unavailable")
        labels=np.arange(20,dtype=np.uint8).reshape(4,5)%5
        points=np.array([[[.51,.6],[1.8,2.6],[4.9,3.9],[2.2,1.7]]])
        a,va=sample_labels(labels,points); b,vb=sample_labels(labels,points,True)
        np.testing.assert_array_equal(a,b); np.testing.assert_array_equal(va,vb)


if __name__=="__main__": unittest.main()
