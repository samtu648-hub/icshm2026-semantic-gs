import unittest
try:
    import numpy as np
    from src.geometry_prior.camera_io import ColmapImage
    from src.semantic_source_expansion import select_plan, summarize_access
except ImportError:
    np = None


@unittest.skipIf(np is None, "optional geometry packages unavailable")
class ExpansionTests(unittest.TestCase):
    def test_all_remaining_targets_and_unchanged_sources(self):
        names = ["a", "b", "c", "s", "z"]
        poses = {n: ColmapImage(i,n,1,np.array([1.,0,0,0]),np.array([float(i),0,0]),()) for i,n in enumerate(names)}
        folds = [{"fold":0,"train":["s"],"oof":["a","b","c"],"guard":["z"]}]
        plans = select_plan(folds,["a","b","c"],poses)
        self.assertEqual([p["target"] for p in plans],["b","c"])
        self.assertTrue(all(p["sources"]==["s"] for p in plans))
        self.assertFalse(any(p["candidate_budget_filled"] for p in plans))

    def test_summary_separates_targets_within_fold_and_keeps_empty(self):
        plans = [{"fold":0,"target":t,"sources":s} for t,s in (("a",["s"]),("b",["s"]),("c",[]))]
        rows = [{"fold":0,"target":t,"source":"s","rank":1,"checks":{"coverage":good,"epipolar_retention":True,"beats_identity_factor2":True},
                 "reliable_grid_fraction":.02,"epipolar_inliers_after_cycle":.9} for t,good in (("a",True),("b",False))]
        result = summarize_access(plans,rows)
        self.assertEqual(result["target_count"],3)
        self.assertEqual(result["top3_accessible_targets"],1)
        self.assertEqual(result["targets"][2]["selected_reliable_grid_fraction"],0)
        self.assertFalse(result["passed"])
        with self.assertRaises(ValueError): summarize_access(plans,rows[::-1])


if __name__ == "__main__": unittest.main()
