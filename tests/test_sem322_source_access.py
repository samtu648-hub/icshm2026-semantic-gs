import tempfile
import unittest
from pathlib import Path
try:
    import numpy as np
    from src.semantic_source_access import read_pose_only, rank_sources, select_plan, summarize_access
    from src.geometry_prior.camera_io import ColmapImage
except ImportError:
    np = None


@unittest.skipIf(np is None, "optional numpy/OpenCV unavailable")
class SourceAccessTests(unittest.TestCase):
    def pose(self, name, x, q=(1, 0, 0, 0)):
        return ColmapImage(1, name, 1, np.array(q, float), np.array([-x, 0, 0.]), ())

    def test_observations_never_parsed_including_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "images.txt"
            for observations in ("1 2 900", "not even valid tracks", ""):
                path.write_text("# model\n1 1 0 0 0 0 0 0 1 t.png\n" + observations + "\n2 1 0 0 0 -1 0 0 1 s.png\n\n")
                poses = read_pose_only(path)
                self.assertEqual(set(poses), {"t.png", "s.png"})
                self.assertEqual(poses["t.png"].points2d, ())
                self.assertEqual(rank_sources("t.png", ["s.png"], poses)[0]["center_distance"], 1)

    def test_distance_orientation_and_tie_break(self):
        poses = {n: self.pose(n, x) for n, x in (("t", 0), ("b", 2), ("a", 2), ("near", 1))}
        poses["back"] = self.pose("back", .1, (0, 0, 1, 0))
        ranking = rank_sources("t", ["b", "a", "near", "back"], poses)
        self.assertFalse(ranking[0]["eligible"])
        self.assertEqual([r["source"] for r in ranking if r["eligible"]], ["near", "a", "b"])

    def test_fold_scope_and_fixed_policy(self):
        poses = {n: self.pose(n, x) for x, n in enumerate(("t", "s", "g", "other"))}
        folds = [{"fold": 0, "train": ["s"], "oof": ["t", "other"], "guard": ["g"]}]
        result = select_plan(folds, ["t"], poses)
        self.assertEqual(result[0]["sources"], ["s"])
        self.assertFalse(result[0]["candidate_budget_filled"])
        with self.assertRaises(ValueError): select_plan(folds, ["t"], poses, top_k=4)
        folds[0]["train"].append("t")
        with self.assertRaises(ValueError): select_plan(folds, ["t"], poses)

    def test_no_target_source_and_duplicate_pose(self):
        with self.assertRaises(ValueError): rank_sources("t", ["t"], {"t": self.pose("t", 0)})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "images.txt"
            path.write_text("1 1 0 0 0 0 0 0 1 t\n\n2 1 0 0 0 1 0 0 1 t\n\n")
            with self.assertRaises(ValueError): read_pose_only(path)

    def test_first_passing_not_highest_coverage_and_missing_fail(self):
        plans = [{"fold": i, "target": f"t{i}", "sources": ["a", "b", "c"], "candidate_budget_filled": True} for i in range(3)]
        rows = [{"fold": i, "target": f"t{i}", "source": n, "rank": j + 1,
                 "checks": {"coverage": j > 0, "epipolar_retention": True, "beats_identity_factor2": True}}
                for i in range(3) for j, n in enumerate(("a", "b", "c"))]
        result = summarize_access(plans, rows)
        self.assertTrue(result["passed"])
        self.assertEqual(result["rank1_accessible_targets"], 0)
        self.assertTrue(all(t["selected_source"] == "b" for t in result["targets"]))
        for r in rows:
            if r["fold"] == 2: r["checks"]["epipolar_retention"] = False
        self.assertFalse(summarize_access(plans, rows)["passed"])
        with self.assertRaises(ValueError): summarize_access(plans, rows[:-1])

    def test_independent_geometry_and_tamper_rejection(self):
        try:
            import torch
            from src.semantic_source_geometry import geometry_row, compare_rows
            from src.semantic_dense_geometry import normalized_grid
            from src.geometry_prior.camera_io import ColmapCamera
        except ImportError:
            self.skipTest("optional torch unavailable")
        grid = normalized_grid(16, 16)
        ones = np.ones((16, 16))
        images = {"t": self.pose("t", 0), "s": self.pose("s", 1)}
        cameras = {1: ColmapCamera(1, "SIMPLE_RADIAL", 1320, 989, (900., 660., 494.5, .009))}
        pair = {"fold": 0, "target": "t", "source": "s", "rank": 1}
        cfg = {"filters": {"overlap_each_direction_minimum": .5, "cycle_maximum_native_pixels": 2., "epipolar_maximum_native_pixels": 2.},
               "geometry_screen": {"minimum_reliable_grid_fraction_each_pair": .01, "minimum_epipolar_inlier_fraction_after_cycle_each_pair": .8}}
        primary = geometry_row(pair, [(grid, ones), (grid, ones)], images, cameras, cfg)
        independent = geometry_row(pair, [(grid, ones), (grid, ones)], images, cameras, cfg, independent=True)
        compare_rows(primary, independent)
        independent["reliable_points"] += 1
        with self.assertRaises(ValueError): compare_rows(primary, independent)


if __name__ == "__main__": unittest.main()
