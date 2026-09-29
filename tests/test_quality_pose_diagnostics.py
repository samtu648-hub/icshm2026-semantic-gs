import json
from pathlib import Path
import tempfile
import unittest

try:
    import cv2
    import numpy as np
    from src.quality_pose_diagnostics import (training_access, load_training_records, fundamental_from_poses,
        sampson_distance, triangulate, reprojection, crossfit_matches, evaluate_pair, summary, symmetric_transfer,
        audit_released_tracks, stable_number, select_pairs)
except ImportError:
    np = None


@unittest.skipIf(np is None, "optional numpy/opencv runtime")
class PoseDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.k = np.array([[900., 0., 660.], [0., 900., 494.5], [0., 0., 1.]])
        self.r = np.eye(3)
        self.t1, self.t2 = np.zeros(3), np.array([-1., 0., 0.])
        rng = np.random.default_rng(1)
        self.xyz = rng.uniform([-2., -2., 4.], [2., 2., 12.], (250, 3))
        def project(t):
            p = (self.xyz + t) @ self.k.T
            return p[:, :2] / p[:, 2:]
        self.a, self.b = project(self.t1), project(self.t2)

    def test_pose_convention_and_direction(self):
        f = fundamental_from_poses(self.r, self.t1, self.r, self.t2, self.k)
        self.assertLess(sampson_distance(f, self.a, self.b).max(), 1e-9)
        np.testing.assert_allclose(sampson_distance(f, self.a, self.b), sampson_distance(f.T, self.b, self.a))

    def test_zero_baseline_rejected(self):
        with self.assertRaises(ValueError):
            fundamental_from_poses(self.r, self.t1, self.r, self.t1, self.k)

    def test_general_rotated_camera_world_to_camera_convention(self):
        theta = .2
        r2 = np.array([[np.cos(theta), 0., np.sin(theta)], [0., 1., 0.], [-np.sin(theta), 0., np.cos(theta)]])
        centre = np.array([1., .3, -.2])
        t2 = -r2 @ centre
        projected = (self.xyz @ r2.T + t2) @ self.k.T
        b = projected[:, :2] / projected[:, 2:]
        f = fundamental_from_poses(self.r, self.t1, r2, t2, self.k)
        self.assertLess(sampson_distance(f, self.a, b).max(), 1e-9)

    def test_triangulation_does_not_reject_large_observation_error(self):
        poses = [(self.r, self.t1), (self.r, self.t2)]
        point = triangulate(poses, np.array([self.a[0], self.b[0]]), self.k)
        np.testing.assert_allclose(point, self.xyz[0], atol=1e-10)
        bad = self.b[0] + [0., 50.]
        point = triangulate(poses, np.array([self.a[0], bad]), self.k)
        delta = reprojection(point, poses, np.array([self.a[0], bad]), self.k)
        self.assertGreater(np.linalg.norm(delta, axis=1).mean(), 20)

    def test_split_deduplicates_location_orientations(self):
        a, b = np.r_[self.a, self.a[:20]], np.r_[self.b, self.b[:20]]
        ids, check = crossfit_matches(a, b, ("001.png", "002.png"), 3)
        self.assertEqual(len(ids), len(self.a))
        self.assertGreater(check.sum(), 20)
        ids2, check2 = crossfit_matches(a, b, ("001.png", "002.png"), 3)
        np.testing.assert_array_equal(ids, ids2)
        np.testing.assert_array_equal(check, check2)

    def test_excluded_observations_not_parsed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "images.txt"
            p.write_text("1 1 0 0 0 0 0 0 1 001.png\n1 2 3\n2 1 0 0 0 0 0 0 1 002.png\nINVALID EXCLUDED OBSERVATIONS\n")
            records = load_training_records(p, ["001.png"])
            self.assertEqual(list(records), ["001.png"])
            self.assertEqual(records["001.png"].observations[0].point_id, 3)

    def test_split_leakage_rejected(self):
        split = {"primary_fold": {"train_rgb_400": ["001.png"], "validation_labeled": ["001.png"],
                                 "buffer_excluded_all_cameras": [], "withheld_unlabeled": []}}
        with self.assertRaises(ValueError):
            training_access(split)

    def test_held_observation_excluded_from_triangulation(self):
        from types import SimpleNamespace
        seed, pid = 3, 101
        held = stable_number(seed, pid) % 3
        records = {}
        point = np.array([0., 0., 5.])
        for i, centre_x in enumerate([-1., 0., 1.]):
            t = np.array([-centre_x, 0., 0.])
            pixel = self.k @ (point + t)
            xy = pixel[:2] / pixel[2]
            if i == held:
                xy += [0., 10.]
            records[f"{i+1:03}.png"] = SimpleNamespace(rotation=np.eye(3), translation=t,
                observations=[SimpleNamespace(point_id=pid, undistorted_xy=tuple(xy))])
        report, arrays = audit_released_tracks(records, self.k, seed)
        self.assertEqual(report["cross_prediction"]["count"], 1)
        self.assertAlmostEqual(np.linalg.norm(arrays["cross_prediction"][0, 4:]), 10., places=7)
        self.assertEqual(int(arrays["cross_prediction"][0, 1]), held)

    def test_crossfit_exposes_bad_official_F_without_filtering_it(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "experiments/runs/RGB-117-CAMERA-CONSISTENCY-DIAG/config.yaml").read_text())["independent_matching"]
        theta = .03
        bad_r = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
        bad_f = fundamental_from_poses(self.r, self.t1, bad_r, self.t2, self.k)
        row, arrays = evaluate_pair(self.a, self.b, bad_f, ("001.png", "002.png"), config, 8, (1320, 989))
        self.assertEqual(row["status"], "completed_correspondence_crossfit_not_pose_truth")
        self.assertEqual(len(arrays["official_check_error"]), row["check_matches"])
        self.assertGreater(row["official_sampson_px"]["median"], 1.5)
        self.assertLess(row["image_f_sampson_px"]["median"], .01)

    def test_summary_preserves_nonfinite_denominator(self):
        result = summary([0., 2., float("inf")])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["finite_count"], 2)
        self.assertEqual(result["nonfinite_count"], 1)

    def test_empty_statistics_are_not_zero_accuracy(self):
        result = summary([])
        self.assertEqual(result["count"], 0)
        self.assertNotIn("median", result)

    def test_insufficient_matches_retained_without_score(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "experiments/runs/RGB-117-CAMERA-CONSISTENCY-DIAG/config.yaml").read_text())["independent_matching"]
        f = fundamental_from_poses(self.r, self.t1, self.r, self.t2, self.k)
        row, arrays = evaluate_pair(self.a[:10], self.b[:10], f, ("001.png", "002.png"), config, 8, (1320, 989))
        self.assertEqual(row["status"], "insufficient_matches")
        self.assertEqual(row["fit_matches"] + row["check_matches"], 10)
        self.assertNotIn("image_f_check_error", arrays)

    def test_homography_diagnostic(self):
        xy = self.a[:10]
        h = np.array([[1., 0., 5.], [0., 1., -4.], [0., 0., 1.]])
        np.testing.assert_allclose(symmetric_transfer(h, xy, xy + [5., -4.]), 0, atol=1e-9)

    def test_candidate_baseline_ranks_and_deduplication(self):
        from types import SimpleNamespace
        records = {f"{i:03}.png": SimpleNamespace(rotation=np.eye(3), translation=np.array([-float(i), 0., 0.])) for i in range(10)}
        near, missing = select_pairs(records, [1, 4], 45, 1e-8)
        wide, _ = select_pairs(records, [8], 45, 1e-8)
        self.assertFalse(missing)
        self.assertEqual(len(near), len(set(near)))
        self.assertIn(("000.png", "004.png"), near)
        self.assertIn(("000.png", "008.png"), wide)
        self.assertTrue(all(a != b for a, b in wide))
