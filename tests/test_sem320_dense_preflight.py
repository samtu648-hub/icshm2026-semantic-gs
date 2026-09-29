import json
from pathlib import Path
import unittest
from src.semantic_dense_preflight import choose_pairs, require_pair_scope, pixel_center


class DensePreflightContracts(unittest.TestCase):
    def fixture(self):
        folds = [{"fold": 0, "oof": ["t"], "train": ["a", "b"], "guard": ["g"]}]
        cameras = {"t": {"center": [0, 0, 0], "tracks": [1, 2]},
                   "a": {"center": [1, 0, 0], "tracks": [1]},
                   "b": {"center": [2, 0, 0], "tracks": [1, 2]}}
        return folds, cameras

    def test_shared_tracks_take_priority_over_distance(self):
        folds, cameras = self.fixture()
        pairs = choose_pairs(folds, ["t"], cameras)
        self.assertEqual(pairs[0]["source"], "b")
        require_pair_scope(pairs, folds, ["t"], ["d"], ["g"])

    def test_guard_and_target_source_leaks_rejected(self):
        folds, cameras = self.fixture()
        pairs = choose_pairs(folds, ["t"], cameras)
        pairs[0]["source"] = "g"
        with self.assertRaises(ValueError): require_pair_scope(pairs, folds, ["t"], ["d"], ["g"])
        folds[0]["train"].append("t")
        with self.assertRaises(ValueError): choose_pairs(folds, ["t"], cameras)

    def test_pixel_centers_not_array_indices(self):
        for size in (320, 512, 989, 1320):
            self.assertAlmostEqual(pixel_center(-1 + 1 / size, size), .5)
            self.assertAlmostEqual(pixel_center(1 - 1 / size, size), size - .5)

    def test_recipe_has_no_quality_selection_or_semantic_authority(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / "experiments/runs/SEM-320-ROMAV2-DENSE-PREFLIGHT/config.yaml").read_text())
        self.assertEqual(cfg["runtime"]["setting"], "fast")
        self.assertEqual(cfg["runtime"]["low_resolution"], [512, 512])
        self.assertEqual(cfg["sources"]["weight_bytes"], 1095883548)
        self.assertIn("No source/target labels", cfg["access_policy"])


if __name__ == "__main__": unittest.main()
