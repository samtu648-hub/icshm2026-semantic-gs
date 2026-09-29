import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.quality_protocol import full_chain_folds, sha256, validation_matrix, validate_component_access, verify_hashes, write_new_json

ROOT = Path(__file__).resolve().parents[1]


class QualityProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rgb = json.loads((ROOT / "experiments/splits/rgb_r1_spatial_v1.json").read_text())["primary_fold"]
        cls.oof = json.loads((ROOT / "experiments/splits/sem_005_oof_spatial_v1.json").read_text())
        cls.nodes = json.loads((ROOT / "experiments/geometry_camera_graph.json").read_text())["nodes"]

    def build(self, **changes):
        return full_chain_folds(changes.get("rgb", self.rgb), changes.get("folds", self.oof["folds"]), changes.get("nodes", self.nodes), self.oof["guard_policy"]["radius"])

    def test_real_outer_rgb_and_semantic_counts(self):
        folds = self.build()
        self.assertEqual([len(f["rgb_train"]) for f in folds], [196, 201, 240])
        self.assertEqual([len(f["semantic_train"]) for f in folds], [114, 128, 177])
        self.assertEqual([len(f["evaluation"]) for f in folds], [95, 84, 40])
        for f in folds:
            self.assertGreaterEqual(f["minimum_train_target_distance"], self.oof["guard_policy"]["radius"])
            self.assertFalse(set(f["reference_label_views"]) & set(f["evaluation"]))
            self.assertEqual(f["initialization_observation_views"], f["rgb_train"])

    def test_heldout_access_rejected_for_every_component(self):
        f = self.build()[0]
        for role in ("rgb", "initialization", "features", "semantics", "reference_rgb", "reference_labels"):
            with self.subTest(role=role), self.assertRaises(ValueError):
                validate_component_access(f, role, [f["evaluation"][0]])
        validate_component_access(f, "rgb", f["rgb_train"])

    def test_guard_unlabelled_excluded(self):
        f = self.build()[2]
        excluded = set(f["rgb_guard"]) - set(f["semantic_guard"])
        self.assertTrue(excluded)
        for name in excluded:
            with self.assertRaises(ValueError):
                validate_component_access(f, "rgb", [name])

    def test_duplicate_fold_rejected(self):
        with self.assertRaises(ValueError):
            self.build(folds=self.oof["folds"] + [self.oof["folds"][0]])

    def test_reserved_rgb_rejected(self):
        rgb = copy.deepcopy(self.rgb)
        rgb["train_rgb_400"].append(rgb["withheld_unlabeled"][0])
        with self.assertRaises(ValueError):
            self.build(rgb=rgb)

    def test_camera_and_guard_corruption_rejected(self):
        nodes = copy.deepcopy(self.nodes)
        nodes[0]["center"][0] = float("nan")
        with self.assertRaises(ValueError):
            self.build(nodes=nodes)
        folds = copy.deepcopy(self.oof["folds"])
        folds[0]["train"].append(folds[0]["guard"].pop())
        with self.assertRaises(ValueError):
            self.build(folds=folds)

    def test_twelve_validation_units(self):
        matrix = validation_matrix()
        self.assertEqual(len(matrix), 12)
        self.assertEqual(len({(x["system"], x["kind"], x["index"]) for x in matrix}), 12)
        self.assertEqual(sum(x["outer_fold"] is not None for x in matrix), 6)

    def test_evidence_is_exclusive_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "evidence.json"
            write_new_json(path, {"quality_metrics": None})
            self.assertEqual(verify_hashes(root, {path.name: sha256(path)})[path.name], sha256(path))
            with self.assertRaises(FileExistsError):
                write_new_json(path, {})
            with self.assertRaises(ValueError):
                verify_hashes(root, {path.name: "0" * 64})
            with self.assertRaises(ValueError):
                verify_hashes(root, {"../escape": "0" * 64})


if __name__ == "__main__":
    unittest.main()
