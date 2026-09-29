import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class SpatialConfirmationBaselineTest(unittest.TestCase):
    def setUp(self):
        self.split_path = ROOT / "experiments/splits/spatial_confirmation_v1.json"
        self.split = read(self.split_path)
        self.fold = self.split["primary_fold"]
        self.rgb = read(
            ROOT
            / "experiments/runs/RGB-152-STANDARD-ABSGRAD-SPATIAL30/config.yaml"
        )
        self.sem = read(
            ROOT / "experiments/runs/SEM-396-RGB152-SPATIAL256/config.yaml"
        )

    def test_roles_and_registered_counts_match(self):
        train = set(self.fold["train_rgb_400"])
        confirmation = set(self.fold["validation_labeled"])
        buffer_views = set(self.fold["buffer_excluded_all_cameras"])
        semantic_train = set(self.fold["semantic_train_labeled"])

        self.assertEqual((len(train), len(confirmation), len(buffer_views)), (353, 30, 17))
        self.assertFalse(train & confirmation)
        self.assertFalse(train & buffer_views)
        self.assertFalse(confirmation & buffer_views)
        self.assertEqual(len(train | confirmation | buffer_views), 400)
        self.assertEqual(len(semantic_train), 256)
        self.assertLessEqual(semantic_train, train)

        self.assertEqual(self.rgb["data"]["rgb_train_views"], len(train))
        self.assertEqual(
            self.rgb["data"]["rgb_confirmation_views"], len(confirmation)
        )
        self.assertEqual(
            self.rgb["data"]["buffer_excluded_views"], len(buffer_views)
        )
        self.assertEqual(self.sem["semantic_train_views"], len(semantic_train))
        self.assertEqual(
            self.sem["semantic_confirmation_views"], len(confirmation)
        )
        self.assertEqual(self.sem["buffer_excluded_views"], len(buffer_views))

    def test_frozen_recipes_and_access_policy(self):
        self.assertEqual(self.rgb["seed"], 0)
        self.assertEqual(self.rgb["training"]["iterations"], 30000)
        self.assertEqual(self.rgb["training"]["absolute_split_threshold"], 0.0004)
        self.assertFalse(self.rgb["data"]["confirmation_used_for_initialization"])
        self.assertFalse(self.rgb["data"]["buffer_used_for_initialization"])

        self.assertEqual(self.sem["seed"], 20260805)
        self.assertEqual(self.sem["iterations"], 6000)
        self.assertEqual(self.sem["crop"], 768)
        self.assertFalse(self.sem["confirmation_access"]["rgb_during_semantic_training"])
        self.assertFalse(
            self.sem["confirmation_access"]["labels_during_semantic_training"]
        )
        self.assertFalse(self.sem["confirmation_access"]["buffer_rgb_or_labels"])

    @unittest.skipUnless((ROOT / "outputs/quality/RGB-152/scene/preparation_manifest.json").is_file(), "optional saved run artifact is not distributed")
    def test_prepared_scene_is_training_only(self):
        manifest = read(ROOT / "outputs/quality/RGB-152/scene/preparation_manifest.json")
        self.assertEqual(
            manifest["counts"],
            {
                "training_images": 353,
                "evaluation_images": 30,
                "scene_images": 383,
                "retriangulated_points": 78045,
            },
        )
        self.assertFalse(
            manifest["evaluation"]["initialization_uses_evaluation_images"]
        )


if __name__ == "__main__":
    unittest.main()
