import json
import unittest
from pathlib import Path

from scripts.promote_sem395_spatial_split import build, sha256


ROOT = Path(__file__).resolve().parents[1]


class SpatialConfirmationSplitTest(unittest.TestCase):
    @unittest.skipUnless((ROOT / "outputs/quality/SEM-395/r2/prepare/confirmation-split-candidate.json").is_file(), "optional saved run artifact is not distributed")
    def test_registered_roles_are_complete_and_disjoint(self):
        source_path = ROOT / "outputs/quality/SEM-395/r2/prepare/confirmation-split-candidate.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        split = build(source, sha256(source_path))
        fold = split["primary_fold"]
        confirmation = set(fold["holdout_all_cameras"])
        buffer_views = set(fold["buffer_excluded_all_cameras"])
        train = set(fold["train_rgb_400"])
        self.assertEqual((len(confirmation), len(buffer_views), len(train)), (30, 17, 353))
        self.assertFalse(confirmation & buffer_views)
        self.assertFalse(confirmation & train)
        self.assertFalse(buffer_views & train)
        self.assertEqual(len(confirmation | buffer_views | train), 400)
        self.assertEqual(len(fold["semantic_train_labeled"]), 256)
        self.assertLessEqual(set(fold["semantic_train_labeled"]), train)
        self.assertEqual(
            sorted(name for group in split["groups"] for name in group["images"]),
            sorted(confirmation),
        )


if __name__ == "__main__":
    unittest.main()
