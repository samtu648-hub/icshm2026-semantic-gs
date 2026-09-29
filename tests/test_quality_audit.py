import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_quality_rgb import organize_named_renders
from src.quality_cameras import read_camera_requests
from scripts.run_quality_rgb import audit_pgsr_neighbors


class QualityAuditTests(unittest.TestCase):
    def test_pgsr_neighbors_train_only_and_zero_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multi_view.json"
            path.write_text(json.dumps({"ref_name": "001", "nearest_name": []}) + "\n")
            report = audit_pgsr_neighbors(path, ["001.png"])
            self.assertEqual(report["eligible_camera_fraction"], 0)
            path.write_text(json.dumps({"ref_name": "001", "nearest_name": ["140"]}) + "\n")
            with self.assertRaises(ValueError):
                audit_pgsr_neighbors(path, ["001.png"])

    def test_pgsr_name_mapping_and_missing_camera_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "140.png").write_bytes(b"fixture")
            mapping = organize_named_renders(source, root / "out", ["140.png"])
            self.assertEqual(mapping["views"][0]["view"], "140.png")
            self.assertEqual((root / "out/140.png").read_bytes(), b"fixture")
            with self.assertRaises(ValueError):
                organize_named_renders(source, root / "bad", ["141.png"])

    def test_native_scene_camera_metadata_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cameras.txt").write_text("1 PINHOLE 1320 989 900 910 660 494.5\n")
            (root / "images.txt").write_text("1 1 0 0 0 0 0 2 1 001.png\n\n")
            cameras = read_camera_requests(root)
            self.assertEqual(cameras["001.png"].matrix[1, 1], 910)
            self.assertEqual(cameras["001.png"].pose.translation[2], 2)
            (root / "cameras.txt").write_text("1 OPENCV 1320 989 900 910 660 494.5\n")
            with self.assertRaises(ValueError):
                read_camera_requests(root)
