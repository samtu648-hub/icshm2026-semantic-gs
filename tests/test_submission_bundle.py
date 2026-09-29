from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_submission_bundle import audit_renders  # noqa: E402
from render_submission import list_external_poses, select_external_poses  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SubmissionBundleTest(unittest.TestCase):
    def test_external_pose_listing_ignores_point_lines_and_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.txt"
            path.write_text(
                "# COLMAP\n"
                "1 1 0 0 0 0 0 0 7 140.png\n"
                "12.5 22.0 -1 18.0 19.0 -1\n"
                "2 1 0 0 0 1 2 3 8 141.png\n\n",
                encoding="utf-8",
            )
            poses = list_external_poses(path)
            self.assertEqual(poses, [("140.png", 7), ("141.png", 8)])
            self.assertEqual(select_external_poses(poses, ["141.png"]), [("141.png", 8)])

    def test_render_audit_accepts_rgb_semantic_and_logits_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rgb").mkdir()
            (root / "semantic").mkdir()
            (root / "logits").mkdir()
            rgb_path = root / "rgb/140.png"
            semantic_path = root / "semantic/140.png"
            logits_path = root / "logits/140.npy"
            Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8), mode="RGB").save(rgb_path)
            Image.fromarray(np.full((3, 4), 2, dtype=np.uint8), mode="L").save(semantic_path)
            np.save(logits_path, np.zeros((5, 3, 4), dtype=np.float32))
            payload = {
                "uses_reference_rgb_or_labels": False,
                "outputs": [
                    {
                        "pose_name": "140.png",
                        "height": 3,
                        "width": 4,
                        "rgb": {"path": "rgb/140.png", "sha256": digest(rgb_path)},
                        "semantic": {"path": "semantic/140.png", "sha256": digest(semantic_path)},
                        "logits": {"path": "logits/140.npy", "sha256": digest(logits_path)},
                    }
                ],
            }
            (root / "submission_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            result = audit_renders(root)
            self.assertEqual(result["rendered_views"], 1)


if __name__ == "__main__":
    unittest.main()
