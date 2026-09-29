import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.geometry_prior.vggt_p0_float64_audit import (
    CANONICAL_PIXEL_LIMIT,
    audit_npz_float64,
    canonical_world_points_float64,
)


class VggtP0Float64AuditTest(unittest.TestCase):
    def test_canonical_depth_roundtrip_and_serialized_world_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.npz"
            depth = np.asarray([[0.061096, 1.0], [2.0, 3.0]], dtype=np.float32)
            confidence = np.ones_like(depth)
            valid = np.ones_like(depth, dtype=np.uint8)
            intrinsics = np.asarray(
                [[583.0, 0.0, 295.5], [0.0, 584.0, 223.5], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            angle = 0.37
            rotation = np.asarray(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            cam2world = np.eye(4, dtype=np.float64)
            cam2world[:3, :3] = rotation
            cam2world[:3, 3] = [13.0, -7.0, 2.0]
            canonical = canonical_world_points_float64(depth, intrinsics, cam2world)
            np.savez_compressed(
                path,
                depth=depth,
                confidence=confidence,
                valid_mask=valid,
                world_points=canonical.astype(np.float32),
                pinhole_intrinsics=intrinsics,
                official_cam2world=cam2world,
                pred_cam2world=np.eye(4, dtype=np.float64),
                depth_scale_ratio=np.asarray(1.0, dtype=np.float64),
            )
            result = audit_npz_float64(path, expected_resolution_wh=(2, 2))
            self.assertLessEqual(
                result["canonical_pixel_error_max"], CANONICAL_PIXEL_LIMIT
            )
            self.assertGreaterEqual(result["stored_pixel_error_max"], 0.0)
            self.assertEqual(result["valid_count"], 4)

    def test_rejects_nonfinite_canonical_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            depth = np.asarray([[np.nan]], dtype=np.float32)
            np.savez_compressed(
                path,
                depth=depth,
                confidence=np.ones((1, 1), dtype=np.float32),
                valid_mask=np.zeros((1, 1), dtype=np.uint8),
                world_points=np.zeros((1, 1, 3), dtype=np.float32),
                pinhole_intrinsics=np.eye(3, dtype=np.float64),
                official_cam2world=np.eye(4, dtype=np.float64),
                depth_scale_ratio=np.asarray(1.0, dtype=np.float64),
            )
            with self.assertRaisesRegex(ValueError, "valid_mask"):
                audit_npz_float64(path, expected_resolution_wh=(1, 1))


if __name__ == "__main__":
    unittest.main()
