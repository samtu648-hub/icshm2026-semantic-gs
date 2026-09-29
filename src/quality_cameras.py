"""Metadata-only COLMAP camera loading for the new native backbone interface."""
from pathlib import Path
import numpy as np

from src.camera_adapter import CameraFromWorld
from src.quality_backbone import CameraRequest
from scripts.prepare_rgb_colmap_scene import parse_colmap_images


def read_camera_requests(sparse):
    sparse = Path(sparse)
    intrinsics = {}
    for raw in (sparse / "cameras.txt").read_text().splitlines():
        row = raw.split()
        if not row or row[0].startswith("#"):
            continue
        ident, model, width, height = int(row[0]), row[1], int(row[2]), int(row[3])
        values = [float(v) for v in row[4:]]
        if model == "PINHOLE" and len(values) == 4:
            fx, fy, cx, cy = values
            radial = 0.
        elif model == "SIMPLE_RADIAL" and len(values) == 4:
            fx, cx, cy, radial = values
            fy = fx
        else:
            raise ValueError("Unsupported native/official COLMAP camera")
        if ident in intrinsics:
            raise ValueError("Duplicate camera ID")
        intrinsics[ident] = dict(width=width, height=height, matrix=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]), model=model, radial_k=radial)
    return {r.name: CameraRequest(pose=CameraFromWorld(r.rotation, r.translation), **intrinsics[r.camera_id])
            for r in parse_colmap_images(sparse / "images.txt")}
