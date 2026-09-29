#!/usr/bin/env python3
"""Run the approved DA3-Large P0 window smoke in the isolated geometry env.

This adapter deliberately consumes the prepared, full-frame undistorted PINHOLE
scene and official CAMERA_FROM_WORLD poses.  DA3 predicted poses are retained
only as diagnostics; all world points are back-projected with official P0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from geometry_prior.camera_io import read_cameras_txt, read_images_txt, world_to_camera


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def camera_matrix(camera) -> np.ndarray:
    if camera.model != "PINHOLE" or len(camera.params) != 4:
        raise ValueError(f"prepared DA3 scene must use PINHOLE, got {camera.model}")
    fx, fy, cx, cy = camera.params
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def cam2world(image) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = image.rotation_camera_to_world
    out[:3, 3] = image.center
    return out


def read_sparse_points(path: Path) -> dict[int, np.ndarray]:
    points = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        points[int(fields[0])] = np.asarray(list(map(float, fields[1:4])), dtype=np.float64)
    return points


def sparse_scale(depth: np.ndarray, K: np.ndarray, image, points: dict[int, np.ndarray]) -> tuple[float, int]:
    ids = sorted(image.point3d_ids.intersection(points))
    if not ids:
        return 1.0, 0
    world = np.stack([points[i] for i in ids])
    cam = world_to_camera(world, image)
    z = cam[:, 2]
    u = K[0, 0] * cam[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
    v = K[1, 1] * cam[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
    x, y = np.rint(u).astype(int), np.rint(v).astype(int)
    inside = (z > 1e-5) & (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
    pred = np.zeros_like(z)
    pred[inside] = depth[y[inside], x[inside]]
    valid = inside & np.isfinite(pred) & (pred > 1e-5)
    if int(valid.sum()) < 20:
        return 1.0, int(valid.sum())
    ratio = float(np.median(z[valid] / pred[valid]))
    return (ratio if np.isfinite(ratio) and ratio > 0 else 1.0), int(valid.sum())


def world_points(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing="xy")
    pc = np.stack(((x - K[0, 2]) * depth / K[0, 0], (y - K[1, 2]) * depth / K[1, 1], depth), axis=-1)
    return (pc @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-root", type=Path, required=True)
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--windows-json", type=Path, required=True)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--checkpoint-sha256", required=True)
    ap.add_argument("--process-res", type=int, default=504)
    args = ap.parse_args()

    import torch
    from depth_anything_3.api import DepthAnything3

    scene = args.scene_root
    sparse = scene / "sparse" / "0"
    cameras = read_cameras_txt(sparse / "cameras.txt")
    images = read_images_txt(sparse / "images.txt")
    by_name = {im.name: im for im in images.values()}
    points = read_sparse_points(sparse / "points3D.txt")
    payload = json.loads(args.windows_json.read_text(encoding="utf-8"))
    windows = payload["windows"]
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("DA3 P0 smoke requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    model = DepthAnything3.from_pretrained(str(args.checkpoint_root), local_files_only=True).to(device).eval()
    load_seconds = time.monotonic() - start
    records = []
    try:
        for window in windows:
            names = window["images"]
            if len(names) != 8:
                raise ValueError("PF-R2 smoke windows must contain exactly 8 training views")
            paths = [str(scene / "images" / n) for n in names]
            exts, ixts = [], []
            for name in names:
                image = by_name[name]
                K = camera_matrix(cameras[image.camera_id])
                c2w = cam2world(image)
                exts.append(np.linalg.inv(c2w))
                ixts.append(K)
            t0 = time.monotonic()
            pred = model.inference(
                paths,
                extrinsics=np.asarray(exts, dtype=np.float32),
                intrinsics=np.asarray(ixts, dtype=np.float32),
                align_to_input_ext_scale=True,
                infer_gs=False,
                use_ray_pose=False,
                ref_view_strategy="first",
                process_res=args.process_res,
                process_res_method="upper_bound_resize",
            )
            elapsed = time.monotonic() - t0
            if pred.conf is None or pred.extrinsics is None or pred.intrinsics is None:
                raise RuntimeError("DA3 prediction lacks confidence or camera outputs")
            for idx, name in enumerate(names):
                depth = np.asarray(pred.depth[idx], dtype=np.float32)
                conf = np.asarray(pred.conf[idx], dtype=np.float32)
                Kp = np.asarray(pred.intrinsics[idx], dtype=np.float64)
                official_c2w = cam2world(by_name[name])
                ratio, corr = sparse_scale(depth, Kp, by_name[name], points)
                depth = depth * ratio
                valid = np.isfinite(depth) & np.isfinite(conf) & (depth > 1e-5)
                wp = world_points(depth, Kp, official_c2w)
                wp[~valid] = np.nan
                pred_w2c = np.asarray(pred.extrinsics[idx], dtype=np.float64)
                if pred_w2c.shape == (3, 4):
                    pred_w2c = np.vstack([pred_w2c, [0, 0, 0, 1]])
                if pred_w2c.shape != (4, 4):
                    raise RuntimeError(f"unexpected DA3 extrinsics shape: {pred_w2c.shape}")
                pred_c2w = np.linalg.inv(pred_w2c)
                out = args.output_dir / "windows" / window["window_id"] / f"{Path(name).stem}.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out,
                    depth=depth.astype(np.float32),
                    confidence=conf.astype(np.float32),
                    valid_mask=valid.astype(np.uint8),
                    world_points=wp.astype(np.float32),
                    pinhole_intrinsics=Kp,
                    official_cam2world=official_c2w,
                    pred_cam2world=pred_c2w,
                    depth_scale_ratio=np.asarray(ratio, dtype=np.float64),
                )
                records.append({
                    "window_manifest_id": window["window_id"], "image_name": name,
                    "official_camera_id": by_name[name].camera_id,
                    "input_resolution": [int(depth.shape[1]), int(depth.shape[0])],
                    "coordinate_frame": "official_colmap_reference",
                    "depth_scale": {"method": "per_view_median_official_DLT_sparse_depth_ratio", "ratio": ratio, "correspondence_count": corr},
                    "source_model": "DA3-LARGE-1.1", "source_commit": args.source_commit,
                    "checkpoint_sha256": args.checkpoint_sha256,
                    "output_npz": str(out.relative_to(args.output_dir)),
                    "finite_depth_fraction": float(np.isfinite(depth).mean()), "valid_fraction": float(valid.mean()),
                    "nan_or_inf": bool((~np.isfinite(depth)).any() or (~np.isfinite(conf)).any()),
                })
    finally:
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        del model
        torch.cuda.empty_cache()
    summary = {"schema_version": 2, "geometry_output_schema": {"schema_id": "geometry_prior_npz_v2"}, "experiment_id": "GEO-020-PF-R2", "model_execution": "completed", "model_load_seconds": load_seconds, "inference_seconds": time.monotonic() - start, "peak_torch_allocated_mib": peak, "amp_dtype": "bf16" if torch.cuda.is_bf16_supported() else "fp16", "records": records, "windows": windows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
