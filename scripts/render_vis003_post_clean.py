#!/usr/bin/env python3
"""Render the VIS-001 144-pose trajectory with raw/cleaned semantics side-by-side."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem001 as s0
from render_external_colmap_camera import (
    make_view,
    read_pose,
    read_simple_radial_camera,
)

from src.camera_adapter import (
    Sim3,
    official_to_model_camera,
    redistort_rgb,
    redistort_semantic_probabilities,
)
from src.scaffold_semantic_sidecar import (
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    load_frozen_scaffold,
    pipeline,
    render_frozen_rgb,
    render_semantic_probabilities,
)
from src.semantic_postprocess import CLASS_COUNT, load_suppression_mask

PALETTE = np.asarray(
    [[0, 0, 0], [230, 86, 37], [255, 210, 0], [38, 139, 210], [133, 153, 0]], dtype=np.uint8
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_video(frame_dir: Path, output: Path, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame%04d.png"),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("VIS-003 requires ffmpeg for the registered H.264 video output") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="VIS-003 config")
    parser.add_argument("--sem070-config", type=Path, required=True)
    parser.add_argument("--sem001-config", type=Path, default=REPO_ROOT / "experiments/runs/SEM-001/config.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Frozen SEM-001 semantic checkpoint")
    parser.add_argument("--mask", type=Path, required=True, help="Frozen SEM-070 mask")
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True, help="Existing VIS-001 144-pose trajectory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    vis_config = json.loads(args.config.read_text(encoding="utf-8"))
    sem070_config = json.loads(args.sem070_config.read_text(encoding="utf-8"))
    sem001_config = json.loads(args.sem001_config.read_text(encoding="utf-8"))
    if sem070_config.get("experiment_id") != "SEM-070-POST-CLEAN":
        raise ValueError("VIS-003 received a different SEM-070 config")
    expected_checkpoint_sha256 = sem070_config.get("frozen_semantic", {}).get("checkpoint_sha256")
    if expected_checkpoint_sha256 and hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != expected_checkpoint_sha256:
        raise ValueError("SEM-001 checkpoint SHA-256 differs from the frozen SEM-070 config")
    trajectory = json.loads((args.trajectory_dir / "trajectory_manifest.json").read_text(encoding="utf-8"))
    if trajectory.get("trajectory_id") != "frozen_scaffold_sem001_bridge_flythrough_v1" or int(trajectory.get("frame_count", -1)) != 144:
        raise ValueError("VIS-003 must reuse the registered VIS-001 144-pose trajectory")
    if trajectory.get("pose_convention") != "CAMERA_FROM_WORLD" or trajectory.get("uses_reference_rgb_or_labels") is not False:
        raise ValueError("VIS-001 trajectory does not satisfy the official no-reference camera contract")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frozen = sem070_config["frozen_rgb"]
    settings = ScaffoldSettings(train_camera_count=int(frozen["train_camera_count"]), n_offsets=int(frozen["n_offsets"]))
    point_cloud = args.rgb_run_dir / "point_cloud" / f"iteration_{frozen['iteration']}" / "point_cloud.ply"
    rgb_hashes = frozen_rgb_hashes(args.rgb_run_dir, int(frozen["iteration"]))
    configure_scaffold_import(args.source_root)
    model = load_frozen_scaffold(args.source_root, point_cloud, settings)
    sidecar = s0.reload_sidecar(args.checkpoint, int(model.get_anchor.shape[0]), sem001_config, rgb_hashes)
    mask = load_suppression_mask(args.mask, int(model.get_anchor.shape[0]))

    import torch

    cameras_file = args.trajectory_dir / "cameras.txt"
    images_file = args.trajectory_dir / "images.txt"
    _camera_id, camera = read_simple_radial_camera(cameras_file, None)
    names = [entry["name"] for entry in trajectory["generated"]]
    if len(names) != 144 or len(set(names)) != 144:
        raise ValueError("VIS-001 trajectory generated names are not exactly 144 unique frames")
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device="cuda")
    pipe = pipeline()
    raw_root = args.output_dir / "semantic_raw"
    clean_root = args.output_dir / "semantic_cleaned"
    rgb_root = args.output_dir / "rgb"
    frame_root = args.output_dir / "comparison_frames"
    for directory in (raw_root, clean_root, rgb_root, frame_root):
        directory.mkdir(parents=True, exist_ok=True)
    identity = Sim3(1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    started = time.perf_counter()
    frame_records: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        selected_name, official_pose = read_pose(images_file, name, _camera_id)
        view = make_view(camera, official_to_model_camera(official_pose, identity), int(frozen["appearance_camera_id"]))
        with torch.inference_mode():
            rgb_pinhole = s0.tensor_to_rgb(render_frozen_rgb(view, model, pipe))
            raw_pinhole = render_semantic_probabilities(view, model, sidecar, pipe).cpu().numpy().astype(np.float32)
            clean_pinhole = render_semantic_probabilities(view, model, sidecar, pipe, anchor_postprocess_mask=mask_tensor).cpu().numpy().astype(np.float32)
        rgb = np.rint(redistort_rgb(rgb_pinhole.astype(np.float32) / 255.0, camera).clip(0.0, 1.0) * 255.0).astype(np.uint8)
        raw = redistort_semantic_probabilities(raw_pinhole, camera).argmax(axis=0).astype(np.uint8)
        cleaned = redistort_semantic_probabilities(clean_pinhole, camera).argmax(axis=0).astype(np.uint8)
        if raw.shape != (camera.height, camera.width) or cleaned.shape != raw.shape or raw.max() >= CLASS_COUNT or cleaned.max() >= CLASS_COUNT:
            raise RuntimeError(f"Invalid VIS-003 semantic frame: {selected_name}")
        stem = Path(selected_name).stem
        Image.fromarray(rgb, mode="RGB").save(rgb_root / f"{stem}.png")
        Image.fromarray(raw, mode="L").save(raw_root / f"{stem}.png")
        Image.fromarray(cleaned, mode="L").save(clean_root / f"{stem}.png")
        header = 43
        canvas = Image.new("RGB", (camera.width * 2, camera.height + header), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 14), "SEM-001 raw", fill="white")
        draw.text((camera.width + 12, 14), "SEM-070 POST-CLEAN (same mask)", fill="white")
        canvas.paste(Image.fromarray(PALETTE[raw], mode="RGB"), (0, header))
        canvas.paste(Image.fromarray(PALETTE[cleaned], mode="RGB"), (camera.width, header))
        frame_path = frame_root / f"frame{index:04d}.png"
        canvas.save(frame_path)
        frame_records.append({"index": index, "pose_name": selected_name, "raw": f"semantic_raw/{stem}.png", "cleaned": f"semantic_cleaned/{stem}.png", "comparison_frame": f"comparison_frames/{frame_path.name}"})
    video = args.output_dir / "semantic_raw_vs_cleaned_side_by_side.mp4"
    _save_video(frame_root, video, int(vis_config["trajectory"]["fps"]))
    result = {
        "schema_version": 1,
        "experiment_id": vis_config["experiment_id"],
        "parent_experiment": "SEM-070-POST-CLEAN",
        "trajectory_manifest_sha256": sha256(args.trajectory_dir / "trajectory_manifest.json"),
        "mask_sha256": sha256(args.mask),
        "frame_count": len(frame_records),
        "fps": int(vis_config["trajectory"]["fps"]),
        "duration_seconds": len(frame_records) / int(vis_config["trajectory"]["fps"]),
        "render_seconds": time.perf_counter() - started,
        "input_pose_convention": "CAMERA_FROM_WORLD",
        "input_camera_model": "SIMPLE_RADIAL",
        "uses_reference_rgb_or_labels": False,
        "same_filter_for_formal_and_visual": True,
        "video": {"path": video.name, "sha256": sha256(video), "codec": "H.264 yuv420p", "resolution": [camera.width * 2, camera.height + 43]},
        "frames": frame_records,
    }
    (args.output_dir / "render_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "frame_count": len(frame_records), "video_sha256": result["video"]["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
