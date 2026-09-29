#!/usr/bin/env python3
"""Same-pose, reference-free RGB-030/091/092 trajectory comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from render_standard_flythrough import read_poses, render_to_u8
from render_external_colmap_camera import read_simple_radial_camera, make_view
from src.camera_adapter import redistort_rgb
from src.scaffold_semantic_sidecar import ScaffoldSettings, configure_scaffold_import, load_frozen_scaffold, pipeline, render_frozen_rgb
from src.semantic2d import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    root = args.output_root
    trajectory = root / "results/frozen_scaffold_sem001/flythrough_v1/cameras"
    output = root / "r2/rgb_092/flythrough/full-20260906-r1"
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((trajectory / "trajectory_manifest.json").read_text())
    poses = read_poses(trajectory / "images.txt")
    if len(poses) != 144 or manifest["uses_reference_rgb_or_labels"]:
        raise ValueError("Expected frozen 144-view, metadata-only VIS-001 trajectory")
    camera = read_simple_radial_camera(trajectory / "cameras.txt", poses[0][1])[1]
    source = root / "r1/third_party/scaffold-gs"
    configure_scaffold_import(source)
    import torch
    records = {}
    for ident in ["RGB-030-FULL", "RGB-091-CTRL", "RGB-092-TRANSIENT"]:
        run = root / "r1/runs" / ident
        directory = output / ident
        directory.mkdir()
        model = load_frozen_scaffold(source, run / "point_cloud/iteration_30000/point_cloud.ply", ScaffoldSettings())
        records[ident] = []
        for index, (name, _id, pose) in enumerate(poses):
            view = make_view(camera, pose, 0)
            with torch.inference_mode():
                rgb = render_to_u8(render_frozen_rgb(view, model, pipeline()))
            # Same official pixel grid for all models, fixed appearance index 0.
            rgb = np.rint(redistort_rgb(rgb.astype(np.float32) / 255, camera).clip(0, 1) * 255).astype(np.uint8)
            path = directory / f"frame{index:04}.png"
            Image.fromarray(rgb).save(path)
            records[ident].append({"pose": name, "frame": path.name, "sha256": sha256(path)})
        del model
        torch.cuda.empty_cache()
    frames = output / "comparisons"
    frames.mkdir()
    for index in range(144):
        canvas = Image.new("RGB", (3960, 1026), "white")
        for j, ident in enumerate(records):
            canvas.paste(Image.open(output / ident / f"frame{index:04}.png"), (j * 1320, 36))
            ImageDraw.Draw(canvas).text((j * 1320 + 12, 12), ident, fill="black")
        canvas.save(frames / f"frame{index:04}.png")
    video = output / "rgb030_rgb091_rgb092_same_pose.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-framerate", "24", "-i", str(frames / "frame%04d.png"),
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(video)], check=True)
    (output / "result.json").write_text(json.dumps({"status": "completed", "frames": 144,
                "trajectory_sha256": sha256(trajectory / "trajectory_manifest.json"), "records": records,
                "video": video.name, "video_sha256": sha256(video), "appearance_index": 0,
                "coordinate_frame": "official_COLMAP", "pose_convention": "CAMERA_FROM_WORLD",
                "uses_reference_rgb_or_labels": False, "display_filters": False}, indent=2))
    print(video)


if __name__ == "__main__":
    main()
