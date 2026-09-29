#!/usr/bin/env python3
"""Evaluate all five RGB168 Scaffold endpoints with the frozen RGB167 protocol."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_rgb import build_lpips_model, lpips_score
from scripts.evaluate_rgb166_horizon import (
    BRIDGE_ROIS, aggregate, image_metrics, structural_metrics, weighted_structural_aggregate,
)
from scripts.evaluate_rgb167_scaffold_growth20k import lifecycle_stats, load8, temporal_metrics
from src.quality_cameras import read_camera_requests
from src.quality_protocol import sha256, write_new_json
from src.quality_scaffold import ScaffoldQualityRenderer

ENDPOINTS = (30_000, 35_000, 40_000, 45_000, 50_000)
LABELS = tuple(f"RGB168_{value}" for value in ENDPOINTS)
DIFFICULT_GROUPS = {
    "159_160": ["159.png", "160.png"],
    "174": ["174.png"],
    "235_245": [f"{value:03d}.png" for value in range(235, 246)],
    "385_392": [f"{value:03d}.png" for value in range(385, 393)],
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def make_montage(name: str, roots: dict[str, Path], photo_root: Path, output: Path,
                 crop: tuple[int, int, int, int] | None) -> None:
    items = [("photo", photo_root / name), *[(label, root / name) for label, root in roots.items()]]
    cells = []
    for label, path in items:
        image = Image.open(path).convert("RGB")
        if crop is not None:
            image = image.crop(crop)
        image.thumbnail((320, 250))
        cells.append((label, image.copy()))
    canvas = Image.new("RGB", (340 * len(cells), 285), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(cells):
        draw.text((index * 340 + 6, 6), label, fill="black")
        canvas.paste(image, (index * 340 + 6, 27))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)


def trajectory_frame(name: str, roots: dict[str, Path]) -> Image.Image:
    cell_w, cell_h, label_h = 440, 330, 22
    canvas = Image.new("RGB", (cell_w * 3, (cell_h + label_h) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, root) in enumerate(roots.items()):
        x, y = (index % 3) * cell_w, (index // 3) * (cell_h + label_h)
        draw.text((x + 5, y + 4), f"{name} · {label}", fill="black")
        canvas.paste(Image.open(root / name).convert("RGB").resize((cell_w, cell_h)), (x, y + label_h))
    return canvas


def make_video(roots: dict[str, Path], names: list[str], output: Path) -> dict:
    command = [
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", "1320x704", "-framerate", "12", "-i", "-", "-c:v", "libx264",
        "-crf", "18", "-preset", "fast", "-threads", "2", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for name in names:
            process.stdin.write(trajectory_frame(name, roots).tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    except BaseException:
        process.kill()
        process.wait()
        raise
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate", "-of", "json", str(output),
    ], text=True))
    stream = probe["streams"][0]
    if stream["nb_read_frames"] != "144" or stream["r_frame_rate"] != "12/1":
        raise ValueError("Unexpected trajectory video metadata")
    return probe


def imported_comparisons() -> dict:
    scaffold = load(ROOT / "outputs/quality/RGB-167/full-r1/evaluation-r1/result.json")
    improved = load(ROOT / "outputs/quality/RGB-166/full-r1/evaluation-r1/result.json")
    improved_per_view = improved["per_view"]
    improved_174 = {
        label: aggregate([row[label] for row in improved_per_view if row["name"] == "174.png"],
                         ("PSNR", "SSIM", "LPIPS"))
        for label in ("30000", "35000", "40000", "45000", "50000")
    }
    return {
        "RGB136_30000": {
            "all400": scaffold["all400"]["RGB136_30000"],
            "structure": scaffold["all300_labeled_bridge_structure"]["RGB136_30000"],
            "difficult": {key: value["RGB136_30000"] for key, value in scaffold["difficult_groups"].items()},
            "bridge_crop": scaffold["bridge_385_392_manual_crop"]["RGB136_30000"],
            "trajectory": scaffold["trajectory_no_gt_temporal_descriptors"]["RGB136_30000"],
            "resources": scaffold["resources"]["RGB136_30000"],
        },
        "RGB167_30000": {
            "all400": scaffold["all400"]["RGB167_30000"],
            "structure": scaffold["all300_labeled_bridge_structure"]["RGB167_30000"],
            "difficult": {key: value["RGB167_30000"] for key, value in scaffold["difficult_groups"].items()},
            "bridge_crop": scaffold["bridge_385_392_manual_crop"]["RGB167_30000"],
            "trajectory": scaffold["trajectory_no_gt_temporal_descriptors"]["RGB167_30000"],
            "resources": scaffold["resources"]["RGB167"]["30000"],
        },
        "RGB166": {
            "all400": {key: improved["all400"][key] for key in ("30000", "35000", "40000", "45000", "50000")},
            "structure": {key: improved["all300_labeled_bridge_structure"][key] for key in ("30000", "35000", "40000", "45000", "50000")},
            "difficult": {
                "159_160": {key: improved["difficult_groups"]["bridge_159_160"][key] for key in ("30000", "35000", "40000", "45000", "50000")},
                "174": improved_174,
                "235_245": {key: improved["difficult_groups"]["bridge_235_245"][key] for key in ("30000", "35000", "40000", "45000", "50000")},
                "385_392": {key: improved["difficult_groups"]["bridge_385_392"][key] for key in ("30000", "35000", "40000", "45000", "50000")},
            },
            "bridge_crop": {key: improved["bridge_385_392_manual_crop"][key] for key in ("30000", "35000", "40000", "45000", "50000")},
            "source_result_sha256": sha256(ROOT / "outputs/quality/RGB-166/full-r1/evaluation-r1/result.json"),
        },
        "source_sha256": {
            "RGB167_evaluation": sha256(ROOT / "outputs/quality/RGB-167/full-r1/evaluation-r1/result.json"),
            "RGB166_evaluation": sha256(ROOT / "outputs/quality/RGB-166/full-r1/evaluation-r1/result.json"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch

    started = time.monotonic()
    training = load(args.run / "result.json")
    if training.get("status") != "completed_from_scratch_50k" or training.get("start_checkpoint") is not None:
        raise ValueError("RGB168 is not a completed from-scratch 50k run")
    for endpoint in ENDPOINTS:
        verification = load(args.run / f"verification_{endpoint}.json")
        if verification["status"] != "passed_finite_checkpoint_model_reload_and_uid_independence":
            raise ValueError(f"RGB168 endpoint {endpoint} was not independently verified")

    train_cameras = read_camera_requests(args.run / "scene/sparse/0")
    trajectory_path = args.artifact_root / "results/frozen_scaffold_sem001/flythrough_v1/cameras"
    trajectory = read_camera_requests(trajectory_path)
    trajectory = {name: replace(camera, model="PINHOLE", radial_k=0.0) for name, camera in trajectory.items()}
    if len(train_cameras) != 400 or len(trajectory) != 144:
        raise ValueError("Unexpected camera universe")
    requests = {**train_cameras, **trajectory}
    train_names, trajectory_names = sorted(train_cameras), sorted(trajectory)
    labels = ROOT / "outputs/quality/all_data/SEM-370-r1/labels"
    if len(list(labels.glob("*.png"))) != 300:
        raise ValueError("Expected 300 evaluation-only semantic masks")

    roots: dict[str, Path] = {}
    render_stats = {}
    torch.cuda.reset_peak_memory_stats()
    for endpoint, label in zip(ENDPOINTS, LABELS):
        renderer = ScaffoldQualityRenderer(
            args.artifact_root / "r1/third_party/scaffold-gs", args.run / "model", endpoint,
            train_camera_count=400,
        )
        if renderer.model.appearance_dim != 0 or renderer.model.embedding_appearance is not None:
            raise ValueError(f"{label} unexpectedly depends on UID")
        root = args.output / "images" / label
        root.mkdir(parents=True)
        roots[label] = root
        render_seconds, hashes = [], {}
        with torch.inference_mode():
            for index, (name, camera) in enumerate(sorted(requests.items())):
                torch.cuda.synchronize()
                mark = time.monotonic()
                rgb = renderer.render_pinhole(camera).rgb
                torch.cuda.synchronize()
                render_seconds.append(time.monotonic() - mark)
                image8 = np.rint(rgb * 255).astype(np.uint8)
                if image8.shape != (989, 1320, 3) or not np.isfinite(rgb).all():
                    raise ValueError(f"Invalid render {label}/{name}")
                Image.fromarray(image8).save(root / name)
                hashes[name] = sha256(root / name)
                if index % 100 == 0:
                    print(json.dumps({"arm": label, "rendered": index + 1}), flush=True)
        render_stats[label] = {
            "median_seconds_excluding_first": float(np.median(render_seconds[1:])),
            "mean_seconds_excluding_first": float(np.mean(render_seconds[1:])),
            "fps_from_median": float(1.0 / np.median(render_seconds[1:])),
            "rendered_images": len(render_seconds), "image_hashes": hashes,
        }
        del renderer
        torch.cuda.empty_cache()

    lpips_model, lpips_torch, lpips_device = build_lpips_model("cuda")
    per_view = []
    structure_rows = {label: [] for label in LABELS}
    region_config = load(ROOT / "experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml")["regions"]
    roi_rows = []
    for index, name in enumerate(train_names):
        reference8 = load8(args.run / "scene/images" / name)
        reference = reference8.astype(np.float32) / 255.0
        semantic_path = labels / name
        semantic = np.asarray(Image.open(semantic_path)) if semantic_path.is_file() else None
        row: dict[str, object] = {"name": name}
        for label in LABELS:
            prediction8 = load8(roots[label] / name)
            values = image_metrics(reference8, prediction8)
            values["LPIPS"] = lpips_score(reference, prediction8.astype(np.float32) / 255.0,
                                            lpips_model, lpips_torch, lpips_device)
            row[label] = values
            if semantic is not None:
                structure_rows[label].append(structural_metrics(reference8, prediction8, semantic))
        per_view.append(row)
        for region_name, region in region_config.items():
            if region["view"] != name:
                continue
            mask = np.zeros(reference8.shape[:2], dtype=bool)
            for x0, y0, x1, y1 in region["boxes"]:
                mask[y0:y1, x0:x1] = True
            roi_rows.append({"region": region_name, "view": name, "metrics": {
                label: {
                    "mae_255": float(np.abs(load8(roots[label] / name).astype(np.float32) - reference8)[mask].mean()),
                    "psnr": float(-10.0 * np.log10(max(np.mean(((reference8.astype(np.float32)[mask] - load8(roots[label] / name).astype(np.float32)[mask]) / 255.0) ** 2), 1e-12))),
                } for label in LABELS
            }})
        if index % 50 == 0:
            print(json.dumps({"metrics": index + 1}), flush=True)
    del lpips_model
    torch.cuda.empty_cache()

    all400 = {label: aggregate([row[label] for row in per_view], ("PSNR", "SSIM", "LPIPS")) for label in LABELS}
    structures = {label: weighted_structural_aggregate(rows) for label, rows in structure_rows.items()}
    difficult = {group: {label: aggregate([row[label] for row in per_view if row["name"] in names],
                                          ("PSNR", "SSIM", "LPIPS")) for label in LABELS}
                 for group, names in DIFFICULT_GROUPS.items()}
    bridge_crop = {}
    for label in LABELS:
        values = [image_metrics(load8(args.run / "scene/images" / name)[y0:y1, x0:x1],
                                load8(roots[label] / name)[y0:y1, x0:x1])
                  for name, (x0, y0, x1, y1) in BRIDGE_ROIS.items()]
        bridge_crop[label] = aggregate(values, ("PSNR", "SSIM"))
    trajectory_metrics = {label: temporal_metrics(roots[label], trajectory_names) for label in LABELS}

    for name in sorted(set(sum(DIFFICULT_GROUPS.values(), []))):
        crop = (0, 560, 1320, 989) if name in ("159.png", "160.png") else ((0, 0, 1320, 430) if name == "174.png" else (tuple(BRIDGE_ROIS[name]) if name in BRIDGE_ROIS else None))
        make_montage(name, roots, args.run / "scene/images", args.output / "montages" / f"{name[:-4]}.jpg", crop)
    for index in (0, 24, 48, 71, 72, 73, 96, 120, 143):
        frame = trajectory_frame(trajectory_names[index], roots)
        target = args.output / "trajectory_contacts" / f"{index:03d}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.save(target)
    video_probe = make_video(roots, trajectory_names, args.output / "trajectory_rgb168_endpoints.mp4")

    resources = {}
    for endpoint, label in zip(ENDPOINTS, LABELS):
        verification = load(args.run / f"verification_{endpoint}.json")
        resources[str(endpoint)] = {
            **lifecycle_stats(training, endpoint),
            "checkpoint_anchors": verification["anchors"],
            "checkpoint_bytes": verification["checkpoint_bytes"],
            "saved_model_bytes": verification["model_endpoint_bytes"],
            "render": render_stats[label],
        }
    result = {
        "schema_version": 1, "status": "completed_strict_five_endpoint_evaluation",
        "scope": "all400 training-view fit; all300 labels used only for RGB region metrics; fixed 144 external cameras have no GT",
        "generalization_warning": "all400 is training-view fit and must not be described as unknown-view generalization",
        "all400": all400, "all300_labeled_bridge_structure": structures,
        "difficult_groups": difficult, "bridge_385_392_manual_crop": bridge_crop,
        "registered_rois": roi_rows, "trajectory_no_gt_temporal_descriptors": trajectory_metrics,
        "resources": resources, "imported_comparisons": imported_comparisons(),
        "formal_training_total_seconds": training["worker"]["seconds"],
        "formal_training_peak_allocated_mib": training["worker"]["peak_allocated_mib"],
        "formal_training_peak_reserved_mib": training["worker"]["peak_reserved_mib"],
        "trajectory_names": trajectory_names,
        "trajectory_camera_sha256": {name: sha256(trajectory_path / name) for name in ("images.txt", "cameras.txt")},
        "video_probe": video_probe, "per_view": per_view,
        "render_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "seconds": time.monotonic() - started,
    }
    write_new_json(args.output / "result.json", result)
    print(json.dumps({"status": result["status"], "all400": all400, "resources": resources}, indent=2))


if __name__ == "__main__":
    main()
