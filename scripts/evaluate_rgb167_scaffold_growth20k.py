#!/usr/bin/env python3
"""Strict four-arm RGB136/RGB167 endpoint and trajectory evaluation."""
from __future__ import annotations

import argparse
import json
import os
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
from src.quality_cameras import read_camera_requests
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_scaffold import ScaffoldQualityRenderer

ENDPOINTS = (15_000, 20_000, 30_000)
LABELS = ("RGB136_30000", "RGB167_15000", "RGB167_20000", "RGB167_30000")
DIFFICULT_GROUPS = {
    "159_160": ["159.png", "160.png"],
    "174": ["174.png"],
    "235_245": [f"{value:03d}.png" for value in range(235, 246)],
    "385_392": [f"{value:03d}.png" for value in range(385, 393)],
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def load8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def endpoint_files_bytes(model_root: Path, iteration: int) -> int:
    return sum(path.stat().st_size for path in (model_root / f"point_cloud/iteration_{iteration}").iterdir() if path.is_file())


def temporal_metrics(root: Path, names: list[str]) -> dict[str, float]:
    frames = [load8(root / name).astype(np.float32) / 255.0 for name in names]
    adjacent = [np.abs(frames[index] - frames[index - 1]).mean() for index in range(1, len(frames))]
    second = [np.abs(frames[index] - 2 * frames[index - 1] + frames[index - 2]).mean() for index in range(2, len(frames))]
    return {
        "mean_adjacent_l1": float(np.mean(adjacent)),
        "p95_adjacent_l1": float(np.percentile(adjacent, 95)),
        "mean_second_difference_l1": float(np.mean(second)),
        "p95_second_difference_l1": float(np.percentile(second, 95)),
        "interpretation": "no-GT camera-motion-confounded continuity descriptor, not accuracy or flicker ground truth",
    }


def make_montage(name: str, roots: dict[str, Path], photo_root: Path, output: Path, crop: tuple[int, int, int, int] | None) -> None:
    items = [("photo", photo_root / name), *[(label, root / name) for label, root in roots.items()]]
    cells = []
    for label, path in items:
        image = Image.open(path).convert("RGB")
        if crop is not None:
            image = image.crop(crop)
        image.thumbnail((360, 280))
        cells.append((label, image.copy()))
    canvas = Image.new("RGB", (380 * len(cells), 315), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(cells):
        draw.text((index * 380 + 8, 7), label, fill="black")
        canvas.paste(image, (index * 380 + 8, 28))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def make_trajectory_contact(name: str, roots: dict[str, Path], output: Path) -> None:
    canvas = Image.new("RGB", (1320, 1038), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, root) in enumerate(roots.items()):
        x, y = (index % 2) * 660, (index // 2) * 519
        draw.text((x + 6, y + 6), f"{name} {label}", fill="black")
        canvas.paste(Image.open(root / name).convert("RGB").resize((660, 494)), (x, y + 25))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def make_video(roots: dict[str, Path], names: list[str], output: Path) -> dict:
    command = [
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1320x1038",
        "-framerate", "12", "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-threads", "2",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for name in names:
            frame = Image.new("RGB", (1320, 1038), "white")
            draw = ImageDraw.Draw(frame)
            for index, (label, root) in enumerate(roots.items()):
                x, y = (index % 2) * 660, (index // 2) * 519
                draw.text((x + 6, y + 6), f"{name} {label}", fill="black")
                frame.paste(Image.open(root / name).convert("RGB").resize((660, 494)), (x, y + 25))
            process.stdin.write(frame.tobytes())
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


def lifecycle_stats(training: dict, endpoint: int) -> dict:
    events = training["lifecycle"]["adjust_events"]
    selected = []
    for event in events:
        iteration = 1600 + event["call_index"] * 100
        if iteration <= endpoint:
            selected.append({**event, "iteration": iteration})
    save = next(item for item in training["lifecycle"]["save_events"] if item["iteration"] == endpoint)
    return {
        "checkpoint_adjustment_events_through_endpoint": len(selected),
        "checkpoint_cumulative_grown": sum(item["grown"] for item in selected),
        "checkpoint_cumulative_pruned": sum(item["pruned"] for item in selected),
        "saved_model_pre_adjustment_anchors": save["anchors"],
        "saved_model_elapsed_seconds": save["elapsed_seconds"],
        "peak_allocated_mib_through_saved_model": save["peak_allocated_mib"],
        "peak_reserved_mib_through_saved_model": save["peak_reserved_mib"],
        "boundary_note": "upstream saves PLY before the iteration's adjust_anchor and saves checkpoint after it",
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
    if training.get("status") != "completed_from_scratch_30k" or training.get("start_checkpoint") is not None:
        raise ValueError("RGB167 is not a completed from-scratch 30k run")
    parent = ROOT / "outputs/quality/scaffold_noappearance/full-r1"
    parent_result = load(parent / "result.json")
    verify_hashes(parent, parent_result["artifacts"])
    parent_export = load(parent / "export/result.json")
    verify_hashes(parent / "export/images", parent_export["image_hashes"])
    for endpoint in ENDPOINTS:
        verification = load(args.run / f"verification_{endpoint}.json")
        if verification["status"] != "passed_finite_checkpoint_model_reload_and_uid_independence":
            raise ValueError(f"RGB167 endpoint {endpoint} was not independently verified")

    train_cameras = read_camera_requests(args.run / "scene/sparse/0")
    trajectory_path = args.artifact_root / "results/frozen_scaffold_sem001/flythrough_v1/cameras"
    trajectory = read_camera_requests(trajectory_path)
    trajectory = {name: replace(camera, model="PINHOLE", radial_k=0.0) for name, camera in trajectory.items()}
    if len(train_cameras) != 400 or len(trajectory) != 144:
        raise ValueError("Unexpected camera universe")
    requests = {**train_cameras, **trajectory}
    train_names = sorted(train_cameras)
    trajectory_names = sorted(trajectory)
    labels = ROOT / "outputs/quality/all_data/SEM-370-r1/labels"
    if len(list(labels.glob("*.png"))) != 300:
        raise ValueError("Expected 300 evaluation-only semantic masks")

    specs = {
        "RGB136_30000": (parent / "model", 30_000),
        "RGB167_15000": (args.run / "model", 15_000),
        "RGB167_20000": (args.run / "model", 20_000),
        "RGB167_30000": (args.run / "model", 30_000),
    }
    roots: dict[str, Path] = {}
    render_stats = {}
    torch.cuda.reset_peak_memory_stats()
    for label, (model_root, iteration) in specs.items():
        renderer = ScaffoldQualityRenderer(
            args.artifact_root / "r1/third_party/scaffold-gs", model_root, iteration, train_camera_count=400
        )
        if renderer.model.appearance_dim != 0 or renderer.model.embedding_appearance is not None:
            raise ValueError(f"{label} unexpectedly depends on UID")
        root = args.output / "images" / label
        root.mkdir(parents=True)
        roots[label] = root
        render_seconds = []
        hashes = {}
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
        if label == "RGB136_30000":
            for name, digest in hashes.items():
                if digest != parent_export["image_hashes"][name]:
                    raise ValueError(f"Fresh parent render differs: {name}")
        render_stats[label] = {
            "median_seconds_excluding_first": float(np.median(render_seconds[1:])),
            "mean_seconds_excluding_first": float(np.mean(render_seconds[1:])),
            "fps_from_median": float(1.0 / np.median(render_seconds[1:])),
            "rendered_images": len(render_seconds),
            "image_hashes": hashes,
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
            values["LPIPS"] = lpips_score(
                reference, prediction8.astype(np.float32) / 255.0, lpips_model, lpips_torch, lpips_device
            )
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
            roi_rows.append({
                "region": region_name,
                "view": name,
                "metrics": {
                    label: {
                        "mae_255": float(np.abs(load8(roots[label] / name).astype(np.float32) - reference8)[mask].mean()),
                        "psnr": float(-10.0 * np.log10(max(np.mean(
                            ((reference8.astype(np.float32)[mask] - load8(roots[label] / name).astype(np.float32)[mask]) / 255.0) ** 2
                        ), 1e-12))),
                    }
                    for label in LABELS
                },
            })
        if index % 50 == 0:
            print(json.dumps({"metrics": index + 1}), flush=True)
    del lpips_model
    torch.cuda.empty_cache()

    all400 = {label: aggregate([row[label] for row in per_view], ("PSNR", "SSIM", "LPIPS")) for label in LABELS}
    structures = {label: weighted_structural_aggregate(rows) for label, rows in structure_rows.items()}
    difficult = {
        group: {
            label: aggregate([row[label] for row in per_view if row["name"] in names], ("PSNR", "SSIM", "LPIPS"))
            for label in LABELS
        }
        for group, names in DIFFICULT_GROUPS.items()
    }
    bridge_crop = {}
    for label in LABELS:
        values = []
        for name, (x0, y0, x1, y1) in BRIDGE_ROIS.items():
            values.append(image_metrics(load8(args.run / "scene/images" / name)[y0:y1, x0:x1], load8(roots[label] / name)[y0:y1, x0:x1]))
        bridge_crop[label] = aggregate(values, ("PSNR", "SSIM"))
    trajectory_metrics = {label: temporal_metrics(roots[label], trajectory_names) for label in LABELS}

    difficult_names = sorted(set(sum(DIFFICULT_GROUPS.values(), [])))
    for name in difficult_names:
        crop = None
        if name in ("159.png", "160.png"):
            crop = (0, 560, 1320, 989)
        elif name == "174.png":
            crop = (0, 0, 1320, 430)
        elif name in BRIDGE_ROIS:
            crop = tuple(BRIDGE_ROIS[name])
        make_montage(name, roots, args.run / "scene/images", args.output / "montages" / f"{name[:-4]}.jpg", crop)
    for index in (0, 24, 48, 71, 72, 73, 96, 120, 143):
        make_trajectory_contact(trajectory_names[index], roots, args.output / "trajectory_contacts" / f"{index:03d}.png")
    video_probe = make_video(roots, trajectory_names, args.output / "trajectory_rgb136_rgb167.mp4")

    endpoint_resources = {}
    for endpoint in ENDPOINTS:
        verification = load(args.run / f"verification_{endpoint}.json")
        endpoint_resources[str(endpoint)] = {
            **lifecycle_stats(training, endpoint),
            "checkpoint_anchors": verification["anchors"],
            "checkpoint_bytes": verification["checkpoint_bytes"],
            "saved_model_bytes": verification["model_endpoint_bytes"],
            "render": render_stats[f"RGB167_{endpoint}"],
        }
    parent_checkpoint = parent / "model/chkpnt30000.pth"
    parent_state, parent_iteration = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    if parent_iteration != 30_000:
        raise ValueError("Unexpected RGB136 checkpoint iteration")
    parent_worker = load(parent / "worker_train.json")
    parent_resources = {
        "checkpoint_anchors": len(parent_state["anchor"]),
        "checkpoint_bytes": parent_checkpoint.stat().st_size,
        "saved_model_bytes": endpoint_files_bytes(parent / "model", 30_000),
        "training_seconds_total": parent_worker["seconds"],
        "peak_allocated_mib_total": parent_worker["peak_allocated_mib"],
        "peak_reserved_mib_total": parent_worker["peak_reserved_mib"],
        "cumulative_grown": None,
        "cumulative_pruned": None,
        "growth_prune_note": "historical RGB136 was not lifecycle-instrumented; gross counts cannot be reconstructed from net anchors",
        "render": render_stats["RGB136_30000"],
    }

    result = {
        "schema_version": 1,
        "status": "completed_strict_four_arm_evaluation",
        "scope": "all400 training-view fit; all300 labels used only for RGB region metrics; fixed 144 external cameras have no GT",
        "generalization_warning": "all400 is training-view fit and must not be described as unknown-view generalization",
        "all400": all400,
        "all300_labeled_bridge_structure": structures,
        "difficult_groups": difficult,
        "bridge_385_392_manual_crop": bridge_crop,
        "registered_rois": roi_rows,
        "trajectory_no_gt_temporal_descriptors": trajectory_metrics,
        "resources": {"RGB136_30000": parent_resources, "RGB167": endpoint_resources},
        "formal_training_total_seconds": training["worker"]["seconds"],
        "formal_training_peak_allocated_mib": training["worker"]["peak_allocated_mib"],
        "formal_training_peak_reserved_mib": training["worker"]["peak_reserved_mib"],
        "trajectory_names": trajectory_names,
        "trajectory_camera_sha256": {name: sha256(trajectory_path / name) for name in ("images.txt", "cameras.txt")},
        "video_probe": video_probe,
        "per_view": per_view,
        "render_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "seconds": time.monotonic() - started,
    }
    write_new_json(args.output / "result.json", result)
    print(json.dumps({"status": result["status"], "all400": all400, "resources": result["resources"]}, indent=2))


if __name__ == "__main__":
    main()
