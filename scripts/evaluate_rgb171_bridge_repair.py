#!/usr/bin/env python3
"""Evaluate a bridge-repair candidate against RGB166 and RGB170."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_rgb import build_lpips_model, lpips_score, psnr, ssim
from scripts.evaluate_rgb166_horizon import (
    BRIDGE_ROIS, CLASSES, DIFFICULT_GROUPS, aggregate, gradient, gray01, load8,
    make_montage, structural_metrics, weighted_structural_aggregate,
)
from src.quality_protocol import sha256, write_new_json

ARTIFACT_REGIONS = {
    "159-shore": ("159.png", [[20, 720, 290, 870], [920, 720, 1310, 860]]),
    "159-near": ("159.png", [[20, 890, 240, 985]]),
    "160-shore": ("160.png", [[20, 740, 285, 885], [920, 740, 1310, 875]]),
    "160-near": ("160.png", [[20, 905, 230, 985]]),
    "174-sky": ("174.png", [[0, 0, 1320, 380]]),
    "159-far-occlusion": ("159.png", [[850, 100, 1320, 340]]),
    "159-water-ghost": ("159.png", [[250, 350, 330, 620]]),
}


def vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        while True:
            line = handle.readline().decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header" or not line:
                raise ValueError(f"PLY vertex declaration missing: {path}")


def difference(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: float(candidate[key] - baseline[key]) for key in candidate if key in baseline}


def cable_continuity(reference8: np.ndarray, prediction8: np.ndarray, semantic: np.ndarray) -> dict[str, float]:
    """Measure photo-visible support along label-localized cable skeleton components."""
    mask = semantic == 2
    if not mask.any():
        return {"skeleton_pixels": 0.0, "supported_skeleton_fraction": 0.0,
                "component_recovered_fraction": 0.0, "component_count": 0.0}
    # Morphological skeletonization uses the already-required SciPy runtime and
    # avoids making the audit depend on the optional OpenCV contrib modules.
    structure = ndimage.generate_binary_structure(2, 1)
    eroded = mask.copy()
    skeleton = np.zeros_like(mask)
    while eroded.any():
        next_eroded = ndimage.binary_erosion(eroded, structure=structure)
        opened = ndimage.binary_dilation(next_eroded, structure=structure)
        skeleton |= eroded & ~opened
        eroded = next_eroded
    reference_contrast = np.abs(gray01(reference8) - cv2.GaussianBlur(gray01(reference8), (0, 0), 2))
    prediction_contrast = np.abs(gray01(prediction8) - cv2.GaussianBlur(gray01(prediction8), (0, 0), 2))
    visible = skeleton & (reference_contrast >= 3.0 / 255.0)
    supported = visible & (prediction_contrast >= 0.5 * reference_contrast)
    components, count = ndimage.label(skeleton, structure=np.ones((3, 3), dtype=np.uint8))
    recovered = []
    weights = []
    for component_id in range(1, count + 1):
        component = visible & (components == component_id)
        pixels = int(component.sum())
        if pixels >= 8:
            recovered.append(float(supported[component].mean()))
            weights.append(pixels)
    return {
        "skeleton_pixels": float(visible.sum()),
        "supported_skeleton_fraction": float(supported.sum() / max(visible.sum(), 1)),
        "component_recovered_fraction": float(np.mean(np.asarray(recovered) >= 0.75)) if recovered else 0.0,
        "component_support_p25": float(np.percentile(recovered, 25)) if recovered else 0.0,
        "component_count": float(len(recovered)),
    }


def aggregate_continuity(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    weights = np.asarray([row["skeleton_pixels"] for row in rows], dtype=np.float64)
    weights = np.maximum(weights, 1.0)
    output = {"skeleton_pixels": float(sum(row["skeleton_pixels"] for row in rows))}
    for key in ("supported_skeleton_fraction", "component_recovered_fraction", "component_support_p25"):
        output[key] = float(np.average([row[key] for row in rows], weights=weights))
    output["component_count"] = float(sum(row["component_count"] for row in rows))
    return output


def artifact_region(reference8: np.ndarray, prediction8: np.ndarray, boxes: list[list[int]]) -> dict[str, float]:
    ref_pixels, pred_pixels, weights, ssims = [], [], [], []
    gradient_l1, gradient_excess = [], []
    for x0, y0, x1, y1 in boxes:
        reference = reference8[y0:y1, x0:x1]
        prediction = prediction8[y0:y1, x0:x1]
        ref_pixels.append(reference.reshape(-1, 3))
        pred_pixels.append(prediction.reshape(-1, 3))
        weights.append(reference.shape[0] * reference.shape[1])
        ssims.append(ssim(reference.astype(np.float32) / 255.0, prediction.astype(np.float32) / 255.0))
        ref_grad, pred_grad = gradient(gray01(reference)), gradient(gray01(prediction))
        gradient_l1.append(float(np.abs(ref_grad - pred_grad).mean()))
        gradient_excess.append(float(pred_grad.mean() - ref_grad.mean()))
    reference = np.concatenate(ref_pixels).astype(np.float32) / 255.0
    prediction = np.concatenate(pred_pixels).astype(np.float32) / 255.0
    return {
        "pixels": float(sum(weights)),
        "mae_255": float(np.abs(reference - prediction).mean() * 255.0),
        "psnr": psnr(reference, prediction),
        "ssim": float(np.average(ssims, weights=weights)),
        "gradient_l1": float(np.average(gradient_l1, weights=weights)),
        "gradient_excess": float(np.average(gradient_excess, weights=weights)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--rgb166-images", type=Path, required=True)
    parser.add_argument("--rgb170-images", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-key", default="RGB171_50000")
    parser.add_argument("--candidate-display", default="RGB171 50k")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    labels = ("RGB166_50000", "RGB170_50000", args.candidate_key)
    sys.path[:0] = [str(args.source), str(args.source / "submodules/diff-gaussian-rasterization"),
                    str(args.source / "submodules/simple-knn"), str(args.source / "submodules/fused-ssim")]
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from src.quality_cameras import read_camera_requests
    from src.quality_native_gaussian import make_native_view

    cameras = read_camera_requests(args.scene / "sparse/0")
    names = sorted(path.name for path in (args.scene / "images").glob("*.png"))
    label_names = sorted(path.name for path in args.labels.glob("*.png"))
    if len(names) != 400 or len(label_names) != 300 or not set(names) <= set(cameras):
        raise ValueError("Expected 400 official images/cameras and 300 labels")
    for root in (args.rgb166_images, args.rgb170_images):
        if any(not (root / name).is_file() for name in names):
            raise FileNotFoundError(f"Incomplete baseline render root: {root}")
    torch.cuda.reset_peak_memory_stats()
    model = GaussianModel(3)
    model.load_ply(str(args.ply))
    pipe = SimpleNamespace(debug=False, antialiasing=False, depth_ratio=0.0)
    background = torch.zeros(3, device="cuda")
    image_root = args.output / "images" / "50000"
    image_root.mkdir(parents=True)
    render_seconds = []
    started = time.monotonic()
    with torch.inference_mode():
        for index, name in enumerate(names):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            tensor = render(make_native_view(cameras[name]), model, pipe, background,
                            track_gradients=False, inference_only=True)["render"]
            torch.cuda.synchronize()
            render_seconds.append(time.perf_counter() - tick)
            image8 = np.rint(tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(image8).save(image_root / name)
            if index % 100 == 0:
                print(json.dumps({"rendered": index + 1}), flush=True)
    roots = {"RGB166_50000": args.rgb166_images, "RGB170_50000": args.rgb170_images,
             args.candidate_key: image_root}
    lpips_model, lpips_torch, lpips_device = build_lpips_model("cuda")
    per_view = []
    structure_rows = {label: [] for label in labels}
    continuity_rows = {label: [] for label in labels}
    structure_by_name: dict[str, dict[str, dict]] = {}
    continuity_by_name: dict[str, dict[str, dict]] = {}
    for index, name in enumerate(names):
        reference8 = load8(args.scene / "images" / name)
        reference = reference8.astype(np.float32) / 255.0
        semantic_path = args.labels / name
        semantic = np.asarray(Image.open(semantic_path)) if semantic_path.is_file() else None
        row = {"name": name}
        structure_by_name[name], continuity_by_name[name] = {}, {}
        for label in labels:
            prediction8 = load8(roots[label] / name)
            values = {"PSNR": psnr(reference, prediction8.astype(np.float32) / 255.0),
                      "SSIM": ssim(reference, prediction8.astype(np.float32) / 255.0)}
            values["LPIPS"] = lpips_score(reference, prediction8.astype(np.float32) / 255.0,
                                           lpips_model, lpips_torch, lpips_device)
            row[label] = values
            if semantic is not None:
                structure = structural_metrics(reference8, prediction8, semantic)
                continuity = cable_continuity(reference8, prediction8, semantic)
                structure_rows[label].append(structure)
                continuity_rows[label].append(continuity)
                structure_by_name[name][label] = structure
                continuity_by_name[name][label] = continuity
        per_view.append(row)
        if index % 100 == 0:
            print(json.dumps({"evaluated": index + 1}), flush=True)
    del lpips_model, model
    torch.cuda.empty_cache()

    all400 = {label: aggregate([row[label] for row in per_view], ("PSNR", "SSIM", "LPIPS")) for label in labels}
    structures = {label: weighted_structural_aggregate(rows) for label, rows in structure_rows.items()}
    continuity = {label: aggregate_continuity(rows) for label, rows in continuity_rows.items()}
    difficult, difficult_structure, difficult_continuity = {}, {}, {}
    for group, group_names in DIFFICULT_GROUPS.items():
        selected = [row for row in per_view if row["name"] in group_names]
        difficult[group] = {label: aggregate([row[label] for row in selected], ("PSNR", "SSIM", "LPIPS"))
                            for label in labels}
        difficult_structure[group] = {
            label: weighted_structural_aggregate([structure_by_name[name][label] for name in group_names
                                                  if name in structure_by_name and label in structure_by_name[name]])
            for label in labels
        }
        difficult_continuity[group] = {
            label: aggregate_continuity([continuity_by_name[name][label] for name in group_names
                                         if name in continuity_by_name and label in continuity_by_name[name]])
            for label in labels
        }
    crop = {}
    for label in labels:
        values = []
        for name, (x0, y0, x1, y1) in BRIDGE_ROIS.items():
            reference = load8(args.scene / "images" / name)[y0:y1, x0:x1].astype(np.float32) / 255.0
            prediction = load8(roots[label] / name)[y0:y1, x0:x1].astype(np.float32) / 255.0
            values.append({"PSNR": psnr(reference, prediction), "SSIM": ssim(reference, prediction)})
        crop[label] = aggregate(values, ("PSNR", "SSIM"))
    artifact = {}
    for region, (name, boxes) in ARTIFACT_REGIONS.items():
        reference = load8(args.scene / "images" / name)
        artifact[region] = {label: artifact_region(reference, load8(roots[label] / name), boxes) for label in labels}

    montage_names = sorted(set(sum(DIFFICULT_GROUPS.values(), [])) | {"174.png"})
    for name in montage_names:
        make_montage(name, [("photo", args.scene / "images" / name),
                            ("RGB166 50k", args.rgb166_images / name),
                            ("RGB170 50k", args.rgb170_images / name),
                            (args.candidate_display, image_root / name)],
                     args.output / "montages" / f"{name[:-4]}.jpg")

    def comparison(baseline: str) -> dict:
        class_delta = {
            class_name: difference(structures[args.candidate_key][class_name], structures[baseline][class_name])
            for class_name in CLASSES.values()
        }
        return {
            "all400": difference(all400[args.candidate_key], all400[baseline]),
            "bridge_classes": class_delta,
            "stay_cable_continuity": difference(continuity[args.candidate_key], continuity[baseline]),
            "difficult_groups": {group: difference(values[args.candidate_key], values[baseline])
                                   for group, values in difficult.items()},
            "bridge_385_392_crop": difference(crop[args.candidate_key], crop[baseline]),
            "artifact_regions": {region: difference(values[args.candidate_key], values[baseline])
                                  for region, values in artifact.items()},
            "per_view_wins": {
                "PSNR": sum(row[args.candidate_key]["PSNR"] > row[baseline]["PSNR"] for row in per_view),
                "SSIM": sum(row[args.candidate_key]["SSIM"] > row[baseline]["SSIM"] for row in per_view),
                "LPIPS": sum(row[args.candidate_key]["LPIPS"] < row[baseline]["LPIPS"] for row in per_view),
            },
        }

    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1, "status": "completed", "candidate_key": args.candidate_key,
        "scope": "all400 training-view fit; all300 label-localized bridge structure; no unknown-view claim",
        "metric_direction": {"PSNR": "higher", "SSIM": "higher", "LPIPS": "lower",
                             "MAE/gradient_l1": "lower", "support/continuity": "higher",
                             "gradient_excess": "closer to zero"},
        "artifacts": {"candidate_ply_sha256": sha256(args.ply),
                      "candidate_ply_bytes": args.ply.stat().st_size,
                      "training_result_sha256": sha256(args.training_result)},
        "resources": {"gaussians": vertex_count(args.ply),
                      "training_seconds": training.get("worker", {}).get("seconds", training.get("seconds")),
                      "runner_wall_seconds": training.get("seconds"),
                      "training_peak_reserved_mib": training.get("peak_reserved_mib"),
                      "evaluation_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                      "render_seconds_median": float(np.median(render_seconds)),
                      "render_seconds_p90": float(np.percentile(render_seconds, 90)),
                      "render_fps_from_median": float(1.0 / np.median(render_seconds))},
        "all400": all400,
        "all300_labeled_bridge_structure": structures,
        "all300_stay_cable_continuity": continuity,
        "difficult_groups": difficult,
        "difficult_group_bridge_structure": difficult_structure,
        "difficult_group_stay_cable_continuity": difficult_continuity,
        "bridge_385_392_manual_crop": crop,
        "artifact_regions": artifact,
        "candidate_minus_rgb170": comparison("RGB170_50000"),
        "candidate_minus_rgb166": comparison("RGB166_50000"),
        "per_view": per_view,
        "seconds": time.monotonic() - started,
    }
    write_new_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
