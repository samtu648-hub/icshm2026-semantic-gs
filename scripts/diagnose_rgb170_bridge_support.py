#!/usr/bin/env python3
"""Read-only native contribution and projection audit for RGB170 bridge support."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import binary_dilation, distance_transform_edt, sobel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_rgb166_horizon import BRIDGE_ROIS
from src.bridge_gaussian_support import (
    project_gaussians,
    quantiles,
    summarize_records,
    undirected_angle_error,
)
from src.quality_protocol import sha256, write_new_json

EXPERIMENT_ID = "QUAL-022-BRIDGE-GAUSSIAN-SUPPORT-DIAG"
CONFIG = ROOT / "experiments/runs" / EXPERIMENT_ID / "config.yaml"
SOURCE = ROOT / "outputs/quality/source/improvedgs-full"
SCENE = ROOT / "outputs/quality/all_data/RGB-120-r2/scene"
LABELS = ROOT / "outputs/quality/all_data/SEM-370-r1/labels"
PLY170 = ROOT / "outputs/quality/RGB-170/full-r1/model/point_cloud/iteration_50000/point_cloud.ply"
PLY166 = ROOT / "outputs/quality/RGB-166/full-r1/model/point_cloud/iteration_50000/point_cloud.ply"
RGB166 = ROOT / "outputs/quality/RGB-166/full-r1/evaluation-r1/images/50000"
RGB170 = ROOT / "outputs/quality/RGB-170/full-r1/evaluation-r1/images/50000"
CLASSES = {1: "deck", 2: "stay_cable", 3: "tower", 4: "foundation"}
FOCUS = {
    "bridge_159_160": ["159.png", "160.png"],
    "bridge_235_245": [f"{index:03d}.png" for index in range(235, 246)],
    "bridge_385_392": [f"{index:03d}.png" for index in range(385, 393)],
}
REVIEW_NAMES = ["159.png", "160.png", "235.png", "240.png", "245.png", *FOCUS["bridge_385_392"]]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def gray01(image: np.ndarray) -> np.ndarray:
    value = image.astype(np.float32) / 255.0
    return 0.299 * value[..., 0] + 0.587 * value[..., 1] + 0.114 * value[..., 2]


def local_contrast(image: np.ndarray) -> np.ndarray:
    gray = gray01(image)
    blur = np.asarray(
        Image.fromarray(np.rint(gray * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)),
        dtype=np.float32,
    ) / 255.0
    return np.abs(gray - blur)


def photo_tangent(image: np.ndarray) -> np.ndarray:
    gray = gray01(image)
    gx, gy = sobel(gray, axis=1), sobel(gray, axis=0)
    return np.arctan2(gy, gx) + math.pi / 2.0


def photo_line_mask(image: np.ndarray, roi: list[int]) -> np.ndarray:
    gray = gray01(image)
    gx, gy = sobel(gray, axis=1), sobel(gray, axis=0)
    magnitude = np.hypot(gx, gy)
    x0, y0, x1, y1 = roi
    crop = magnitude[y0:y1, x0:x1]
    threshold = max(float(np.quantile(crop, 0.82)), 3.0 / 255.0)
    mask = np.zeros(gray.shape, dtype=bool)
    mask[y0:y1, x0:x1] = crop >= threshold
    return mask


def polygon_ellipse(row: dict[str, Any], count: int = 40) -> list[tuple[float, float]]:
    major = min(float(row["major_radius_px"]), 250.0)
    minor = min(float(row["minor_radius_px"]), 250.0)
    angle = float(row["angle_rad"])
    cosine, sine = math.cos(angle), math.sin(angle)
    points = []
    for step in range(count):
        phase = 2.0 * math.pi * step / count
        x, y = major * math.cos(phase), minor * math.sin(phase)
        points.append((row["u"] + cosine * x - sine * y, row["v"] + sine * x + cosine * y))
    return points


def ellipse_coverage(shape: tuple[int, int], rows: list[dict[str, Any]], target: np.ndarray) -> float:
    canvas = Image.new("1", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    for row in rows:
        if row["projection_valid"] and row["major_radius_px"] > 0:
            draw.polygon(polygon_ellipse(row), fill=1)
    covered = np.asarray(canvas, dtype=bool)
    return float((covered & target).sum() / max(int(target.sum()), 1))


def comparison_page(name: str, overlay: Image.Image, output: Path) -> None:
    sources = [
        ("photo", SCENE / "images" / name),
        ("RGB166", RGB166 / name),
        ("RGB170", RGB170 / name),
    ]
    crop = BRIDGE_ROIS.get(name)
    cells: list[tuple[str, Image.Image]] = []
    for label, path in sources:
        image = Image.open(path).convert("RGB")
        if crop:
            image = image.crop(tuple(crop))
        image.thumbnail((480, 340))
        cells.append((label, image.copy()))
    image = overlay.copy()
    if crop:
        image = image.crop(tuple(crop))
    image.thumbnail((480, 340))
    cells.append(("RGB170 top contributors on photo", image))
    canvas = Image.new("RGB", (500 * len(cells), 380), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(cells):
        x = 500 * index
        draw.text((x + 8, 8), label, fill="black")
        canvas.paste(image, (x + 8, 32))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)


def finite_top(weights: Any, count: int) -> tuple[np.ndarray, np.ndarray]:
    import torch

    values = weights.detach().float().flatten()
    values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    chosen = torch.topk(values, min(count, len(values)), sorted=True)
    positive = chosen.values > 0
    return chosen.indices[positive].cpu().numpy(), chosen.values[positive].cpu().numpy()


def weighted_render(render: Any, view: Any, model: Any, pipe: Any, background: Any, mask: np.ndarray) -> Any:
    import torch

    pixel_weights = torch.as_tensor(mask.astype(np.float32), device="cuda")
    package = render(
        view,
        model,
        pipe,
        background,
        track_gradients=False,
        inference_only=False,
        pixel_weights=pixel_weights,
    )
    if "accum_weights" not in package or package["accum_weights"].shape[0] != model.get_xyz.shape[0]:
        raise RuntimeError("Native rasterizer did not return one accumulated weight per Gaussian")
    return package


def build_records(
    *,
    name: str,
    class_name: str,
    mask: np.ndarray,
    ring: np.ndarray,
    image: np.ndarray,
    camera: Any,
    structure_weights: Any,
    ring_weights: Any,
    model: Any,
    rotations: Any,
    top_count: int,
) -> list[dict[str, Any]]:
    import torch

    structure_ids, _ = finite_top(structure_weights, top_count)
    ring_ids, _ = finite_top(ring_weights, top_count)
    ids = np.unique(np.concatenate([structure_ids, ring_ids])).astype(np.int64)
    if not len(ids):
        return []
    index = torch.as_tensor(ids, device=model.get_xyz.device, dtype=torch.long)
    xyz = model.get_xyz[index].detach().cpu().numpy()
    scales = model.get_scaling[index].detach().cpu().numpy()
    opacity = model.get_opacity[index, 0].detach().cpu().numpy()
    rotation = rotations[index].detach().cpu().numpy()
    projected = project_gaussians(xyz, scales, rotation, camera)
    tangent = photo_tangent(image)
    distance = distance_transform_edt(~mask)
    structure_np = structure_weights[ids].detach().float().cpu().numpy() / max(int(mask.sum()), 1)
    ring_np = ring_weights[ids].detach().float().cpu().numpy() / max(int(ring.sum()), 1)
    rows = []
    for local, gaussian_id in enumerate(ids):
        u, v = float(projected["u"][local]), float(projected["v"][local])
        x, y = int(round(u)), int(round(v))
        inside_image = 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]
        center_in_mask = bool(mask[y, x]) if inside_image else False
        center_distance = float(distance[y, x]) if inside_image else None
        orientation_error = None
        if inside_image:
            orientation_error = float(np.degrees(undirected_angle_error(
                np.array([projected["angle_rad"][local]]), np.array([tangent[y, x]])
            )[0]))
        total_density = max(float(structure_np[local] + ring_np[local]), 1e-20)
        near_background_fraction = float(ring_np[local] / total_density)
        minor = float(projected["minor_radius"][local])
        major = float(projected["major_radius"][local])
        if center_in_mask and structure_np[local] > ring_np[local] and (
            class_name != "stay_cable" or orientation_error is None or orientation_error <= 35.0
        ):
            kind = "true_like"
        elif near_background_fraction >= 0.5 and (not center_in_mask or minor >= 2.0):
            kind = "artifact_like"
        else:
            kind = "ambiguous"
        rows.append({
            "view": name,
            "class": class_name,
            "gaussian_id": int(gaussian_id),
            "top_structure": bool(gaussian_id in set(structure_ids.tolist())),
            "top_near_background": bool(gaussian_id in set(ring_ids.tolist())),
            "structure_weight_density": float(structure_np[local]),
            "near_background_weight_density": float(ring_np[local]),
            "near_background_fraction": near_background_fraction,
            "kind": kind,
            "u": u,
            "v": v,
            "depth": float(projected["depth"][local]),
            "projection_valid": bool(projected["valid"][local]),
            "center_in_mask": center_in_mask,
            "center_distance_px": center_distance,
            "major_radius_px": major,
            "minor_radius_px": minor,
            "area_proxy_px2": float(math.pi * major * minor),
            "angle_rad": float(projected["angle_rad"][local]),
            "orientation_error_deg": orientation_error,
            "opacity": float(opacity[local]),
            "scales": [float(value) for value in scales[local]],
            "anisotropy": float(scales[local].max() / scales[local].min()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/quality/QUAL-022/diagnostic-r1")
    parser.add_argument("--max-labeled-views", type=int, default=300)
    parser.add_argument("--top-count", type=int, default=48)
    args = parser.parse_args()
    if args.max_labeled_views < 1 or args.max_labeled_views > 300:
        raise ValueError("max-labeled-views must be in [1,300]")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(CONFIG.read_text())
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "started",
        "read_only": True,
        "config_sha256": sha256(CONFIG),
        "code_sha256": {"driver": sha256(Path(__file__)), "helpers": sha256(ROOT / "src/bridge_gaussian_support.py")},
    }
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            active = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True,
            )
            other = [line for line in active.splitlines() if line.strip() and "ToDesk" not in line]
            if other:
                raise RuntimeError(f"Other GPU compute process active: {other}")

            sys.path[:0] = [
                str(SOURCE),
                str(SOURCE / "submodules/diff-gaussian-rasterization"),
                str(SOURCE / "submodules/simple-knn"),
                str(SOURCE / "submodules/fused-ssim"),
            ]
            import torch
            from gaussian_renderer import render
            from scene.gaussian_model import GaussianModel
            from src.quality_cameras import read_camera_requests
            from src.quality_native_gaussian import make_native_view
            from utils.general_utils import build_rotation

            torch.set_num_threads(4)
            torch.manual_seed(0)
            torch.cuda.reset_peak_memory_stats()
            parent_hashes = {"rgb170": sha256(PLY170), "rgb166": sha256(PLY166), "split": sha256(ROOT / "outputs/quality/all_data/RGB-120-r2/split.json")}
            if parent_hashes["rgb170"] != "54390020839cadb238e6a81bb40b4790c6c3da34d2d0b3cc3c74f6d2fc6e3277":
                raise ValueError("RGB170 parent PLY changed")
            model = GaussianModel(3)
            model.load_ply(str(PLY170))
            if model.get_xyz.shape[0] != 985_530:
                raise ValueError("RGB170 Gaussian count changed")
            for value in vars(model).values():
                if isinstance(value, torch.Tensor):
                    value.requires_grad_(False)
            rotations = build_rotation(model.get_rotation).detach()
            cameras = read_camera_requests(SCENE / "sparse/0")
            names = sorted(path.name for path in LABELS.glob("*.png"))[: args.max_labeled_views]
            if args.max_labeled_views == 300 and len(names) != 300:
                raise ValueError("Expected all 300 labeled views")
            if len(cameras) != 400 or any(name not in cameras for name in names):
                raise ValueError("Official camera universe changed")
            pipe = SimpleNamespace(debug=False, antialiasing=False, depth_ratio=0.0)
            background = torch.zeros(3, device="cuda")
            all_records: list[dict[str, Any]] = []
            per_view: list[dict[str, Any]] = []
            review_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

            jobs: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, Any]] = []
            for name in names:
                image = load_rgb(SCENE / "images" / name)
                labels = np.asarray(Image.open(LABELS / name), dtype=np.uint8)
                for class_id, class_name in CLASSES.items():
                    mask = labels == class_id
                    ring = binary_dilation(mask, iterations=config["scope"]["near_background_ring_pixels"]) & ~mask
                    jobs.append((name, class_name, mask, ring, image, cameras[name]))
            for name in FOCUS["bridge_385_392"]:
                image = load_rgb(SCENE / "images" / name)
                mask = photo_line_mask(image, BRIDGE_ROIS[name])
                ring = binary_dilation(mask, iterations=config["scope"]["near_background_ring_pixels"]) & ~mask
                jobs.append((name, "photo_line", mask, ring, image, cameras[name]))

            total_jobs = len(jobs)
            for job_index, (name, class_name, mask, ring, image, camera) in enumerate(jobs):
                view = make_native_view(camera)
                structure_package = weighted_render(render, view, model, pipe, background, mask)
                ring_package = weighted_render(render, view, model, pipe, background, ring)
                rows = build_records(
                    name=name,
                    class_name=class_name,
                    mask=mask,
                    ring=ring,
                    image=image,
                    camera=camera,
                    structure_weights=structure_package["accum_weights"],
                    ring_weights=ring_package["accum_weights"],
                    model=model,
                    rotations=rotations,
                    top_count=args.top_count,
                )
                all_records.extend(rows)
                if name in REVIEW_NAMES:
                    review_rows[name].extend(rows)
                prediction = load_rgb(RGB170 / name)
                contrast_photo = local_contrast(image)
                contrast_prediction = local_contrast(prediction)
                if class_name == "stay_cable":
                    target = mask & (contrast_photo >= 3.0 / 255.0)
                    visible_support = float((contrast_prediction[target] >= 0.5 * contrast_photo[target]).mean()) if target.any() else None
                else:
                    target = mask & ~(
                        binary_dilation(mask, iterations=2) & ~binary_dilation(mask, iterations=1)
                    )
                    visible_support = None
                structure_top = [row for row in rows if row["top_structure"]]
                true_like = [row for row in structure_top if row["kind"] == "true_like"]
                per_view.append({
                    "view": name,
                    "class": class_name,
                    "mask_pixels": int(mask.sum()),
                    "ring_pixels": int(ring.sum()),
                    "top_records": len(rows),
                    "visible_photo_pixels": int(target.sum()),
                    "visible_support_fraction": visible_support,
                    "top_ellipse_coverage_fraction": ellipse_coverage(mask.shape, structure_top, target),
                    "true_like_ellipse_coverage_fraction": ellipse_coverage(mask.shape, true_like, target),
                })
                if job_index % 50 == 0 or job_index + 1 == total_jobs:
                    progress = {
                        "status": "running_read_only",
                        "jobs_completed": job_index + 1,
                        "jobs_total": total_jobs,
                        "records": len(all_records),
                        "seconds": time.monotonic() - started,
                    }
                    (args.output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                    print(json.dumps(progress), flush=True)
                del structure_package, ring_package

            top_counts: dict[tuple[str, int], set[str]] = defaultdict(set)
            for row in all_records:
                if row["top_structure"]:
                    top_counts[(row["class"], row["gaussian_id"])].add(row["view"])
            for row in all_records:
                row["cross_view_top_count"] = len(top_counts[(row["class"], row["gaussian_id"])])

            summary = {}
            for class_name in [*CLASSES.values(), "photo_line"]:
                selected = [row for row in all_records if row["class"] == class_name and row["top_structure"]]
                summary[class_name] = summarize_records(selected)
                views = [row for row in per_view if row["class"] == class_name]
                summary[class_name]["view_count"] = len(views)
                summary[class_name]["top_ellipse_coverage_fraction"] = quantiles(row["top_ellipse_coverage_fraction"] for row in views)
                summary[class_name]["true_like_ellipse_coverage_fraction"] = quantiles(row["true_like_ellipse_coverage_fraction"] for row in views)
                summary[class_name]["visible_support_fraction"] = quantiles(
                    row["visible_support_fraction"] for row in views if row["visible_support_fraction"] is not None
                )

            review_root = args.output / "review"
            for name in REVIEW_NAMES:
                image = Image.open(SCENE / "images" / name).convert("RGB")
                draw = ImageDraw.Draw(image, "RGBA")
                candidates = sorted(
                    [row for row in review_rows[name] if row["top_structure"]],
                    key=lambda row: row["structure_weight_density"], reverse=True,
                )[:32]
                for row in candidates:
                    color = {
                        "true_like": (0, 220, 80, 135),
                        "artifact_like": (255, 60, 50, 135),
                        "ambiguous": (255, 205, 40, 120),
                    }[row["kind"]]
                    draw.line(polygon_ellipse(row) + [polygon_ellipse(row)[0]], fill=color, width=2)
                    draw.ellipse((row["u"] - 2, row["v"] - 2, row["u"] + 2, row["v"] + 2), fill=color)
                comparison_page(name, image, review_root / f"{name[:-4]}.jpg")

            with gzip.open(args.output / "contributors.jsonl.gz", "wt", encoding="utf-8") as handle:
                for row in all_records:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            write_new_json(args.output / "per-view.json", {"rows": per_view})
            parent_after = {"rgb170": sha256(PLY170), "rgb166": sha256(PLY166), "split": sha256(ROOT / "outputs/quality/all_data/RGB-120-r2/split.json")}
            if parent_after != parent_hashes:
                raise RuntimeError("Protected parent or split changed during diagnosis")
            result.update(
                status="completed_read_only",
                parent_hashes=parent_hashes,
                protected_exact=True,
                model_gaussians=int(model.get_xyz.shape[0]),
                labeled_views=len(names),
                photo_line_views=len(FOCUS["bridge_385_392"]),
                jobs=total_jobs,
                contributor_records=len(all_records),
                summary=summary,
                focus_groups={
                    group: {
                        class_name: summarize_records([
                            row for row in all_records
                            if row["view"] in group_names and row["class"] == class_name and row["top_structure"]
                        ])
                        for class_name in [*CLASSES.values(), "photo_line"]
                    }
                    for group, group_names in FOCUS.items()
                },
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                interpretation_scope="Training-view native contribution and official-camera projection evidence; labels only localize 2D regions; no 3D truth or unknown-view claim.",
            )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
