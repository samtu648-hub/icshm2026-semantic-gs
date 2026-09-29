#!/usr/bin/env python3
"""Evaluate the matched ImprovedGS 1.0M/1.5M spatial-confirmation pair."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_rgb import build_lpips_model, lpips_score, psnr, ssim
from scripts.evaluate_rgb166_horizon import (
    BRIDGE_ROIS, CLASSES, DIFFICULT_GROUPS, load8, structural_metrics,
    weighted_structural_aggregate,
)
from scripts.evaluate_rgb171_bridge_repair import (
    ARTIFACT_REGIONS, aggregate_continuity, artifact_region, cable_continuity,
)
from src.quality_protocol import sha256, write_new_json

ARMS = {
    "RGB175_1M": ROOT / "outputs/quality/RGB-175/full-r1",
    "RGB176_15M": ROOT / "outputs/quality/RGB-176/full-r1",
}
CONTEXT_RENDERS = {
    "RGB166 · full400 1.0M": ROOT / "outputs/quality/RGB-166/full-r1/evaluation-r1/images/50000",
    "RGB170 · full400 1.5M": ROOT / "outputs/quality/RGB-170/full-r1/evaluation-r1/images/50000",
}


def vertices(path: Path) -> int:
    with path.open("rb") as handle:
        while True:
            line = handle.readline().decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header" or not line:
                raise ValueError(f"PLY vertex count missing: {path}")


def means(rows: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def delta(candidate: dict[str, float], control: dict[str, float]) -> dict[str, float]:
    return {key: float(candidate[key] - control[key]) for key in candidate if key in control}


def masked_rgb(reference8: np.ndarray, prediction8: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    reference = reference8.astype(np.float32)[mask] / 255.0
    prediction = prediction8.astype(np.float32)[mask] / 255.0
    return {
        "pixels": float(mask.sum()),
        "PSNR": psnr(reference, prediction),
        "MAE_255": float(np.abs(reference - prediction).mean() * 255.0),
    }


def weighted_masked(rows: list[dict[str, float]]) -> dict[str, float]:
    weights = np.asarray([row["pixels"] for row in rows], dtype=np.float64)
    return {
        "pixels": float(weights.sum()),
        "PSNR": float(np.average([row["PSNR"] for row in rows], weights=weights)),
        "MAE_255": float(np.average([row["MAE_255"] for row in rows], weights=weights)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bridge_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        return (0, 0, mask.shape[1], mask.shape[0])
    margin = 45
    return (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
            min(mask.shape[1], int(xs.max()) + margin + 1), min(mask.shape[0], int(ys.max()) + margin + 1))


def montage(name: str, paths: list[tuple[str, Path]], semantic: np.ndarray, output: Path, crop: bool) -> None:
    box = tuple(BRIDGE_ROIS[name]) if crop and name in BRIDGE_ROIS else bridge_box(semantic > 0) if crop else None
    cells = []
    for label, path in paths:
        image = Image.open(path).convert("RGB")
        if box:
            image = image.crop(box)
        image.thumbnail((520, 390))
        cells.append((label, image.copy()))
    canvas = Image.new("RGB", (540 * len(cells), 430), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(cells):
        x = index * 540
        draw.text((x + 8, 8), f"{name} · {label}", fill="black")
        canvas.paste(image, (x + 8, 30))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/quality/source/improvedgs-full")
    parser.add_argument("--scene", type=Path, default=ROOT / "outputs/quality/all_data/RGB-120-r2/scene")
    parser.add_argument("--spatial-scene", type=Path, default=ROOT / "outputs/quality/RGB-152/scene")
    parser.add_argument("--access", type=Path, default=ROOT / "outputs/quality/RGB-152/access.json")
    parser.add_argument("--split", type=Path, default=ROOT / "experiments/splits/spatial_confirmation_v1.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "outputs/quality/all_data/SEM-370-r1/labels")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/quality/QUAL-026/r1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    image_roots = {label: args.output / "images" / label for label in ARMS}
    for root in image_roots.values():
        root.mkdir(parents=True)

    access = json.loads(args.access.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_names, holdout_names, buffer_names = list(access["train"]), list(access["evaluation"]), list(access["buffer"])
    names = sorted(train_names + holdout_names + buffer_names)
    if len(train_names) != 353 or len(holdout_names) != 30 or len(buffer_names) != 17 or len(names) != 400:
        raise ValueError("Unexpected spatial_confirmation_v1 roles")
    if set(holdout_names) != set(split["primary_fold"]["validation_labeled"]):
        raise ValueError("Access and split holdout roles differ")
    for root in CONTEXT_RENDERS.values():
        if any(not (root / name).is_file() for name in names):
            raise FileNotFoundError(f"Incomplete contextual full400 render root: {root}")

    sys.path[:0] = [str(args.source), str(args.source / "submodules/diff-gaussian-rasterization"),
                    str(args.source / "submodules/simple-knn"), str(args.source / "submodules/fused-ssim")]
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from src.quality_cameras import read_camera_requests
    from src.quality_native_gaussian import make_native_view

    cameras = read_camera_requests(args.scene / "sparse/0")
    if set(cameras) != set(names):
        raise ValueError("Official all400 camera universe changed")
    spatial_cameras = read_camera_requests(args.spatial_scene / "sparse/0")
    if set(spatial_cameras) != set(train_names) | set(holdout_names):
        raise ValueError("Spatial training scene camera universe changed")
    for name, camera in spatial_cameras.items():
        other = cameras[name]
        if (camera.width, camera.height, camera.model, camera.radial_k) != (
            other.width, other.height, other.model, other.radial_k
        ) or not (
            np.array_equal(camera.matrix, other.matrix)
            and np.array_equal(camera.pose.rotation, other.pose.rotation)
            and np.array_equal(camera.pose.translation, other.pose.translation)
        ):
            raise ValueError(f"Spatial/all400 official camera mismatch: {name}")
    pipe = SimpleNamespace(debug=False, antialiasing=False, depth_ratio=0.0)
    background = torch.zeros(3, device="cuda")
    render_seconds: dict[str, list[float]] = {}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for arm, run in ARMS.items():
        ply = run / "model/point_cloud/iteration_50000/point_cloud.ply"
        model = GaussianModel(3)
        model.load_ply(str(ply))
        times = []
        with torch.inference_mode():
            for index, name in enumerate(names):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                tensor = render(make_native_view(cameras[name]), model, pipe, background,
                                track_gradients=False, inference_only=True)["render"]
                torch.cuda.synchronize()
                times.append(time.perf_counter() - tick)
                array = np.rint(tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(array).save(image_roots[arm] / name)
                if index % 100 == 0:
                    print(json.dumps({"arm": arm, "rendered": index + 1}), flush=True)
        render_seconds[arm] = times
        del model
        torch.cuda.empty_cache()

    lpips_model, lpips_torch, lpips_device = build_lpips_model("cuda")
    rows: list[dict[str, object]] = []
    by_name: dict[str, dict[str, dict[str, float]]] = {}
    for index, name in enumerate(names):
        reference = load8(args.scene / "images" / name).astype(np.float32) / 255.0
        by_name[name] = {}
        role = "holdout" if name in holdout_names else "buffer" if name in buffer_names else "train"
        row: dict[str, object] = {"view": name, "role": role}
        for arm in ARMS:
            prediction = load8(image_roots[arm] / name).astype(np.float32) / 255.0
            values = {"PSNR": psnr(reference, prediction), "SSIM": ssim(reference, prediction),
                      "LPIPS": lpips_score(reference, prediction, lpips_model, lpips_torch, lpips_device)}
            by_name[name][arm] = values
            for key, value in values.items():
                row[f"{arm}_{key}"] = value
        rows.append(row)
        if index % 100 == 0:
            print(json.dumps({"evaluated": index + 1}), flush=True)
    del lpips_model
    torch.cuda.empty_cache()
    write_csv(args.output / "per_view.csv", rows)

    partitions: dict[str, dict[str, dict[str, float]]] = {}
    for role, role_names in (("all400", names), ("train", train_names), ("holdout", holdout_names),
                             ("buffer", buffer_names)):
        partitions[role] = {arm: means([by_name[name][arm] for name in role_names], ("PSNR", "SSIM", "LPIPS"))
                            for arm in ARMS}
    groups = {}
    for group in split["groups"]:
        groups[group["group_id"]] = {
            arm: means([by_name[name][arm] for name in group["images"]], ("PSNR", "SSIM", "LPIPS"))
            for arm in ARMS
        }

    label_names = sorted(path.name for path in args.labels.glob("*.png"))
    if len(label_names) != 300:
        raise ValueError("Expected 300 semantic masks")
    structures = {scope: {arm: [] for arm in ARMS} for scope in ("all300", "holdout")}
    continuities = {scope: {arm: [] for arm in ARMS} for scope in ("all300", "holdout")}
    bridge_union = {scope: {arm: [] for arm in ARMS} for scope in ("all300", "holdout")}
    background_regions = {scope: {arm: [] for arm in ARMS} for scope in ("all300", "holdout")}
    per_holdout_structure = {}
    structure_by_name = {}
    continuity_by_name = {}
    for name in label_names:
        semantic = np.asarray(Image.open(args.labels / name))
        reference8 = load8(args.scene / "images" / name)
        structure_by_name[name], continuity_by_name[name] = {}, {}
        if name in holdout_names:
            per_holdout_structure[name] = {}
        for arm in ARMS:
            prediction8 = load8(image_roots[arm] / name)
            structure = structural_metrics(reference8, prediction8, semantic)
            continuity = cable_continuity(reference8, prediction8, semantic)
            structure_by_name[name][arm], continuity_by_name[name][arm] = structure, continuity
            for scope in (["all300", "holdout"] if name in holdout_names else ["all300"]):
                structures[scope][arm].append(structure)
                continuities[scope][arm].append(continuity)
                bridge_union[scope][arm].append(masked_rgb(reference8, prediction8, semantic > 0))
                background_regions[scope][arm].append(masked_rgb(reference8, prediction8, semantic == 0))
            if name in holdout_names:
                per_holdout_structure[name][arm] = {"classes": structure, "stay_cable_continuity": continuity}
        if name in holdout_names:
            paths = [("photo", args.scene / "images" / name),
                     *[(label, root / name) for label, root in CONTEXT_RENDERS.items()],
                     ("RGB175 · 1.0M", image_roots["RGB175_1M"] / name),
                     ("RGB176 · 1.5M", image_roots["RGB176_15M"] / name)]
            montage(name, paths, semantic, args.output / "review/full" / f"{name[:-4]}.jpg", crop=False)
            montage(name, paths, semantic, args.output / "review/bridge" / f"{name[:-4]}.jpg", crop=True)

    structure_aggregate = {scope: {arm: weighted_structural_aggregate(values) for arm, values in arms.items()}
                           for scope, arms in structures.items()}
    continuity_aggregate = {scope: {arm: aggregate_continuity(values) for arm, values in arms.items()}
                            for scope, arms in continuities.items()}
    bridge_aggregate = {scope: {arm: weighted_masked(values) for arm, values in arms.items()}
                        for scope, arms in bridge_union.items()}
    background_aggregate = {scope: {arm: weighted_masked(values) for arm, values in arms.items()}
                            for scope, arms in background_regions.items()}

    difficult = {}
    difficult_structure = {}
    difficult_continuity = {}
    for group, group_names in DIFFICULT_GROUPS.items():
        difficult[group] = {
            arm: means([by_name[name][arm] for name in group_names], ("PSNR", "SSIM", "LPIPS")) for arm in ARMS
        }
        labeled_group = [name for name in group_names if name in structure_by_name]
        difficult_structure[group] = {
            arm: weighted_structural_aggregate([structure_by_name[name][arm] for name in labeled_group]) for arm in ARMS
        } if labeled_group else {}
        difficult_continuity[group] = {
            arm: aggregate_continuity([continuity_by_name[name][arm] for name in labeled_group]) for arm in ARMS
        } if labeled_group else {}
    crop = {}
    for arm in ARMS:
        values = []
        for name, (x0, y0, x1, y1) in BRIDGE_ROIS.items():
            reference = load8(args.scene / "images" / name)[y0:y1, x0:x1].astype(np.float32) / 255.0
            prediction = load8(image_roots[arm] / name)[y0:y1, x0:x1].astype(np.float32) / 255.0
            values.append({"PSNR": psnr(reference, prediction), "SSIM": ssim(reference, prediction)})
        crop[arm] = means(values, ("PSNR", "SSIM"))
    artifact = {}
    for region, (name, boxes) in ARTIFACT_REGIONS.items():
        reference = load8(args.scene / "images" / name)
        artifact[region] = {
            arm: artifact_region(reference, load8(image_roots[arm] / name), boxes) for arm in ARMS
        }
    difficult_review_names = sorted(set(sum(DIFFICULT_GROUPS.values(), [])))
    for name in difficult_review_names:
        semantic_path = args.labels / name
        semantic = np.asarray(Image.open(semantic_path)) if semantic_path.is_file() else np.ones((989, 1320), dtype=np.uint8)
        paths = [("photo", args.scene / "images" / name),
                 *[(label, root / name) for label, root in CONTEXT_RENDERS.items()],
                 ("RGB175 · 1.0M", image_roots["RGB175_1M"] / name),
                 ("RGB176 · 1.5M", image_roots["RGB176_15M"] / name)]
        montage(name, paths, semantic, args.output / "review/difficult" / f"{name[:-4]}.jpg", crop=name in BRIDGE_ROIS)

    control, candidate = "RGB175_1M", "RGB176_15M"
    comparison = {
        "all400": delta(partitions["all400"][candidate], partitions["all400"][control]),
        "train": delta(partitions["train"][candidate], partitions["train"][control]),
        "holdout": delta(partitions["holdout"][candidate], partitions["holdout"][control]),
        "buffer": delta(partitions["buffer"][candidate], partitions["buffer"][control]),
        "groups": {key: delta(value[candidate], value[control]) for key, value in groups.items()},
        "all300_bridge_union": delta(bridge_aggregate["all300"][candidate], bridge_aggregate["all300"][control]),
        "holdout_bridge_union": delta(bridge_aggregate["holdout"][candidate], bridge_aggregate["holdout"][control]),
        "all300_background": delta(background_aggregate["all300"][candidate], background_aggregate["all300"][control]),
        "holdout_background": delta(background_aggregate["holdout"][candidate], background_aggregate["holdout"][control]),
        "all300_classes": {class_name: delta(structure_aggregate["all300"][candidate][class_name],
                                             structure_aggregate["all300"][control][class_name])
                           for class_name in CLASSES.values()},
        "holdout_classes": {class_name: delta(structure_aggregate["holdout"][candidate][class_name],
                                              structure_aggregate["holdout"][control][class_name])
                            for class_name in CLASSES.values()},
        "all300_stay_cable_continuity": delta(continuity_aggregate["all300"][candidate],
                                              continuity_aggregate["all300"][control]),
        "holdout_stay_cable_continuity": delta(continuity_aggregate["holdout"][candidate],
                                               continuity_aggregate["holdout"][control]),
        "difficult_groups": {key: delta(value[candidate], value[control]) for key, value in difficult.items()},
        "bridge_385_392_crop": delta(crop[candidate], crop[control]),
        "artifact_regions": {key: delta(value[candidate], value[control]) for key, value in artifact.items()},
        "wins": {
            role: {
                "PSNR": sum(by_name[name][candidate]["PSNR"] > by_name[name][control]["PSNR"] for name in role_names),
                "SSIM": sum(by_name[name][candidate]["SSIM"] > by_name[name][control]["SSIM"] for name in role_names),
                "LPIPS": sum(by_name[name][candidate]["LPIPS"] < by_name[name][control]["LPIPS"] for name in role_names),
                "views": len(role_names),
            }
            for role, role_names in (("all400", names), ("train", train_names), ("holdout", holdout_names),
                                     ("buffer", buffer_names))
        },
    }
    resources = {}
    for arm, run in ARMS.items():
        training = json.loads((run / "result.json").read_text(encoding="utf-8"))
        ply = run / "model/point_cloud/iteration_50000/point_cloud.ply"
        times = render_seconds[arm][5:]
        resources[arm] = {
            "gaussians": vertices(ply), "ply_bytes": ply.stat().st_size, "ply_sha256": sha256(ply),
            "training_seconds": training["worker"]["seconds"], "runner_wall_seconds": training["wall_seconds"],
            "training_peak_reserved_mib": training["worker"]["peak_reserved_mib"],
            "render_median_seconds": float(np.median(times)), "render_fps": float(1.0 / np.median(times)),
        }

    html = ["<!doctype html><meta charset='utf-8'><title>QUAL026</title>",
            "<style>body{font-family:sans-serif;max-width:1500px;margin:auto}img{width:100%;margin:8px 0}</style>",
            "<h1>QUAL026 · RGB170 budget spatial confirmation</h1>",
            "<p>Photo / contextual full400 RGB166 and RGB170 / matched spatial RGB175 1.0M and RGB176 1.5M. "
            "The 30 views were excluded only from RGB175/RGB176 training and initialization; full400 models are visual context, not holdout controls.</p>"]
    for name in holdout_names:
        html.append(f"<h2>{name}</h2><img src='bridge/{name[:-4]}.jpg'><img src='full/{name[:-4]}.jpg'>")
    (args.output / "review/index.html").write_text("\n".join(html) + "\n", encoding="utf-8")

    result = {
        "schema_version": 1, "status": "completed", "experiment_id": "QUAL-026-RGB170-SPATIAL-CONFIRMATION",
        "scope": "historically exposed development spatial confirmation; no blind-test claim",
        "roles": {"train": train_names, "holdout": holdout_names, "buffer": buffer_names},
        "hashes": {"split": sha256(args.split), "access": sha256(args.access),
                   "initialization": sha256(args.spatial_scene / "sparse/0/points3D.ply")},
        "code_sha256": {
            "evaluator": sha256(ROOT / "scripts/evaluate_rgb175_rgb176_spatial_pair.py"),
            "structure_metrics": sha256(ROOT / "scripts/evaluate_rgb166_horizon.py"),
            "continuity_metrics": sha256(ROOT / "scripts/evaluate_rgb171_bridge_repair.py"),
        },
        "camera_audit": {"shared_exact": True, "shared_views": len(spatial_cameras),
                         "all400_views": len(cameras)},
        "partitions": partitions, "groups": groups,
        "bridge_structure": structure_aggregate,
        "stay_cable_continuity": continuity_aggregate,
        "bridge_union": bridge_aggregate, "background": background_aggregate,
        "difficult_groups": difficult, "difficult_group_bridge_structure": difficult_structure,
        "difficult_group_stay_cable_continuity": difficult_continuity,
        "bridge_385_392_manual_crop": crop,
        "artifact_regions": artifact,
        "comparison_15M_minus_1M": comparison, "per_holdout_structure": per_holdout_structure,
        "resources": resources, "evaluation_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "seconds": time.monotonic() - started,
    }
    write_new_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
