#!/usr/bin/env python3
"""Build the frozen paper-facing profile for the released Project 2 data."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label as connected_components


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, verify_hashes, write_new_json


EID = "QUAL-009-DATASET-PROFILE"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
SCRIPT = ROOT / "scripts/audit_qual009_dataset_profile.py"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quantiles(values: list[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {key: None for key in ("min", "p10", "p25", "median", "p75", "p90", "max")}
    points = np.quantile(array, [0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
    return {
        key: float(value)
        for key, value in zip(
            ("min", "p10", "p25", "median", "p75", "p90", "max"), points
        )
    }


def quaternion_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid camera quaternion")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def read_camera_centers(path: Path) -> dict[str, np.ndarray]:
    cameras: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#") or len(fields) < 10:
            continue
        name = fields[9]
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        values = list(map(float, fields[1:8]))
        rotation = quaternion_rotation(*values[:4])
        translation = np.asarray(values[4:7], dtype=np.float64)
        cameras[name] = -rotation.T @ translation
    return cameras


def pca_projection(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ axes.T
    extent = projection.max(axis=0) - projection.min(axis=0)
    return projection, extent, axes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> dict[str, str]:
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": sha256(png), "pdf": sha256(pdf)}


def main() -> None:
    config = read(CONFIG)
    if config["experiment_id"] != EID:
        raise ValueError("QUAL009 config identity changed")
    output = ROOT / config["output_path"]
    output.mkdir(parents=True, exist_ok=False)
    inputs = config["inputs"]
    label_root = ROOT / inputs["label_root"]
    label_manifest_path = ROOT / inputs["label_manifest"]
    camera_path = ROOT / inputs["camera_images_txt"]
    split_path = ROOT / inputs["spatial_split"]
    dataset_manifest_path = ROOT / config["dataset_manifest"]
    manifest = read(label_manifest_path)
    dataset_manifest = read(dataset_manifest_path)
    split = read(split_path)
    analysis = config["analysis"]
    class_mapping = {int(key): value for key, value in analysis["class_mapping"].items()}
    records = {row["image"]: row for row in manifest["records"]}
    expected_names = sorted(records)
    if len(expected_names) != analysis["expected_labeled_views"]:
        raise ValueError("Unexpected labeled-view count")
    if manifest["class_mapping"] != {name: index for index, name in class_mapping.items()}:
        raise ValueError("Class mapping differs from the frozen label manifest")
    label_hashes = {name: records[name]["label_png_sha256"] for name in expected_names}
    verify_hashes(label_root, label_hashes)

    structure = np.ones((3, 3), dtype=np.uint8)
    total_pixels = analysis["image_width"] * analysis["image_height"] * len(expected_names)
    class_pixels: dict[int, int] = defaultdict(int)
    class_present_views: dict[int, int] = defaultdict(int)
    class_boundary_pixels: dict[int, dict[int, int]] = {
        class_id: {distance: 0 for distance in analysis["interior_distance_thresholds_px"]}
        for class_id in class_mapping
    }
    component_areas: dict[int, list[int]] = defaultdict(list)
    component_rows: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []

    for view_index, name in enumerate(expected_names):
        target = np.asarray(Image.open(label_root / name), dtype=np.uint8)
        if target.shape != (analysis["image_height"], analysis["image_width"]):
            raise ValueError(f"Unexpected label shape for {name}: {target.shape}")
        unique = set(map(int, np.unique(target)))
        if not unique <= set(class_mapping):
            raise ValueError(f"Unknown class IDs in {name}: {sorted(unique)}")
        row: dict[str, Any] = {"image": name, "view_index": view_index + 1}
        for class_id, class_name in class_mapping.items():
            mask = target == class_id
            pixels = int(mask.sum())
            class_pixels[class_id] += pixels
            class_present_views[class_id] += int(pixels > 0)
            row[f"{class_name}_pixels"] = pixels
            row[f"{class_name}_fraction"] = pixels / target.size
            if pixels == 0:
                row[f"{class_name}_components"] = 0
                continue
            distance = distance_transform_edt(mask)
            for threshold in analysis["interior_distance_thresholds_px"]:
                class_boundary_pixels[class_id][threshold] += int(
                    np.sum(mask & (distance <= threshold))
                )
            labels, count = connected_components(mask, structure=structure)
            areas = np.bincount(labels.ravel())[1:].astype(np.int64)
            row[f"{class_name}_components"] = int(count)
            for component_index, area in enumerate(areas, start=1):
                area_int = int(area)
                component_areas[class_id].append(area_int)
                component_rows.append(
                    {
                        "image": name,
                        "class_id": class_id,
                        "class_name": class_name,
                        "component_index": component_index,
                        "area_pixels": area_int,
                    }
                )
        view_rows.append(row)

    if sum(class_pixels.values()) != total_pixels:
        raise ValueError("Class pixels do not cover the full labeled grid")
    write_csv(output / "per_view.csv", view_rows)
    write_csv(output / "components.csv", component_rows)

    bins = analysis["component_area_bins"]
    class_summary: dict[str, Any] = {}
    for class_id, class_name in class_mapping.items():
        fractions = [float(row[f"{class_name}_fraction"]) for row in view_rows]
        areas = np.asarray(component_areas[class_id], dtype=np.int64)
        bin_masks = {
            "1_64": areas <= bins[0],
            "65_256": (areas > bins[0]) & (areas <= bins[1]),
            "257_1024": (areas > bins[1]) & (areas <= bins[2]),
            "gt_1024": areas > bins[2],
        }
        pixels = class_pixels[class_id]
        class_summary[class_name] = {
            "class_id": class_id,
            "pixels": pixels,
            "pixel_share_all_labeled": pixels / total_pixels,
            "present_views": class_present_views[class_id],
            "present_view_fraction": class_present_views[class_id] / len(expected_names),
            "source_shape_count": int(dataset_manifest["shape_label_counts"].get(class_name, 0)),
            "per_view_fraction_quantiles": quantiles(fractions),
            "component_count": int(areas.size),
            "component_area_quantiles": quantiles(areas),
            "component_bins": {
                key: {
                    "count": int(mask.sum()),
                    "count_fraction": float(mask.mean()) if areas.size else 0.0,
                    "pixel_count": int(areas[mask].sum()),
                    "class_pixel_fraction": float(areas[mask].sum() / pixels) if pixels else 0.0,
                }
                for key, mask in bin_masks.items()
            },
            "interior_boundary_fraction": {
                str(distance): class_boundary_pixels[class_id][distance] / pixels
                for distance in analysis["interior_distance_thresholds_px"]
            },
        }

    cameras = read_camera_centers(camera_path)
    if len(cameras) != analysis["expected_camera_views"]:
        raise ValueError(f"Expected 400 camera poses, found {len(cameras)}")
    if not set(expected_names) <= set(cameras):
        raise ValueError("A labeled view has no registered camera")
    camera_names = sorted(cameras)
    all_points = np.stack([cameras[name] for name in camera_names])
    labeled_points = np.stack([cameras[name] for name in expected_names])
    all_projection, all_extent, axes = pca_projection(all_points)
    all_center = all_points.mean(axis=0, keepdims=True)
    labeled_projection = (labeled_points - all_center) @ axes.T
    labeled_extent = labeled_projection.max(axis=0) - labeled_projection.min(axis=0)
    unlabeled_names = sorted(set(camera_names) - set(expected_names))
    unlabeled_points = np.stack([cameras[name] for name in unlabeled_names])
    nearest_labeled = np.sqrt(
        np.min(
            np.sum((unlabeled_points[:, None, :] - labeled_points[None, :, :]) ** 2, axis=2),
            axis=1,
        )
    )
    camera_summary = {
        "camera_count": len(camera_names),
        "labeled_camera_count": len(expected_names),
        "unlabeled_camera_count": len(unlabeled_names),
        "all_camera_pca_extent": all_extent.tolist(),
        "labeled_camera_pca_extent_in_all_axes": labeled_extent.tolist(),
        "labeled_to_all_extent_ratio": (labeled_extent / all_extent).tolist(),
        "unlabeled_nearest_labeled_distance_quantiles": quantiles(nearest_labeled),
    }

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    foreground = [class_mapping[index] for index in range(1, 5)]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    figure_hashes: dict[str, dict[str, str]] = {}

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    shares = [class_summary[name]["pixel_share_all_labeled"] * 100 for name in foreground]
    ax.bar(foreground, shares, color=colors)
    ax.set_ylabel("Share of all labeled pixels (%)")
    ax.set_title("Foreground class imbalance across 300 labeled views")
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(shares):
        ax.text(index, value, f"{value:.2f}%", ha="center", va="bottom")
    figure_hashes["class_pixel_share"] = save_figure(fig, output, "class_pixel_share")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    cable_bins = class_summary["stay_cable"]["component_bins"]
    labels = ["1–64", "65–256", "257–1024", ">1024"]
    keys = ["1_64", "65_256", "257_1024", "gt_1024"]
    counts = [cable_bins[key]["count"] for key in keys]
    pixel_shares = [cable_bins[key]["class_pixel_fraction"] * 100 for key in keys]
    x = np.arange(len(keys))
    ax.bar(x - 0.18, counts, width=0.36, color="#4C78A8", label="Component count")
    ax.set_ylabel("Connected components")
    ax.set_xticks(x, labels)
    ax.set_xlabel("stay_cable component area (pixels)")
    other = ax.twinx()
    other.bar(x + 0.18, pixel_shares, width=0.36, color="#F58518", label="Cable-pixel share")
    other.set_ylabel("Share of stay_cable pixels (%)")
    handles, legend_labels = ax.get_legend_handles_labels()
    other_handles, other_labels = other.get_legend_handles_labels()
    ax.legend(handles + other_handles, legend_labels + other_labels, loc="upper left")
    ax.set_title("stay_cable scale distribution")
    ax.grid(axis="y", alpha=0.25)
    figure_hashes["stay_cable_components"] = save_figure(fig, output, "stay_cable_components")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    labeled_set = set(expected_names)
    labeled_mask = np.asarray([name in labeled_set for name in camera_names])
    ax.scatter(
        all_projection[~labeled_mask, 0],
        all_projection[~labeled_mask, 1],
        s=16,
        alpha=0.65,
        color="#BAB0AC",
        label="Unlabeled RGB",
    )
    ax.scatter(
        all_projection[labeled_mask, 0],
        all_projection[labeled_mask, 1],
        s=18,
        alpha=0.75,
        color="#4C78A8",
        label="Labeled RGB",
    )
    confirmation = set(split["primary_fold"]["validation_labeled"])
    confirmation_mask = np.asarray([name in confirmation for name in camera_names])
    ax.scatter(
        all_projection[confirmation_mask, 0],
        all_projection[confirmation_mask, 1],
        s=34,
        facecolors="none",
        edgecolors="#E45756",
        linewidths=1.1,
        label="Spatial confirmation",
    )
    ax.set_xlabel("Camera-centre PCA axis 1")
    ax.set_ylabel("Camera-centre PCA axis 2")
    ax.set_title("Released camera coverage and existing spatial confirmation")
    ax.legend(frameon=False)
    ax.grid(alpha=0.20)
    ax.set_aspect("equal", adjustable="datalim")
    figure_hashes["camera_coverage"] = save_figure(fig, output, "camera_coverage")

    result = {
        "schema_version": 1,
        "status": "completed_read_only_dataset_profile",
        "experiment_id": EID,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": sha256(CONFIG),
        "protocol_sha256": sha256(ROOT / config["protocol"]),
        "script_sha256": sha256(SCRIPT),
        "dataset_manifest_sha256": sha256(dataset_manifest_path),
        "label_manifest_sha256": sha256(label_manifest_path),
        "camera_images_txt_sha256": sha256(camera_path),
        "spatial_split_sha256": sha256(split_path),
        "label_hashes_verified": len(label_hashes),
        "labeled_pixels": total_pixels,
        "class_summary": class_summary,
        "camera_summary": camera_summary,
        "figure_hashes": figure_hashes,
        "per_view_csv_sha256": sha256(output / "per_view.csv"),
        "components_csv_sha256": sha256(output / "components.csv"),
        "claim_scope": "released-data descriptive analysis only; no model quality or hidden-test claim",
    }
    write_new_json(output / "result.json", result)


if __name__ == "__main__":
    main()
