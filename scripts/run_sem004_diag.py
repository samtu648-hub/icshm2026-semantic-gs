#!/usr/bin/env python3
"""Read-only SEM-004-DIAG audit of frozen SEM-001 error mechanisms.

This program deliberately has no optimizer, loss, checkpoint writer, model
mutation, pseudo-label input, or unlabeled-image input.  It renders only the
already-frozen SEM-001 sidecar, compares it to released labels on the fixed
spatial split, and writes diagnostics to a Git-ignored output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import prepare_semantic_labels as label_builder  # noqa: E402
import run_sem001 as s0  # noqa: E402
from src.scaffold_semantic_sidecar import (  # noqa: E402
    AnchorSemanticSidecar,
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    frozen_rgb_named_parameters,
    load_frozen_scaffold,
    pipeline,
    render_frozen_rgb,
    render_semantic_probabilities,
    sha256,
)
from src.scaffold_visibility_supervision import (  # noqa: E402
    load_visibility_extension,
    render_frozen_contribution_diagnostics,
)
from src.sem004_diagnostics import (  # noqa: E402
    BACKGROUND,
    CLASS_COUNT,
    STAY_CABLE,
    component_distribution,
    error_structure,
    semantic_triptych,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sem001-checkpoint", type=Path, required=True)
    parser.add_argument("--sem003-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Git-ignored external output directory")
    parser.add_argument("--extension-root", type=Path, required=True, help="Git-ignored CUDA extension cache")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record_progress(path: Path, **payload: Any) -> None:
    payload["timestamp_unix"] = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def local_path(path: Path) -> str:
    """Use repository-relative paths in output metadata whenever possible."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def environment_summary() -> dict[str, Any]:
    import torch

    nvidia_smi: str | None
    try:
        nvidia_smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        nvidia_smi = None
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": nvidia_smi,
    }


def _histogram_template() -> dict[str, list[float]]:
    return {
        "contributor_count": [0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, float("inf")],
        "dominance": list(np.linspace(0.0, 1.0, 21)),
        "support_views": [0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, float("inf")],
    }


def _new_group() -> dict[str, Any]:
    return {
        "pixels": 0,
        "dominant_anchor_pixels": 0,
        "sum_foreground_mass": 0.0,
        "sum_maximum_contribution": 0.0,
        "sum_dominance": 0.0,
        "sum_contributor_count": 0.0,
        "sum_support_views": 0.0,
        "sum_background_evidence_fraction": 0.0,
        "sum_stay_cable_evidence_fraction": 0.0,
        "sum_protected_evidence_fraction": 0.0,
        "sem003_conflict_eligible_dominant_pixels": 0,
        "histograms": {name: [0] * (len(edges) - 1) for name, edges in _histogram_template().items()},
    }


def _histogram_quantile(counts: list[int], edges: list[float], quantile: float) -> float | None:
    total = int(sum(counts))
    if total == 0:
        return None
    threshold = total * float(quantile)
    running = 0
    for index, count in enumerate(counts):
        running += int(count)
        if running >= threshold:
            left, right = float(edges[index]), float(edges[index + 1])
            if math.isinf(right):
                return left
            return (left + right) / 2.0
    return float(edges[-2])


def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    pixels = int(group["pixels"])
    anchored = int(group["dominant_anchor_pixels"])
    result = {
        "pixels": pixels,
        "dominant_anchor_pixels": anchored,
        "dominant_anchor_pixel_fraction": float(anchored / pixels) if pixels else None,
        "means_over_all_pixels": {
            "foreground_mass": float(group["sum_foreground_mass"] / pixels) if pixels else None,
            "maximum_contribution": float(group["sum_maximum_contribution"] / pixels) if pixels else None,
            "maximum_over_foreground_concentration": float(group["sum_dominance"] / pixels) if pixels else None,
            "contributor_count": float(group["sum_contributor_count"] / pixels) if pixels else None,
        },
        "means_over_dominant_anchor_pixels": {
            "train_support_views": float(group["sum_support_views"] / anchored) if anchored else None,
            "background_evidence_fraction": float(group["sum_background_evidence_fraction"] / anchored) if anchored else None,
            "stay_cable_evidence_fraction": float(group["sum_stay_cable_evidence_fraction"] / anchored) if anchored else None,
            "protected_evidence_fraction": float(group["sum_protected_evidence_fraction"] / anchored) if anchored else None,
        },
        "sem003_conflict_eligibility_fraction_over_dominant_anchor_pixels": (
            float(group["sem003_conflict_eligible_dominant_pixels"] / anchored) if anchored else None
        ),
        "histograms": {},
    }
    templates = _histogram_template()
    for name, edges in templates.items():
        counts = group["histograms"][name]
        result["histograms"][name] = {
            "edges": [None if math.isinf(value) else value for value in edges],
            "counts": counts,
            "approx_median": _histogram_quantile(counts, edges, 0.50),
            "approx_p90": _histogram_quantile(counts, edges, 0.90),
        }
    return result


def _update_group(
    group: dict[str, Any],
    mask: np.ndarray,
    *,
    foreground: np.ndarray,
    maximum: np.ndarray,
    contributor_count: np.ndarray,
    anchor_indices: np.ndarray,
    support_views: np.ndarray,
    evidence_fraction: np.ndarray,
    eligibility: np.ndarray,
) -> None:
    if not bool(mask.any()):
        return
    foreground_values = foreground[mask].astype(np.float64, copy=False)
    maximum_values = maximum[mask].astype(np.float64, copy=False)
    contributor_values = contributor_count[mask].astype(np.float64, copy=False)
    dominance = maximum_values / np.maximum(foreground_values, np.finfo(np.float64).tiny)
    group["pixels"] += int(mask.sum())
    group["sum_foreground_mass"] += float(foreground_values.sum())
    group["sum_maximum_contribution"] += float(maximum_values.sum())
    group["sum_dominance"] += float(dominance.sum())
    group["sum_contributor_count"] += float(contributor_values.sum())
    templates = _histogram_template()
    for name, values in {
        "contributor_count": contributor_values,
        "dominance": dominance,
    }.items():
        counts, _ = np.histogram(values, bins=np.asarray(templates[name], dtype=np.float64))
        group["histograms"][name] = (np.asarray(group["histograms"][name], dtype=np.int64) + counts).astype(np.int64).tolist()
    selected_anchors = anchor_indices[mask]
    valid = selected_anchors >= 0
    if not bool(valid.any()):
        return
    selected_anchors = selected_anchors[valid]
    selected_support = support_views[selected_anchors].astype(np.float64, copy=False)
    selected_evidence = evidence_fraction[selected_anchors]
    group["dominant_anchor_pixels"] += int(selected_anchors.size)
    group["sum_support_views"] += float(selected_support.sum())
    group["sum_background_evidence_fraction"] += float(selected_evidence[:, BACKGROUND].sum())
    group["sum_stay_cable_evidence_fraction"] += float(selected_evidence[:, STAY_CABLE].sum())
    group["sum_protected_evidence_fraction"] += float(selected_evidence[:, [1, 3, 4]].sum())
    group["sem003_conflict_eligible_dominant_pixels"] += int(eligibility[selected_anchors].sum())
    counts, _ = np.histogram(selected_support, bins=np.asarray(templates["support_views"], dtype=np.float64))
    group["histograms"]["support_views"] = (
        np.asarray(group["histograms"]["support_views"], dtype=np.int64) + counts
    ).astype(np.int64).tolist()


class DirectionAccumulator:
    """Training-only, fixed-bin dominant-anchor bg/stay evidence table."""

    def __init__(self, anchors: np.ndarray, support_views: np.ndarray, policy: dict[str, Any]) -> None:
        self.anchors = anchors.astype(np.float64, copy=False)
        self.support_views = support_views.astype(np.int32, copy=False)
        self.policy = policy
        self.bin_count = int(policy["azimuth_bins"])
        self.anchor_count = int(anchors.shape[0])
        shape = (2, self.bin_count, 2, self.anchor_count)  # half, azimuth, bg/stay, anchor
        self.mass = np.zeros(shape, dtype=np.float64)
        self.pixels = np.zeros(shape, dtype=np.int32)
        self.views_per_half = [0, 0]

    @staticmethod
    def half_for_view(name: str) -> int:
        return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[0], 16) & 1

    def update(self, name: str, camera_center: np.ndarray, anchor_indices: np.ndarray, maximum: np.ndarray, target: np.ndarray) -> None:
        half = self.half_for_view(name)
        self.views_per_half[half] += 1
        flat_anchor = anchor_indices.reshape(-1)
        flat_target = target.reshape(-1)
        flat_maximum = maximum.reshape(-1).astype(np.float64, copy=False)
        valid = (flat_anchor >= 0) & ((flat_target == BACKGROUND) | (flat_target == STAY_CABLE))
        if not bool(valid.any()):
            return
        anchors = flat_anchor[valid]
        labels = flat_target[valid]
        weights = flat_maximum[valid]
        vectors = self.anchors[anchors] - camera_center[None, :]
        azimuth = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2.0 * math.pi)
        direction = np.minimum((azimuth * self.bin_count / (2.0 * math.pi)).astype(np.int32), self.bin_count - 1)
        classes = (labels == STAY_CABLE).astype(np.int32)
        # Accumulate each populated (direction, class) pair.  This avoids an
        # image-sized raw evidence dump while remaining exact for the chosen
        # dominant-anchor diagnostic statistic.
        for direction_id in np.unique(direction):
            direction_mask = direction == direction_id
            for class_id in (0, 1):
                current = direction_mask & (classes == class_id)
                if not bool(current.any()):
                    continue
                ids = anchors[current]
                self.mass[half, int(direction_id), class_id] += np.bincount(
                    ids, weights=weights[current], minlength=self.anchor_count
                )
                self.pixels[half, int(direction_id), class_id] += np.bincount(ids, minlength=self.anchor_count).astype(np.int32)

    def summary(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        min_support = int(self.policy["stable_anchor_minimum_support_views"])
        min_pixels = int(self.policy["stable_anchor_minimum_dominant_pixels"])
        min_bin_pixels = int(self.policy["minimum_dominant_pixels_per_direction_bin"])
        min_common = int(self.policy["minimum_common_direction_bins_between_train_halves"])
        min_range = float(self.policy["minimum_cable_fraction_range"])
        min_correlation = float(self.policy["minimum_half_pattern_correlation"])
        mass_total = self.mass.sum(axis=2)
        pixel_total = self.pixels.sum(axis=2)
        cable_fraction = self.mass[:, :, 1] / np.maximum(mass_total, np.finfo(np.float64).tiny)
        valid_bin = pixel_total >= min_bin_pixels
        total_pixels = pixel_total.sum(axis=(0, 1))
        common = valid_bin[0] & valid_bin[1]
        common_count = common.sum(axis=0)
        stable = (self.support_views >= min_support) & (total_pixels >= min_pixels) & (common_count >= min_common)
        records: list[dict[str, Any]] = []
        reproducible = 0
        directional = 0
        for anchor_id in np.flatnonzero(stable):
            selected = common[:, anchor_id]
            first = cable_fraction[0, selected, anchor_id]
            second = cable_fraction[1, selected, anchor_id]
            first_range = float(first.max() - first.min()) if first.size else 0.0
            second_range = float(second.max() - second.min()) if second.size else 0.0
            if first.size >= 2 and float(np.std(first)) > 0.0 and float(np.std(second)) > 0.0:
                correlation = float(np.corrcoef(first, second)[0, 1])
            else:
                correlation = None
            is_directional = first_range >= min_range and second_range >= min_range
            is_reproducible = bool(is_directional and correlation is not None and correlation >= min_correlation)
            directional += int(is_directional)
            reproducible += int(is_reproducible)
            if is_reproducible:
                records.append(
                    {
                        "anchor_index": int(anchor_id),
                        "support_views": int(self.support_views[anchor_id]),
                        "dominant_pixels": int(total_pixels[anchor_id]),
                        "common_direction_bins": int(common_count[anchor_id]),
                        "half0_cable_fraction_range": first_range,
                        "half1_cable_fraction_range": second_range,
                        "half_pattern_correlation": correlation,
                    }
                )
        per_bin: list[dict[str, Any]] = []
        for half in (0, 1):
            for direction_id in range(self.bin_count):
                background_mass = float(self.mass[half, direction_id, 0].sum())
                cable_mass = float(self.mass[half, direction_id, 1].sum())
                per_bin.append(
                    {
                        "train_half": half,
                        "azimuth_bin": direction_id,
                        "background_dominant_mass": background_mass,
                        "stay_cable_dominant_mass": cable_mass,
                        "stay_cable_mass_fraction": cable_mass / max(background_mass + cable_mass, np.finfo(np.float64).tiny),
                        "dominant_pixels": int(self.pixels[half, direction_id].sum()),
                    }
                )
        stable_count = int(stable.sum())
        return {
            "training_only": True,
            "train_views_per_half": self.views_per_half,
            "stable_anchors": stable_count,
            "directional_anchors": directional,
            "reproducible_directional_anchors": reproducible,
            "reproducible_directional_anchor_fraction": float(reproducible / stable_count) if stable_count else 0.0,
            "policy": {
                "minimum_support_views": min_support,
                "minimum_dominant_pixels": min_pixels,
                "minimum_pixels_per_direction_bin": min_bin_pixels,
                "minimum_common_direction_bins": min_common,
                "minimum_cable_fraction_range": min_range,
                "minimum_half_pattern_correlation": min_correlation,
            },
            "per_bin": per_bin,
        }, records


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    return float(abs(cv2.contourArea(points.astype(np.float32).reshape(-1, 1, 2))))


def _polygon_perimeter(points: np.ndarray) -> float:
    return float(cv2.arcLength(points.astype(np.float32).reshape(-1, 1, 2), True)) if len(points) >= 3 else 0.0


def _polygon_thinness(points: np.ndarray) -> float:
    area = _polygon_area(points)
    perimeter = _polygon_perimeter(points)
    return float(1.0 - (4.0 * math.pi * area / (perimeter * perimeter))) if perimeter > 0.0 else 1.0


def _raster_polygon(points: np.ndarray) -> np.ndarray:
    canvas = np.zeros((989, 1320), dtype=np.uint8)
    cv2.fillPoly(canvas, [points.astype(np.int32)], color=1)
    return canvas


def _label_geometry_audit(
    *,
    names: list[str],
    split_name: str,
    data_root: Path,
    labels_root: Path,
    matrix: np.ndarray,
    distortion: np.ndarray,
    adjacency_radius: int,
    output_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_rebuilt_exact = 0
    shape_counts: Counter[str] = Counter()
    aggregate = {"raw_area_px": 0.0, "pinhole_area_px": 0.0, "overlap_pixels": 0, "shapes": 0}
    adjacency: Counter[str] = Counter()
    for name in names:
        source = data_root / "json" / f"{Path(name).stem}.json"
        payload = load_json(source)
        rebuilt, _shape_counts = label_builder.make_label(source, matrix, distortion)
        existing = s0.read_label(labels_root / name)
        source_rebuilt_exact += int(np.array_equal(rebuilt, existing))
        assigned = np.zeros_like(existing)
        for shape_index, shape in enumerate(payload["shapes"]):
            label_name = str(shape["label"])
            class_id = int(label_builder.CLASS_IDS[label_name])
            raw = np.asarray(shape["points"], dtype=np.float64)
            raw[:, 0] = np.clip(raw[:, 0], 0.0, 1319.0)
            raw[:, 1] = np.clip(raw[:, 1], 0.0, 988.0)
            raw_pixels = np.rint(raw).astype(np.int32)
            pinhole_pixels = label_builder.polygon_points(shape, matrix, distortion)
            raw_mask = _raster_polygon(raw_pixels)
            pinhole_mask = _raster_polygon(pinhole_pixels)
            overlap = (assigned > 0) & (pinhole_mask > 0)
            overlap_by_prior = Counter(int(value) for value in assigned[overlap] if int(value) != class_id)
            cv2.fillPoly(assigned, [pinhole_pixels], color=class_id)
            x, y, width, height = cv2.boundingRect(pinhole_pixels.reshape(-1, 1, 2))
            raw_area = float(raw_mask.sum())
            pinhole_area = float(pinhole_mask.sum())
            shape_counts[label_name] += 1
            aggregate["raw_area_px"] += raw_area
            aggregate["pinhole_area_px"] += pinhole_area
            aggregate["overlap_pixels"] += int(overlap.sum())
            aggregate["shapes"] += 1
            rows.append(
                {
                    "split": split_name,
                    "view": name,
                    "shape_index": shape_index,
                    "label": label_name,
                    "raw_polygon_area_px": raw_area,
                    "pinhole_polygon_area_px": pinhole_area,
                    "rasterization_area_delta_px": pinhole_area - raw_area,
                    "rasterization_area_delta_fraction": (pinhole_area - raw_area) / max(raw_area, 1.0),
                    "pinhole_bbox_aspect_ratio": float(max(width, height) / max(min(width, height), 1)),
                    "pinhole_thinness": _polygon_thinness(pinhole_pixels),
                    "pinhole_perimeter_px": _polygon_perimeter(pinhole_pixels),
                    "overlap_pixels_with_prior_polygons": int(overlap.sum()),
                    "overlap_prior_class_ids": json.dumps(dict(sorted(overlap_by_prior.items()))),
                }
            )
        cable = existing == STAY_CABLE
        dilated = cv2.dilate(cable.astype(np.uint8), np.ones((2 * adjacency_radius + 1, 2 * adjacency_radius + 1), dtype=np.uint8)) > 0
        for class_id, class_name in ((1, "deck"), (3, "tower"), (4, "foundation")):
            adjacency[class_name] += int((dilated & (existing == class_id)).sum())
    with (output_dir / f"labelme_shape_audit_{split_name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["split", "view"])
        writer.writeheader()
        writer.writerows(rows)
    cable_rows = [row for row in rows if row["label"] == "stay_cable"]
    thinness = np.asarray([float(row["pinhole_thinness"]) for row in cable_rows], dtype=np.float64)
    area_delta = np.asarray([float(row["rasterization_area_delta_fraction"]) for row in cable_rows], dtype=np.float64)
    return {
        "split": split_name,
        "views": len(names),
        "source_json_rebuild_exact_labels": source_rebuilt_exact,
        "source_json_rebuild_exact_fraction": float(source_rebuilt_exact / len(names)) if names else None,
        "shape_counts": dict(sorted(shape_counts.items())),
        "aggregate": aggregate,
        "stay_cable_polygon_thinness": _array_summary(thinness),
        "stay_cable_rasterization_area_delta_fraction": _array_summary(area_delta),
        "stay_cable_adjacency_pixels_within_radius": dict(sorted(adjacency.items())),
        "rows_csv": f"labelme_shape_audit_{split_name}.csv",
    }


def _array_summary(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _split_error_summary(
    *,
    confusion: np.ndarray,
    per_view: list[dict[str, Any]],
    fp_component_rows: list[dict[str, Any]],
    pred_component_rows: list[dict[str, Any]],
    gt_component_rows: list[dict[str, Any]],
    category_pixels: Counter[str],
) -> dict[str, Any]:
    metrics = s0.confusion_metrics(confusion)
    metrics["confusion_target_rows_prediction_columns"] = confusion.tolist()
    metrics["per_view_count"] = len(per_view)
    metrics["false_positive_category_pixels"] = dict(sorted(category_pixels.items()))
    for key, rows in {
        "false_positive_components": fp_component_rows,
        "prediction_stay_cable_components": pred_component_rows,
        "ground_truth_stay_cable_components": gt_component_rows,
    }.items():
        metrics[key] = {
            "count": len(rows),
            "area_px": _array_summary(np.asarray([float(row["area_px"]) for row in rows], dtype=np.float64)),
            "bbox_aspect_ratio": _array_summary(np.asarray([float(row["bbox_aspect_ratio"]) for row in rows], dtype=np.float64)),
            "thinness": _array_summary(np.asarray([float(row["thinness"]) for row in rows], dtype=np.float64)),
        }
    return metrics


def _save_distribution_plot(output: Path, *, title: str, values: dict[str, float]) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    labels = list(values)
    axis.bar(labels, [values[label] for label in labels], color=["#d73027", "#4575b4", "#74add1", "#fdae61"][: len(labels)])
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _save_direction_plot(output: Path, summary: dict[str, Any]) -> None:
    rows = summary["per_bin"]
    figure, axis = plt.subplots(figsize=(9, 4))
    for half, color in ((0, "#4575b4"), (1, "#d73027")):
        selected = [row for row in rows if row["train_half"] == half]
        axis.plot(
            [row["azimuth_bin"] for row in selected],
            [row["stay_cable_mass_fraction"] for row in selected],
            marker="o",
            label=f"training half {half}",
            color=color,
        )
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("fixed global-azimuth bin")
    axis.set_ylabel("dominant-anchor stay_cable mass fraction")
    axis.set_title("Training-only observation-direction diagnostic")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _output_manifest(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "output_manifest.json"):
        files.append({"relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {"schema_version": 1, "file_count": len(files), "files": files}


def _validate_config(config: dict[str, Any]) -> None:
    require(config["experiment_id"] == "SEM-004-DIAG", "SEM-004 diagnostic requires its fixed experiment ID")
    require(config["status"] == "diagnostic_only_not_a_semantic_candidate", "SEM-004 status must remain diagnostic-only")
    for key in ("trains_new_parameters", "updates_RGB", "updates_semantic_parameters", "uses_unlabeled_RGB", "uses_pseudo_labels"):
        require(config[key] is False, f"SEM-004 config must set {key}=false")
    require(config["frozen_semantic"]["class_names"] == list(s0.CLASS_NAMES), "SEM-004 class mapping must match SEM-001")
    require(config["frozen_rgb"]["pose_convention"] == "CAMERA_FROM_WORLD", "SEM-004 must retain official pose convention")


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    _validate_config(config)
    split = load_json(REPO_ROOT / config["data"]["semantic_split"])
    label_manifest = load_json(REPO_ROOT / config["data"]["label_manifest"])
    require(split["split_id"] == config["data"]["split_id"], "Semantic split ID mismatch")
    require(label_manifest["split_id"] == split["split_id"], "Label manifest split ID mismatch")
    require(len(split["semantic_train_labeled"]) == config["data"]["training_images"], "Unexpected training-view count")
    require(len(split["semantic_validation_labeled"]) == config["data"]["validation_images"], "Unexpected validation-view count")
    for path in (args.data_root, args.labels_root, args.rgb_run_dir, args.source_root, args.sem001_checkpoint, args.sem003_evidence):
        require(path.exists(), f"Missing required SEM-004 input: {path}")

    import torch

    require(torch.cuda.is_available(), "SEM-004 frozen contribution audit requires the host CUDA execution context")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.reset_peak_memory_stats()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = args.output_dir / "progress.jsonl"
    write_json(
        args.output_dir / "invocation.json",
        {
            "experiment_id": config["experiment_id"],
            "diagnostic_only": True,
            "command_paths": {"config": local_path(args.config)},
            "data_boundary": config["data"],
        },
    )
    record_progress(progress, status="started", experiment_id=config["experiment_id"])

    rgb_iteration = int(config["frozen_rgb"]["iteration"])
    rgb_hashes_before = frozen_rgb_hashes(args.rgb_run_dir, rgb_iteration)
    input_artifacts = {
        "config": file_record(args.config),
        "dataset_manifest": file_record(REPO_ROOT / config["dataset_manifest"]),
        "semantic_split": file_record(REPO_ROOT / config["data"]["semantic_split"]),
        "label_manifest": file_record(REPO_ROOT / config["data"]["label_manifest"]),
        "sem001_checkpoint": file_record(args.sem001_checkpoint),
        "sem003_train_only_evidence": file_record(args.sem003_evidence),
        "frozen_rgb_artifacts": rgb_hashes_before,
    }
    require(
        input_artifacts["sem001_checkpoint"]["sha256"] == config["frozen_semantic"]["semantic_checkpoint_sha256"],
        "SEM-001 checkpoint SHA-256 differs from the frozen SEM-004 config",
    )
    require(
        input_artifacts["sem003_train_only_evidence"]["sha256"] == config["sem003_train_only_evidence"]["artifact_sha256"],
        "SEM-003 evidence artifact SHA-256 differs from SEM-004 config",
    )
    evidence_payload = torch.load(args.sem003_evidence, map_location="cpu", weights_only=True)
    require(evidence_payload.get("experiment_id") == "SEM-003-DIAG", "Wrong SEM-003 evidence source")
    require(evidence_payload.get("uses_validation_rgb_or_labels") is False, "SEM-003 evidence must remain training-only")
    require(evidence_payload.get("uses_unlabeled_rgb") is False, "SEM-003 evidence must exclude unlabeled RGB")

    settings = ScaffoldSettings(
        train_camera_count=int(config["frozen_rgb"]["train_camera_count"]), n_offsets=int(config["frozen_rgb"]["n_offsets"])
    )
    point_cloud = args.rgb_run_dir / "point_cloud" / f"iteration_{rgb_iteration}" / "point_cloud.ply"
    configure_scaffold_import(args.source_root)
    model = load_frozen_scaffold(args.source_root, point_cloud, settings)
    require(not frozen_rgb_named_parameters(model) or not any(parameter.requires_grad for _name, parameter in frozen_rgb_named_parameters(model)), "Frozen RGB parameter unexpectedly trainable")
    sidecar = s0.reload_sidecar(args.sem001_checkpoint, int(model.get_anchor.shape[0]), load_json(REPO_ROOT / "experiments/runs/SEM-001/config.yaml"), rgb_hashes_before)
    sidecar.logits.requires_grad_(False)
    pipe = pipeline()
    anchor_count = int(model.get_anchor.shape[0])
    require(tuple(evidence_payload["evidence"].shape) == (anchor_count, CLASS_COUNT), "SEM-003 evidence anchor count mismatch")
    evidence = evidence_payload["evidence"].numpy().astype(np.float64, copy=False)
    evidence_fraction = evidence / np.maximum(evidence.sum(axis=1, keepdims=True), np.finfo(np.float64).tiny)
    support_views = evidence_payload["support_views"].numpy().astype(np.int32, copy=False)
    eligibility = evidence_payload["eligibility"].numpy().astype(bool, copy=False)
    anchors = model.get_anchor.detach().cpu().numpy().astype(np.float64, copy=False)
    direction = DirectionAccumulator(anchors, support_views, config["direction_diagnostic"])
    extension = load_visibility_extension(args.extension_root)
    record_progress(progress, status="frozen_model_loaded", anchors=anchor_count, sem001_checkpoint_sha256=input_artifacts["sem001_checkpoint"]["sha256"])

    camera_matrix, distortion = label_builder.read_camera(args.data_root / "camera_parameters" / "cameras.txt")
    label_root = args.output_dir / "label_geometry"
    label_root.mkdir(parents=True, exist_ok=True)
    train_names = list(split["semantic_train_labeled"])
    validation_names = list(split["semantic_validation_labeled"])
    label_train = _label_geometry_audit(
        names=train_names,
        split_name="training",
        data_root=args.data_root,
        labels_root=args.labels_root,
        matrix=camera_matrix,
        distortion=distortion,
        adjacency_radius=int(config["label_geometry"]["adjacency_radius_px"]),
        output_dir=label_root,
    )
    label_validation = _label_geometry_audit(
        names=validation_names,
        split_name="validation_read_only",
        data_root=args.data_root,
        labels_root=args.labels_root,
        matrix=camera_matrix,
        distortion=distortion,
        adjacency_radius=int(config["label_geometry"]["adjacency_radius_px"]),
        output_dir=label_root,
    )
    representative = sorted(
        [row for row in csv.DictReader((label_root / "labelme_shape_audit_training.csv").open(encoding="utf-8")) if row["label"] == "stay_cable"],
        key=lambda row: (-abs(float(row["rasterization_area_delta_fraction"])), row["view"], int(row["shape_index"])),
    )[: int(config["label_geometry"]["representative_training_cases"])]
    write_json(label_root / "summary.json", {"training": label_train, "validation_read_only": label_validation, "representative_training_shapes": representative})
    record_progress(progress, status="label_geometry_complete", training_views=len(train_names), validation_views=len(validation_names))

    rules = config["error_region_rules"]
    split_outputs: dict[str, Any] = {}
    all_per_view: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    contribution_groups = {"training": {"background_to_stay_cable": _new_group(), "correct_background": _new_group()}, "validation_read_only": {"background_to_stay_cable": _new_group(), "correct_background": _new_group()}}
    started = time.perf_counter()
    for split_name, names in (("training", train_names), ("validation_read_only", validation_names)):
        combined = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
        category_pixels: Counter[str] = Counter()
        fp_components: list[dict[str, Any]] = []
        pred_components: list[dict[str, Any]] = []
        gt_components: list[dict[str, Any]] = []
        view_root = args.output_dir / "visualizations" / split_name
        view_root.mkdir(parents=True, exist_ok=True)
        split_rows: list[dict[str, Any]] = []
        for index, name in enumerate(names, start=1):
            target = s0.read_label(args.labels_root / name)
            _selected, _camera, view = s0.official_view(args.data_root, name, int(config["frozen_rgb"]["appearance_camera_id"]))
            with torch.no_grad():
                prediction = render_semantic_probabilities(view, model, sidecar, pipe).argmax(dim=0).cpu().numpy().astype(np.uint8)
                diagnostics = render_frozen_contribution_diagnostics(view, model, pipe, extension=extension)
            foreground = diagnostics["foreground_mass"].detach().cpu().numpy().astype(np.float32, copy=False)
            maximum = diagnostics["max_contribution"].detach().cpu().numpy().astype(np.float32, copy=False)
            contributor_count = diagnostics["contributor_count"].detach().cpu().numpy().astype(np.int32, copy=False)
            dominant_offset = diagnostics["dominant_offset_index"].detach().cpu().numpy().astype(np.int64, copy=False)
            offset_to_anchor = diagnostics["offset_anchor_indices"].detach().cpu().numpy().astype(np.int64, copy=False)
            anchor_indices = np.full(dominant_offset.shape, -1, dtype=np.int64)
            valid_offset = (dominant_offset >= 0) & (dominant_offset < offset_to_anchor.size)
            anchor_indices[valid_offset] = offset_to_anchor[dominant_offset[valid_offset]]
            structure = error_structure(
                target,
                prediction,
                near_boundary_px=float(rules["near_gt_stay_cable_boundary_px"]),
                small_component_area_px=int(rules["small_false_positive_component_area_px"]),
                large_component_area_px=int(rules["large_false_positive_component_area_px"]),
            )
            local_confusion = np.bincount(
                CLASS_COUNT * target.reshape(-1).astype(np.int64) + prediction.reshape(-1).astype(np.int64), minlength=CLASS_COUNT**2
            ).reshape(CLASS_COUNT, CLASS_COUNT)
            combined += local_confusion
            fp_mask = (target == BACKGROUND) & (prediction == STAY_CABLE)
            correct_background_mask = (target == BACKGROUND) & (prediction == BACKGROUND)
            _update_group(
                contribution_groups[split_name]["background_to_stay_cable"],
                fp_mask,
                foreground=foreground,
                maximum=maximum,
                contributor_count=contributor_count,
                anchor_indices=anchor_indices,
                support_views=support_views,
                evidence_fraction=evidence_fraction,
                eligibility=eligibility,
            )
            _update_group(
                contribution_groups[split_name]["correct_background"],
                correct_background_mask,
                foreground=foreground,
                maximum=maximum,
                contributor_count=contributor_count,
                anchor_indices=anchor_indices,
                support_views=support_views,
                evidence_fraction=evidence_fraction,
                eligibility=eligibility,
            )
            if split_name == "training":
                direction.update(name, view.camera_center.detach().cpu().numpy().astype(np.float64, copy=False), anchor_indices, maximum, target)
            for category, value in structure["false_positive_categories"].items():
                category_pixels[category] += int(value.get("pixels", 0))
            for kind, rows in (("false_positive", structure["false_positive_components"]["rows"]), ("prediction_stay_cable", structure["prediction_stay_cable_components"]["rows"]), ("ground_truth_stay_cable", structure["ground_truth_stay_cable_components"]["rows"])):
                for row in rows:
                    component_rows.append({"split": split_name, "view": name, "kind": kind, **row})
                if kind == "false_positive":
                    fp_components.extend(rows)
                elif kind == "prediction_stay_cable":
                    pred_components.extend(rows)
                else:
                    gt_components.extend(rows)
            metrics = s0.confusion_metrics(local_confusion)
            split_rows.append(
                {
                    "split": split_name,
                    "view": name,
                    "background_to_stay_cable_pixels": structure["background_to_stay_cable_pixels"],
                    "stay_cable_to_background_pixels": structure["stay_cable_to_background_pixels"],
                    "false_positive_distance_mean_px": structure["false_positive_distance_to_gt_stay_cable_px"]["mean"],
                    "false_positive_components": structure["false_positive_components"]["count"],
                    "prediction_stay_cable_components": structure["prediction_stay_cable_components"]["count"],
                    "ground_truth_stay_cable_components": structure["ground_truth_stay_cable_components"]["count"],
                    "mIoU_with_background": metrics["mIoU_with_background"],
                    "mIoU_without_background": metrics["mIoU_without_background"],
                    "stay_cable_IoU": metrics["stay_cable"]["IoU"],
                    "stay_cable_precision": metrics["stay_cable"]["precision"],
                    "stay_cable_recall": metrics["stay_cable"]["recall"],
                }
            )
            Image.fromarray(semantic_triptych(target, prediction), mode="RGB").save(view_root / name)
            if index == 1 or index % 20 == 0 or index == len(names):
                record_progress(progress, status="view_complete", split=split_name, index=index, total=len(names), view=name)
                print(json.dumps({"split": split_name, "index": index, "total": len(names), "view": name}, sort_keys=True), flush=True)
            del diagnostics
        all_per_view.extend(split_rows)
        split_outputs[split_name] = _split_error_summary(
            confusion=combined,
            per_view=split_rows,
            fp_component_rows=fp_components,
            pred_component_rows=pred_components,
            gt_component_rows=gt_components,
            category_pixels=category_pixels,
        )
    _write_rows(args.output_dir / "per_view_error_metrics.csv", all_per_view)
    _write_rows(args.output_dir / "component_metrics.csv", component_rows)
    contribution_summary = {split_name: {group_name: _finalize_group(group) for group_name, group in groups.items()} for split_name, groups in contribution_groups.items()}
    direction_summary, direction_records = direction.summary()
    _write_rows(args.output_dir / "direction_reproducible_anchors.csv", direction_records)
    _write_rows(args.output_dir / "direction_bins.csv", direction_summary["per_bin"])
    write_json(args.output_dir / "contribution_summary.json", contribution_summary)
    write_json(args.output_dir / "direction_summary.json", direction_summary)
    _save_distribution_plot(
        args.output_dir / "false_positive_structure_training.png",
        title="Training-side frozen SEM-001 background-to-stay_cable structure",
        values={
            "near boundary": float(split_outputs["training"]["false_positive_category_pixels"].get("near_gt_stay_cable_boundary_expansion", 0)),
            "remote": float(split_outputs["training"]["false_positive_category_pixels"].get("spatially_remote_from_gt_stay_cable", 0)),
            "small fragments": float(split_outputs["training"]["false_positive_category_pixels"].get("small_fragments", 0)),
            "large expansions": float(split_outputs["training"]["false_positive_category_pixels"].get("large_area_connected_expansions", 0)),
        },
    )
    _save_direction_plot(args.output_dir / "direction_training_only.png", direction_summary)

    # A repeat render and artifact hashes prove that this read-only diagnostic
    # did not mutate frozen RGB state.  The external renderer consumes an
    # official CAMERA_FROM_WORLD camera and no reference RGB/label.
    external_root = args.output_dir / "external_camera_audit"
    before_rgb = s0.tensor_to_rgb(render_frozen_rgb(s0.official_view(args.data_root, config["external_camera_audit"]["image_name"], int(config["frozen_rgb"]["appearance_camera_id"]))[2], model, pipe))
    external = s0.render_external(
        external_root,
        args.data_root,
        config["external_camera_audit"]["image_name"],
        int(config["frozen_rgb"]["appearance_camera_id"]),
        model,
        sidecar,
        pipe,
    )
    after_rgb = s0.tensor_to_rgb(render_frozen_rgb(s0.official_view(args.data_root, config["external_camera_audit"]["image_name"], int(config["frozen_rgb"]["appearance_camera_id"]))[2], model, pipe))
    rgb_max_difference = int(np.abs(before_rgb.astype(np.int16) - after_rgb.astype(np.int16)).max())
    rgb_hashes_after = frozen_rgb_hashes(args.rgb_run_dir, rgb_iteration)
    require(rgb_hashes_before == rgb_hashes_after and rgb_max_difference == 0, "SEM-004 diagnostic changed frozen RGB evidence")
    require(external["semantic_png_dtype"] == "uint8" and external["semantic_png_min"] >= 0 and external["semantic_png_max"] < CLASS_COUNT, "External semantic PNG audit failed")
    require(external["rgb_shape"][:2] == [989, 1320] and external["semantic_logits_shape"] == [5, 989, 1320], "External camera output dimensions failed")

    training_fp = contribution_summary["training"]["background_to_stay_cable"]
    training_bg = contribution_summary["training"]["correct_background"]
    fp_pixels = int(split_outputs["training"]["background_to_stay_cable_false_positive_pixels"])
    category_pixels = split_outputs["training"]["false_positive_category_pixels"]
    near_fraction = float(category_pixels.get("near_gt_stay_cable_boundary_expansion", 0) / max(fp_pixels, 1))
    remote_fraction = float(category_pixels.get("spatially_remote_from_gt_stay_cable", 0) / max(fp_pixels, 1))
    thinness = label_train["stay_cable_polygon_thinness"]
    # The shape-level CSV carries the exact distribution needed for this
    # train-only threshold, rather than borrowing validation samples.
    training_shapes = list(csv.DictReader((label_root / "labelme_shape_audit_training.csv").open(encoding="utf-8")))
    cable_shapes = [row for row in training_shapes if row["label"] == "stay_cable"]
    thin_fraction = float(
        sum(float(row["pinhole_thinness"]) >= float(config["hypothesis_readiness"]["thin_polygon_minimum_thinness"]) for row in cable_shapes)
        / max(len(cable_shapes), 1)
    )
    fp_concentration = training_fp["means_over_all_pixels"]["maximum_over_foreground_concentration"]
    bg_concentration = training_bg["means_over_all_pixels"]["maximum_over_foreground_concentration"]
    concentration_difference = abs(float(fp_concentration) - float(bg_concentration)) if fp_concentration is not None and bg_concentration is not None else 0.0
    fp_eligibility = training_fp["sem003_conflict_eligibility_fraction_over_dominant_anchor_pixels"]
    bg_eligibility = training_bg["sem003_conflict_eligibility_fraction_over_dominant_anchor_pixels"]
    eligibility_lift = float(fp_eligibility / max(float(bg_eligibility), np.finfo(np.float64).tiny)) if fp_eligibility is not None and bg_eligibility is not None else 0.0
    rules_ready = config["hypothesis_readiness"]
    signals = {
        "dominant_training_error_geometry": max(near_fraction, remote_fraction) >= float(rules_ready["dominant_error_structure_fraction"]),
        "fine_stay_cable_label_geometry": thin_fraction >= float(rules_ready["minimum_thin_polygon_fraction"]),
        "frozen_contribution_separation": concentration_difference >= float(rules_ready["minimum_contribution_concentration_absolute_difference"]) or eligibility_lift >= float(rules_ready["minimum_conflict_eligibility_lift"]),
        "training_half_directional_reproducibility": direction_summary["reproducible_directional_anchor_fraction"] >= float(rules_ready["minimum_reproducible_direction_anchor_fraction"]),
    }
    passed_signals = [name for name, passed in signals.items() if passed]
    conclusion = "hypothesis_ready_for_new_preregistration" if len(passed_signals) >= int(rules_ready["minimum_training_signals"]) else "no_go_for_new_semantic_mechanism"
    next_step = (
        {
            "allowed_action": "prepare but do not start a separately authorized SEM-004/S3 preregistration",
            "only_main_variable": rules_ready["proposal_if_ready"],
            "matched_parent_control": "SEM-001 static sidecar under the identical 219/74 spatial protocol",
            "failure_conditions": "Any RGB difference, protected-class decline beyond a pre-registered bound, lack of stay_cable improvement, or increased background-to-stay_cable false positives drops the candidate.",
            "estimated_batch": {"tasks": 2, "iterations_per_task": 10000, "estimated_gpu_hours": 1.8, "estimated_peak_gpu_memory_mib": 1000, "estimated_external_disk_gib": 8},
            "authorization_required_before_training": True,
        }
        if conclusion == "hypothesis_ready_for_new_preregistration"
        else {
            "allowed_action": "do not start a new semantic mechanism; preserve SEM-001 and direct optional future GPU budget to independently authorized A6000 GeometryPrior work",
            "authorization_required_before_training": True,
        }
    )
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": config["status"],
        "trains_new_parameters": False,
        "updates_RGB": False,
        "updates_semantic_parameters": False,
        "uses_unlabeled_RGB": False,
        "uses_pseudo_labels": False,
        "environment": environment_summary(),
        "input_artifacts": input_artifacts,
        "data_boundary": {"training_views": len(train_names), "validation_views_read_only": len(validation_names), "guard_band_used": False, "hidden_test_used": False},
        "label_geometry": {"training": label_train, "validation_read_only": label_validation, "representative_training_shapes": representative},
        "error_audit": split_outputs,
        "contribution_audit": contribution_summary,
        "direction_audit_training_only": direction_summary,
        "hypothesis_readiness": {
            "signals": signals,
            "passed_signals": passed_signals,
            "required_signals": int(rules_ready["minimum_training_signals"]),
            "near_boundary_fraction": near_fraction,
            "remote_fraction": remote_fraction,
            "thin_polygon_fraction": thin_fraction,
            "contribution_concentration_absolute_difference": concentration_difference,
            "sem003_conflict_eligibility_lift_fp_over_correct_background": eligibility_lift,
            "conclusion": conclusion,
            "next_step": next_step,
            "causality_note": "Frozen contribution and direction summaries are correlations. They can motivate one future pre-registered variable but do not establish that a semantic regularizer will improve blind-test accuracy.",
        },
        "freeze_and_interface_audit": {
            "frozen_rgb_artifact_hashes_equal": rgb_hashes_before == rgb_hashes_after,
            "rgb_max_absolute_pixel_difference": rgb_max_difference,
            "frozen_rgb_trainable_parameters": [name for name, parameter in frozen_rgb_named_parameters(model) if parameter.requires_grad],
            "semantic_checkpoint_loaded": True,
            "semantic_checkpoint_writer_or_optimizer_created": False,
            "external_camera": external,
            "external_camera_uses_test_rgb_or_labels": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_mib": int(torch.cuda.max_memory_allocated() / 1024**2),
    }
    write_json(args.output_dir / "result.json", result)
    manifest = _output_manifest(args.output_dir)
    write_json(args.output_dir / "output_manifest.json", manifest)
    record_progress(progress, status="completed", conclusion=conclusion, passed_signals=passed_signals)
    print(json.dumps({"experiment_id": config["experiment_id"], "conclusion": conclusion, "passed_signals": passed_signals, "peak_cuda_allocated_mib": result["peak_cuda_allocated_mib"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
