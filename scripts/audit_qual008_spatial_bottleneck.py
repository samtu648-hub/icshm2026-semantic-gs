#!/usr/bin/env python3
"""Join frozen RGB152 and SEM396 spatial-confirmation errors."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_semantic_audit import independent_confusion, metrics


EID = "QUAL-008-SPATIAL-BOTTLENECK-DIAG"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def average_ranks(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have the same nontrivial length")
    value = float(np.corrcoef(average_ranks(left), average_ranks(right))[0, 1])
    if not np.isfinite(value):
        raise ValueError("Non-finite rank correlation")
    return value


def boundary_counts(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    cable = target == 2
    predicted_cable = prediction == 2
    gt_boundary = cable ^ binary_erosion(cable)
    predicted_boundary = predicted_cable ^ binary_erosion(predicted_cable)
    gt_distance = distance_transform_edt(~gt_boundary)
    predicted_distance = distance_transform_edt(~predicted_boundary)
    return np.asarray(
        [
            np.sum(predicted_boundary & (gt_distance <= 2)),
            np.sum(predicted_boundary),
            np.sum(gt_boundary & (predicted_distance <= 2)),
            np.sum(gt_boundary),
        ],
        dtype=np.int64,
    )


def boundary_metrics(counts: np.ndarray) -> dict:
    precision = float(counts[0] / counts[1]) if counts[1] else None
    recall = float(counts[2] / counts[3]) if counts[3] else None
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "counts": counts.tolist(),
    }


def classify(
    better_metrics: dict,
    low_recall_views: int,
    correlations: dict[str, dict[str, float]],
    analysis: dict,
) -> dict:
    rule = analysis["semantic_residual_rule"]
    residual = (
        better_metrics["boundary"]["f1"]
        < rule["better_half_boundary_f1_below"]
        and low_recall_views
        >= rule["better_half_views_with_stay_cable_recall_below_0_90_at_least"]
    ) or (
        better_metrics["semantic"]["mIoU_with_background"]
        < rule["or_better_half_five_class_miou_below"]
    )
    threshold = analysis["strong_rgb_coupling_absolute_spearman_at_least"]
    coupling = max(
        abs(correlations["PSNR_dB"]["mIoU_with_background"]),
        abs(correlations["PSNR_dB"]["boundary_f1"]),
        abs(correlations["LPIPS_Alex"]["mIoU_with_background"]),
        abs(correlations["LPIPS_Alex"]["boundary_f1"]),
    ) >= threshold
    outcome = (
        "mixed_rgb_and_semantic_bottleneck"
        if residual and coupling
        else "semantic_residual_supported"
        if residual
        else "rgb_dominant_or_no_actionable_semantic_residual"
    )
    return {
        "semantic_residual_present": residual,
        "strong_rgb_coupling_present": coupling,
        "outcome": outcome,
        "boundary_candidate_supported": residual,
    }


def main() -> None:
    config = read(CONFIG)
    output = ROOT / config["output_path"]
    output.mkdir(parents=True, exist_ok=False)
    inputs = config["inputs"]
    fixed = {
        "rgb_holdout_csv": inputs["rgb_holdout_csv_sha256"],
        "rgb_evaluation_result": inputs["rgb_evaluation_result_sha256"],
        "semantic_inference_result": inputs["semantic_inference_result_sha256"],
    }
    paths = {
        key: ROOT / inputs[key]
        for key in ("rgb_holdout_csv", "rgb_evaluation_result", "semantic_inference_result")
    }
    if any(sha256(paths[key]) != expected for key, expected in fixed.items()):
        raise ValueError("A frozen diagnostic input changed")
    split_path = ROOT / config["split_path"]
    split = read(split_path)
    holdout = set(split["primary_fold"]["semantic_validation_labeled"])
    if len(holdout) != config["analysis"]["view_count"]:
        raise ValueError("Expected the registered 30-view confirmation role")

    with paths["rgb_holdout_csv"].open(newline="", encoding="utf-8") as handle:
        rgb = {row["view"]: row for row in csv.DictReader(handle)}
    semantic_result = read(paths["semantic_inference_result"])
    semantic = {
        row["name"]: row
        for row in semantic_result["per_view"]
        if row["role"] == "holdout"
    }
    if set(rgb) != holdout or set(semantic) != holdout:
        raise ValueError("RGB and semantic confirmation identities differ")
    predictions = ROOT / inputs["prediction_root"]
    verify_hashes(predictions, {name: semantic_result["prediction_hashes"][name] for name in holdout})
    labels = ROOT / inputs["label_root"]
    manifest = read(ROOT / inputs["label_manifest"])
    label_hashes = {row["image"]: row["label_png_sha256"] for row in manifest["records"]}
    verify_hashes(labels, {name: label_hashes[name] for name in holdout})

    rows = []
    for name in sorted(holdout):
        target = np.asarray(Image.open(labels / name), dtype=np.uint8)
        prediction = np.asarray(Image.open(predictions / name), dtype=np.uint8)
        confusion = independent_confusion(target, prediction)
        semantic_metrics = metrics(confusion)
        counts = boundary_counts(target, prediction)
        bmetrics = boundary_metrics(counts)
        cable = semantic_metrics["per_class"]["stay_cable"]
        rows.append(
            {
                "name": name,
                "group": semantic[name]["group"],
                "PSNR_dB": float(rgb[name]["PSNR_dB"]),
                "SSIM": float(rgb[name]["SSIM"]),
                "LPIPS_Alex": float(rgb[name]["LPIPS_Alex"]),
                "mIoU_with_background": semantic_metrics["mIoU_with_background"],
                "mIoU_without_background": semantic_metrics["mIoU_without_background"],
                "stay_cable_IoU": cable["IoU"],
                "stay_cable_precision": cable["precision"],
                "stay_cable_recall": cable["recall"],
                "boundary_f1": bmetrics["f1"],
                "confusion": confusion.tolist(),
                "boundary_counts": counts.tolist(),
            }
        )

    rgb_keys = ("PSNR_dB", "SSIM", "LPIPS_Alex")
    semantic_keys = (
        "mIoU_with_background",
        "mIoU_without_background",
        "stay_cable_IoU",
        "stay_cable_recall",
        "boundary_f1",
    )
    correlations = {
        rkey: {
            skey: spearman(
                [row[rkey] for row in rows], [row[skey] for row in rows]
            )
            for skey in semantic_keys
        }
        for rkey in rgb_keys
    }
    ranked = sorted(rows, key=lambda row: (-row["PSNR_dB"], row["name"]))
    better = ranked[: config["analysis"]["rgb_better_half_count"]]
    worse = ranked[config["analysis"]["rgb_better_half_count"] :]

    def aggregate(selected: list[dict]) -> dict:
        confusion = np.sum(
            [np.asarray(row["confusion"], dtype=np.int64) for row in selected], axis=0
        )
        counts = np.sum(
            [np.asarray(row["boundary_counts"], dtype=np.int64) for row in selected],
            axis=0,
        )
        return {
            "names": [row["name"] for row in selected],
            "mean_rgb": {
                key: float(np.mean([row[key] for row in selected]))
                for key in rgb_keys
            },
            "semantic": metrics(confusion),
            "boundary": boundary_metrics(counts),
        }

    better_metrics = aggregate(better)
    worse_metrics = aggregate(worse)
    low_recall_views = sum(row["stay_cable_recall"] < 0.90 for row in better)
    decision = classify(
        better_metrics,
        low_recall_views,
        correlations,
        config["analysis"],
    )
    with (output / "per-view.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "group",
                *rgb_keys,
                *semantic_keys,
                "stay_cable_precision",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    write_new_json(
        output / "result.json",
        {
            "schema_version": 1,
            "status": "completed",
            "experiment_id": EID,
            "config_sha256": sha256(CONFIG),
            "split_sha256": sha256(split_path),
            "input_sha256": {key: sha256(path) for key, path in paths.items()},
            "view_count": len(rows),
            "correlations": correlations,
            "better_rgb_half": better_metrics,
            "worse_rgb_half": worse_metrics,
            "better_half_low_stay_cable_recall_view_count": low_recall_views,
            "decision": decision,
            "per_view_csv_sha256": sha256(output / "per-view.csv"),
        },
    )


if __name__ == "__main__":
    main()
