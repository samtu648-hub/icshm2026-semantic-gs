#!/usr/bin/env python3
"""Run SEM-004 restricted first-order observation-direction residual gates.

The runner loads the final frozen SEM-001 checkpoint and trains only a
``[7069, 3]`` direction-coefficient tensor selected by the immutable
SEM-004-DIAG artifact.  No RGB or static semantic parameter is optimized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem001 as s0  # noqa: E402
from src.scaffold_directional_residual import (  # noqa: E402
    DirectionalResidualSidecar,
    load_direction_anchor_indices,
    render_directional_semantic_probabilities,
)
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


CLASS_NAMES = s0.CLASS_NAMES
CLASS_COUNT = len(CLASS_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sem001-checkpoint", type=Path, required=True)
    parser.add_argument("--sem001-result", type=Path, required=True)
    parser.add_argument("--sem001-per-view", type=Path, required=True)
    parser.add_argument("--diag-result", type=Path, required=True)
    parser.add_argument("--diag-manifest", type=Path, required=True)
    parser.add_argument("--direction-anchors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Git-ignored experiment output root")
    parser.add_argument("--mode", choices=("equivalence", "preflight", "full"), required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True)
    return result.stdout.decode("utf-8", errors="replace")


def _git_evidence() -> dict[str, Any]:
    diff = _git_value("diff", "--binary").encode("utf-8")
    return {
        "head": _git_value("rev-parse", "HEAD").strip(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "status_short": _git_value("status", "--short").splitlines(),
    }


def _environment(torch: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory_mib": int(torch.cuda.get_device_properties(0).total_memory / 1024**2)
        if torch.cuda.is_available()
        else None,
    }


def _sem001_payload(path: Path, config: dict[str, Any], rgb_hashes: dict[str, str]) -> dict[str, Any]:
    import torch

    if sha256(path) != config["inputs"]["sem001_checkpoint_sha256"]:
        raise ValueError("Final SEM-001 checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("experiment_id") != "SEM-001":
        raise ValueError("SEM-004 must use the final SEM-001 checkpoint")
    if payload.get("semantic_storage") != "per_anchor_logits" or payload.get("class_names") != list(CLASS_NAMES):
        raise ValueError("SEM-001 checkpoint schema or official class mapping mismatch")
    if payload.get("frozen_rgb_hashes") != rgb_hashes:
        raise ValueError("SEM-001 checkpoint does not belong to the loaded frozen RGB artifact")
    logits = payload.get("semantic_anchor_logits")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != CLASS_COUNT:
        raise ValueError("SEM-001 checkpoint has no [anchors,5] static logits")
    return payload


def _validate_diagnostic_inputs(args: argparse.Namespace, config: dict[str, Any], anchor_count: int) -> list[int]:
    expected = config["inputs"]
    paths_and_hashes = (
        (args.diag_result, expected["sem004_diag_result_sha256"]),
        (args.diag_manifest, expected["sem004_diag_output_manifest_sha256"]),
        (args.direction_anchors, expected["direction_anchor_csv_sha256"]),
        (args.sem001_result, expected["sem001_result_sha256"]),
        (args.sem001_per_view, expected["sem001_per_view_sha256"]),
    )
    for path, digest in paths_and_hashes:
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Frozen input is missing or has the wrong SHA-256: {path.name}")
    result = load_json(args.diag_result)
    direction = result.get("direction_audit_training_only", {})
    if result.get("experiment_id") != "SEM-004-DIAG" or result.get("status") != "diagnostic_only_not_a_semantic_candidate":
        raise ValueError("Diagnostic result provenance/status mismatch")
    if direction.get("training_only") is not True or int(direction.get("reproducible_directional_anchors", -1)) != int(
        config["residual"]["enabled_anchor_count"]
    ):
        raise ValueError("Diagnostic result does not contain the frozen training-only 7,069-anchor mask")
    manifest = load_json(args.diag_manifest)
    matching = [entry for entry in manifest.get("files", []) if entry.get("relative_path") == "direction_reproducible_anchors.csv"]
    if len(matching) != 1 or matching[0].get("sha256") != expected["direction_anchor_csv_sha256"]:
        raise ValueError("Diagnostic output manifest does not authenticate the direction anchor CSV")
    return load_direction_anchor_indices(
        args.direction_anchors,
        expected_count=int(config["residual"]["enabled_anchor_count"]),
        anchor_count=anchor_count,
    )


def _new_sidecar(config: dict[str, Any], payload: dict[str, Any], indices: list[int]) -> DirectionalResidualSidecar:
    return DirectionalResidualSidecar(
        payload["semantic_anchor_logits"], indices, cap=float(config["residual"]["cap"]), device="cuda"
    )


def _save_checkpoint(
    path: Path,
    sidecar: DirectionalResidualSidecar,
    config: dict[str, Any],
    rgb_hashes: dict[str, str],
    input_hashes: dict[str, str],
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "parent_experiment": "SEM-001",
            "semantic_storage": "frozen_per_anchor_logits_plus_sparse_first_order_direction_residual",
            "class_names": list(CLASS_NAMES),
            "frozen_rgb_hashes": rgb_hashes,
            "input_artifact_sha256": input_hashes,
            "residual_cap": sidecar.cap,
            **sidecar.checkpoint_state(),
        },
        path,
    )


def _reload_checkpoint(
    path: Path,
    config: dict[str, Any],
    payload: dict[str, Any],
    indices: list[int],
    rgb_hashes: dict[str, str],
    input_hashes: dict[str, str],
) -> DirectionalResidualSidecar:
    import torch

    stored = torch.load(path, map_location="cpu", weights_only=True)
    if stored.get("experiment_id") != config["experiment_id"]:
        raise ValueError("Residual checkpoint experiment mismatch")
    if stored.get("semantic_storage") != "frozen_per_anchor_logits_plus_sparse_first_order_direction_residual":
        raise ValueError("Residual checkpoint storage schema mismatch")
    if stored.get("frozen_rgb_hashes") != rgb_hashes or stored.get("input_artifact_sha256") != input_hashes:
        raise ValueError("Residual checkpoint frozen input provenance mismatch")
    if float(stored.get("residual_cap", -1.0)) != float(config["residual"]["cap"]):
        raise ValueError("Residual checkpoint cap mismatch")
    reloaded = _new_sidecar(config, payload, indices)
    reloaded.load_checkpoint_state(stored)
    return reloaded


def _render_external(
    output_dir: Path,
    data_root: Path,
    image_name: str,
    appearance_camera_id: int,
    model: Any,
    sidecar: DirectionalResidualSidecar,
    pipe: Any,
) -> dict[str, Any]:
    import torch

    selected, camera, view = s0.official_view(data_root, image_name, appearance_camera_id)
    rgb = s0.tensor_to_rgb(render_frozen_rgb(view, model, pipe))
    with torch.no_grad():
        probabilities = render_directional_semantic_probabilities(view, model, sidecar, pipe).cpu().numpy().astype(np.float32)
    official_rgb = s0.redistort_rgb(rgb.astype(np.float32) / 255.0, camera)
    official_probabilities = s0.redistort_semantic_probabilities(probabilities, camera)
    logits = np.log(np.clip(official_probabilities, np.finfo(np.float32).tiny, 1.0))
    labels = logits.argmax(axis=0).astype(np.uint8)
    if labels.shape != (camera.height, camera.width) or int(labels.max()) >= CLASS_COUNT:
        raise RuntimeError("External semantic output is not a single-channel uint8 0..4 image")
    root = output_dir / "external_camera"
    root.mkdir(parents=True, exist_ok=True)
    rgb_path = root / "rgb.png"
    semantic_path = root / "semantic.png"
    logits_path = root / "semantic_logits.npy"
    Image.fromarray(np.rint(official_rgb.clip(0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(rgb_path)
    Image.fromarray(labels, mode="L").save(semantic_path)
    np.save(logits_path, logits)
    return {
        "selected_pose_name": selected,
        "input_pose_convention": "CAMERA_FROM_WORLD",
        "official_to_model_sim3": {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]},
        "input_camera_model": "SIMPLE_RADIAL",
        "output_size_from_input_camera": True,
        "rgb_shape": list(official_rgb.shape),
        "semantic_logits_shape": list(logits.shape),
        "semantic_png_shape": list(labels.shape),
        "semantic_png_dtype": str(labels.dtype),
        "semantic_png_min": int(labels.min()),
        "semantic_png_max": int(labels.max()),
        "uses_test_rgb_or_labels": False,
        "sha256": {"rgb": sha256(rgb_path), "semantic_png": sha256(semantic_path), "logits": sha256(logits_path)},
    }


def _residual_summary(sidecar: DirectionalResidualSidecar, deltas: list[np.ndarray]) -> dict[str, Any]:
    coefficient = sidecar.direction_coefficients.detach().cpu().numpy()
    norms = np.linalg.norm(coefficient, axis=1)
    values = np.concatenate(deltas) if deltas else np.empty(0, dtype=np.float32)
    return {
        "enabled_anchors": sidecar.selected_count,
        "trainable_scalars": int(sidecar.direction_coefficients.numel()),
        "coefficient_l2_mean": float(np.mean(norms)),
        "coefficient_l2_median": float(np.median(norms)),
        "coefficient_l2_max": float(np.max(norms)),
        "nonzero_coefficient_anchor_fraction": float(np.mean(norms > 0.0)),
        "observed_delta_abs_mean": float(np.mean(np.abs(values))) if values.size else 0.0,
        "observed_delta_abs_p95": float(np.quantile(np.abs(values), 0.95)) if values.size else 0.0,
        "observed_delta_abs_max": float(np.max(np.abs(values))) if values.size else 0.0,
        "direct_logit_effect": {
            "background": "-delta",
            "deck_max_abs": 0.0,
            "stay_cable": "+delta",
            "tower_max_abs": 0.0,
            "foundation_max_abs": 0.0,
        },
    }


def _evaluate(
    output_dir: Path,
    names: list[str],
    labels_root: Path,
    data_root: Path,
    appearance_camera_id: int,
    model: Any,
    sidecar: DirectionalResidualSidecar,
    pipe: Any,
) -> dict[str, Any]:
    import torch

    combined = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    per_view: list[dict[str, Any]] = []
    delta_samples: list[np.ndarray] = []
    render_root = output_dir / "validation_semantic_png"
    render_root.mkdir(parents=True, exist_ok=True)
    render_times: list[float] = []
    for name in names:
        target = s0.read_label(labels_root / name)
        _, _camera, view = s0.official_view(data_root, name, appearance_camera_id)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            probabilities, stats = render_directional_semantic_probabilities(
                view, model, sidecar, pipe, return_stats=True
            )
        torch.cuda.synchronize()
        render_times.append(time.perf_counter() - started)
        prediction = probabilities.argmax(dim=0).cpu().numpy().astype(np.uint8)
        Image.fromarray(prediction, mode="L").save(render_root / name)
        local = np.bincount(
            CLASS_COUNT * target.reshape(-1).astype(np.int64) + prediction.reshape(-1).astype(np.int64),
            minlength=CLASS_COUNT**2,
        ).reshape(CLASS_COUNT, CLASS_COUNT)
        combined += local
        enabled_delta = stats["delta"][stats["enabled"]].detach().cpu().numpy().astype(np.float32, copy=False)
        if enabled_delta.size:
            delta_samples.append(enabled_delta)
        per_view.append({"view": name, **s0.confusion_metrics(local)})
    metrics = s0.confusion_metrics(combined)
    worst_count = max(1, int(np.ceil(len(per_view) * 0.10)))
    worst = sorted(per_view, key=lambda record: record["mIoU_without_background"])[:worst_count]
    difficult = [record for record in per_view if 140 <= int(Path(record["view"]).stem) <= 146]
    metrics.update(
        {
            "view_count": len(names),
            "confusion_target_rows_prediction_columns": combined.tolist(),
            "per_view": per_view,
            "worst_10_percent": {
                "count": worst_count,
                "views": [item["view"] for item in worst],
                "mean_mIoU_without_background": float(np.mean([item["mIoU_without_background"] for item in worst])),
            },
            "challenge_140_146": {
                "views": [item["view"] for item in difficult],
                "mean_mIoU_without_background": float(np.mean([item["mIoU_without_background"] for item in difficult]))
                if difficult
                else None,
            },
            "single_view_render_seconds": {
                "mean": float(np.mean(render_times)),
                "median": float(np.median(render_times)),
            },
            "residual": _residual_summary(sidecar, delta_samples),
            "local_protocol_note": "Fixed SEM-001 full-image confusion protocol; validation did not select model hyperparameters.",
        }
    )
    with (output_dir / "validation_per_view.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "view",
            "mIoU_with_background",
            "mIoU_without_background",
            "stay_cable_IoU",
            "stay_cable_precision",
            "stay_cable_recall",
            "background_to_stay_cable_false_positive_pixels",
            "background_to_stay_cable_false_positive_fraction_of_gt_background",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in per_view:
            stay = record["stay_cable"]
            writer.writerow(
                {
                    "view": record["view"],
                    "mIoU_with_background": record["mIoU_with_background"],
                    "mIoU_without_background": record["mIoU_without_background"],
                    "stay_cable_IoU": stay["IoU"],
                    "stay_cable_precision": stay["precision"],
                    "stay_cable_recall": stay["recall"],
                    "background_to_stay_cable_false_positive_pixels": record[
                        "background_to_stay_cable_false_positive_pixels"
                    ],
                    "background_to_stay_cable_false_positive_fraction_of_gt_background": record[
                        "background_to_stay_cable_false_positive_fraction_of_gt_background"
                    ],
                }
            )
    return metrics


def _paired_comparison(
    output_dir: Path,
    candidate: dict[str, Any],
    sem001_result: dict[str, Any],
    sem001_per_view_path: Path,
) -> dict[str, Any]:
    baseline = sem001_result["evaluation"]
    baseline_rows: dict[str, dict[str, str]] = {}
    with sem001_per_view_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            baseline_rows[row["view"]] = row
    rows: list[dict[str, Any]] = []
    for record in candidate["per_view"]:
        base = baseline_rows[record["view"]]
        rows.append(
            {
                "view": record["view"],
                "delta_mIoU_with_background": float(record["mIoU_with_background"]) - float(base["mIoU_with_background"]),
                "delta_mIoU_without_background": float(record["mIoU_without_background"])
                - float(base["mIoU_without_background"]),
                "delta_stay_cable_IoU": float(record["stay_cable"]["IoU"]) - float(base["stay_cable_IoU"]),
                "delta_background_to_stay_cable_pixels": int(record["background_to_stay_cable_false_positive_pixels"])
                - int(base["background_to_stay_cable_false_positive_pixels"]),
            }
        )
    with (output_dir / "paired_vs_sem001.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_class_delta = {
        name: float(candidate["per_class"][name]["IoU"] - baseline["per_class"][name]["IoU"])
        for name in CLASS_NAMES
    }
    return {
        "candidate_minus_sem001": {
            "mIoU_with_background": float(candidate["mIoU_with_background"] - baseline["mIoU_with_background"]),
            "mIoU_without_background": float(candidate["mIoU_without_background"] - baseline["mIoU_without_background"]),
            "per_class_IoU": per_class_delta,
            "stay_cable_precision": float(
                candidate["stay_cable"]["precision"] - baseline["stay_cable"]["precision"]
            ),
            "stay_cable_recall": float(candidate["stay_cable"]["recall"] - baseline["stay_cable"]["recall"]),
            "background_to_stay_cable_pixels": int(
                candidate["background_to_stay_cable_false_positive_pixels"]
                - baseline["background_to_stay_cable_false_positive_pixels"]
            ),
            "background_to_stay_cable_relative_reduction": float(
                (baseline["background_to_stay_cable_false_positive_pixels"] - candidate["background_to_stay_cable_false_positive_pixels"])
                / baseline["background_to_stay_cable_false_positive_pixels"]
            ),
            "worst_10_percent_mIoU_without_background": float(
                candidate["worst_10_percent"]["mean_mIoU_without_background"]
                - baseline["worst_10_percent"]["mean_mIoU_without_background"]
            ),
            "challenge_140_146_mIoU_without_background": float(
                candidate["challenge_140_146"]["mean_mIoU_without_background"]
                - baseline["challenge_140_146"]["mean_mIoU_without_background"]
            ),
        },
        "views_improved_mIoU_with_background": int(sum(row["delta_mIoU_with_background"] > 0.0 for row in rows)),
        "views_reduced_background_to_stay_cable": int(
            sum(row["delta_background_to_stay_cable_pixels"] < 0 for row in rows)
        ),
        "paired_views": len(rows),
    }


def _decision(config: dict[str, Any], comparison: dict[str, Any], isolation: dict[str, Any], external: dict[str, Any]) -> dict[str, Any]:
    gate = config["keep_gate"]
    delta = comparison["candidate_minus_sem001"]
    checks = {
        "mIoU_with_background_plus_1pp": delta["mIoU_with_background"] >= float(gate["minimum_mIoU_with_background_gain"]),
        "stay_cable_IoU_not_lower": delta["per_class_IoU"]["stay_cable"] >= 0.0,
        "background_to_stay_cable_clearly_lower": delta["background_to_stay_cable_relative_reduction"]
        >= float(gate["minimum_background_to_stay_cable_relative_reduction"]),
        "deck_protected": delta["per_class_IoU"]["deck"] >= -float(gate["maximum_protected_class_IoU_drop"]),
        "tower_protected": delta["per_class_IoU"]["tower"] >= -float(gate["maximum_protected_class_IoU_drop"]),
        "foundation_protected": delta["per_class_IoU"]["foundation"]
        >= -float(gate["maximum_protected_class_IoU_drop"]),
        "worst_10_not_materially_lower": delta["worst_10_percent_mIoU_without_background"]
        >= -float(gate["maximum_worst_or_hard_mIoU_drop"]),
        "hard_140_146_not_materially_lower": delta["challenge_140_146_mIoU_without_background"]
        >= -float(gate["maximum_worst_or_hard_mIoU_drop"]),
        "rgb_exact": isolation["rgb_max_absolute_pixel_difference"] == 0
        and isolation["rgb_png_hashes_equal"]
        and isolation["rgb_artifact_hashes_equal"],
        "checkpoint_and_external_camera": external["input_pose_convention"] == "CAMERA_FROM_WORLD"
        and external["semantic_png_dtype"] == "uint8"
        and 0 <= external["semantic_png_min"] <= external["semantic_png_max"] <= 4,
    }
    keep = all(checks.values())
    return {
        "checks": checks,
        "passed": keep,
        "keep_or_drop": "keep_main_fold_pending_confirmation_authorization" if keep else "drop_unproven_main_fold",
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "failure_boundary": "No cap/mask/L2/learning-rate/seed retry and no S4/pseudo-label/three-fold launch if any check fails.",
    }


def _write_output_manifest(output_dir: Path) -> str:
    entries: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "output_manifest.json":
            entries.append({"relative_path": str(path.relative_to(output_dir)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    path = output_dir / "output_manifest.json"
    write_json(path, {"schema_version": 1, "file_count": len(entries), "files": entries})
    return sha256(path)


def _run_equivalence(
    args: argparse.Namespace,
    config: dict[str, Any],
    model: Any,
    payload: dict[str, Any],
    indices: list[int],
    pipe: Any,
) -> dict[str, Any]:
    import torch

    candidate = _new_sidecar(config, payload, indices)
    reference = AnchorSemanticSidecar(int(model.get_anchor.shape[0]))
    reference.load_state_dict(payload)
    maximum = 0.0
    classes_equal = True
    rgb_maximum = 0
    per_view: list[dict[str, Any]] = []
    for name in config["equivalence"]["views"]:
        _, _camera, view = s0.official_view(args.data_root, name, int(config["frozen_rgb"]["appearance_camera_id"]))
        with torch.no_grad():
            baseline = render_semantic_probabilities(view, model, reference, pipe)
            residual = render_directional_semantic_probabilities(view, model, candidate, pipe)
            rgb_before = s0.tensor_to_rgb(render_frozen_rgb(view, model, pipe))
            rgb_after = s0.tensor_to_rgb(render_frozen_rgb(view, model, pipe))
        difference = float((baseline - residual).abs().max().item())
        labels_equal = bool(torch.equal(baseline.argmax(dim=0), residual.argmax(dim=0)))
        rgb_difference = int(np.abs(rgb_before.astype(np.int16) - rgb_after.astype(np.int16)).max())
        maximum = max(maximum, difference)
        classes_equal = classes_equal and labels_equal
        rgb_maximum = max(rgb_maximum, rgb_difference)
        per_view.append({"view": name, "max_probability_difference": difference, "class_png_bit_identical": labels_equal, "rgb_max_absolute_pixel_difference": rgb_difference})
    tolerance = float(config["equivalence"]["max_probability_difference"])
    if maximum > tolerance or not classes_equal or rgb_maximum != 0 or bool((candidate.direction_coefficients != 0).any()):
        raise RuntimeError("SEM-040-EQ zero-residual equivalence gate failed")
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_equivalence",
        "views": per_view,
        "max_probability_difference": maximum,
        "class_png_bit_identical": classes_equal,
        "rgb_max_absolute_pixel_difference": rgb_maximum,
        "mask_anchor_count": candidate.selected_count,
        "trainable_parameter_shape": list(candidate.direction_coefficients.shape),
        "validation_labels_used": False,
    }


def _run_training(
    args: argparse.Namespace,
    config: dict[str, Any],
    split: dict[str, Any],
    label_manifest: dict[str, Any],
    model: Any,
    payload: dict[str, Any],
    indices: list[int],
    rgb_hashes: dict[str, str],
    input_hashes: dict[str, str],
    pipe: Any,
) -> dict[str, Any]:
    import torch

    budget = config[args.mode]
    iterations = int(budget["iterations"])
    if args.mode == "full" and iterations != 10000:
        raise ValueError("SEM-004 full run must use the registered 10,000-iteration budget")
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.reset_peak_memory_stats()
    sidecar = _new_sidecar(config, payload, indices)
    base_logits_before = sidecar.base_logits.detach().cpu().clone()
    optimizer = torch.optim.AdamW(
        [sidecar.direction_coefficients],
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(float(value) for value in config["training"]["betas"]),
        eps=float(config["training"]["epsilon"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    weights = torch.tensor(s0.class_weights(label_manifest), device="cuda")
    sampler = np.random.default_rng(seed)
    names = list(split["semantic_train_labeled"])
    validation = list(split["semantic_validation_labeled"])
    if args.mode == "preflight":
        validation = validation[: int(budget["validation_views"])]
    checkpoint_steps = {int(value) for value in budget.get("checkpoint_iterations", [])}
    losses: list[float] = []
    nll_losses: list[float] = []
    l2_losses: list[float] = []
    milestone_results: dict[str, Any] = {}
    started = time.perf_counter()
    for step in range(1, iterations + 1):
        name = names[int(sampler.integers(len(names)))]
        target = torch.from_numpy(s0.read_label(args.labels_root / name).astype(np.int64, copy=False)).to(
            device="cuda", non_blocking=True
        )
        _, _camera, view = s0.official_view(args.data_root, name, int(config["frozen_rgb"]["appearance_camera_id"]))
        probabilities = render_directional_semantic_probabilities(view, model, sidecar, pipe)
        nll = torch.nn.functional.nll_loss(
            torch.log(probabilities.clamp_min(1e-8)).unsqueeze(0), target.unsqueeze(0), weight=weights
        )
        l2 = sidecar.direction_coefficients.square().mean()
        loss = nll + float(config["training"]["l2_lambda"]) * l2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        frozen_with_grad = [
            name for name, parameter in frozen_rgb_named_parameters(model) if parameter.requires_grad or parameter.grad is not None
        ]
        if frozen_with_grad:
            raise RuntimeError("Frozen RGB received gradients: " + ", ".join(frozen_with_grad))
        if sidecar.base_logits.requires_grad or sidecar.base_logits.grad is not None:
            raise RuntimeError("Frozen SEM-001 static logits received gradients")
        gradient = sidecar.direction_coefficients.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Direction residual gradient is missing or non-finite")
        optimizer.step()
        if not bool(torch.isfinite(sidecar.direction_coefficients).all()):
            raise RuntimeError("Direction residual coefficients became non-finite")
        losses.append(float(loss.detach().cpu()))
        nll_losses.append(float(nll.detach().cpu()))
        l2_losses.append(float(l2.detach().cpu()))

        if step in checkpoint_steps:
            checkpoint = args.output_dir / "checkpoints" / f"direction_residual_iter_{step}.pth"
            _save_checkpoint(checkpoint, sidecar, config, rgb_hashes, input_hashes)
            milestone = _evaluate(
                args.output_dir / "milestones" / f"iter_{step}",
                validation,
                args.labels_root,
                args.data_root,
                int(config["frozen_rgb"]["appearance_camera_id"]),
                model,
                sidecar,
                pipe,
            )
            milestone_results[str(step)] = {
                "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size},
                "evaluation": milestone,
            }
        if step == 1 or step % int(budget["log_every"]) == 0 or step == iterations:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": losses[-1],
                        "nll": nll_losses[-1],
                        "l2": l2_losses[-1],
                        "mean_recent_loss": float(np.mean(losses[-min(len(losses), int(budget["log_every"])) :])),
                        "max_memory_allocated_mib": int(torch.cuda.max_memory_allocated() / 1024**2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    training_seconds = time.perf_counter() - started
    if not torch.equal(base_logits_before, sidecar.base_logits.detach().cpu()):
        raise RuntimeError("Frozen SEM-001 static logits changed during SEM-004")

    isolation = s0.RGB_isolation_evidence(
        args.output_dir,
        args.data_root,
        list(config["isolation"]["views"]),
        int(config["frozen_rgb"]["appearance_camera_id"]),
        model,
        pipe,
        rgb_hashes,
        args.rgb_run_dir,
        int(config["frozen_rgb"]["iteration"]),
    )
    isolation["semantic_optimizer_parameter_allowlist"] = ["direction_coefficients[7069,3]"]
    isolation["frozen_sem001_static_logits_bit_identical"] = True
    checkpoint = args.output_dir / "checkpoints" / "direction_residual.pth"
    _save_checkpoint(checkpoint, sidecar, config, rgb_hashes, input_hashes)
    reloaded = _reload_checkpoint(checkpoint, config, payload, indices, rgb_hashes, input_hashes)
    if not torch.equal(sidecar.direction_coefficients.detach().cpu(), reloaded.direction_coefficients.detach().cpu()):
        raise RuntimeError("Residual checkpoint reload is not bit-identical")
    external = _render_external(
        args.output_dir,
        args.data_root,
        str(config["external_camera"]["image_name"]),
        int(config["frozen_rgb"]["appearance_camera_id"]),
        model,
        reloaded,
        pipe,
    )
    if str(iterations) in milestone_results:
        evaluation = milestone_results[str(iterations)]["evaluation"]
    else:
        evaluation = _evaluate(
            args.output_dir,
            validation,
            args.labels_root,
            args.data_root,
            int(config["frozen_rgb"]["appearance_camera_id"]),
            model,
            reloaded,
            pipe,
        )
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed" if args.mode == "full" else "completed_preflight_not_a_quality_baseline",
        "mode": args.mode,
        "iterations": iterations,
        "seed": seed,
        "train_views": len(names),
        "validation_views": len(validation),
        "class_weights": {name: float(value) for name, value in zip(CLASS_NAMES, weights.cpu().numpy())},
        "loss": {
            "first": losses[0],
            "last": losses[-1],
            "mean": float(np.mean(losses)),
            "nll_first": nll_losses[0],
            "nll_last": nll_losses[-1],
            "l2_last": l2_losses[-1],
        },
        "training_seconds": training_seconds,
        "gpu_hours": training_seconds / 3600.0,
        "peak_cuda_allocated_mib": int(torch.cuda.max_memory_allocated() / 1024**2),
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size, "reloaded": True},
        "milestones": milestone_results,
        "rgb_isolation": isolation,
        "external_camera": external,
        "evaluation": evaluation,
    }
    if args.mode == "full":
        comparison = _paired_comparison(args.output_dir, evaluation, load_json(args.sem001_result), args.sem001_per_view)
        result["paired_comparison"] = comparison
        result["decision"] = _decision(config, comparison, isolation, external)
    return result


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    expected = {"equivalence": "SEM-040-EQ", "preflight": "SEM-041-PF", "full": "SEM-004"}
    if config.get("experiment_id") != expected[args.mode]:
        raise ValueError(f"{args.mode} mode requires experiment {expected[args.mode]}")
    split = load_json(REPO_ROOT / config["data"]["semantic_split"])
    manifest = load_json(REPO_ROOT / config["data"]["label_manifest"])
    if split.get("split_id") != config["data"]["split_id"] or manifest.get("split_id") != split.get("split_id"):
        raise ValueError("SEM-004 config, split and label manifest mismatch")
    if manifest.get("class_mapping") != {name: index for index, name in enumerate(CLASS_NAMES)}:
        raise ValueError("Official semantic mapping mismatch; class ID 2 must be stay_cable")
    if len(split.get("semantic_train_labeled", [])) != 219 or len(split.get("semantic_validation_labeled", [])) != 74:
        raise ValueError("SEM-004 requires the frozen 219/74 semantic split")
    if not args.labels_root.is_dir():
        raise FileNotFoundError("SEM-004 LabelMe-derived training labels are missing")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("SEM-004 output directory is not empty; use a unique retry directory")

    import torch

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not torch.cuda.is_available():
        raise RuntimeError("SEM-004 requires the host CUDA execution context")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_hashes = {
        "sem001_checkpoint": sha256(args.sem001_checkpoint),
        "sem004_diag_result": sha256(args.diag_result),
        "sem004_diag_output_manifest": sha256(args.diag_manifest),
        "direction_anchor_csv": sha256(args.direction_anchors),
        "sem001_result": sha256(args.sem001_result),
        "sem001_per_view": sha256(args.sem001_per_view),
    }
    write_json(
        args.output_dir / "invocation.json",
        {
            "argv": sys.argv,
            "experiment_id": config["experiment_id"],
            "mode": args.mode,
            "git": _git_evidence(),
            "environment": _environment(torch),
            "input_sha256": input_hashes,
            "uses_validation_for_hyperparameter_or_mask_selection": False,
            "uses_guard_band": False,
            "uses_unlabeled_rgb": False,
            "uses_hidden_test_rgb_or_labels": False,
            "global_visibility_gate_enabled": False,
            "dirichlet_regularizer_enabled": False,
        },
    )
    try:
        settings = ScaffoldSettings(
            train_camera_count=int(config["frozen_rgb"]["train_camera_count"]),
            n_offsets=int(config["frozen_rgb"]["n_offsets"]),
        )
        point_cloud = args.rgb_run_dir / "point_cloud" / f"iteration_{config['frozen_rgb']['iteration']}" / "point_cloud.ply"
        rgb_hashes = frozen_rgb_hashes(args.rgb_run_dir, int(config["frozen_rgb"]["iteration"]))
        configure_scaffold_import(args.source_root)
        model = load_frozen_scaffold(args.source_root, point_cloud, settings)
        payload = _sem001_payload(args.sem001_checkpoint, config, rgb_hashes)
        if int(payload["semantic_anchor_logits"].shape[0]) != int(model.get_anchor.shape[0]):
            raise ValueError("SEM-001 checkpoint and frozen Scaffold anchor counts differ")
        indices = _validate_diagnostic_inputs(args, config, int(model.get_anchor.shape[0]))
        pipe = pipeline()
        if args.mode == "equivalence":
            result = _run_equivalence(args, config, model, payload, indices, pipe)
        else:
            result = _run_training(
                args, config, split, manifest, model, payload, indices, rgb_hashes, input_hashes, pipe
            )
        result["git"] = _git_evidence()
        result["environment"] = _environment(torch)
        result["input_sha256"] = input_hashes
        write_json(args.output_dir / "result.json", result)
        manifest_sha = _write_output_manifest(args.output_dir)
        print(
            json.dumps(
                {
                    "experiment_id": result["experiment_id"],
                    "status": result["status"],
                    "decision": result.get("decision", {}).get("keep_or_drop"),
                    "output_manifest_sha256": manifest_sha,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        write_json(
            args.output_dir / "failure.json",
            {
                "experiment_id": config["experiment_id"],
                "mode": args.mode,
                "status": "blocked_runtime_or_provenance_gate",
                "exception_type": type(error).__name__,
                "exception": str(error),
                "uses_hidden_test_rgb_or_labels": False,
            },
        )
        _write_output_manifest(args.output_dir)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
