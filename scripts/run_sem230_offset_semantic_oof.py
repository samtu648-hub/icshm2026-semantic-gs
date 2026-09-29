#!/usr/bin/env python3
"""Run SEM-230's offset-granular Scaffold semantic OOF qualification.

This runner has an intentionally strict order:

1. verify frozen parents and prove zero-residual SEM-001 equivalence;
2. train three isolated spatial folds and materialize every OOF prediction;
3. seal all prediction and contribution hashes;
4. open calibration labels once; open check/developer labels only after their
   preceding registered gate passes.

No image pixels are an input to the semantic learner.  RGB, geometry, opacity,
appearance, densification and cameras remain frozen Scaffold-GS inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem001 as sem001
from src.scaffold_offset_semantic import (
    OffsetSemanticResidualSidecar,
    render_offset_binary_probabilities,
    render_offset_residual_semantic_probabilities,
)
from src.scaffold_semantic_sidecar import (
    AnchorSemanticSidecar,
    ScaffoldSettings,
    frozen_rgb_hashes,
    frozen_rgb_trainable_parameters,
    load_frozen_scaffold,
    pipeline,
    render_frozen_rgb,
    render_semantic_probabilities,
)
from src.semantic2d import CLASS_COUNT, CLASS_NAMES, fold_for_index, label_array, read_json, sha256, write_json
from src.semantic_protected_fusion import local_confusion, merge_confusions, metrics_from_confusion, partition_training_names
from src.semantic_render_domain import semantic_rgb
from src.semantic_rendered_training import select_rendered_training_views, validate_rendered_dataset_manifest


EXPERIMENT_ID = "SEM-230-OFFSET-SEMANTIC-OOF"
BACKGROUND, DECK, STAY_CABLE, TOWER, FOUNDATION = range(5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--oof-split", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sem001-checkpoint", type=Path, required=True)
    parser.add_argument("--sem001-training-root", type=Path, required=True)
    parser.add_argument("--sem150-root", type=Path, required=True)
    parser.add_argument("--sem160-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-sealed-oof-candidate",
        action="store_true",
        help="resume only the calibration of an already sealed, hash-verified OOF candidate",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def png_map(path: Path, *, context: str) -> np.ndarray:
    value = np.asarray(Image.open(path), dtype=np.uint8)
    require(value.shape == (989, 1320) and value.size > 0 and int(value.max()) < CLASS_COUNT, f"{context} is not an official five-class PNG: {path}")
    return value


def tensor_rgb(value: Any) -> np.ndarray:
    return np.rint(value.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


def png_sha256(array: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)
    return sha256(path)


def read_sem001_logits(path: Path, *, expected_sha256: str) -> Any:
    import torch

    require(path.is_file() and sha256(path) == expected_sha256, "SEM-230 SEM-001 checkpoint hash differs from SUBMIT-020")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(
        payload.get("experiment_id") == "SEM-001"
        and payload.get("semantic_storage") == "per_anchor_logits"
        and payload.get("class_names") == list(CLASS_NAMES),
        "SEM-230 SEM-001 checkpoint schema differs",
    )
    logits = payload.get("semantic_anchor_logits")
    require(isinstance(logits, torch.Tensor) and logits.ndim == 2 and logits.shape[1] == CLASS_COUNT, "SEM-230 missing SEM-001 anchor logits")
    return logits


def parent_maps(root: Path, *, manifest_name: str, expected_sha256: str, expected_experiment: str, names: list[str]) -> dict[str, Path]:
    manifest_path = root / manifest_name
    require(manifest_path.is_file() and sha256(manifest_path) == expected_sha256, f"SEM-230 {expected_experiment} manifest hash differs")
    manifest = read_json(manifest_path)
    require(manifest.get("experiment_id") == expected_experiment, f"SEM-230 {expected_experiment} manifest experiment ID differs")
    records = manifest.get("predictions", manifest.get("rendered"))
    require(isinstance(records, list), f"SEM-230 {expected_experiment} prediction records are missing")
    key = "prediction" if records and "prediction" in records[0] else "semantic_png"
    digest_key = "prediction_sha256" if key == "prediction" else "sha256"
    paths: dict[str, Path] = {}
    for record in records:
        name = record["name"]
        path = root / record[key]
        require(path.is_file() and sha256(path) == record[digest_key], f"SEM-230 {expected_experiment} prediction hash differs: {name}")
        paths[name] = path
    require(set(paths) == set(names), f"SEM-230 {expected_experiment} predictions do not cover 219 training views")
    return paths


def semantic230_eligibility(*, base: np.ndarray, sem001_map: np.ndarray, sem150_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mutable bkg/deck mask and immutable foundation-rejection mask.

    SEM-150 turned some original foundation predictions into background.  These
    pixels are still protected: SEM-230 cannot re-label them as deck merely
    because the final five-class map currently calls them background.
    """
    pre_foundation = sem150_map.copy()
    pre_foundation[(sem150_map == BACKGROUND) & (sem001_map == FOUNDATION)] = FOUNDATION
    foundation_rejection = (base == BACKGROUND) & (pre_foundation == FOUNDATION)
    protected = np.isin(base, [STAY_CABLE, TOWER, FOUNDATION]) | foundation_rejection
    mutable = ~protected
    require(np.all(np.isin(base[mutable], [BACKGROUND, DECK])), "SEM-230 mutable lane includes a protected class")
    return mutable, foundation_rejection


def fuse_binary_lane(base: np.ndarray, binary_deck: np.ndarray, mutable: np.ndarray) -> np.ndarray:
    require(binary_deck.shape == base.shape and binary_deck.dtype == np.uint8 and np.isin(binary_deck, [0, 1]).all(), "SEM-230 binary prediction is invalid")
    candidate = base.copy()
    candidate[mutable] = binary_deck[mutable]
    require(np.array_equal(candidate[~mutable], base[~mutable]), "SEM-230 changed a protected pixel")
    return candidate


def two_class_weights(labels_root: Path, names: list[str]) -> tuple[Any, Any]:
    counts = np.zeros(2, dtype=np.int64)
    for name in names:
        target = label_array(labels_root / name)
        counts += np.bincount(target[target <= DECK], minlength=2)
    require(np.all(counts > 0), f"SEM-230 fold lacks background/deck pixels: {counts.tolist()}")
    weights = np.median(counts.astype(np.float64)) / counts
    return np.clip(weights, 0.05, 20.0).astype(np.float32), counts


def save_checkpoint(path: Path, sidecar: OffsetSemanticResidualSidecar, metadata: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "metadata": metadata, "sidecar": sidecar.state_dict()}, path)


def reload_checkpoint(path: Path, *, anchor_count: int, n_offsets: int, metadata: dict[str, Any], device: Any) -> OffsetSemanticResidualSidecar:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload.get("metadata") == metadata, "SEM-230 checkpoint metadata changed on reload")
    sidecar = OffsetSemanticResidualSidecar(torch.zeros((anchor_count, CLASS_COUNT), dtype=torch.float32), n_offsets=n_offsets, device=device)
    sidecar.load_state_dict(payload["sidecar"])
    return sidecar


def zero_residual_equivalence(
    *, args: argparse.Namespace, config: dict[str, Any], model: Any, pipe: Any, old_logits: Any, rgb_before: dict[str, str]
) -> dict[str, Any]:
    import torch

    old = AnchorSemanticSidecar(int(model.get_anchor.shape[0]))
    old.load_state_dict({"semantic_anchor_logits": old_logits})
    residual = OffsetSemanticResidualSidecar(old_logits, n_offsets=int(config["frozen_rgb"]["n_offsets"]), device="cuda")
    views = list(config["equivalence"]["views"])
    semantic_max_difference = 0.0
    semantic_equal = True
    RGB_after: dict[str, str] = {}
    for name in views:
        _, _, view = sem001.official_view(args.data_root, name, int(config["frozen_rgb"]["appearance_camera_id"]))
        with torch.no_grad():
            previous = render_semantic_probabilities(view, model, old, pipe)
            current = render_offset_residual_semantic_probabilities(view, model, residual, pipe)
            semantic_max_difference = max(semantic_max_difference, float((previous - current).abs().max().cpu()))
            semantic_equal = semantic_equal and bool(torch.equal(previous, current))
            RGB_after[name] = png_sha256(tensor_rgb(render_frozen_rgb(view, model, pipe)), args.output_dir / "zero_residual_equivalence" / "rgb" / name)
    rgb_after_artifacts = frozen_rgb_hashes(args.rgb_run_dir, int(config["frozen_rgb"]["iteration"]))
    evidence = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "semantic_reference": "SEM-001 per_anchor_logits checkpoint from frozen SUBMIT-020",
        "residual_initialisation": "all_zero",
        "views": views,
        "semantic_tensor_bit_identical": semantic_equal,
        "semantic_max_absolute_difference": semantic_max_difference,
        "rgb_png_sha256_before": rgb_before,
        "rgb_png_sha256_after": RGB_after,
        "rgb_png_hashes_equal": rgb_before == RGB_after,
        "frozen_rgb_hashes_after": rgb_after_artifacts,
        "frozen_model_trainable_parameters": frozen_rgb_trainable_parameters(model),
    }
    require(semantic_equal and semantic_max_difference == 0.0, "SEM-230 zero-residual semantics do not reproduce SEM-001")
    require(evidence["rgb_png_hashes_equal"] and not evidence["frozen_model_trainable_parameters"], "SEM-230 zero-residual RGB isolation failed")
    # This is label-free and uses exactly the established official adapter.  It
    # verifies that the representation change has not changed the required
    # COLMAP CAMERA_FROM_WORLD + SIMPLE_RADIAL external interface.
    external = sem001.render_external(
        args.output_dir / "zero_residual_equivalence",
        args.data_root,
        str(config["external_camera"]["image_name"]),
        int(config["frozen_rgb"]["appearance_camera_id"]),
        model,
        old,
        pipe,
    )
    require(
        external["input_pose_convention"] == "CAMERA_FROM_WORLD"
        and external["input_camera_model"] == "SIMPLE_RADIAL"
        and external["semantic_logits_shape"] == [5, 989, 1320]
        and external["semantic_png_shape"] == [989, 1320],
        "SEM-230 external COLMAP compatibility test failed",
    )
    evidence["external_colmap_compatibility"] = external
    write_json(args.output_dir / "zero_residual_equivalence.json", evidence)
    return evidence


def train_fold(
    *, fold_index: int, args: argparse.Namespace, config: dict[str, Any], split: dict[str, Any], model: Any, pipe: Any, rgb_hashes: dict[str, str]
) -> dict[str, Any]:
    import torch

    fold = fold_for_index(split, fold_index)
    train_names, oof_names = list(fold["train"]), list(fold["oof"])
    require(not (set(train_names) & set(oof_names)) and not (set(train_names) & set(fold["guard"])), "SEM-230 fold leakage")
    n_offsets = int(config["frozen_rgb"]["n_offsets"])
    device = torch.device("cuda:0")
    seed = int(config["training"]["seed"])
    random.seed(seed + fold_index * 100003)
    np.random.seed(seed + fold_index * 100003)
    torch.manual_seed(seed + fold_index * 100003)
    torch.cuda.manual_seed_all(seed + fold_index * 100003)
    sidecar = OffsetSemanticResidualSidecar(torch.zeros((int(model.get_anchor.shape[0]), CLASS_COUNT), dtype=torch.float32), n_offsets=n_offsets, device=device)
    optimizer = torch.optim.AdamW([sidecar.offset_background_deck_residual], lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    weights_np, counts = two_class_weights(args.labels_root, train_names)
    weights = torch.as_tensor(weights_np, device=device)
    rng = random.Random(seed + fold_index * 100003)
    torch.cuda.reset_peak_memory_stats(device)
    losses: list[float] = []
    started = time.perf_counter()
    for iteration in range(1, int(config["training"]["iterations_per_fold"]) + 1):
        name = train_names[rng.randrange(len(train_names))]
        target = torch.from_numpy(label_array(args.labels_root / name).astype(np.int64, copy=False)).to(device=device, non_blocking=True)
        _, _, view = sem001.official_view(args.data_root, name, int(config["frozen_rgb"]["appearance_camera_id"]))
        probability = render_offset_binary_probabilities(view, model, sidecar, pipe)
        mutable_target = target <= DECK
        log_probability = torch.log(probability.clamp_min(1e-8))
        pixel_log_probability = log_probability.permute(1, 2, 0)[mutable_target, target[mutable_target]]
        pixel_weight = weights[target[mutable_target]]
        loss = -(pixel_log_probability * pixel_weight).sum() / pixel_weight.sum().clamp_min(1e-8)
        require(bool(torch.isfinite(loss)), f"SEM-230 non-finite loss in fold {fold_index} iteration {iteration}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if iteration == 1 or iteration % int(config["training"]["log_every"]) == 0 or iteration == int(config["training"]["iterations_per_fold"]):
            print(json.dumps({"phase": "train", "fold": fold_index, "iteration": iteration, "loss": losses[-1], "peak_mib": int(torch.cuda.max_memory_allocated(device) / 1024**2)}), flush=True)
    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "fold": fold_index,
        "seed": seed,
        "n_offsets": n_offsets,
        "iterations": int(config["training"]["iterations_per_fold"]),
        "oof_split_sha256": sha256(args.oof_split),
        "rgb_hashes": rgb_hashes,
        "semantic_storage": "per_generated_offset_background_deck_residual",
    }
    root = args.output_dir / f"fold-{fold_index}"
    checkpoint = root / "checkpoints" / "offset_semantic_residual.pth"
    save_checkpoint(checkpoint, sidecar, metadata)
    reference_residual = sidecar.offset_background_deck_residual.detach().cpu().clone()
    del sidecar, optimizer
    torch.cuda.empty_cache()
    sidecar = reload_checkpoint(checkpoint, anchor_count=int(model.get_anchor.shape[0]), n_offsets=n_offsets, metadata=metadata, device=device)
    require(torch.equal(reference_residual, sidecar.offset_background_deck_residual.detach().cpu()), f"SEM-230 fold {fold_index} residual reload differs")
    predictions: list[dict[str, Any]] = []
    contributions: list[dict[str, Any]] = []
    inference_started = time.perf_counter()
    for index, name in enumerate(oof_names, start=1):
        _, _, view = sem001.official_view(args.data_root, name, int(config["frozen_rgb"]["appearance_camera_id"]))
        with torch.no_grad():
            probability, contribution = render_offset_binary_probabilities(view, model, sidecar, pipe, return_contributions=True)
        value = probability[DECK].detach().cpu().numpy().astype(np.float32)
        binary = (probability[DECK] >= probability[BACKGROUND]).detach().cpu().numpy().astype(np.uint8)
        probability_path = root / "oof_deck_probabilities" / f"{Path(name).stem}.npz"
        prediction_path = root / "oof_binary_predictions" / name
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(probability_path, deck_probability=value.astype(np.float16))
        Image.fromarray(binary, mode="L").save(prediction_path)
        predictions.append({"name": name, "fold": fold_index, "probability": str(probability_path.relative_to(args.output_dir)), "probability_sha256": sha256(probability_path), "binary_prediction": str(prediction_path.relative_to(args.output_dir)), "binary_prediction_sha256": sha256(prediction_path), "decision": "background_deck_argmax_no_threshold"})
        contributions.append({"name": name, "fold": fold_index, **contribution})
        if index == 1 or index % 20 == 0 or index == len(oof_names):
            print(json.dumps({"phase": "oof_inference", "fold": fold_index, "completed": index, "total": len(oof_names)}), flush=True)
    torch.cuda.synchronize(device)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "fold": fold_index,
        "status": "completed_fold_oof",
        "training_views": len(train_names),
        "oof_views": len(oof_names),
        "guard_views": len(fold["guard"]),
        "binary_pixel_counts_train": counts.astype(int).tolist(),
        "binary_class_weights": weights_np.astype(float).tolist(),
        "loss": {"first": losses[0], "last": losses[-1], "mean_last_100": float(np.mean(losses[-100:]))},
        "training_seconds": training_seconds,
        "inference_seconds": float(time.perf_counter() - inference_started),
        "gpu_hours_training": training_seconds / 3600.0,
        "peak_cuda_allocated_mib": int(torch.cuda.max_memory_allocated(device) / 1024**2),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_reload": "bit_identical_residual_passed",
        "trainable_parameter_allowlist": ["offset_background_deck_residual"],
        "fixed_validation_opened": False,
        "real_rgb_opened": False,
        "oof_target_labels_used_for_this_fold_training": False,
        "predictions": predictions,
        "contributions": contributions,
    }
    write_json(root / "result.json", result)
    return result


def materialize_candidate(
    *, args: argparse.Namespace, names: list[str], sem001_paths: dict[str, Path], sem150_paths: dict[str, Path], sem160_paths: dict[str, Path], fold_results: list[dict[str, Any]]
) -> dict[str, Any]:
    output = args.output_dir / "oof_candidate"
    output.mkdir(parents=True, exist_ok=True)
    binary_paths = {record["name"]: args.output_dir / record["binary_prediction"] for fold in fold_results for record in fold["predictions"]}
    require(set(binary_paths) == set(names), "SEM-230 OOF binary predictions incomplete")
    records: list[dict[str, Any]] = []
    for name in names:
        base = png_map(sem160_paths[name], context="SEM-230 SEM-160 baseline")
        sem001_map = png_map(sem001_paths[name], context="SEM-230 SEM-001 provenance")
        sem150_map = png_map(sem150_paths[name], context="SEM-230 SEM-150 provenance")
        mutable, rejection = semantic230_eligibility(base=base, sem001_map=sem001_map, sem150_map=sem150_map)
        binary = png_map(binary_paths[name], context="SEM-230 binary prediction")
        candidate = fuse_binary_lane(base, binary, mutable)
        path = output / name
        Image.fromarray(candidate, mode="L").save(path)
        changed = candidate != base
        records.append({"name": name, "prediction": str(path.relative_to(args.output_dir)), "prediction_sha256": sha256(path), "mutable_background_deck_pixels": int(mutable.sum()), "foundation_rejection_pixels_protected": int(rejection.sum()), "background_to_deck": int(((base == BACKGROUND) & (candidate == DECK)).sum()), "deck_to_background": int(((base == DECK) & (candidate == BACKGROUND)).sum()), "protected_predictions_exact": bool(np.array_equal(candidate[~mutable], base[~mutable]))})
    manifest = {"schema_version": 1, "experiment_id": EXPERIMENT_ID, "status": "sealed_oof_candidate_before_calibration", "target_labels_opened": False, "fixed_validation_opened": False, "all_protected_predictions_exact": all(r["protected_predictions_exact"] for r in records), "predictions": records}
    write_json(args.output_dir / "prediction_manifest.json", manifest)
    return manifest


def evaluate_names(*, names: list[str], labels_root: Path, baseline_paths: dict[str, Path], candidate_paths: dict[str, Path], sem001_paths: dict[str, Path], sem150_paths: dict[str, Path]) -> dict[str, Any]:
    baseline_confusions: list[np.ndarray] = []
    candidate_confusions: list[np.ndarray] = []
    protected_exact = True
    foundation_rejection_exact = True
    per_view: list[dict[str, Any]] = []
    changes_by_target = np.zeros(CLASS_COUNT, dtype=np.int64)
    for name in names:
        target = label_array(labels_root / name)
        baseline = png_map(baseline_paths[name], context="SEM-230 baseline evaluation")
        candidate = png_map(candidate_paths[name], context="SEM-230 candidate evaluation")
        _mutable, rejection = semantic230_eligibility(base=baseline, sem001_map=png_map(sem001_paths[name], context="SEM-230 SEM-001 evaluation"), sem150_map=png_map(sem150_paths[name], context="SEM-230 SEM-150 evaluation"))
        protected_exact = protected_exact and bool(np.array_equal(candidate[np.isin(baseline, [STAY_CABLE, TOWER, FOUNDATION])], baseline[np.isin(baseline, [STAY_CABLE, TOWER, FOUNDATION])]))
        foundation_rejection_exact = foundation_rejection_exact and bool(np.array_equal(candidate[rejection], baseline[rejection]))
        changed = candidate != baseline
        changes_by_target += np.bincount(target[changed], minlength=CLASS_COUNT)
        base_local, candidate_local = local_confusion(target, baseline), local_confusion(target, candidate)
        baseline_confusions.append(base_local)
        candidate_confusions.append(candidate_local)
        # A single official image need not contain all five classes.  The
        # five-class metric contract applies to the aggregate below, while a
        # per-view record is intentionally limited to values that remain
        # meaningful for a sparse local confusion.
        per_view.append(
            {
                "name": name,
                "baseline_confusion": base_local.astype(int).tolist(),
                "candidate_confusion": candidate_local.astype(int).tolist(),
                "changed_pixels": int(changed.sum()),
            }
        )
    return {"view_count": len(names), "baseline_metrics": metrics_from_confusion(merge_confusions(baseline_confusions)), "candidate_metrics": metrics_from_confusion(merge_confusions(candidate_confusions)), "protected_predictions_exact": protected_exact, "foundation_rejection_exact": foundation_rejection_exact, "changed_pixels_by_ground_truth": {CLASS_NAMES[index]: int(value) for index, value in enumerate(changes_by_target)}, "per_view": per_view}


def gate(evaluation: dict[str, Any], gates: dict[str, Any], *, rgb_unchanged: bool) -> dict[str, Any]:
    baseline, candidate = evaluation["baseline_metrics"], evaluation["candidate_metrics"]
    base_deck, candidate_deck = baseline["per_class"]["deck"], candidate["per_class"]["deck"]
    checks = {
        "deck_iou_gain_at_least_0_2pp": float(candidate_deck["iou"]) >= float(base_deck["iou"]) + float(gates["minimum_deck_iou_gain"]),
        "deck_recall_gain_at_least_0_5pp": float(candidate_deck["recall"]) >= float(base_deck["recall"]) + float(gates["minimum_deck_recall_gain"]),
        "deck_precision_at_least_94pct": float(candidate_deck["precision"]) >= float(gates["minimum_deck_precision"]),
        "miou_non_decreasing": float(candidate["mIoU_with_background"]) >= float(baseline["mIoU_with_background"]),
        "stay_cable_tower_foundation_exact": bool(evaluation["protected_predictions_exact"]),
        "foundation_rejection_exact": bool(evaluation["foundation_rejection_exact"]),
        "rgb_exact": bool(rgb_unchanged),
    }
    return {"thresholds": gates, "checks": checks, "passed": all(checks.values()), "delta_mIoU_pp": 100.0 * (float(candidate["mIoU_with_background"]) - float(baseline["mIoU_with_background"])), "delta_deck_iou_pp": 100.0 * (float(candidate_deck["iou"]) - float(base_deck["iou"])), "delta_deck_recall_pp": 100.0 * (float(candidate_deck["recall"]) - float(base_deck["recall"])), "delta_deck_precision_pp": 100.0 * (float(candidate_deck["precision"]) - float(base_deck["precision"]))}


def write_panels(*, names: list[str], labels_root: Path, baseline_paths: dict[str, Path], candidate_paths: dict[str, Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = label_array(labels_root / name)
        baseline, candidate = png_map(baseline_paths[name], context="SEM-230 panel baseline"), png_map(candidate_paths[name], context="SEM-230 panel candidate")
        change = np.zeros((*target.shape, 3), dtype=np.uint8)
        changed = candidate != baseline
        change[changed & (candidate == target)] = (48, 220, 96)
        change[changed & (candidate != target)] = (255, 48, 48)
        panel = np.concatenate((semantic_rgb(target), semantic_rgb(baseline), semantic_rgb(candidate), change), axis=1)
        Image.fromarray(panel, mode="RGB").save(output_dir / name)
        deck = (target == DECK) | (baseline == DECK) | (candidate == DECK)
        ys, xs = np.where(deck)
        if len(xs):
            pad, y0, y1, x0, x1 = 32, max(0, int(ys.min()) - 32), min(target.shape[0], int(ys.max()) + 33), max(0, int(xs.min()) - 32), min(target.shape[1], int(xs.max()) + 33)
            _ = pad
            crop = panel[y0:y1, x0:x1]
            Image.fromarray(crop, mode="RGB").save(output_dir / f"{Path(name).stem}_deck_zoom.png")


def resume_sealed_oof_calibration(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Evaluate the exact saved candidate after a reporting-only interruption.

    This is deliberately narrower than the normal runner: it refuses to train,
    write predictions, or touch the check/developer partitions.  It exists for
    recovery from a failure after ``prediction_manifest.json`` was sealed and
    before the aggregate calibration record could be written.
    """
    require(args.output_dir.is_dir(), "SEM-230 resume requires the existing OOF output")
    require(not (args.output_dir / "calibration_result.json").exists(), "SEM-230 calibration is already sealed")
    require(not (args.output_dir / "result.json").exists(), "SEM-230 result is already sealed")
    require(sha256(args.split) == config["data"]["semantic_split_sha256"], "SEM-230 split hash differs")
    require(sha256(args.oof_split) == config["data"]["oof_split_sha256"], "SEM-230 OOF split hash differs")
    split = read_json(args.split)
    names = select_rendered_training_views(split, expected_count=219)
    partition = partition_training_names(names, expected_count=219)
    require(args.labels_root.is_dir(), "SEM-230 calibration labels are missing")
    require(sha256(args.sem001_checkpoint) == config["inputs"]["sem001_checkpoint_sha256"], "SEM-230 SEM-001 checkpoint hash differs")
    sem001_paths = parent_maps(
        args.sem001_training_root,
        manifest_name="training_sem001_render_manifest.json",
        expected_sha256=config["inputs"]["sem001_training_manifest_sha256"],
        expected_experiment="SEM-110-PROTECTED-FUSION-OOF",
        names=names,
    )
    sem150_paths = parent_maps(
        args.sem150_root,
        manifest_name="prediction_manifest.json",
        expected_sha256=config["inputs"]["sem150_prediction_manifest_sha256"],
        expected_experiment="SEM-150-FOUNDATION-GATE-OOF",
        names=names,
    )
    sem160_paths = parent_maps(
        args.sem160_root,
        manifest_name="prediction_manifest.json",
        expected_sha256=config["inputs"]["sem160_prediction_manifest_sha256"],
        expected_experiment="SEM-160-DECK-BINARY-OOF",
        names=names,
    )
    candidate_manifest_path = args.output_dir / "prediction_manifest.json"
    candidate_manifest = read_json(candidate_manifest_path)
    require(
        candidate_manifest.get("experiment_id") == EXPERIMENT_ID
        and candidate_manifest.get("status") == "sealed_oof_candidate_before_calibration"
        and candidate_manifest.get("target_labels_opened") is False
        and candidate_manifest.get("fixed_validation_opened") is False,
        "SEM-230 candidate was not sealed before calibration",
    )
    candidate_paths: dict[str, Path] = {}
    for record in candidate_manifest.get("predictions", []):
        name, relative = record.get("name"), record.get("prediction")
        path = args.output_dir / str(relative)
        require(
            isinstance(name, str)
            and path.is_file()
            and sha256(path) == record.get("prediction_sha256"),
            "SEM-230 sealed candidate prediction hash differs",
        )
        candidate_paths[name] = path
    require(set(candidate_paths) == set(names), "SEM-230 sealed candidate does not cover all 219 OOF views")
    fold_results: list[dict[str, Any]] = []
    for fold_index in range(3):
        fold_path = args.output_dir / f"fold-{fold_index}" / "result.json"
        fold = read_json(fold_path)
        require(
            fold.get("experiment_id") == EXPERIMENT_ID
            and fold.get("fold") == fold_index
            and fold.get("status") == "completed_fold_oof"
            and fold.get("checkpoint_reload") == "bit_identical_residual_passed",
            f"SEM-230 fold {fold_index} is not a completed reload-verified OOF fold",
        )
        fold_results.append(fold)
    contributions = read_json(args.output_dir / "contribution_manifest.json")
    require(
        contributions.get("experiment_id") == EXPERIMENT_ID
        and contributions.get("status") == "completed_oof_offset_contribution_weights"
        and len(contributions.get("records", [])) == len(names),
        "SEM-230 contribution manifest is incomplete",
    )
    equivalence = read_json(args.output_dir / "zero_residual_equivalence.json")
    require(
        equivalence.get("semantic_tensor_bit_identical") is True
        and float(equivalence.get("semantic_max_absolute_difference", -1.0)) == 0.0
        and equivalence.get("rgb_png_hashes_equal") is True
        and equivalence.get("frozen_model_trainable_parameters") == [],
        "SEM-230 zero-residual/RGB equivalence is not intact",
    )
    current_rgb_hashes = frozen_rgb_hashes(args.rgb_run_dir, int(config["frozen_rgb"]["iteration"]))
    rgb_unchanged = current_rgb_hashes == config["frozen_rgb"]["artifact_sha256"]
    calibration = evaluate_names(
        names=partition["calibration"],
        labels_root=args.labels_root,
        baseline_paths=sem160_paths,
        candidate_paths=candidate_paths,
        sem001_paths=sem001_paths,
        sem150_paths=sem150_paths,
    )
    calibration_gate = gate(calibration, config["gates"], rgb_unchanged=rgb_unchanged)
    calibration_record = {
        "experiment_id": EXPERIMENT_ID,
        "status": "sealed_calibration_result",
        "prediction_manifest_sha256": sha256(candidate_manifest_path),
        "view_names": partition["calibration"],
        "evaluation": calibration,
        "gate": calibration_gate,
        "execution_note": "The first execution stopped before an aggregate metric or calibration result was produced because a per-view report incorrectly required every image to contain all five classes. This reporting-only recovery evaluates the unchanged, hash-verified sealed candidate; no predictions, model, gate, threshold, or check/developer labels were changed or used.",
    }
    write_json(args.output_dir / "calibration_result.json", calibration_record)
    panel_names = [name for name in config["outputs"]["calibration_visual_samples"] if name in partition["calibration"]]
    write_panels(
        names=panel_names,
        labels_root=args.labels_root,
        baseline_paths=sem160_paths,
        candidate_paths=candidate_paths,
        output_dir=args.output_dir / "calibration_panels",
    )
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "zero_residual_equivalence_sha256": sha256(args.output_dir / "zero_residual_equivalence.json"),
        "contribution_manifest_sha256": sha256(args.output_dir / "contribution_manifest.json"),
        "prediction_manifest_sha256": sha256(candidate_manifest_path),
        "fold_results": fold_results,
        "calibration": calibration,
        "calibration_gate": calibration_gate,
        "check_labels_opened": False,
        "developer_labels_opened": False,
        "fixed_validation_opened": False,
        "rgb_artifacts_unchanged": rgb_unchanged,
        "rgb_hashes": current_rgb_hashes,
        "candidate_materialized_before_calibration": True,
        "final_training_started": False,
        "execution_note": calibration_record["execution_note"],
    }
    if not calibration_gate["passed"]:
        result.update(
            {
                "status": "completed_calibration_gate_failed",
                "qualified": False,
                "next_step": "do_not_open_check_or_developer_or_train_final; retain_RGB030_FULL_plus_SEM161",
            }
        )
        write_json(args.output_dir / "result.json", result)
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": result["status"], "calibration_gate": calibration_gate}, sort_keys=True))
        return 0
    result.update(
        {
            "status": "qualified_oof_final_training_not_implemented",
            "qualified": True,
            "next_step": "separate_final_run_required_after_OOF_gate",
        }
    )
    write_json(args.output_dir / "result.json", result)
    return 0


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    require(config.get("experiment_id") == EXPERIMENT_ID and config.get("status") == "registered_not_run", "SEM-230 requires its untouched registered configuration")
    if args.resume_sealed_oof_candidate:
        return resume_sealed_oof_calibration(args, config)
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), "SEM-230 refuses to overwrite an output directory")
    require(sha256(args.split) == config["data"]["semantic_split_sha256"] and sha256(args.oof_split) == config["data"]["oof_split_sha256"], "SEM-230 split hash differs")
    split, oof_split = read_json(args.split), read_json(args.oof_split)
    names = select_rendered_training_views(split, expected_count=219)
    require(set(names).isdisjoint(split["semantic_validation_labeled"]) and len(split["semantic_validation_labeled"]) == 74, "SEM-230 training/developer isolation failed")
    require(read_json(args.oof_split)["global_checks"]["every_parent_train_view_exactly_once_oof"] is True, "SEM-230 OOF coverage invalid")
    require(args.labels_root.is_dir() and args.data_root.is_dir() and args.source_root.is_dir(), "SEM-230 required external input is missing")
    import torch

    require(torch.cuda.is_available(), "SEM-230 requires a visible CUDA GPU")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_hashes = frozen_rgb_hashes(args.rgb_run_dir, int(config["frozen_rgb"]["iteration"]))
    require(rgb_hashes == config["frozen_rgb"]["artifact_sha256"], "SEM-230 frozen RGB hash differs from SUBMIT-020")
    old_logits = read_sem001_logits(args.sem001_checkpoint, expected_sha256=config["inputs"]["sem001_checkpoint_sha256"])
    sem001_paths = parent_maps(args.sem001_training_root, manifest_name="training_sem001_render_manifest.json", expected_sha256=config["inputs"]["sem001_training_manifest_sha256"], expected_experiment="SEM-110-PROTECTED-FUSION-OOF", names=names)
    sem150_paths = parent_maps(args.sem150_root, manifest_name="prediction_manifest.json", expected_sha256=config["inputs"]["sem150_prediction_manifest_sha256"], expected_experiment="SEM-150-FOUNDATION-GATE-OOF", names=names)
    sem160_paths = parent_maps(args.sem160_root, manifest_name="prediction_manifest.json", expected_sha256=config["inputs"]["sem160_prediction_manifest_sha256"], expected_experiment="SEM-160-DECK-BINARY-OOF", names=names)
    rendered_manifest = args.sem001_training_root / "training_sem001_render_manifest.json"
    require(rendered_manifest.is_file(), "SEM-230 missing SEM-001 training render manifest")
    settings = ScaffoldSettings(train_camera_count=int(config["frozen_rgb"]["train_camera_count"]), n_offsets=int(config["frozen_rgb"]["n_offsets"]))
    model = load_frozen_scaffold(args.source_root, args.rgb_run_dir / "point_cloud" / f"iteration_{config['frozen_rgb']['iteration']}" / "point_cloud.ply", settings)
    require(int(model.get_anchor.shape[0]) == int(old_logits.shape[0]), "SEM-230 SEM-001 checkpoint anchor count differs from RGB")
    pipe = pipeline()
    rgb_before: dict[str, str] = {}
    for name in config["equivalence"]["views"]:
        _, _, view = sem001.official_view(args.data_root, name, int(config["frozen_rgb"]["appearance_camera_id"]))
        rgb_before[name] = png_sha256(tensor_rgb(render_frozen_rgb(view, model, pipe)), args.output_dir / "rgb_isolation" / "before" / name)
    equivalence = zero_residual_equivalence(args=args, config=config, model=model, pipe=pipe, old_logits=old_logits, rgb_before=rgb_before)
    fold_results = [train_fold(fold_index=index, args=args, config=config, split=oof_split, model=model, pipe=pipe, rgb_hashes=rgb_hashes) for index in range(3)]
    rgb_after = frozen_rgb_hashes(args.rgb_run_dir, int(config["frozen_rgb"]["iteration"]))
    rgb_unchanged = rgb_after == rgb_hashes and not frozen_rgb_trainable_parameters(model)
    contributions = [record for fold in fold_results for record in fold["contributions"]]
    write_json(args.output_dir / "contribution_manifest.json", {"experiment_id": EXPERIMENT_ID, "status": "completed_oof_offset_contribution_weights", "records": contributions})
    candidate_manifest = materialize_candidate(args=args, names=names, sem001_paths=sem001_paths, sem150_paths=sem150_paths, sem160_paths=sem160_paths, fold_results=fold_results)
    candidate_paths = {record["name"]: args.output_dir / record["prediction"] for record in candidate_manifest["predictions"]}
    partition = partition_training_names(names, expected_count=219)
    calibration = evaluate_names(names=partition["calibration"], labels_root=args.labels_root, baseline_paths=sem160_paths, candidate_paths=candidate_paths, sem001_paths=sem001_paths, sem150_paths=sem150_paths)
    calibration_gate = gate(calibration, config["gates"], rgb_unchanged=rgb_unchanged)
    calibration_record = {"experiment_id": EXPERIMENT_ID, "status": "sealed_calibration_result", "prediction_manifest_sha256": sha256(args.output_dir / "prediction_manifest.json"), "view_names": partition["calibration"], "evaluation": calibration, "gate": calibration_gate}
    write_json(args.output_dir / "calibration_result.json", calibration_record)
    panel_names = [name for name in config["outputs"]["calibration_visual_samples"] if name in partition["calibration"]]
    write_panels(names=panel_names, labels_root=args.labels_root, baseline_paths=sem160_paths, candidate_paths=candidate_paths, output_dir=args.output_dir / "calibration_panels")
    result = {"schema_version": 1, "experiment_id": EXPERIMENT_ID, "zero_residual_equivalence_sha256": sha256(args.output_dir / "zero_residual_equivalence.json"), "contribution_manifest_sha256": sha256(args.output_dir / "contribution_manifest.json"), "prediction_manifest_sha256": sha256(args.output_dir / "prediction_manifest.json"), "fold_results": fold_results, "calibration": calibration, "calibration_gate": calibration_gate, "check_labels_opened": False, "developer_labels_opened": False, "fixed_validation_opened": False, "rgb_artifacts_unchanged": rgb_unchanged, "rgb_hashes": rgb_hashes, "candidate_materialized_before_calibration": True, "final_training_started": False}
    if not calibration_gate["passed"]:
        result.update({"status": "completed_calibration_gate_failed", "qualified": False, "next_step": "do_not_open_check_or_developer_or_train_final; retain_RGB030_FULL_plus_SEM161"})
        write_json(args.output_dir / "result.json", result)
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": result["status"], "calibration_gate": calibration_gate}, sort_keys=True))
        return 0
    # The path below is intentionally unreachable until the calibration record
    # is sealed and passed.  It is kept explicit so a future qualifying run has
    # no opportunity to read check/developer labels early.
    check = evaluate_names(names=partition["check"], labels_root=args.labels_root, baseline_paths=sem160_paths, candidate_paths=candidate_paths, sem001_paths=sem001_paths, sem150_paths=sem150_paths)
    check_gate = gate(check, config["gates"], rgb_unchanged=rgb_unchanged)
    result.update({"check_labels_opened": True, "check": check, "check_gate": check_gate})
    if not check_gate["passed"]:
        result.update({"status": "completed_check_gate_failed", "qualified": False, "next_step": "do_not_open_developer_or_train_final; retain_RGB030_FULL_plus_SEM161"})
        write_json(args.output_dir / "result.json", result)
        return 0
    result.update({"status": "qualified_oof_final_training_not_implemented", "qualified": True, "next_step": "separate_final_run_required_after_OOF_gate"})
    write_json(args.output_dir / "result.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
