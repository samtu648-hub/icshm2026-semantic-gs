#!/usr/bin/env python3
"""Run the frozen SEM394 dual-seed 0.5/0.5 probability ensemble."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.quality_protocol import CLASSES, sha256, verify_hashes, write_new_json


EID = "SEM-394-RGB150-DEV270-SEED-ENSEMBLE"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
PARENT_KEYS = ("SEM-384-RGB150-DEV270", "SEM-392-RGB150-DEV270-SEED20260919-MEMORY")
PARENT_TAGS = {
    "SEM-384-RGB150-DEV270": "sem384",
    "SEM-392-RGB150-DEV270-SEED20260919-MEMORY": "sem392",
}
MASK2FORMER_SHA256 = "debd14ba97e6e8b89db5e281ccc023b53159718a284b332fcce21813345a7d6e"
PROBABILITY_SUM_ATOL = 2e-5
CODE_PATHS = (
    "scripts/run_sem394_seed_ensemble.py",
    "scripts/run_sem384_dev270.py",
    "src/quality_mask2former.py",
    "src/quality_semantic_audit.py",
    "src/quality_protocol.py",
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def code_hashes() -> dict[str, str]:
    return {path: sha256(ROOT / path) for path in CODE_PATHS}


def ensemble_probability(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 3 or x.shape[0] != 5:
        raise ValueError("Expected two matching five-class probability stacks")
    if x.dtype != np.float64 or y.dtype != np.float64:
        raise ValueError("Ensemble arithmetic must be float64")
    return 0.5 * x + 0.5 * y


def smallest_class_argmax(probability: np.ndarray) -> np.ndarray:
    return np.argmax(probability, axis=0).astype(np.uint8)


def tie_pixel_count(probability: np.ndarray) -> int:
    peak = probability.max(axis=0)
    return int(np.count_nonzero((probability == peak[None]).sum(axis=0) > 1))


def boundary_counts(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion, distance_transform_edt

    cable = target == 2
    gt_boundary = cable ^ binary_erosion(cable)
    predicted_cable = prediction == 2
    predicted_boundary = predicted_cable ^ binary_erosion(predicted_cable)
    gt_distance = np.asarray(distance_transform_edt(~gt_boundary))
    predicted_distance = np.asarray(distance_transform_edt(~predicted_boundary))
    return np.asarray(
        [
            np.sum(predicted_boundary & (gt_distance <= 2)),
            np.sum(predicted_boundary),
            np.sum(gt_boundary & (predicted_distance <= 2)),
            np.sum(gt_boundary),
        ],
        dtype=np.int64,
    )


def boundary_metrics(values: np.ndarray) -> dict[str, Any]:
    precision = float(values[0] / values[1]) if values[1] else None
    recall = float(values[2] / values[3]) if values[3] else None
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "tolerance_px": 2,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "counts": values.tolist(),
    }


def disagreement_diagnostics(
    gt: np.ndarray, a: np.ndarray, b: np.ndarray
) -> dict[str, int]:
    total = int(gt.size)
    disagree = a != b
    only_a = (a == gt) & (b != gt)
    only_b = (b == gt) & (a != gt)
    both_wrong_disagree = disagree & (a != gt) & (b != gt)
    return {
        "total_pixels": total,
        "disagreement_pixels": int(disagree.sum()),
        "only_a_correct_pixels": int(only_a.sum()),
        "only_b_correct_pixels": int(only_b.sum()),
        "both_wrong_disagreement_pixels": int(both_wrong_disagree.sum()),
    }


def hard_oracle_prediction(gt: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a == gt, a, b).astype(np.uint8)


def paired_bootstrap(records, seed: int, draws: int = 10000) -> dict[str, Any]:
    from src.quality_semantic_audit import metrics

    if len(records) != 30:
        raise ValueError("Paired bootstrap requires exactly 30 holdout views")
    rng = np.random.default_rng(seed)
    keys = (
        "five_class_miou_pp",
        "foreground_miou_pp",
        "stay_cable_iou_pp",
        "stay_cable_boundary_f1_pp",
    )
    deltas = {key: np.empty(draws) for key in keys}
    for draw in range(draws):
        selected = rng.integers(0, len(records), len(records))
        baseline_confusion = sum(
            (records[index][0] for index in selected), np.zeros((5, 5), np.int64)
        )
        candidate_confusion = sum(
            (records[index][1] for index in selected), np.zeros((5, 5), np.int64)
        )
        baseline = metrics(baseline_confusion)
        candidate = metrics(candidate_confusion)
        for metric_key, key in (
            ("mIoU_with_background", "five_class_miou_pp"),
            ("mIoU_without_background", "foreground_miou_pp"),
        ):
            deltas[key][draw] = 100 * (candidate[metric_key] - baseline[metric_key])
        deltas["stay_cable_iou_pp"][draw] = 100 * (
            candidate["per_class"]["stay_cable"]["IoU"]
            - baseline["per_class"]["stay_cable"]["IoU"]
        )
        baseline_boundary = sum(
            (records[index][2] for index in selected), np.zeros(4, np.int64)
        )
        candidate_boundary = sum(
            (records[index][3] for index in selected), np.zeros(4, np.int64)
        )
        deltas["stay_cable_boundary_f1_pp"][draw] = 100 * (
            boundary_metrics(candidate_boundary)["boundary_f1"]
            - boundary_metrics(baseline_boundary)["boundary_f1"]
        )
    return {
        "method": "paired_view_bootstrap_with_replacement",
        "views": len(records),
        "draws": draws,
        "seed": seed,
        "interval": "percentile_95",
        "metrics": {
            key: {
                "lower_pp": float(np.nanpercentile(values, 2.5)),
                "median_pp": float(np.nanpercentile(values, 50.0)),
                "upper_pp": float(np.nanpercentile(values, 97.5)),
                "probability_strictly_positive": float(np.nanmean(values > 0)),
            }
            for key, values in deltas.items()
        },
    }


def evaluate_gates(
    holdout: dict[str, Any], bootstrap: dict[str, Any]
) -> dict[str, Any]:
    five_class_pp = 100 * (
        holdout["ensemble"]["five_class_miou"] - holdout["baseline"]["five_class_miou"]
    )
    foreground_pp = 100 * (
        holdout["ensemble"]["foreground_miou"] - holdout["baseline"]["foreground_miou"]
    )
    class_drops = {
        name: 100 * (holdout["ensemble"]["per_class_iou"][name] - value)
        for name, value in holdout["baseline"]["per_class_iou"].items()
    }
    group_drops = {
        name: 100 * (row["ensemble"] - row["baseline"])
        for name, row in holdout["groups_five_class_miou"].items()
    }
    boundary_f1_pp = 100 * (
        holdout["ensemble"]["boundary_f1"] - holdout["baseline"]["boundary_f1"]
    )
    five_class_stats = bootstrap["metrics"]["five_class_miou_pp"]
    checks = {
        "g1_holdout_five_class_miou_vs_sem392": {
            "delta_pp": five_class_pp,
            "threshold_pp": 0.10,
            "passed": five_class_pp >= 0.10,
        },
        "g2_holdout_foreground_miou_vs_sem392": {
            "delta_pp": foreground_pp,
            "threshold_pp": 0.0,
            "passed": foreground_pp >= 0.0,
        },
        "g3_holdout_class_iou_drop_vs_sem392": {
            "per_class_delta_pp": class_drops,
            "max_drop_pp": min(class_drops.values()),
            "threshold_pp": -0.10,
            "passed": all(value >= -0.10 for value in class_drops.values()),
        },
        "g4_holdout_group_five_class_miou_drop_vs_sem392": {
            "per_group_delta_pp": group_drops,
            "max_drop_pp": min(group_drops.values()),
            "threshold_pp": -0.15,
            "passed": all(value >= -0.15 for value in group_drops.values()),
        },
        "g5_holdout_stay_cable_2px_boundary_f1_drop_vs_sem392": {
            "delta_pp": boundary_f1_pp,
            "threshold_pp": -0.10,
            "passed": boundary_f1_pp >= -0.10,
        },
        "g6_paired_bootstrap_median_and_probability": {
            "median_pp": five_class_stats["median_pp"],
            "probability_strictly_positive": five_class_stats[
                "probability_strictly_positive"
            ],
            "passed": (
                five_class_stats["median_pp"] > 0
                and five_class_stats["probability_strictly_positive"] >= 0.95
            ),
        },
    }
    return {
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks.values()),
        "outcome": (
            "worth_training_rgb140_second_seed_for_full_ensemble_validation"
            if all(check["passed"] for check in checks.values())
            else "closed_negative_result"
        ),
    }


def parent_context(config: dict[str, Any], key: str) -> dict[str, Any]:
    parent = config["parents"][key]
    return {
        "key": key,
        "tag": PARENT_TAGS[key],
        "checkpoint": ROOT / parent["checkpoint"],
        "checkpoint_sha256": parent["checkpoint_sha256"],
        "result": ROOT / parent["inference_result"],
        "result_sha256": parent["inference_result_sha256"],
        "predictions": ROOT / parent["predictions"],
        "directory": ROOT / config["output_path"] / f"parent-{PARENT_TAGS[key]}",
    }


def verify_parent_inputs(context: dict[str, Any], split_path: Path) -> dict[str, Any]:
    if sha256(context["checkpoint"]) != context["checkpoint_sha256"]:
        raise ValueError(f"{context['key']}: sealed checkpoint hash differs")
    if sha256(context["result"]) != context["result_sha256"]:
        raise ValueError(f"{context['key']}: sealed inference result hash differs")
    if sha256(ROOT / "src/quality_mask2former.py") != MASK2FORMER_SHA256:
        raise ValueError("Current native_predict source differs from the parents")
    parent = read(context["result"])
    if parent["status"] != "completed":
        raise ValueError(f"{context['key']}: parent inference is not completed")
    if (
        parent["checkpoint_sha256_before_holdout_access"]
        != context["checkpoint_sha256"]
    ):
        raise ValueError(
            f"{context['key']}: parent result binds a different checkpoint"
        )
    if len(parent["prediction_hashes"]) != 400:
        raise ValueError(f"{context['key']}: expected 400 parent predictions")
    verify_hashes(context["predictions"], parent["prediction_hashes"])
    return parent


def gpu_lock():
    return (ROOT / "outputs/quality/gpu.lock").open("a")


def assert_gpu_free() -> None:
    applications = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    )
    if any(line.strip() and "ToDesk" not in line for line in applications.splitlines()):
        raise RuntimeError("Other GPU compute process active outside the shared lock")


def reproduce(config: dict[str, Any], artifact_root: Path, parent_key: str) -> None:
    context = parent_context(config, parent_key)
    output = context["directory"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "probabilities").mkdir()
    (output / "predictions").mkdir()
    split_path = ROOT / config["split_path"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "stage": "reproduce",
        "experiment_id": EID,
        "parent_experiment": parent_key,
        "config_sha256": sha256(CONFIG),
        "split_sha256": sha256(split_path),
        "code_sha256": code_hashes(),
        "labels_read_in_this_stage": 0,
    }
    started = time.monotonic()
    try:
        parent = verify_parent_inputs(context, split_path)
        split = read(split_path)
        fold = split["primary_fold"]
        labeled = set(fold["semantic_train_labeled"]) | set(
            fold["semantic_validation_labeled"]
        )
        rgb_evaluation = ROOT / config["rgb_evaluation"]
        rgb_result = read(rgb_evaluation / "result.json")
        if (
            rgb_result["status"] != "completed"
            or len(rgb_result["image_hashes"]) != 400
        ):
            raise ValueError("Frozen RGB150 evaluation renders are unavailable")
        verify_hashes(rgb_evaluation / "images", rgb_result["image_hashes"])
        result.update(
            parent_result_sha256=context["result_sha256"],
            checkpoint_sha256=context["checkpoint_sha256"],
            frozen_rgb_hashes_exact=True,
        )
        with gpu_lock() as lock:
            queued = time.monotonic()
            fcntl.flock(lock, fcntl.LOCK_EX)
            result["queue_seconds"] = time.monotonic() - queued
            assert_gpu_free()
            import run_sem384_dev270 as shared

            runtime, torch, np = shared.setup_runtime(artifact_root)
            from PIL import Image
            from src.quality_mask2former import (
                config_for_model,
                load_official_model,
                native_predict,
            )
            from src.semantic_cable_binary import read_rgb

            model_config = config_for_model(
                runtime / "vendor/Mask2Former",
                "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
                False,
            )
            model, _ = load_official_model(
                model_config, runtime / "weights/model_final_6b4a3a.pkl"
            )
            model.load_state_dict(
                torch.load(
                    context["checkpoint"], map_location="cpu", weights_only=True
                )["model"]
            )
            model.cuda()
            model.eval()
            torch.cuda.reset_peak_memory_stats()
            names = sorted(parent["prediction_hashes"])
            inference_seconds = []
            probability_hashes = {}
            prediction_hashes = {}
            byte_identical_to_parent = 0
            for index, name in enumerate(names):
                rgb = read_rgb(rgb_evaluation / "images" / name)
                torch.cuda.synchronize()
                infer_started = time.monotonic()
                probability = native_predict(model, rgb)
                torch.cuda.synchronize()
                inference_seconds.append(time.monotonic() - infer_started)
                if (
                    probability.dtype != np.float32
                    or probability.shape != (5, 989, 1320)
                    or not np.isfinite(probability).all()
                    or (probability < 0).any()
                    or not np.allclose(
                        probability.sum(0), 1.0, rtol=0, atol=PROBABILITY_SUM_ATOL
                    )
                ):
                    raise ValueError(f"{name}: reproduced probability is invalid")
                prediction = probability.argmax(0).astype(np.uint8)
                parent_prediction = np.asarray(
                    Image.open(context["predictions"] / name), dtype=np.uint8
                )
                if not np.array_equal(prediction, parent_prediction):
                    raise ValueError(
                        f"{name}: reproduced argmax differs from the sealed parent prediction"
                    )
                np.savez_compressed(
                    output / "probabilities" / f"{Path(name).stem}.npz",
                    probability=probability,
                )
                Image.fromarray(prediction).save(output / "predictions" / name)
                probability_hashes[f"{Path(name).stem}.npz"] = sha256(
                    output / "probabilities" / f"{Path(name).stem}.npz"
                )
                prediction_hashes[name] = sha256(output / "predictions" / name)
                if prediction_hashes[name] == parent["prediction_hashes"][name]:
                    byte_identical_to_parent += 1
                if index % 25 == 0:
                    print(
                        json.dumps({"reproduced": index + 1, "total": 400}),
                        flush=True,
                    )
            if len(probability_hashes) != 400 or len(prediction_hashes) != 400:
                raise ValueError("Expected exactly 400 reproduced views")
            if sha256(context["checkpoint"]) != context["checkpoint_sha256"]:
                raise ValueError("Parent checkpoint changed during reproduction")
            verify_hashes(context["predictions"], parent["prediction_hashes"])
            verify_hashes(rgb_evaluation / "images", rgb_result["image_hashes"])
            result.update(
                status="completed",
                pixel_exact_predictions=400,
                byte_identical_predictions=byte_identical_to_parent,
                probability_hashes=probability_hashes,
                prediction_hashes=prediction_hashes,
                inference_seconds=inference_seconds,
                inference_median_seconds=float(np.median(inference_seconds[1:])),
                labeled_views_deliberately_untouched=sorted(labeled),
                holdout_label_reads=0,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "reproduce-result.json", result)


def require_reproductions(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reproduced = {}
    for key in PARENT_KEYS:
        context = parent_context(config, key)
        record = read(context["directory"] / "reproduce-result.json")
        if record["status"] != "completed" or record["parent_experiment"] != key:
            raise ValueError(f"A completed sealed reproduction of {key} is required")
        verify_hashes(
            context["directory"] / "probabilities", record["probability_hashes"]
        )
        verify_hashes(context["directory"] / "predictions", record["prediction_hashes"])
        if record["pixel_exact_predictions"] != 400:
            raise ValueError(f"{key}: reproduction is not pixel-exact")
        reproduced[key] = record
    return reproduced


def ensemble(config: dict[str, Any]) -> None:
    output = ROOT / config["output_path"] / "ensemble"
    output.mkdir(parents=True, exist_ok=False)
    (output / "probabilities").mkdir()
    (output / "predictions").mkdir()
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "stage": "ensemble",
        "experiment_id": EID,
        "config_sha256": sha256(CONFIG),
        "split_sha256": sha256(ROOT / config["split_path"]),
        "code_sha256": code_hashes(),
        "formula": config["ensemble_protocol"]["formula"],
        "labels_read_in_this_stage": 0,
    }
    started = time.monotonic()
    try:
        reproduced = require_reproductions(config)
        contexts = {key: parent_context(config, key) for key in PARENT_KEYS}
        names = sorted(reproduced[PARENT_KEYS[0]]["prediction_hashes"])
        if names != sorted(reproduced[PARENT_KEYS[1]]["prediction_hashes"]):
            raise ValueError("The two parents do not cover the same 400 views")
        from PIL import Image

        probability_hashes = {}
        prediction_hashes = {}
        stored_precision_argmax_mismatches = 0
        tie_pixels = 0
        for index, name in enumerate(names):
            stem = Path(name).stem
            a = np.load(
                contexts[PARENT_KEYS[0]]["directory"] / "probabilities" / f"{stem}.npz"
            )["probability"]
            b = np.load(
                contexts[PARENT_KEYS[1]]["directory"] / "probabilities" / f"{stem}.npz"
            )["probability"]
            merged = ensemble_probability(a, b)
            if (
                not np.isfinite(merged).all()
                or (merged < 0).any()
                or not np.allclose(
                    merged.sum(0), 1.0, rtol=0, atol=PROBABILITY_SUM_ATOL
                )
            ):
                raise ValueError(f"{name}: ensemble probability is invalid")
            prediction = smallest_class_argmax(merged)
            stored = merged.astype(np.float32)
            if not np.array_equal(stored.argmax(0).astype(np.uint8), prediction):
                stored_precision_argmax_mismatches += 1
            tie_pixels += tie_pixel_count(merged)
            np.savez_compressed(
                output / "probabilities" / f"{stem}.npz", probability=stored
            )
            Image.fromarray(prediction).save(output / "predictions" / name)
            probability_hashes[f"{stem}.npz"] = sha256(
                output / "probabilities" / f"{stem}.npz"
            )
            prediction_hashes[name] = sha256(output / "predictions" / name)
            if index % 25 == 0:
                print(json.dumps({"ensembled": index + 1, "total": 400}), flush=True)
        if stored_precision_argmax_mismatches:
            raise ValueError("Stored float32 precision changes the ensemble argmax")
        result.update(
            status="completed",
            views=len(names),
            probability_hashes=probability_hashes,
            prediction_hashes=prediction_hashes,
            float64_tie_pixels=tie_pixels,
            stored_precision="float32",
            stored_precision_argmax_mismatches=stored_precision_argmax_mismatches,
            holdout_label_reads=0,
        )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


def require_ensemble(config: dict[str, Any]) -> dict[str, Any]:
    output = ROOT / config["output_path"] / "ensemble"
    record = read(output / "result.json")
    if record["status"] != "completed" or record["views"] != 400:
        raise ValueError("A completed sealed 400-view ensemble is required")
    verify_hashes(output / "probabilities", record["probability_hashes"])
    verify_hashes(output / "predictions", record["prediction_hashes"])
    return record


def evaluate(config: dict[str, Any]) -> None:
    from PIL import Image
    from src.quality_semantic_audit import independent_confusion, metrics

    output = ROOT / config["output_path"] / "evaluation"
    output.mkdir(parents=True, exist_ok=False)
    split_path = ROOT / config["split_path"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "stage": "evaluate",
        "experiment_id": EID,
        "config_sha256": sha256(CONFIG),
        "split_sha256": sha256(split_path),
        "code_sha256": code_hashes(),
        "label_access_begins_after_ensemble_sealed": True,
    }
    started = time.monotonic()
    try:
        ensemble_record = require_ensemble(config)
        reproduced = require_reproductions(config)
        contexts = {key: parent_context(config, key) for key in PARENT_KEYS}
        split = read(split_path)
        fold = split["primary_fold"]
        train_names = set(fold["semantic_train_labeled"])
        holdout_names = set(fold["semantic_validation_labeled"])
        group_by_name = {
            name: group["group_id"]
            for group in split["groups"]
            for name in group["images"]
        }
        label_root = ROOT / config["label_root"]
        ensemble_predictions = ROOT / config["output_path"] / "ensemble/predictions"
        names = sorted(ensemble_record["prediction_hashes"])
        systems = {
            "SEM-384-RGB150-DEV270": contexts[PARENT_KEYS[0]]["predictions"],
            "SEM-392-RGB150-DEV270-SEED20260919-MEMORY": contexts[PARENT_KEYS[1]][
                "predictions"
            ],
            EID: ensemble_predictions,
        }
        totals = {
            system: {
                "train": np.zeros((5, 5), dtype=np.int64),
                "holdout": np.zeros((5, 5), dtype=np.int64),
                **{
                    group["group_id"]: np.zeros((5, 5), dtype=np.int64)
                    for group in split["groups"]
                },
            }
            for system in systems
        }
        boundary = {
            system: {key: np.zeros(4, dtype=np.int64) for key in totals[system]}
            for system in systems
        }
        per_system_views = {system: {} for system in systems}
        parent_records = {
            key: {row["name"]: row for row in read(contexts[key]["result"])["per_view"]}
            for key in PARENT_KEYS
        }
        parent_confusion_checks = 0
        ensemble_confusions = {}
        disagreement_totals = np.zeros(5, dtype=np.int64)
        oracle_confusion = np.zeros((5, 5), dtype=np.int64)
        corrected_damaged = {
            system: {
                role: dict(
                    corrected_pixels=0,
                    damaged_pixels=0,
                    improved_views=0,
                    worse_views=0,
                    tied_views=0,
                )
                for role in ("train", "holdout")
            }
            for system in PARENT_KEYS
        }
        bootstrap_records = []
        for name in names:
            is_labeled = name in train_names or name in holdout_names
            if not is_labeled:
                continue
            target = np.asarray(Image.open(label_root / name), dtype=np.uint8)
            role = "train" if name in train_names else "holdout"
            keys = [role] + ([group_by_name[name]] if role == "holdout" else [])
            predictions = {
                system: np.asarray(Image.open(path / name), dtype=np.uint8)
                for system, path in systems.items()
            }
            confusions = {
                system: independent_confusion(target, prediction)
                for system, prediction in predictions.items()
            }
            for key in PARENT_KEYS:
                recorded = np.asarray(
                    parent_records[key][name]["confusion"], dtype=np.int64
                )
                if not np.array_equal(confusions[key], recorded):
                    raise ValueError(
                        f"{name}: recomputed {key} confusion differs from the sealed parent result"
                    )
                parent_confusion_checks += 1
            ensemble_confusions[name] = confusions[EID].tolist()
            counts = {
                system: boundary_counts(target, prediction)
                for system, prediction in predictions.items()
            }
            for system in systems:
                for key in keys:
                    totals[system][key] += confusions[system]
                    boundary[system][key] += counts[system]
                per_system_views[system][name] = metrics(confusions[system])
            if role == "holdout":
                a = predictions[PARENT_KEYS[0]]
                b = predictions[PARENT_KEYS[1]]
                diagnostics = disagreement_diagnostics(target, a, b)
                disagreement_totals += np.asarray(
                    [
                        diagnostics["total_pixels"],
                        diagnostics["disagreement_pixels"],
                        diagnostics["only_a_correct_pixels"],
                        diagnostics["only_b_correct_pixels"],
                        diagnostics["both_wrong_disagreement_pixels"],
                    ],
                    dtype=np.int64,
                )
                oracle_confusion += independent_confusion(
                    target, hard_oracle_prediction(target, a, b)
                )
                bootstrap_records.append(
                    (
                        confusions[PARENT_KEYS[1]],
                        confusions[EID],
                        counts[PARENT_KEYS[1]],
                        counts[EID],
                    )
                )
            for key in PARENT_KEYS:
                baseline = per_system_views[key][name]["mIoU_with_background"]
                candidate = per_system_views[EID][name]["mIoU_with_background"]
                delta = 100 * (candidate - baseline)
                corrected = int(
                    ((predictions[key] != target) & (predictions[EID] == target)).sum()
                )
                damaged = int(
                    ((predictions[key] == target) & (predictions[EID] != target)).sum()
                )
                bucket = corrected_damaged[key][role]
                bucket["corrected_pixels"] += corrected
                bucket["damaged_pixels"] += damaged
                bucket[
                    "improved_views"
                    if delta > 0
                    else "worse_views"
                    if delta < 0
                    else "tied_views"
                ] += 1
        if parent_confusion_checks != 600:
            raise ValueError("Expected 600 independent parent confusion checks")
        if len(bootstrap_records) != 30:
            raise ValueError("Expected exactly 30 holdout bootstrap records")
        holdout_pixels = int(disagreement_totals[0])
        diagnostics_summary = {
            "holdout_views": 30,
            "total_pixels": holdout_pixels,
            "disagreement_pixel_share": float(disagreement_totals[1] / holdout_pixels),
            "only_sem384_correct_share": float(disagreement_totals[2] / holdout_pixels),
            "only_sem392_correct_share": float(disagreement_totals[3] / holdout_pixels),
            "both_wrong_disagreement_share": float(
                disagreement_totals[4] / holdout_pixels
            ),
            "hard_oracle_holdout_five_class_miou": float(
                metrics(oracle_confusion)["mIoU_with_background"]
            ),
            "hard_oracle_role": "complementarity_upper_bound_and_input_check_only",
        }
        oracle_foreground = metrics(oracle_confusion)["mIoU_without_background"]
        diagnostics_summary["hard_oracle_holdout_foreground_miou"] = float(
            oracle_foreground
        )
        system_metrics = {
            system: {
                key: {
                    **metrics(matrix),
                    "boundary": boundary_metrics(boundary[system][key]),
                }
                for key, matrix in totals[system].items()
            }
            for system in systems
        }
        parent_result_crosscheck = {}
        for key in PARENT_KEYS:
            parent_result = read(contexts[key]["result"])
            for scope in ("train", "holdout"):
                if not np.array_equal(
                    np.asarray(system_metrics[key][scope]["confusion"], dtype=np.int64),
                    np.asarray(
                        parent_result["metrics"][scope]["confusion"], dtype=np.int64
                    ),
                ):
                    raise ValueError(f"{key}: aggregate {scope} confusion differs")
            parent_result_crosscheck[key] = {
                "aggregate_confusions_exact": True,
                "holdout_boundary_counts_exact": np.array_equal(
                    boundary[key]["holdout"],
                    np.asarray(
                        parent_result["boundary_metrics"]["holdout"]["counts"],
                        dtype=np.int64,
                    ),
                ),
            }
        bootstrap = paired_bootstrap(
            bootstrap_records,
            config["bootstrap_seed"],
            config["evaluation_protocol"]["paired_bootstrap"]["draws"],
        )
        baseline = system_metrics[PARENT_KEYS[1]]
        ensemble_metrics = system_metrics[EID]
        holdout_view = {
            "baseline": {
                "five_class_miou": baseline["holdout"]["mIoU_with_background"],
                "foreground_miou": baseline["holdout"]["mIoU_without_background"],
                "per_class_iou": {
                    name: baseline["holdout"]["per_class"][name]["IoU"]
                    for name in CLASSES
                },
                "boundary_f1": baseline["holdout"]["boundary"]["boundary_f1"],
            },
            "ensemble": {
                "five_class_miou": ensemble_metrics["holdout"]["mIoU_with_background"],
                "foreground_miou": ensemble_metrics["holdout"][
                    "mIoU_without_background"
                ],
                "per_class_iou": {
                    name: ensemble_metrics["holdout"]["per_class"][name]["IoU"]
                    for name in CLASSES
                },
                "boundary_f1": ensemble_metrics["holdout"]["boundary"]["boundary_f1"],
            },
            "groups_five_class_miou": {
                group["group_id"]: {
                    "baseline": baseline[group["group_id"]]["mIoU_with_background"],
                    "ensemble": ensemble_metrics[group["group_id"]][
                        "mIoU_with_background"
                    ],
                }
                for group in split["groups"]
            },
        }
        gates = evaluate_gates(holdout_view, bootstrap)
        sem384 = system_metrics[PARENT_KEYS[0]]
        result.update(
            status="completed",
            metrics=system_metrics,
            parent_confusion_checks=parent_confusion_checks,
            parent_result_crosscheck=parent_result_crosscheck,
            input_integrity_diagnostics=diagnostics_summary,
            paired_holdout_bootstrap_vs_sem392=bootstrap,
            promotion_gates=gates,
            per_view_ensemble_confusions=ensemble_confusions,
            corrected_damaged_vs_parents=corrected_damaged,
            comparisons_vs_sem392={
                scope: {
                    "five_class_delta_pp": 100
                    * (
                        ensemble_metrics[scope]["mIoU_with_background"]
                        - baseline[scope]["mIoU_with_background"]
                    ),
                    "foreground_delta_pp": 100
                    * (
                        ensemble_metrics[scope]["mIoU_without_background"]
                        - baseline[scope]["mIoU_without_background"]
                    ),
                    "per_class_iou_delta_pp": {
                        name: 100
                        * (
                            ensemble_metrics[scope]["per_class"][name]["IoU"]
                            - baseline[scope]["per_class"][name]["IoU"]
                        )
                        for name in CLASSES
                    },
                    "boundary_f1_delta_pp": 100
                    * (
                        ensemble_metrics[scope]["boundary"]["boundary_f1"]
                        - baseline[scope]["boundary"]["boundary_f1"]
                    ),
                }
                for scope in (
                    "train",
                    "holdout",
                    *(g["group_id"] for g in split["groups"]),
                )
            },
            comparisons_vs_sem384={
                scope: {
                    "five_class_delta_pp": 100
                    * (
                        ensemble_metrics[scope]["mIoU_with_background"]
                        - sem384[scope]["mIoU_with_background"]
                    ),
                    "foreground_delta_pp": 100
                    * (
                        ensemble_metrics[scope]["mIoU_without_background"]
                        - sem384[scope]["mIoU_without_background"]
                    ),
                }
                for scope in (
                    "train",
                    "holdout",
                    *(g["group_id"] for g in split["groups"]),
                )
            },
            interpretation="Read-only frozen-checkpoint ensemble on one fixed split; the hard-oracle number is an unreachable complementarity bound, not a candidate score.",
        )
        write_new_json(
            ROOT / config["output_path"] / "evaluation/gate.json",
            {
                "schema_version": 1,
                "experiment_id": EID,
                "gates": gates,
                "baseline": "SEM-392-RGB150-DEV270-SEED20260919-MEMORY",
                "bootstrap_seed": config["bootstrap_seed"],
            },
        )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)
        print(
            json.dumps(
                {
                    "status": result.get("status"),
                    "gates": result.get("promotion_gates", {}).get("outcome"),
                },
                ensure_ascii=False,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("reproduce", "ensemble", "evaluate"), required=True
    )
    parser.add_argument("--parent", choices=("sem384", "sem392"), default=None)
    args = parser.parse_args()
    config = read(CONFIG)
    if config["experiment_id"] != EID:
        raise ValueError("SEM-394 config identity changed")
    artifact_root = args.artifact_root.resolve()
    if args.stage == "reproduce":
        if args.parent is None:
            raise ValueError("The reproduce stage requires --parent sem384 or sem392")
        parent_key = [key for key, tag in PARENT_TAGS.items() if tag == args.parent][0]
        reproduce(config, artifact_root, parent_key)
    elif args.stage == "ensemble":
        ensemble(config)
    else:
        evaluate(config)


if __name__ == "__main__":
    main()
