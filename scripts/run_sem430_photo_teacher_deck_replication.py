#!/usr/bin/env python3
"""Replicate SEM429's deck-only photo-teacher rule on spatial fold v2."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_sem429_photo_teacher_gate import load_model, load_rgb, one_hot, save_id_and_visual
from src.photo_teacher_gate import PhotoTeacherGate, apply_photo_teacher_gate
from src.quality_protocol import sha256, write_new_json
from src.quality_semantic_audit import independent_confusion, metrics

EID = "SEM-430-PHOTO-TEACHER-DECK-REPLICATION"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
CODE = (
    "scripts/run_sem430_photo_teacher_deck_replication.py",
    "scripts/run_sem429_photo_teacher_gate.py",
    "src/photo_teacher_gate.py",
    "src/quality_mask2former.py",
    "src/quality_semantic_audit.py",
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_runtime(config: dict[str, Any], artifact_root: Path):
    runtime = ROOT / config["runtime_root"]
    sys.path[:0] = [
        str(runtime / "runtime"),
        str(runtime / "vendor/detectron2"),
        str(runtime / "vendor/Mask2Former"),
        str(runtime / "vendor/Mask2Former/mask2former/modeling/pixel_decoder/ops"),
    ]
    cuda = artifact_root / "r1/toolchain/cuda-12.8"
    os.environ.update(
        CUDA_HOME=str(cuda),
        PATH=str(cuda / "bin") + os.pathsep + os.environ.get("PATH", ""),
        LD_LIBRARY_PATH=str(cuda / "lib64") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""),
        TORCH_HOME=str(artifact_root / "r1/weights/torch"),
        WANDB_MODE="disabled",
        HF_HUB_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    import torch

    return runtime, torch


def infer(model, names: list[str], spec: dict[str, Any], output: Path, native_predict):
    gate = PhotoTeacherGate(teacher_classes=(1,), minimum_component_pixels=((1, 128),))
    rows = []
    for index, name in enumerate(names):
        photo_path = ROOT / spec["photo_root"] / name
        render_path = ROOT / spec["render_root"] / name
        prediction_path = ROOT / spec["saved_predictions"] / name
        photo = load_rgb(photo_path)
        load_rgb(render_path)
        baseline = np.asarray(Image.open(prediction_path), dtype=np.uint8)
        normal = native_predict(model, photo)
        flipped = native_predict(model, np.ascontiguousarray(photo[:, ::-1]))[:, :, ::-1].copy()
        result = apply_photo_teacher_gate(one_hot(baseline), normal, flipped, gate)
        save_id_and_visual(output / "render", name, baseline)
        save_id_and_visual(output / "photo-teacher", name, result["photo_prediction"])
        save_id_and_visual(output / "candidate", name, result["candidate_prediction"])
        save_id_and_visual(output / "accepted-mask", name, result["accepted_mask"].astype(np.uint8), visual=False)
        rows.append(
            {
                "name": name,
                "counts": result["counts"],
                "photo_sha256": sha256(photo_path),
                "render_rgb_sha256": sha256(render_path),
                "saved_prediction_sha256": sha256(prediction_path),
                "candidate_sha256": sha256(output / "candidate" / name),
                "mask_sha256": sha256(output / "accepted-mask" / name),
            }
        )
        if index % 10 == 0:
            print(json.dumps({"done": index + 1, "total": len(names)}), flush=True)
    return rows


def evaluate(config: dict[str, Any], names: list[str], output: Path):
    baseline_total = np.zeros((5, 5), np.int64)
    candidate_total = np.zeros((5, 5), np.int64)
    changed = corrected = harmed = neutral = 0
    per_view = []
    for name in names:
        target = np.asarray(Image.open(ROOT / config["validation"]["label_root"] / name), dtype=np.uint8)
        baseline = np.asarray(Image.open(output / "validation/render" / name), dtype=np.uint8)
        candidate = np.asarray(Image.open(output / "validation/candidate" / name), dtype=np.uint8)
        mask = baseline != candidate
        base_ok, candidate_ok = baseline == target, candidate == target
        vc = int(np.sum(mask & ~base_ok & candidate_ok))
        vh = int(np.sum(mask & base_ok & ~candidate_ok))
        vn = int(np.sum(mask & ~base_ok & ~candidate_ok))
        changed += int(mask.sum()); corrected += vc; harmed += vh; neutral += vn
        baseline_total += independent_confusion(target, baseline)
        candidate_total += independent_confusion(target, candidate)
        per_view.append({"name": name, "changed": int(mask.sum()), "corrected": vc, "harmed": vh, "neutral_wrong_to_wrong": vn})
    baseline_metrics, candidate_metrics = metrics(baseline_total), metrics(candidate_total)
    five_delta = 100 * (candidate_metrics["mIoU_with_background"] - baseline_metrics["mIoU_with_background"])
    deck_delta = 100 * (candidate_metrics["per_class"]["deck"]["IoU"] - baseline_metrics["per_class"]["deck"]["IoU"])
    changed_precision = corrected / changed if changed else 0.0
    non_deck_identity = True
    for name in names:
        baseline = np.asarray(Image.open(output / "validation/render" / name), dtype=np.uint8)
        candidate = np.asarray(Image.open(output / "validation/candidate" / name), dtype=np.uint8)
        # Background is intentionally consumed; structural non-deck classes remain exact.
        non_deck_identity &= all(np.array_equal(baseline == class_id, candidate == class_id) for class_id in (2, 3, 4))
    gates = {
        "minimum_changed_precision": changed_precision >= config["promotion_gates"]["minimum_changed_precision"],
        "minimum_five_class_miou_delta_pp": five_delta >= config["promotion_gates"]["minimum_five_class_miou_delta_pp"],
        "minimum_deck_iou_delta_pp": deck_delta >= config["promotion_gates"]["minimum_deck_iou_delta_pp"],
        "positive_net_corrections": corrected > harmed,
        "non_deck_structural_prediction_identity": bool(non_deck_identity),
    }
    return {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "five_class_miou_delta_pp": five_delta,
        "foreground_miou_delta_pp": 100 * (candidate_metrics["mIoU_without_background"] - baseline_metrics["mIoU_without_background"]),
        "deck_iou_delta_pp": deck_delta,
        "changed_pixels": changed,
        "changed_fraction": changed / (len(names) * 989 * 1320),
        "corrected_pixels": corrected,
        "harmed_pixels": harmed,
        "neutral_wrong_to_wrong": neutral,
        "net_corrections": corrected - harmed,
        "changed_precision": changed_precision,
        "gates": gates,
        "training_design_authorized": all(gates.values()),
        "per_view": per_view,
    }


def filter_public(config: dict[str, Any], output: Path):
    source = ROOT / config["public_source"] / "public100"
    rows = []
    for index in range(301, 401):
        name = f"{index:03d}.png"
        baseline = np.asarray(Image.open(source / "render" / name), dtype=np.uint8)
        teacher = np.asarray(Image.open(source / "photo-teacher" / name), dtype=np.uint8)
        parent_mask = np.asarray(Image.open(source / "accepted-mask" / name), dtype=np.uint8).astype(bool)
        mask = parent_mask & (teacher == 1)
        candidate = baseline.copy(); candidate[mask] = 1
        save_id_and_visual(output / "public100/render", name, baseline)
        save_id_and_visual(output / "public100/photo-teacher", name, teacher)
        save_id_and_visual(output / "public100/candidate", name, candidate)
        save_id_and_visual(output / "public100/accepted-mask", name, mask.astype(np.uint8), visual=False)
        rows.append({"name": name, "accepted_deck_pixels": int(mask.sum()), "candidate_sha256": sha256(output / "public100/candidate" / name), "mask_sha256": sha256(output / "public100/accepted-mask" / name)})
    return {
        "accepted_deck_pixels": int(sum(row["accepted_deck_pixels"] for row in rows)),
        "accepted_fraction": float(sum(row["accepted_deck_pixels"] for row in rows) / (100 * 989 * 1320)),
        "views_with_changes": int(sum(row["accepted_deck_pixels"] > 0 for row in rows)),
        "ground_truth_accessed": False,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    config = read(CONFIG)
    output = ROOT / config["output_path"]
    output.mkdir(parents=True, exist_ok=False)
    split = read(ROOT / config["split_path"])
    names = sorted(split["primary_fold"]["semantic_validation_labeled"])
    if len(names) != config["validation"]["views"]:
        raise ValueError("v2 confirmation count mismatch")
    result = {
        "schema_version": 1, "status": "running", "experiment_id": EID,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seed": config["seed"], "dataset_manifest": config["dataset_manifest"], "split_id": config["split_id"],
        "config_sha256": sha256(CONFIG), "split_sha256": sha256(ROOT / config["split_path"]),
        "code_sha256": {name: sha256(ROOT / name) for name in CODE}, "trained_steps": 0,
        "labels_decoded_during_inference": False, "validation_names": names,
    }
    started = time.monotonic()
    try:
        runtime, torch = setup_runtime(config, args.artifact_root.resolve())
        from src.quality_mask2former import native_predict
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            queued = time.monotonic(); fcntl.flock(lock, fcntl.LOCK_EX); result["queue_seconds"] = time.monotonic() - queued
            torch.set_num_threads(4); torch.cuda.reset_peak_memory_stats()
            model, model_record = load_model(runtime, torch, config["validation"], config)
            rows = infer(model, names, config["validation"], output / "validation", native_predict)
            del model; torch.cuda.empty_cache()
            result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        seal = {"schema_version": 1, "experiment_id": EID, "config_sha256": result["config_sha256"], "code_sha256": result["code_sha256"], "labels_decoded": False, "model": model_record, "rows": rows}
        write_new_json(output / "inference-seal.json", seal)
        result["inference_seal_sha256_before_label_access"] = sha256(output / "inference-seal.json")
        result["validation"] = evaluate(config, names, output)
        if sha256(output / "inference-seal.json") != result["inference_seal_sha256_before_label_access"]:
            raise ValueError("Sealed teacher outputs changed during evaluation")
        result["public100"] = filter_public(config, output)
        result["status"] = "completed_training_design_authorized" if result["validation"]["training_design_authorized"] else "completed_replication_failed"
    except BaseException as error:
        result.update(status="failed_preserved", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
