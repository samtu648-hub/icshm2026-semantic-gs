#!/usr/bin/env python3
"""Run the frozen SEM437-12k multiscale and bridge-strip inference audit."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_sem396_spatial as spatial_base
from src.quality_multiscale_inference import (
    fixed_ensemble,
    scaled_probability,
    strip_probability,
    tiled_probability,
)
from src.quality_protocol import sha256, verify_hashes, write_new_json


EID = "SEM-438-RGB170-MASK2FORMER-MULTISCALE-AUDIT"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
ARMS = ("native", "global_1p5", "global_2p0", "strip_2p0", "fixed_ensemble")
CLASSES = ("background", "deck", "stay_cable", "tower", "foundation")
PALETTE = np.asarray(
    [[0, 0, 0], [230, 86, 37], [255, 210, 0], [38, 139, 210], [133, 153, 0]],
    dtype=np.uint8,
)
CODE_PATHS = (
    "scripts/run_sem438_multiscale_audit.py",
    "src/quality_multiscale_inference.py",
    "src/quality_mask2former.py",
    "src/quality_semantic_audit.py",
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(config):
    if (
        config["experiment_id"] != EID
        or config["semantic_training"]
        or config["tile"] != 768
        or config["stride"] != 512
        or config["arms"]["native"]["scale"] != 1.0
        or config["arms"]["global_1p5"]["scale"] != 1.5
        or config["arms"]["global_2p0"]["scale"] != 2.0
        or config["arms"]["strip_2p0"]["source_height"] != 384
        or config["arms"]["strip_2p0"]["anchor_y_percentile"] != 85
        or config["arms"]["fixed_ensemble"]["weights"]
        != {"native": 0.5, "global_1p5": 0.25, "strip_2p0": 0.25}
    ):
        raise ValueError("Registered SEM438 protocol changed")


def paths(config):
    export_result = ROOT / config["parent_export_result"]
    export = read(export_result)
    image_root = export_result.parent / "images"
    return export, image_root, ROOT / config["parent_prediction_root"]


def prepare(config):
    output = ROOT / config["output_path"]
    if output.exists():
        raise FileExistsError(output)
    export, image_root, parent_predictions = paths(config)
    checkpoint = ROOT / config["checkpoint"]
    if sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("SEM437 checkpoint changed")
    inference = read(ROOT / config["parent_inference_result"])
    if inference["status"] != "completed" or inference["checkpoint_sha256_before_inference"]["12000"] != config["checkpoint_sha256"]:
        raise ValueError("Parent inference binding changed")
    if len(export["image_hashes"]) != 544:
        raise ValueError("Expected 300 labeled + 100 unlabeled + 144 trajectory inputs")
    verify_hashes(image_root, export["image_hashes"])
    missing_predictions = [name for name in export["image_hashes"] if not (parent_predictions / name).is_file()]
    if missing_predictions:
        raise FileNotFoundError(missing_predictions[:8])
    label_manifest = ROOT / config["label_manifest"]
    label_records = {row["image"]: row for row in read(label_manifest)["records"]}
    label_hashes = {
        name: sha256(ROOT / config["label_root"] / name)
        for name in export["semantic_train"]
    }
    if any(label_hashes[name] != label_records[name]["label_png_sha256"] for name in label_hashes):
        raise ValueError("Frozen semantic labels changed")
    output.mkdir(parents=True)
    write_new_json(
        output / "prepare-result.json",
        {
            "schema_version": 1,
            "status": "completed",
            "experiment_id": EID,
            "config_sha256": sha256(CONFIG),
            "checkpoint_sha256": sha256(checkpoint),
            "parent_export_result_sha256": sha256(ROOT / config["parent_export_result"]),
            "parent_inference_result_sha256": sha256(ROOT / config["parent_inference_result"]),
            "image_hashes": export["image_hashes"],
            "label_hashes": label_hashes,
            "image_count": len(export["image_hashes"]),
            "train_count": len(export["semantic_train"]),
            "unlabeled_count": len(export["unlabeled"]),
            "trajectory_count": len(export["trajectory"]),
            "code_sha256": {path: sha256(ROOT / path) for path in CODE_PATHS},
            "scales_and_weights_frozen_before_outputs": True,
        },
    )


def load_model(config, artifact_root: Path):
    runtime, torch, _ = spatial_base.setup_runtime(artifact_root)
    from src.quality_mask2former import config_for_model, load_official_model

    model_config = config_for_model(
        runtime / "vendor/Mask2Former", config["model_config"], False
    )
    model, _ = load_official_model(model_config, runtime / "weights/model_final_6b4a3a.pkl")
    saved = torch.load(ROOT / config["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"])
    del saved
    model.cuda().eval()
    return model, torch


def infer_arms(model, torch, rgb, config):
    tile, stride = config["tile"], config["stride"]
    durations = {}

    def timed(name, function):
        torch.cuda.synchronize()
        stamp = time.monotonic()
        value = function()
        torch.cuda.synchronize()
        durations[name] = time.monotonic() - stamp
        return value

    native = timed("native", lambda: tiled_probability(model, rgb, tile, stride))
    global_1p5 = timed(
        "global_1p5",
        lambda: scaled_probability(model, rgb, config["arms"]["global_1p5"]["scale"], tile, stride),
    )
    global_2p0 = timed(
        "global_2p0",
        lambda: scaled_probability(model, rgb, config["arms"]["global_2p0"]["scale"], tile, stride),
    )
    strip = config["arms"]["strip_2p0"]
    torch.cuda.synchronize()
    stamp = time.monotonic()
    strip_2p0, strip_record = strip_probability(
        model,
        rgb,
        native,
        source_height=strip["source_height"],
        scale=strip["scale"],
        anchor_classes=tuple(strip["anchor_classes"]),
        anchor_y_percentile=strip["anchor_y_percentile"],
        fallback_anchor_y=strip["fallback_anchor_y"],
        tile=tile,
        stride=stride,
    )
    torch.cuda.synchronize()
    durations["strip_2p0"] = time.monotonic() - stamp
    ensemble = timed(
        "fixed_ensemble",
        lambda: fixed_ensemble(
            {"native": native, "global_1p5": global_1p5, "strip_2p0": strip_2p0},
            config["arms"]["fixed_ensemble"]["weights"],
        ),
    )
    return {
        "native": native,
        "global_1p5": global_1p5,
        "global_2p0": global_2p0,
        "strip_2p0": strip_2p0,
        "fixed_ensemble": ensemble,
    }, durations, strip_record


def save_prediction(probability, path: Path):
    prediction = probability.argmax(axis=0).astype(np.uint8)
    Image.fromarray(prediction).save(path)
    return prediction


def write_review(output: Path, image_root: Path, names: list[str], rows: dict):
    assets = output / "review/assets"
    assets.mkdir(parents=True, exist_ok=True)
    cards = []
    for name in names:
        stem = Path(name).stem
        rgb_path = assets / f"{stem}-rgb.jpg"
        rgb = Image.open(image_root / name).convert("RGB")
        rgb.thumbnail((440, 330), Image.Resampling.LANCZOS)
        rgb.save(rgb_path, quality=90)
        figures = [f'<figure><img src="assets/{rgb_path.name}"><figcaption>RGB170</figcaption></figure>']
        for arm in ARMS:
            label = np.asarray(Image.open(output / f"probe/predictions/{arm}/{name}"), dtype=np.uint8)
            target = assets / f"{stem}-{arm}.png"
            image = Image.fromarray(PALETTE[label], mode="RGB")
            image.thumbnail((440, 330), Image.Resampling.NEAREST)
            image.save(target)
            figures.append(f'<figure><img src="assets/{target.name}"><figcaption>{arm}</figcaption></figure>')
        cards.append(
            f'<article><h2>{stem}</h2><div class="grid">{"".join(figures)}</div>'
            f'<pre>{json.dumps(rows[name], ensure_ascii=False, indent=2)}</pre></article>'
        )
    html = f'''<!doctype html><meta charset="utf-8"><title>SEM438 probe</title><style>
body{{background:#101318;color:#edf2f7;font:14px system-ui;margin:0}}header{{position:sticky;top:0;background:#101318f2;padding:14px;z-index:2}}main{{padding:12px}}article{{border:1px solid #34404d;border-radius:8px;padding:10px;margin:12px 0}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}figure{{margin:0}}img{{width:100%;display:block;background:#000}}figcaption{{color:#aab4c2}}pre{{white-space:pre-wrap}}@media(min-width:1700px){{.grid{{grid-template-columns:repeat(6,1fr)}}}}
</style><header><h1>SEM438 344–358 multiscale probe</h1><p>无GT；只检查尺度可恢复信号，不代表准确率或端点选择。</p></header><main>{''.join(cards)}</main>'''
    (output / "review/index.html").write_text(html, encoding="utf-8")


def gpu_guard():
    applications = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True,
    )
    if any(line.strip() and "ToDesk" not in line for line in applications.splitlines()):
        raise RuntimeError("Other GPU compute process active outside shared lock")
    return applications


def probe(config, artifact_root: Path):
    batch = ROOT / config["output_path"]
    prepared = read(batch / "prepare-result.json")
    output = batch / "probe"
    output.mkdir(exist_ok=False)
    for arm in ARMS:
        (output / "predictions" / arm).mkdir(parents=True)
    export, image_root, parent_predictions = paths(config)
    verify_hashes(image_root, prepared["image_hashes"])
    result = {
        "schema_version": 1,
        "status": "starting",
        "experiment_id": EID,
        "quality_scope": "344-358 no-GT scale-recoverability probe only",
    }
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            queued = time.monotonic()
            fcntl.flock(lock, fcntl.LOCK_EX)
            result["queue_seconds"] = time.monotonic() - queued
            result["gpu_processes_before"] = gpu_guard()
            model, torch = load_model(config, artifact_root)
            torch.cuda.reset_peak_memory_stats()
            from src.semantic_cable_binary import read_rgb

            rows = {}
            all_simplex = True
            native_exact = True
            for index, name in enumerate(config["probe_names"]):
                rgb = read_rgb(image_root / name)
                probabilities, durations, strip_record = infer_arms(model, torch, rgb, config)
                predictions = {}
                for arm, probability in probabilities.items():
                    all_simplex &= bool(
                        np.isfinite(probability).all()
                        and (probability >= 0).all()
                        and np.allclose(probability.sum(0), 1, atol=2e-5)
                    )
                    predictions[arm] = save_prediction(probability, output / f"predictions/{arm}/{name}")
                parent = np.asarray(Image.open(parent_predictions / name), dtype=np.uint8)
                native_exact &= bool(np.array_equal(predictions["native"], parent))
                rows[name] = {
                    "strip": strip_record,
                    "seconds": durations,
                    "arms": {
                        arm: {
                            "changed_from_native_fraction": float(np.mean(prediction != predictions["native"])),
                            "added_deck_pixels": int(np.sum((prediction == 1) & (predictions["native"] != 1))),
                            "removed_deck_pixels": int(np.sum((prediction != 1) & (predictions["native"] == 1))),
                            "class_pixels": np.bincount(prediction.ravel(), minlength=5).tolist(),
                            "sha256": sha256(output / f"predictions/{arm}/{name}"),
                        }
                        for arm, prediction in predictions.items()
                    },
                }
                print(json.dumps({"experiment_id": EID, "stage": "probe", "inferred": index + 1, "total": len(config["probe_names"]), "name": name}), flush=True)
            difficult = ("355.png", "356.png", "357.png")
            gate = config["probe_expansion_gate"]
            qualifying = []
            max_change_ok = True
            for arm in ARMS[1:]:
                recovered = sum(rows[name]["arms"][arm]["added_deck_pixels"] >= gate["minimum_added_deck_pixels_355_357"] for name in difficult)
                if recovered >= gate["minimum_recovered_views_355_357"]:
                    qualifying.append(arm)
                max_change_ok &= all(rows[name]["arms"][arm]["changed_from_native_fraction"] <= gate["maximum_per_view_changed_fraction"] for name in config["probe_names"])
            checkpoint_unchanged = sha256(ROOT / config["checkpoint"]) == config["checkpoint_sha256"]
            verify_hashes(image_root, prepared["image_hashes"])
            result.update(
                status="completed_probe",
                rows=rows,
                native_exact_parent=native_exact,
                finite_simplex=all_simplex,
                maximum_change_gate=max_change_ok,
                qualifying_recovery_arms=qualifying,
                checkpoint_unchanged=checkpoint_unchanged,
                rgb_unchanged=True,
                expansion_authorized=bool(native_exact and all_simplex and max_change_ok and qualifying and checkpoint_unchanged),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
            write_review(batch, image_root, list(config["probe_names"]), rows)
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


def boundary_counts(target, prediction):
    from scipy.ndimage import binary_erosion, distance_transform_edt

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


def metric_summary(record):
    return {
        "mIoU_with_background": record["metrics"]["mIoU_with_background"],
        "mIoU_without_background": record["metrics"]["mIoU_without_background"],
        "per_class_iou": {name: record["metrics"]["per_class"][name]["IoU"] for name in CLASSES},
        "stay_cable_precision": record["metrics"]["per_class"]["stay_cable"]["precision"],
        "stay_cable_recall": record["metrics"]["per_class"]["stay_cable"]["recall"],
        "stay_cable_fp": record["metrics"]["per_class"]["stay_cable"]["fp"],
        "stay_cable_fn": record["metrics"]["per_class"]["stay_cable"]["fn"],
        "boundary_f1": record["boundary_metrics"]["boundary_f1"],
    }


def delta_pp(candidate, control):
    return 100.0 * (candidate - control)


def full(config, artifact_root: Path):
    batch = ROOT / config["output_path"]
    prepared = read(batch / "prepare-result.json")
    probe_result = read(batch / "probe/result.json")
    if not probe_result.get("expansion_authorized"):
        raise ValueError("Preregistered no-GT expansion gate did not pass")
    output = batch / "full"
    output.mkdir(exist_ok=False)
    for arm in ARMS:
        (output / "predictions" / arm).mkdir(parents=True)
    export, image_root, parent_predictions = paths(config)
    names = sorted(export["image_hashes"])
    train_names = set(export["semantic_train"])
    result = {
        "schema_version": 1,
        "status": "starting",
        "experiment_id": EID,
        "quality_scope": config["quality_scope"],
    }
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            queued = time.monotonic()
            fcntl.flock(lock, fcntl.LOCK_EX)
            result["queue_seconds"] = time.monotonic() - queued
            result["gpu_processes_before"] = gpu_guard()
            model, torch = load_model(config, artifact_root)
            torch.cuda.reset_peak_memory_stats()
            from src.quality_semantic_audit import independent_confusion, metrics
            from src.semantic_cable_binary import read_rgb

            confusion = {arm: np.zeros((5, 5), dtype=np.int64) for arm in ARMS}
            boundaries = {arm: np.zeros(4, dtype=np.int64) for arm in ARMS}
            per_view = {arm: [] for arm in ARMS}
            durations = {arm: [] for arm in ARMS}
            no_gt = {arm: {"changed_pixels": 0, "total_pixels": 0, "deck_pixel_delta": 0, "stay_cable_pixel_delta": 0} for arm in ARMS[1:]}
            strip_records = {}
            native_exact = True
            prediction_hashes = {arm: {} for arm in ARMS}
            label_root = ROOT / config["label_root"]
            for index, name in enumerate(names):
                rgb = read_rgb(image_root / name)
                probabilities, elapsed, strip_record = infer_arms(model, torch, rgb, config)
                predictions = {}
                for arm, probability in probabilities.items():
                    target_path = output / f"predictions/{arm}/{name}"
                    predictions[arm] = save_prediction(probability, target_path)
                    prediction_hashes[arm][name] = sha256(target_path)
                    durations[arm].append(elapsed[arm])
                parent = np.asarray(Image.open(parent_predictions / name), dtype=np.uint8)
                native_exact &= bool(np.array_equal(predictions["native"], parent))
                strip_records[name] = strip_record
                if name in train_names:
                    target = np.asarray(Image.open(label_root / name), dtype=np.uint8)
                    for arm in ARMS:
                        view_confusion = independent_confusion(target, predictions[arm])
                        confusion[arm] += view_confusion
                        boundaries[arm] += boundary_counts(target, predictions[arm])
                        per_view[arm].append({"name": name, **metrics(view_confusion)})
                else:
                    native = predictions["native"]
                    for arm in ARMS[1:]:
                        candidate = predictions[arm]
                        no_gt[arm]["changed_pixels"] += int(np.sum(candidate != native))
                        no_gt[arm]["total_pixels"] += int(native.size)
                        no_gt[arm]["deck_pixel_delta"] += int(np.sum(candidate == 1) - np.sum(native == 1))
                        no_gt[arm]["stay_cable_pixel_delta"] += int(np.sum(candidate == 2) - np.sum(native == 2))
                if index % 10 == 0:
                    print(json.dumps({"experiment_id": EID, "stage": "full", "inferred": index + 1, "total": len(names), "name": name}), flush=True)
            arm_results = {}
            for arm in ARMS:
                arm_results[arm] = {
                    "metrics": metrics(confusion[arm]),
                    "boundary_metrics": spatial_base.boundary_metrics(boundaries[arm]),
                    "per_view": per_view[arm],
                    "inference_seconds": durations[arm],
                    "inference_median_seconds": float(np.median(durations[arm][1:])),
                    "prediction_hashes": prediction_hashes[arm],
                }
            for arm in no_gt:
                no_gt[arm]["changed_fraction"] = no_gt[arm]["changed_pixels"] / no_gt[arm]["total_pixels"]
                no_gt[arm]["ground_truth_available"] = False
            control = metric_summary(arm_results["native"])
            candidate = metric_summary(arm_results["fixed_ensemble"])
            deltas = {
                "five_class_miou": delta_pp(candidate["mIoU_with_background"], control["mIoU_with_background"]),
                "foreground_miou": delta_pp(candidate["mIoU_without_background"], control["mIoU_without_background"]),
                "deck_iou": delta_pp(candidate["per_class_iou"]["deck"], control["per_class_iou"]["deck"]),
                "stay_cable_iou": delta_pp(candidate["per_class_iou"]["stay_cable"], control["per_class_iou"]["stay_cable"]),
                "tower_iou": delta_pp(candidate["per_class_iou"]["tower"], control["per_class_iou"]["tower"]),
                "foundation_iou": delta_pp(candidate["per_class_iou"]["foundation"], control["per_class_iou"]["foundation"]),
                "boundary_f1": delta_pp(candidate["boundary_f1"], control["boundary_f1"]),
            }
            gate = config["fixed_ensemble_retention_gate_pp"]
            gate_checks = {
                "five_class_miou": deltas["five_class_miou"] >= gate["five_class_miou_min_delta"],
                "foreground_miou": deltas["foreground_miou"] >= gate["foreground_miou_min_delta"],
                "deck_iou": deltas["deck_iou"] > 0,
                "stay_cable_iou": deltas["stay_cable_iou"] >= gate["stay_cable_iou_min_delta"],
                "tower_iou": deltas["tower_iou"] >= gate["tower_iou_min_delta"],
                "foundation_iou": deltas["foundation_iou"] >= gate["foundation_iou_min_delta"],
                "boundary_f1": deltas["boundary_f1"] >= gate["boundary_f1_min_delta"],
            }
            verify_hashes(image_root, prepared["image_hashes"])
            result.update(
                status="completed",
                native_exact_parent=native_exact,
                checkpoint_unchanged=sha256(ROOT / config["checkpoint"]) == config["checkpoint_sha256"],
                rgb_unchanged=True,
                arms=arm_results,
                no_gt_changes_vs_native=no_gt,
                strip_records=strip_records,
                fixed_ensemble_delta_pp=deltas,
                fixed_ensemble_gate_checks=gate_checks,
                fixed_ensemble_retained=all(gate_checks.values()),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "probe", "full"), required=True)
    args = parser.parse_args()
    config = read(CONFIG)
    validate(config)
    if args.stage == "prepare":
        prepare(config)
    elif args.stage == "probe":
        probe(config, args.artifact_root.resolve())
    else:
        full(config, args.artifact_root.resolve())


if __name__ == "__main__":
    main()
