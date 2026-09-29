#!/usr/bin/env python3
"""Compare photo and RGB140 inputs for frozen SEM378/382/383 models."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import backbone_comparison_stage as shared
from src.quality_dinov3_mask2former import module_state_sha256
from src.quality_protocol import sha256, verify_hashes, write_new_json

EID = "SEM-383-DINOV3-VITB-SPATIAL-SFP-ALL300"
BATCH = ROOT / "outputs/quality/SEM-383/r2"
CONTROL = ROOT / "outputs/quality/SEM-382/r1"
BASE = ROOT / "outputs/quality/SEM-378/r1"
PHOTO = ROOT / "outputs/quality/all_data/RGB-120-r2/scene/images"
RUNTIME = ROOT / "outputs/quality/mask2former/SEM-340"
PALETTE = None
ROIS = {
    "311.png": [180, 160, 900, 900],
    "350.png": [0, 300, 1320, 650],
    "356.png": [0, 300, 1320, 650],
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_photo_hashes(expected: dict[str, str]) -> None:
    for name, digest in expected.items():
        if sha256(PHOTO / name) != digest:
            raise ValueError(f"Diagnostic photo hash mismatch: {name}")


def distribution_metrics(render_probability, photo_probability) -> dict:
    import numpy as np

    render_prediction = render_probability.argmax(0)
    photo_prediction = photo_probability.argmax(0)
    transition = np.bincount(
        (5 * render_prediction + photo_prediction).ravel(), minlength=25
    ).reshape(5, 5)
    epsilon = 1e-7
    midpoint = 0.5 * (render_probability + photo_probability)
    js = 0.5 * np.sum(
        render_probability
        * np.log((render_probability + epsilon) / (midpoint + epsilon))
        + photo_probability
        * np.log((photo_probability + epsilon) / (midpoint + epsilon)),
        axis=0,
    )

    def entropy(probability):
        return float(
            np.mean(-np.sum(probability * np.log(probability + epsilon), axis=0))
        )

    def confidence(probability):
        return float(np.mean(np.max(probability, axis=0)))

    return {
        "hard_label_disagreement": float(
            np.mean(render_prediction != photo_prediction)
        ),
        "render_to_photo_transition": transition.tolist(),
        "render_class_pixels": np.bincount(
            render_prediction.ravel(), minlength=5
        ).tolist(),
        "photo_class_pixels": np.bincount(
            photo_prediction.ravel(), minlength=5
        ).tolist(),
        "probability_mae": float(
            np.mean(np.abs(render_probability - photo_probability))
        ),
        "mean_js_divergence": float(np.mean(js)),
        "render_entropy": entropy(render_probability),
        "photo_entropy": entropy(photo_probability),
        "render_confidence": confidence(render_probability),
        "photo_confidence": confidence(photo_probability),
    }


def save_panel(
    path: Path, name: str, inputs: dict, predictions: dict, box=None
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    palette = np.asarray(
        [[30, 30, 30], [80, 160, 245], [245, 210, 65], [220, 80, 80], [170, 100, 190]],
        dtype=np.uint8,
    )
    tiles = [
        ("Prepared photo", Image.fromarray(inputs["prepared_photo"])),
        ("RGB140 render", Image.fromarray(inputs["rgb140_render"])),
    ]
    for model_name in predictions:
        for domain in ("prepared_photo", "rgb140_render"):
            tiles.append(
                (
                    f"{model_name} {domain}",
                    Image.fromarray(palette[predictions[model_name][domain]]),
                )
            )
    box = box or [0, 0, 1320, 989]
    width = 330
    height = round(width * (box[3] - box[1]) / (box[2] - box[0]))
    header = 20
    columns = 2
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * width, rows * (height + header)),
        cast(Any, "white"),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(tiles):
        x = (index % columns) * width
        y = (index // columns) * (height + header)
        draw.text((x + 4, y + 2), f"{name} {label}", fill="black")
        tile = image.crop(box).resize((width, height), Image.Resampling.NEAREST)
        canvas.paste(tile, (x, y + header))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--include-candidate", action="store_true")
    args = parser.parse_args()
    args.artifact_root = args.artifact_root.resolve()
    args.method = "standard-3dgs"
    config = read(ROOT / "experiments/runs" / EID / "config.yaml")
    binding = read(BATCH / "binding.json")
    names = config["input_diagnostic"]["views"]
    output = BATCH / (
        "input-diagnostic-candidate"
        if args.include_candidate
        else "input-diagnostic-control"
    )
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    result = {
        "schema_version": 1,
        "status": "waiting_gpu",
        "experiment_id": EID,
        "stage": output.name,
        "views": names,
        "domains": ["prepared_photo", "rgb140_render"],
        "no_ground_truth": True,
        "interpretation": "domain sensitivity and qualitative localization only; not accuracy or unique causality",
        "started_unix": time.time(),
    }
    try:
        verify_hashes(ROOT, binding["code_sha256"])
        verify_photo_hashes(binding["diagnostic_photo_hashes"])
        verify_hashes(
            BATCH / "export/images",
            {name: binding["rgb140_image_hashes"][name] for name in names},
        )
        verify_hashes(CONTROL, binding["sem382_sha256"])
        verify_hashes(BASE, binding["sem378_sha256"])
        verify_hashes(args.artifact_root, binding["frozen18"])
        if args.include_candidate:
            candidate_inference = read(BATCH / "inference-6000/result.json")
            if candidate_inference["status"] != "completed":
                raise ValueError("Completed SEM383-6000 inference is required")

        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            queued = time.monotonic()
            while True:
                fcntl.flock(lock, fcntl.LOCK_EX)
                apps = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader",
                    ],
                    text=True,
                )
                if not any(
                    line.strip() and "ToDesk" not in line for line in apps.splitlines()
                ):
                    break
                fcntl.flock(lock, fcntl.LOCK_UN)
                time.sleep(30)
            result["queue_seconds"] = time.monotonic() - queued
            result["gpu_processes_before"] = apps
            _, torch, np = shared.setup(args)
            from PIL import Image
            from scripts import run_sem381, run_sem382, run_sem383
            from src.quality_mask2former import load_official_model, native_predict
            from src.semantic_cable_binary import read_rgb

            torch.set_num_threads(4)
            torch.cuda.reset_peak_memory_stats()
            inputs = {
                name: {
                    "prepared_photo": read_rgb(PHOTO / name),
                    "rgb140_render": read_rgb(BATCH / "export/images" / name),
                }
                for name in names
            }
            model_specs = ["SEM378-6000", "SEM382-6000"]
            if args.include_candidate:
                model_specs.append("SEM383-6000")
            all_probabilities = {}
            all_predictions = {}
            model_records = {}
            for model_name in model_specs:
                if model_name == "SEM378-6000":
                    cfg = run_sem381.model_config()
                    model, loaded = load_official_model(
                        cfg, RUNTIME / "weights/model_final_6b4a3a.pkl"
                    )
                    checkpoint = BASE / "mask2former/model-6000.pt"
                    saved_predictions = BASE / "inference/predictions"
                elif model_name == "SEM382-6000":
                    model_dir = Path(
                        os.environ[config["dinov3"]["model_dir_env"]]
                    ).resolve()
                    _, model, loaded = run_sem382.make_model(
                        read(
                            ROOT
                            / "experiments/runs/SEM-382-DINOV3-VITB-SFP-LR1E4-ALL300/config.yaml"
                        ),
                        model_dir,
                    )
                    checkpoint = CONTROL / "mask2former/model-6000.pt"
                    saved_predictions = CONTROL / "inference-6000/predictions"
                else:
                    model_dir = Path(
                        os.environ[config["dinov3"]["model_dir_env"]]
                    ).resolve()
                    _, model, loaded = run_sem383.make_model(config, model_dir)
                    checkpoint = BATCH / "mask2former/model-6000.pt"
                    saved_predictions = BATCH / "inference-6000/predictions"
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model.load_state_dict(payload["model"], strict=True)
                model.cuda().eval().requires_grad_(False)
                state_before = module_state_sha256(model)
                dino_before = (
                    module_state_sha256(model.backbone.dinov3)
                    if hasattr(model.backbone, "dinov3")
                    else None
                )
                model_probabilities = {}
                model_predictions = {}
                timings = []
                for name in names:
                    model_probabilities[name] = {}
                    model_predictions[name] = {}
                    for domain, rgb in inputs[name].items():
                        torch.cuda.synchronize()
                        inference_started = time.monotonic()
                        probability = native_predict(model, rgb)
                        torch.cuda.synchronize()
                        timings.append(time.monotonic() - inference_started)
                        prediction = probability.argmax(0).astype(np.uint8)
                        model_probabilities[name][domain] = probability
                        model_predictions[name][domain] = prediction
                        probability_path = (
                            output / "probabilities" / model_name / domain
                        )
                        prediction_path = output / "predictions" / model_name / domain
                        color_path = output / "colors" / model_name / domain
                        probability_path.mkdir(parents=True, exist_ok=True)
                        prediction_path.mkdir(parents=True, exist_ok=True)
                        color_path.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            probability_path / f"{Path(name).stem}.npz",
                            probabilities=probability,
                        )
                        Image.fromarray(prediction).save(prediction_path / name)
                        palette = np.asarray(
                            [
                                [30, 30, 30],
                                [80, 160, 245],
                                [245, 210, 65],
                                [220, 80, 80],
                                [170, 100, 190],
                            ],
                            dtype=np.uint8,
                        )
                        Image.fromarray(palette[prediction]).save(color_path / name)
                    expected = np.asarray(Image.open(saved_predictions / name))
                    if not np.array_equal(
                        model_predictions[name]["rgb140_render"], expected
                    ):
                        raise ValueError(f"RGB140 replay mismatch: {model_name}/{name}")
                state_after = module_state_sha256(model)
                dino_after = (
                    module_state_sha256(model.backbone.dinov3)
                    if hasattr(model.backbone, "dinov3")
                    else None
                )
                if state_before != state_after or dino_before != dino_after:
                    raise ValueError(
                        f"Frozen model changed during diagnostic: {model_name}"
                    )
                model_records[model_name] = {
                    "checkpoint_sha256": sha256(checkpoint),
                    "checkpoint_step": payload["steps"],
                    "pretrained_load": loaded,
                    "state_sha256_before": state_before,
                    "state_sha256_after": state_after,
                    "dinov3_sha256_before": dino_before,
                    "dinov3_sha256_after": dino_after,
                    "rgb140_prediction_replay_exact": len(names),
                    "inference_seconds": timings,
                }
                all_probabilities[model_name] = model_probabilities
                all_predictions[model_name] = model_predictions
                del payload, model
                torch.cuda.empty_cache()

            rows = []
            for name in names:
                row = {"name": name, "roi": ROIS[name], "models": {}}
                panel_predictions = {}
                for model_name in model_specs:
                    row["models"][model_name] = distribution_metrics(
                        all_probabilities[model_name][name]["rgb140_render"],
                        all_probabilities[model_name][name]["prepared_photo"],
                    )
                    panel_predictions[model_name] = all_predictions[model_name][name]
                rows.append(row)
                panels = output / "panels"
                panels.mkdir(exist_ok=True)
                save_panel(
                    panels / f"{Path(name).stem}-full.png",
                    name,
                    inputs[name],
                    panel_predictions,
                )
                save_panel(
                    panels / f"{Path(name).stem}-roi.png",
                    name,
                    inputs[name],
                    panel_predictions,
                    ROIS[name],
                )
            result.update(
                status="completed",
                models=model_records,
                per_view=rows,
                rgb140_replay_exact=True,
                no_optimizer_created=True,
                no_labels_read=True,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
        verify_photo_hashes(binding["diagnostic_photo_hashes"])
        verify_hashes(CONTROL, binding["sem382_sha256"])
        verify_hashes(BASE, binding["sem378_sha256"])
        verify_hashes(args.artifact_root, binding["frozen18"])
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


if __name__ == "__main__":
    main()
