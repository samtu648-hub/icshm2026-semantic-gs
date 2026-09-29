#!/usr/bin/env python3
"""Resume SEM437 for a frozen 2k real-label weak-structure tail."""

from __future__ import annotations

import argparse
import fcntl
import json
import random
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter")


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_sem396_spatial as spatial_base
from run_mask2former_long_spatial import save_model_checkpoint, save_resume_checkpoint
from audit_mask2former_long_spatial import boundary_counts, boundary_summary
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.semantic_weak_structure_tail import TRACE_KEYS, WeakStructureTailSampler, tail_plan


EID = "SEM-440-RGB170-WEAK-STRUCTURE-TAIL14K"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
ENDPOINTS = (13000, 14000)
CODE_PATHS = (
    "scripts/run_sem440_rgb170_weak_structure_tail.py",
    "scripts/audit_sem440_rgb170_weak_structure_tail.py",
    "src/semantic_weak_structure_tail.py",
    "src/semantic_context_sampling.py",
    "scripts/run_mask2former_long_spatial.py",
    "src/quality_mask2former.py",
    "src/quality_mask2former_training.py",
)
PALETTE = np.asarray(
    [[0, 0, 0], [230, 86, 37], [255, 210, 0], [38, 139, 210], [133, 153, 0]],
    dtype=np.uint8,
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(config):
    if (
        config["experiment_id"] != EID
        or config["start_step"] != 12000
        or config["iterations"] != 14000
        or config["tail_updates"] != 2000
        or tuple(config["endpoints"]) != ENDPOINTS
        or config["extension_lr_multiplier"] != 0.1
        or config["weak_fraction"] != 0.5
        or tuple(config["sampling_trace_keys"]) != TRACE_KEYS
        or config["weak_resize"]["target_width"] != 512
        or config["weak_resize"]["target_height"] != 384
    ):
        raise ValueError("SEM440 frozen protocol changed")


def gpu_guard():
    applications = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True,
    )
    if any(line.strip() and "ToDesk" not in line for line in applications.splitlines()):
        raise RuntimeError("Other GPU compute process active outside shared lock")
    return applications


def parent_inputs(config):
    export_path = ROOT / config["parent_export_result"]
    if sha256(export_path) != config["parent_export_sha256"]:
        raise ValueError("Parent export result changed")
    export = read(export_path)
    image_root = ROOT / config["rgb_images"]
    verify_hashes(image_root, export["image_hashes"])
    if sha256(ROOT / config["parent_resume_checkpoint"]) != config["parent_resume_sha256"]:
        raise ValueError("Parent resume checkpoint changed")
    if sha256(ROOT / config["parent_inference_result"]) != config["parent_inference_sha256"]:
        raise ValueError("Parent inference result changed")
    return export, image_root


def derive_hard_views(config, train_names):
    root = ROOT / config["label_root"]
    selected = []
    for name in train_names:
        label = np.asarray(Image.open(root / name), dtype=np.uint8)
        if np.count_nonzero(label == 1) > 0 and np.count_nonzero(label == 3) == 0:
            selected.append(name)
    if selected != config["hard_views"]:
        raise ValueError(f"Frozen hard-view pool differs: {selected}")
    return selected


def build_plan(config, train_names):
    native = spatial_base.sampling_plan(train_names, config["seed"], config["iterations"], config["crop"])
    parent = read(ROOT / config["parent_sampling_plan"])["samples"]
    keys = ("name", "top", "left", "flip")
    if [{k: row[k] for k in keys} for row in native[:12000]] != [
        {k: row[k] for k in keys} for row in parent
    ]:
        raise ValueError("Regenerated native prefix differs from SEM437")
    return tail_plan(native[12000:], config["hard_views"], config["tail_seed"])


def prepare(config):
    output = ROOT / config["output_path"]
    if output.exists():
        raise FileExistsError(output)
    export, _ = parent_inputs(config)
    train_names = list(export["semantic_train"])
    derive_hard_views(config, train_names)
    plan = build_plan(config, train_names)
    output.mkdir(parents=True)
    write_new_json(
        output / "sampling-plan.json",
        {
            "schema_version": 1,
            "experiment_id": EID,
            "start_step": 12001,
            "end_step": 14000,
            "tail_seed": config["tail_seed"],
            "samples": plan,
            "mode_counts": dict(Counter(row["mode"] for row in plan)),
            "hard_view_counts": dict(Counter(row["name"] for row in plan if row["mode"] == "weak_full_frame")),
        },
    )
    label_root = ROOT / config["label_root"]
    label_hashes = {name: sha256(label_root / name) for name in train_names}
    write_new_json(
        output / "prepare-result.json",
        {
            "schema_version": 1,
            "status": "completed",
            "experiment_id": EID,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "config_sha256": sha256(CONFIG),
            "sampling_plan_sha256": sha256(output / "sampling-plan.json"),
            "parent_resume_sha256": sha256(ROOT / config["parent_resume_checkpoint"]),
            "parent_inference_sha256": sha256(ROOT / config["parent_inference_result"]),
            "image_hashes": export["image_hashes"],
            "label_hashes": label_hashes,
            "hard_views": config["hard_views"],
            "recipe_frozen_before_new_metrics": True,
            "code_sha256": {path: sha256(ROOT / path) for path in CODE_PATHS if (ROOT / path).is_file()},
        },
    )


def setup_training(config, artifact_root: Path):
    runtime, torch, _ = spatial_base.setup_runtime(artifact_root)
    from src.quality_mask2former import author_optimizer, config_for_model, load_official_model
    from run_sem437_rgb170_context_long import enable_recomputation

    model_config = config_for_model(runtime / "vendor/Mask2Former", config["model_config"], False)
    model_config.defrost()
    model_config.DATASETS.TRAIN = ("sem341_rendered_training",)
    model_config.SOLVER.MAX_ITER = 6000
    model_config.SOLVER.POLY_LR_POWER = 0.9
    model_config.freeze()
    model, loaded = load_official_model(model_config, runtime / "weights/model_final_6b4a3a.pkl")
    enable_recomputation(model)
    model.cuda()
    optimizer = author_optimizer(model, model_config)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    saved = torch.load(ROOT / config["parent_resume_checkpoint"], map_location="cpu", weights_only=True)
    required = {"model", "optimizer", "scaler", "torch_cpu_rng", "torch_cuda_rng", "sampler_rng", "steps", "next_step"}
    if not required.issubset(saved) or saved["steps"] != 12000 or saved["next_step"] != 12001:
        raise ValueError("Parent resume checkpoint is incomplete")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"])
    sampler_rng = random.Random()
    sampler_rng.setstate(saved["sampler_rng"])
    torch.set_rng_state(saved["torch_cpu_rng"])
    torch.cuda.set_rng_state(saved["torch_cuda_rng"])
    restored_rates = sorted({group["lr"] for group in optimizer.param_groups})
    if len(restored_rates) != 2 or not np.allclose(restored_rates, [1e-06, 1e-05], rtol=0, atol=1e-15):
        raise ValueError("Parent optimizer is not at the fixed low learning rate")
    del saved
    return runtime, torch, model_config, model, optimizer, scaler, sampler_rng, loaded


def run_updates(config, artifact_root: Path, output: Path, limit: int | None, preserve: bool):
    export, image_root = parent_inputs(config)
    prepared = read(ROOT / config["output_path"] / "prepare-result.json")
    verify_hashes(ROOT / config["label_root"], prepared["label_hashes"])
    train_names = list(export["semantic_train"])
    plan = read(ROOT / config["output_path"] / "sampling-plan.json")["samples"]
    runtime, torch, model_config, model, optimizer, scaler, sampler_rng, loaded = setup_training(config, artifact_root)
    from detectron2.structures import Instances
    from src.quality_mask2former import class_masks
    from src.quality_mask2former_training import amp_update, sampled_crop
    from src.quality_static_training import TrainingLabels
    from src.semantic_cable_binary import read_rgb

    labels = TrainingLabels(ROOT / config["label_root"], train_names, read(ROOT / config["label_manifest"]), [1320, 989])
    images = {name: read_rgb(image_root / name) for name in train_names}
    targets = {name: labels.read(name) for name in train_names}
    sampler = WeakStructureTailSampler(sampled_crop, plan)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    total = len(plan) if limit is None else limit
    started = time.monotonic()
    journal = output / "training.jsonl"
    with journal.open("x") as stream:
        for local_index in range(total):
            step = 12001 + local_index
            image_np, label_np, trace = sampler(sampler_rng, train_names, images, targets, config["crop"])
            ids, masks = class_masks(label_np)
            instances = Instances(tuple(label_np.shape))
            instances.gt_classes = torch.from_numpy(ids)
            instances.gt_masks = torch.from_numpy(masks)
            image = torch.from_numpy(image_np.transpose(2, 0, 1).copy()).float()
            row = amp_update(model, optimizer, scaler, image, instances, max_attempts=12, clip=0.01)
            row.update(
                step=step,
                sample=trace,
                schedule_phase="fixed_low_lr_weak_structure_tail",
                minimum_group_lr=min(group["lr"] for group in optimizer.param_groups),
                maximum_group_lr=max(group["lr"] for group in optimizer.param_groups),
                elapsed_seconds=time.monotonic() - started,
            )
            stream.write(json.dumps(row) + "\n")
            rows.append(row)
            if step % 100 == 0 or local_index == 0:
                stream.flush()
                print(json.dumps({"experiment_id": EID, "step": step, "mode": trace["mode"], "seconds": time.monotonic() - started}), flush=True)
            if preserve and step in ENDPOINTS:
                binding = {
                    "config_sha256": sha256(CONFIG),
                    "sampling_plan_sha256": sha256(ROOT / config["output_path"] / "sampling-plan.json"),
                    "parent_resume_sha256": config["parent_resume_sha256"],
                }
                access = {"train": train_names, "unlabeled_rgb_allowed_during_fit": False, "trajectory_rgb_allowed_during_fit": False}
                save_model_checkpoint(model, binding, access, step, output / f"model-{step}.pt")
                if step == 14000:
                    save_resume_checkpoint(model, optimizer, scaler, sampler_rng, binding, access, step, output / "resume-14000.pt")
    expected = [{key: row[key] for key in TRACE_KEYS} for row in plan[:total]]
    actual = [row["sample"] for row in rows]
    if actual != expected:
        raise ValueError("Actual SEM440 tail trace differs from frozen plan")
    return {
        "trained_steps": total,
        "last_global_step": 12000 + total,
        "training_seconds": time.monotonic() - started,
        "amp_skips": sum(len(row["skips"]) for row in rows),
        "all_losses_finite": all(np.isfinite(row["loss"]) for row in rows),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "pretrained_load": loaded,
        "restored_model_optimizer_scaler_rng": True,
        "training_labels_accessed": sorted(labels.accessed),
    }


def training_stage(config, artifact_root: Path, preflight: bool):
    batch = ROOT / config["output_path"]
    output = batch / ("preflight" if preflight else "mask2former")
    output.mkdir(exist_ok=False)
    result = {"schema_version": 1, "status": "starting", "experiment_id": EID}
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result["gpu_processes_before"] = gpu_guard()
            result.update(run_updates(config, artifact_root, output, 4 if preflight else None, not preflight))
            if not preflight:
                parent_inputs(config)
                result["checkpoint_sha256"] = {str(step): sha256(output / f"model-{step}.pt") for step in ENDPOINTS}
                result["resume_sha256"] = sha256(output / "resume-14000.pt")
            result["status"] = "completed_discarded" if preflight else "completed_sealed_14000"
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


def hard_metrics(prediction, name):
    from scipy.ndimage import label

    deck = prediction == 1
    components, count = label(deck, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(components.ravel())[1:] if count else np.zeros(0, dtype=np.int64)
    return {
        "name": name,
        "class_pixel_counts": np.bincount(prediction.ravel(), minlength=5).tolist(),
        "deck_pixels": int(deck.sum()),
        "deck_component_count": int(count),
        "deck_largest_component_pixels": int(sizes.max()) if sizes.size else 0,
    }


def make_review(config, endpoint_rows):
    batch = ROOT / config["output_path"]
    review = batch / "review/hard344_358"
    assets = review / "assets"
    assets.mkdir(parents=True)
    parent = ROOT / "outputs/quality/SEM-437/r2/inference/predictions/12000"
    image_root = ROOT / config["rgb_images"]
    cards = []
    for index in range(344, 359):
        name = f"{index:03d}.png"
        figures = []
        rgb = Image.open(image_root / name).convert("RGB")
        rgb.thumbnail((440, 330), Image.Resampling.LANCZOS)
        rgb.save(assets / f"{index}-rgb.jpg", quality=90)
        figures.append(f'<figure><img src="assets/{index}-rgb.jpg"><figcaption>RGB170</figcaption></figure>')
        for label_name, source in (("SEM437-12k", parent), ("SEM440-13k", batch / "inference/predictions/13000"), ("SEM440-14k", batch / "inference/predictions/14000")):
            mask = np.asarray(Image.open(source / name), dtype=np.uint8)
            Image.fromarray(PALETTE[mask], mode="RGB").save(assets / f"{index}-{label_name}.png")
            figures.append(f'<figure><img src="assets/{index}-{label_name}.png"><figcaption>{label_name}</figcaption></figure>')
        details = {step: endpoint_rows[step][name] for step in endpoint_rows}
        cards.append(f'<article><h2>{index}</h2><div class="grid">{"".join(figures)}</div><pre>{json.dumps(details, ensure_ascii=False, indent=2)}</pre></article>')
    html = f'''<!doctype html><meta charset="utf-8"><title>SEM440 final hard-view attempt</title><style>
body{{background:#101318;color:#edf2f7;font:14px system-ui;margin:0}}header{{position:sticky;top:0;background:#101318f2;padding:14px;z-index:2}}main{{padding:12px}}article{{border:1px solid #34404d;border-radius:8px;padding:10px;margin:12px 0}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}figure{{margin:0}}img{{width:100%;display:block;background:#000}}figcaption{{color:#aab4c2}}pre{{white-space:pre-wrap}}
</style><header><h1>SEM440: 12k control vs 13k/14k weak-structure tail</h1><p>344--358无GT，属于已查看的开发诊断，不是准确率。</p></header><main>{''.join(cards)}</main>'''
    (review / "index.html").write_text(html, encoding="utf-8")


def inference(config, artifact_root: Path):
    batch = ROOT / config["output_path"]
    output = batch / "inference"
    output.mkdir(exist_ok=False)
    (output / "predictions").mkdir()
    export, image_root = parent_inputs(config)
    train_names = set(export["semantic_train"])
    training = read(batch / "mask2former/result.json")
    if training["status"] != "completed_sealed_14000":
        raise ValueError("Completed SEM440 training required")
    result = {"schema_version": 1, "status": "starting", "experiment_id": EID, "quality_scope": "all300 training fit; 301-400 and trajectory have no GT"}
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result["gpu_processes_before"] = gpu_guard()
            runtime, torch, _ = spatial_base.setup_runtime(artifact_root)
            from src.quality_mask2former import config_for_model, load_official_model, native_predict
            from src.quality_semantic_audit import independent_confusion, metrics
            from src.semantic_cable_binary import read_rgb

            model_config = config_for_model(runtime / "vendor/Mask2Former", config["model_config"], False)
            model, _ = load_official_model(model_config, runtime / "weights/model_final_6b4a3a.pkl")
            model.cuda().eval()
            torch.cuda.reset_peak_memory_stats()
            endpoint_results = {}
            endpoint_hard = {}
            for step in ENDPOINTS:
                checkpoint = batch / f"mask2former/model-{step}.pt"
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model.load_state_dict(saved["model"])
                del saved
                target_dir = output / "predictions" / str(step)
                target_dir.mkdir()
                matrix = np.zeros((5, 5), dtype=np.int64)
                boundary = np.zeros(4, dtype=np.int64)
                hashes = {}
                per_view = []
                durations = []
                hard = {}
                for index, name in enumerate(sorted(export["image_hashes"])):
                    rgb = read_rgb(image_root / name)
                    torch.cuda.synchronize()
                    stamp = time.monotonic()
                    prediction = native_predict(model, rgb).argmax(0).astype(np.uint8)
                    torch.cuda.synchronize()
                    durations.append(time.monotonic() - stamp)
                    Image.fromarray(prediction).save(target_dir / name)
                    hashes[name] = sha256(target_dir / name)
                    if name in train_names:
                        target = np.asarray(Image.open(ROOT / config["label_root"] / name), dtype=np.uint8)
                        one = independent_confusion(target, prediction)
                        matrix += one
                        boundary += boundary_counts(target, prediction)
                        per_view.append({"name": name, **metrics(one)})
                    if name.endswith(".png") and name[:3].isdigit() and 344 <= int(name[:3]) <= 358:
                        hard[name] = hard_metrics(prediction, name)
                    if index % 25 == 0:
                        print(json.dumps({"experiment_id": EID, "endpoint": step, "inferred": index + 1, "total": len(export["image_hashes"])}), flush=True)
                endpoint_results[str(step)] = {
                    "checkpoint_sha256": sha256(checkpoint),
                    "prediction_hashes": hashes,
                    "inference_median_seconds": float(np.median(durations[1:])),
                    "metrics": metrics(matrix),
                    "boundary_metrics": boundary_summary(boundary),
                    "per_view": per_view,
                }
                endpoint_hard[str(step)] = hard
            parent = read(ROOT / config["parent_inference_result"])["endpoints"]["12000"]
            gate = config["retention_gate"]
            protection = {}
            for step in map(str, ENDPOINTS):
                current = endpoint_results[step]
                drops = {
                    "five_class_miou_pp": 100 * (parent["metrics"]["mIoU_with_background"] - current["metrics"]["mIoU_with_background"]),
                    "foreground_miou_pp": 100 * (parent["metrics"]["mIoU_without_background"] - current["metrics"]["mIoU_without_background"]),
                    "boundary_f1_pp": 100 * (parent["boundary_metrics"]["boundary_f1"] - current["boundary_metrics"]["boundary_f1"]),
                    "per_class_iou_pp": {name: 100 * (parent["metrics"]["per_class"][name]["IoU"] - current["metrics"]["per_class"][name]["IoU"]) for name in ("deck", "stay_cable", "tower", "foundation")},
                }
                fit_pass = (
                    drops["five_class_miou_pp"] <= gate["maximum_five_class_miou_drop_pp"]
                    and drops["foreground_miou_pp"] <= gate["maximum_foreground_miou_drop_pp"]
                    and drops["boundary_f1_pp"] <= gate["maximum_boundary_f1_drop_pp"]
                    and max(drops["per_class_iou_pp"].values()) <= gate["maximum_per_foreground_class_iou_drop_pp"]
                )
                view357 = endpoint_hard[step]["357.png"]
                protection[step] = {
                    "drops_vs_sem437_12k_pp": drops,
                    "labeled_fit_protection_pass": fit_pass,
                    "view357_quantitative_target_pass": view357["deck_pixels"] >= gate["required_357_deck_pixels"] and view357["deck_largest_component_pixels"] >= gate["required_357_deck_largest_component_pixels"],
                    "visual_plausibility_requires_manual_review": True,
                }
            parent_inputs(config)
            result.update(
                status="completed",
                endpoints=endpoint_results,
                hard344_358=endpoint_hard,
                gates=protection,
                frozen_rgb_hashes_exact=True,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
            make_review(config, endpoint_hard)
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "preflight", "train", "inference"), required=True)
    args = parser.parse_args()
    config = read(CONFIG)
    validate(config)
    if args.stage == "prepare":
        prepare(config)
    elif args.stage == "preflight":
        training_stage(config, args.artifact_root.resolve(), True)
    elif args.stage == "train":
        if read(ROOT / config["output_path"] / "preflight/result.json")["status"] != "completed_discarded":
            raise ValueError("Successful discarded preflight required")
        training_stage(config, args.artifact_root.resolve(), False)
    else:
        inference(config, args.artifact_root.resolve())


if __name__ == "__main__":
    main()
