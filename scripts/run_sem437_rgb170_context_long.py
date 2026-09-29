#!/usr/bin/env python3
"""SEM437: RGB170 all300 long training with a fixed 50% context mix."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_sem436_rgb170_long_all300 as base
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.semantic_context_sampling import FullFrameContextSampler, TRACE_KEYS, context_plan


EID = "SEM-437-RGB170-MASK2FORMER-CONTEXT50-LONG12K-ALL300"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
CODE_PATHS = (
    "scripts/run_sem437_rgb170_context_long.py",
    "scripts/run_sem436_rgb170_long_all300.py",
    "scripts/run_mask2former_long_spatial.py",
    "scripts/run_sem396_spatial.py",
    "src/semantic_context_sampling.py",
    "src/quality_mask2former.py",
    "src/quality_mask2former_training.py",
    "src/quality_static_training.py",
    "src/quality_semantic_audit.py",
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def enable_recomputation(model):
    layers = list(model.backbone.layers)
    if len(layers) != 4 or any(not hasattr(layer, "use_checkpoint") for layer in layers):
        raise ValueError("Expected four native Swin stages")
    for layer in layers:
        layer.use_checkpoint = True
    return {
        "implementation": "native_swin_basiclayer_checkpoint",
        "stages": 4,
        "blocks": [len(layer.blocks) for layer in layers],
        "runtime_override": "backbone.layers[*].use_checkpoint=True",
        "base_resolved_yaml_unchanged": True,
        "inference_uses_native_noncheckpoint_model": True,
    }


@contextmanager
def identity_and_policy(config, artifact_root: Path, stage: str):
    base.setup_runtime = base.spatial_base.setup_runtime
    base.spatial_base.setup_runtime(artifact_root)
    import src.quality_mask2former as api

    original = {
        "eid": base.EID,
        "config": base.CONFIG,
        "code": base.CODE_PATHS,
        "plan": base.build_sampling_plan,
        "keys": base.sampling_trace_keys,
        "sampler": base.make_training_sampler,
        "loader": api.load_official_model,
    }
    policy = {"experiment_id": EID, "stage": stage, "model_loads": 0}

    def loader(*args, **kwargs):
        model, loaded = original["loader"](*args, **kwargs)
        if stage in {"preflight", "train"}:
            policy.update(enable_recomputation(model))
        policy["model_loads"] += 1
        return model, loaded

    def build_plan(names, seed, iterations, crop):
        parent = base.spatial_base.sampling_plan(names, seed, iterations, crop)
        return context_plan(parent)

    base.EID, base.CONFIG, base.CODE_PATHS = EID, CONFIG, CODE_PATHS
    base.build_sampling_plan = build_plan
    base.sampling_trace_keys = lambda _: TRACE_KEYS
    base.make_training_sampler = lambda parent, _: FullFrameContextSampler(parent)
    api.load_official_model = loader
    try:
        yield policy
    finally:
        api.load_official_model = original["loader"]
        base.EID, base.CONFIG, base.CODE_PATHS = original["eid"], original["config"], original["code"]
        base.build_sampling_plan = original["plan"]
        base.sampling_trace_keys = original["keys"]
        base.make_training_sampler = original["sampler"]
        if stage in {"preflight", "train"}:
            target = ROOT / config["output_path"] / ("preflight" if stage == "preflight" else "mask2former")
            if target.exists() and not (target / "memory-policy.json").exists():
                write_new_json(target / "memory-policy.json", policy)


def validate(config):
    if (
        config["experiment_id"] != EID
        or config["iterations"] != 12000
        or config["context_fraction"] != 0.5
        or tuple(config["sampling_trace_keys"]) != TRACE_KEYS
        or not config["memory_policy"]["swin_activation_recomputation"]
    ):
        raise ValueError("SEM437 frozen identity or context contract changed")
    original = base.EID
    base.EID = EID
    try:
        base.validate_config(config)
    finally:
        base.EID = original


def source_probe(config, artifact_root: Path):
    from PIL import Image

    batch = ROOT / config["output_path"]
    output = batch / config.get("source_probe_output", "source-probe")
    output.mkdir(exist_ok=False)
    (output / "photo").mkdir()
    (output / "render").mkdir()
    names = [f"{index:03d}.png" for index in range(344, 359)]
    photo_root = ROOT / config["photo_images"]
    render_root = ROOT / config["rgb_images"]
    result = {
        "schema_version": 1,
        "status": "starting",
        "experiment_id": EID,
        "scope": "344-358 original-photo versus frozen RGB170 input probe; no GT and no recipe selection",
        "checkpoint": config["source_probe_checkpoint"],
        "names": names,
    }
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            applications = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True
            )
            if any(line.strip() and "ToDesk" not in line for line in applications.splitlines()):
                raise RuntimeError("Other GPU compute process active outside shared lock")
            runtime, torch, np = base.spatial_base.setup_runtime(artifact_root)
            from src.quality_mask2former import config_for_model, load_official_model, native_predict
            from src.semantic_cable_binary import read_rgb

            model_config = config_for_model(
                runtime / "vendor/Mask2Former",
                "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
                False,
            )
            model, _ = load_official_model(model_config, runtime / "weights/model_final_6b4a3a.pkl")
            saved = torch.load(ROOT / config["source_probe_checkpoint"], map_location="cpu", weights_only=True)
            model.load_state_dict(saved["model"])
            model.cuda().eval()
            rows = []
            for name in names:
                predictions = {}
                hashes = {}
                counts = {}
                for domain, root in (("photo", photo_root), ("render", render_root)):
                    source = root / name
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    prediction = native_predict(model, read_rgb(source)).argmax(0).astype(np.uint8)
                    target = output / domain / name
                    Image.fromarray(prediction).save(target)
                    predictions[domain] = prediction
                    hashes[domain] = {"input": sha256(source), "prediction": sha256(target)}
                    counts[domain] = np.bincount(prediction.ravel(), minlength=5).tolist()
                rows.append({
                    "name": name,
                    "changed_fraction": float(np.mean(predictions["photo"] != predictions["render"])),
                    "class_pixel_counts": counts,
                    "sha256": hashes,
                })
            result.update(
                status="completed_no_gt",
                rows=rows,
                mean_changed_fraction=float(np.mean([row["changed_fraction"] for row in rows])),
                checkpoint_sha256=sha256(ROOT / config["source_probe_checkpoint"]),
            )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


def preflight(config, artifact_root: Path):
    from src.quality_static_training import TrainingLabels

    batch = ROOT / config["output_path"]
    output = batch / "preflight"
    output.mkdir(exist_ok=False)
    result = {"schema_version": 1, "status": "starting", "experiment_id": EID, "trained_steps": 0}
    started = time.monotonic()
    try:
        export = read(batch / "export/result.json")
        prepared = read(batch / "prepare-result.json")
        plan = read(batch / "sampling-plan.json")
        verify_hashes(batch / "export/images", export["image_hashes"])
        label_root = ROOT / config["label_root"]
        verify_hashes(label_root, prepared["training_label_hashes"])
        train_names = list(export["semantic_train"])
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            applications = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True
            )
            if any(line.strip() and "ToDesk" not in line for line in applications.splitlines()):
                raise RuntimeError("Other GPU compute process active outside shared lock")
            runtime, torch, np = base.spatial_base.setup_runtime(artifact_root)
            from detectron2.structures import Instances
            from src.quality_mask2former import author_optimizer, class_masks, config_for_model, load_official_model
            from src.quality_mask2former_training import amp_update, sampled_crop
            from src.semantic_cable_binary import read_rgb

            random.seed(config["seed"])
            np.random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats()
            labels = TrainingLabels(label_root, train_names, read(ROOT / config["label_manifest"]), [1320, 989])
            model_config = config_for_model(
                runtime / "vendor/Mask2Former",
                "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
                False,
            )
            model_config.defrost()
            model_config.DATASETS.TRAIN = ("sem341_rendered_training",)
            model_config.SOLVER.MAX_ITER = 6000
            model_config.SOLVER.POLY_LR_POWER = 0.9
            model_config.freeze()
            (output / "resolved_model.yaml").write_text(model_config.dump())
            reference = ROOT / config["recipe_reference_run"] / "mask2former/resolved_model.yaml"
            if (output / "resolved_model.yaml").read_bytes() != reference.read_bytes():
                raise ValueError("Resolved model recipe differs from SEM436")
            model, loaded = load_official_model(model_config, runtime / "weights/model_final_6b4a3a.pkl")
            initial_state = base.model_state_sha256(model.state_dict())
            if initial_state != read(ROOT / config["recipe_reference_run"] / "mask2former/result.json")["initial_model_state_sha256"]:
                raise ValueError("Fresh ADE initialization differs from SEM436")
            model.cuda().train()
            optimizer = author_optimizer(model, model_config)
            rates = [group["lr"] for group in optimizer.param_groups]
            scaler = torch.amp.GradScaler("cuda", init_scale=1024)
            images = {name: read_rgb(batch / "export/images" / name) for name in train_names}
            targets = {name: labels.read(name) for name in train_names}
            sampler = FullFrameContextSampler(sampled_crop)
            rng = random.Random(config["seed"])
            rows = []
            with (output / "training.jsonl").open("x") as journal:
                for step in range(1, 9):
                    image, target, trace = sampler(rng, train_names, images, targets, config["crop"])
                    ids, masks = class_masks(target)
                    instances = Instances(tuple(target.shape))
                    instances.gt_classes = torch.from_numpy(ids)
                    instances.gt_masks = torch.from_numpy(masks)
                    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
                    _, factor = base.phase_lr_factor(step, config["extension_lr_multiplier"])
                    for group, rate in zip(optimizer.param_groups, rates):
                        group["lr"] = rate * factor
                    row = amp_update(model, optimizer, scaler, tensor, instances, max_attempts=12, clip=0.01)
                    row.update(step=step, sample=trace, lr_factor=factor)
                    journal.write(json.dumps(row) + "\n")
                    rows.append(row)
                    result["trained_steps"] = step
            expected = [{key: row[key] for key in TRACE_KEYS} for row in plan["samples"][:8]]
            actual = [row["sample"] for row in rows]
            modes = Counter(row["mode"] for row in actual)
            if actual != expected or modes != {"native_crop": 4, "full_frame_context": 4}:
                raise ValueError("Eight-step context trace differs from frozen plan")
            result.update(
                status="completed_fresh_model_discarded",
                mode_counts=dict(sorted(modes.items())),
                initial_model_state_sha256=initial_state,
                pretrained_load=loaded,
                amp_skips=sum(len(row["skips"]) for row in rows),
                all_losses_finite=all(np.isfinite(row["loss"]) for row in rows),
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
    parser.add_argument("--stage", choices=("prepare", "source-probe", "preflight", "train", "inference"), required=True)
    args = parser.parse_args()
    config = read(CONFIG)
    validate(config)
    artifact_root = args.artifact_root.resolve()
    if args.stage == "prepare":
        with identity_and_policy(config, artifact_root, "prepare"):
            base.prepare(config)
    elif args.stage == "source-probe":
        source_probe(config, artifact_root)
    elif args.stage == "preflight":
        with identity_and_policy(config, artifact_root, "preflight"):
            preflight(config, artifact_root)
    elif args.stage == "train":
        preflight_result = read(ROOT / config["output_path"] / "preflight/result.json")
        if preflight_result["status"] != "completed_fresh_model_discarded":
            raise ValueError("Successful discarded-model preflight is required")
        with identity_and_policy(config, artifact_root, "train"):
            base.train(config, artifact_root)
    else:
        with identity_and_policy(config, artifact_root, "inference"):
            base.inference(config, artifact_root)


if __name__ == "__main__":
    main()
