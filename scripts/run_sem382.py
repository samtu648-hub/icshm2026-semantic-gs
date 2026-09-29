#!/usr/bin/env python3
"""Run the SEM382 SFP-LR preflight, training, and checkpoint inference."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import backbone_comparison_stage as shared
from scripts import run_sem381 as parent_runner
from src.quality_dinov3_mask2former import (
    load_dinov3_mask2former,
    module_state_sha256,
    validate_model_dir,
)
from src.quality_protocol import sha256, verify_hashes, write_new_json

EID = "SEM-382-DINOV3-VITB-SFP-LR1E4-ALL300"
PARENT = ROOT / "outputs/quality/SEM-381/r1"
BASE = ROOT / "outputs/quality/SEM-378/r1"
SEM = ROOT / "outputs/quality/all_data/SEM-370-r1"
RUNTIME = ROOT / "outputs/quality/mask2former/SEM-340"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def model_config():
    cfg = parent_runner.model_config()
    cfg.defrost()
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.freeze()
    return cfg


def make_model(config: dict, model_dir: Path):
    cfg = model_config()
    model, loaded = load_dinov3_mask2former(
        cfg,
        RUNTIME / "weights/model_final_6b4a3a.pkl",
        model_dir,
        adapter_seed=config["sfp"]["adapter_seed"],
    )
    return cfg, model, loaded


def parameter_partition(name: str, requires_grad: bool) -> str:
    if name.startswith("backbone.dinov3."):
        return "dinov3"
    if name.startswith("backbone.norms."):
        return "sfp_norm"
    if name.startswith("backbone.adapters."):
        return "sfp_adapter"
    if requires_grad:
        return "decoder"
    raise ValueError(f"Unexpected frozen parameter outside DINOv3: {name}")


def optimizer_contract(model, optimizer, config: dict) -> dict:
    owner = {id(parameter): name for name, parameter in model.named_parameters()}
    optimizer_rows = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter in group["params"]:
            name = owner[id(parameter)]
            if name in optimizer_rows:
                raise ValueError(
                    f"Parameter appears in multiple optimizer groups: {name}"
                )
            optimizer_rows[name] = {
                "group_index": group_index,
                "initial_lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "numel": parameter.numel(),
            }

    partitions = {
        name: {"parameter_tensors": 0, "parameters": 0, "names": []}
        for name in ("sfp_adapter", "sfp_norm", "decoder", "dinov3")
    }
    for name, parameter in model.named_parameters():
        partition = parameter_partition(name, parameter.requires_grad)
        row = partitions[partition]
        row["parameter_tensors"] += 1
        row["parameters"] += parameter.numel()
        row["names"].append(name)
        if partition == "dinov3":
            if parameter.requires_grad or name in optimizer_rows:
                raise ValueError(f"Frozen DINOv3 parameter entered optimizer: {name}")
            continue
        if not parameter.requires_grad or name not in optimizer_rows:
            raise ValueError(f"Trainable parameter missing from optimizer: {name}")
        row.setdefault("initial_lrs", set()).add(optimizer_rows[name]["initial_lr"])
        row.setdefault("weight_decays", set()).add(optimizer_rows[name]["weight_decay"])

    expected = config["expected_parameter_counts"]
    sfp_count = (
        partitions["sfp_adapter"]["parameters"] + partitions["sfp_norm"]["parameters"]
    )
    if sfp_count != expected["sfp"]:
        raise ValueError(f"SFP parameter count mismatch: {sfp_count}")
    if partitions["decoder"]["parameters"] != expected["decoder"]:
        raise ValueError("Decoder parameter count mismatch")
    if partitions["dinov3"]["parameters"] != expected["frozen_dinov3"]:
        raise ValueError("DINOv3 parameter count mismatch")
    if sum(row["parameters"] for row in partitions.values()) != expected["total"]:
        raise ValueError("Total parameter count mismatch")
    expected_lr = config["optimizer"]["sfp_learning_rate"]
    for partition in ("sfp_adapter", "sfp_norm"):
        if partitions[partition]["initial_lrs"] != {expected_lr}:
            raise ValueError(f"{partition} learning-rate mismatch")
    if partitions["decoder"]["initial_lrs"] != {
        config["optimizer"]["decoder_learning_rate"]
    }:
        raise ValueError("Decoder learning-rate mismatch")

    for row in partitions.values():
        row["names"] = sorted(row["names"])
        for key in ("initial_lrs", "weight_decays"):
            if key in row:
                row[key] = sorted(row[key])
    return {
        "optimizer_groups": len(optimizer.param_groups),
        "optimizer_parameter_tensors": len(optimizer_rows),
        "partitions": partitions,
        "all_sfp_in_optimizer": True,
        "all_sfp_norm_in_optimizer": True,
        "all_decoder_in_optimizer": True,
        "dinov3_absent_from_optimizer": True,
    }


def trainable_snapshot(model) -> dict:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def update_statistics(model, initial: dict) -> dict:
    import torch

    accumulators = {
        "sfp": {
            "parameter_tensors": 0,
            "parameters": 0,
            "initial_l2_squared": 0.0,
            "final_l2_squared": 0.0,
            "update_l2_squared": 0.0,
            "max_abs_update": 0.0,
            "changed_tensors": 0,
        },
        "decoder": {
            "parameter_tensors": 0,
            "parameters": 0,
            "initial_l2_squared": 0.0,
            "final_l2_squared": 0.0,
            "update_l2_squared": 0.0,
            "max_abs_update": 0.0,
            "changed_tensors": 0,
        },
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        partition = parameter_partition(name, True)
        group = "sfp" if partition.startswith("sfp_") else "decoder"
        before = initial[name].double()
        after = parameter.detach().cpu().double()
        delta = after - before
        row = accumulators[group]
        row["parameter_tensors"] += 1
        row["parameters"] += parameter.numel()
        row["initial_l2_squared"] += float(torch.sum(before * before))
        row["final_l2_squared"] += float(torch.sum(after * after))
        row["update_l2_squared"] += float(torch.sum(delta * delta))
        row["max_abs_update"] = max(
            row["max_abs_update"], float(torch.max(torch.abs(delta)))
        )
        row["changed_tensors"] += int(not torch.equal(before, after))
    for row in accumulators.values():
        row["initial_l2"] = row.pop("initial_l2_squared") ** 0.5
        row["final_l2"] = row.pop("final_l2_squared") ** 0.5
        row["update_l2"] = row.pop("update_l2_squared") ** 0.5
        row["relative_update_l2"] = row["update_l2"] / row["initial_l2"]
    return accumulators


def partition_learning_rates(model, optimizer) -> dict[str, list[float]]:
    owner = {id(parameter): name for name, parameter in model.named_parameters()}
    values = {"sfp": set(), "decoder": set()}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            partition = parameter_partition(owner[id(parameter)], True)
            key = "sfp" if partition.startswith("sfp_") else "decoder"
            values[key].add(float(group["lr"]))
    return {key: sorted(group_values) for key, group_values in values.items()}


def run_updates(
    args,
    output: Path,
    result: dict,
    config: dict,
    model_dir: Path,
    torch,
    np,
    *,
    steps: int,
) -> None:
    from detectron2.structures import Instances
    from src.quality_mask2former import author_optimizer, class_masks
    from src.quality_mask2former_training import (
        amp_update,
        poly_factor,
        sampled_crop,
        save_checkpoint,
    )
    from src.semantic_cable_binary import read_rgb

    cfg, model, loaded = make_model(config, model_dir)
    (output / "resolved_model.yaml").write_text(cfg.dump())
    expected_resolved = (
        (PARENT / "mask2former/resolved_model.yaml")
        .read_text()
        .replace("BACKBONE_MULTIPLIER: 0.1", "BACKBONE_MULTIPLIER: 1.0")
    )
    if (output / "resolved_model.yaml").read_text() != expected_resolved:
        raise ValueError("Resolved Mask2Former recipe differs from SEM381")
    result["pretrained_load"] = loaded
    result["initial_trainable_state_sha256"] = parent_runner.trainable_state_sha256(
        model
    )
    result["initial_dinov3_state_sha256"] = module_state_sha256(model.backbone.dinov3)
    expected_state = config["expected_initial_state"]
    if result["initial_trainable_state_sha256"] != expected_state["trainable_sha256"]:
        raise ValueError("Initial trainable state differs from SEM381")
    if result["initial_dinov3_state_sha256"] != expected_state["dinov3_sha256"]:
        raise ValueError("Initial DINOv3 state differs from SEM381")
    if steps == config["iterations"]:
        preflight = read(args.batch / "preflight/result.json")
        if (
            preflight["initial_trainable_state_sha256"]
            != result["initial_trainable_state_sha256"]
            or preflight["initial_dinov3_state_sha256"]
            != result["initial_dinov3_state_sha256"]
        ):
            raise ValueError("Formal run initial state differs from preflight")
        result["initial_state_matches_preflight"] = True

    model.cuda().train()
    if model.backbone.dinov3.training or any(
        parameter.requires_grad for parameter in model.backbone.dinov3.parameters()
    ):
        raise ValueError("DINOv3 is not frozen in eval mode")
    optimizer = author_optimizer(model, cfg)
    result["optimizer_contract"] = optimizer_contract(model, optimizer, config)
    initial = trainable_snapshot(model)
    base_rates = [group["lr"] for group in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    labels, names = parent_runner.labels_and_names(args.batch, config)

    class Lazy(dict):
        def __init__(self, loader):
            super().__init__()
            self.loader = loader

        def __missing__(self, key):
            self[key] = self.loader(key)
            return self[key]

    images = Lazy(lambda name: read_rgb(args.batch / "export/images" / name))
    targets = Lazy(labels.read)
    plan = [
        json.loads(line)
        for line in (args.batch / "sampling-plan.jsonl").read_text().splitlines()
    ]
    if len(plan) != config["iterations"]:
        raise ValueError("Incomplete sampling plan")
    rng = random.Random(config["seed"])
    started = time.monotonic()
    result.update(trained_steps=0, amp_skips=0, updates=[] if steps < 6000 else None)
    journal_path = output / "training.jsonl"
    with journal_path.open("x") as journal:
        for step in range(1, steps + 1):
            rgb, target, trace = sampled_crop(
                rng, names, images, targets, config["crop"]
            )
            if trace != plan[step - 1]:
                raise ValueError(f"Sampling mismatch at update {step}")
            ids, masks = class_masks(target)
            instances = Instances((config["crop"], config["crop"]))
            instances.gt_classes = torch.from_numpy(ids)
            instances.gt_masks = torch.from_numpy(masks)
            factor = poly_factor(step - 1)
            for group, rate in zip(optimizer.param_groups, base_rates):
                group["lr"] = rate * factor
            actual_lrs = partition_learning_rates(model, optimizer)
            row = amp_update(
                model,
                optimizer,
                scaler,
                torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float(),
                instances,
                max_attempts=12,
                clip=config["optimizer"]["clip_global_norm"],
            )
            if any(
                parameter.grad is not None
                for parameter in model.backbone.dinov3.parameters()
            ):
                raise ValueError("DINOv3 received gradients")
            row.update(
                step=step,
                sample=trace,
                lr_factor=factor,
                actual_learning_rates=actual_lrs,
                elapsed_seconds=time.monotonic() - started,
            )
            journal.write(json.dumps(row) + "\n")
            result["trained_steps"] = step
            result["amp_skips"] += len(row["skips"])
            if result["updates"] is not None:
                result["updates"].append(row)
            if step == 1 or step % 100 == 0:
                journal.flush()
                print(
                    json.dumps(
                        {
                            "stage": "mask2former",
                            "step": step,
                            "amp_skips": result["amp_skips"],
                            "loss": row["loss"],
                            "sfp_lr": actual_lrs["sfp"],
                            "decoder_lr": actual_lrs["decoder"],
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
            if steps == config["iterations"] and step in config["stage_checkpoints"]:
                save_checkpoint(
                    model,
                    SimpleNamespace(binding=result["binding"], access=result["access"]),
                    step,
                    output / f"model-{step}.pt",
                )
    result["training_seconds"] = time.monotonic() - started
    result["parameter_updates"] = update_statistics(model, initial)
    result["final_trainable_state_sha256"] = parent_runner.trainable_state_sha256(model)
    result["final_dinov3_state_sha256"] = module_state_sha256(model.backbone.dinov3)
    if result["final_dinov3_state_sha256"] != result["initial_dinov3_state_sha256"]:
        raise ValueError("DINOv3 state changed")
    result["dinov3_frozen_exact"] = True
    if steps == config["iterations"]:
        result["updates"] = None
        result["sampling6000_matches_sem381"] = True
        result["artifacts"] = {
            name: sha256(output / name)
            for name in (
                "model-3000.pt",
                "model-6000.pt",
                "resolved_model.yaml",
                "training.jsonl",
            )
        }
    del optimizer, scaler, images, targets, initial, model
    torch.cuda.empty_cache()


def infer(
    args,
    output: Path,
    result: dict,
    config: dict,
    model_dir: Path,
    torch,
    np,
    *,
    checkpoint_step: int,
) -> None:
    from PIL import Image
    from src.quality_mask2former import native_predict
    from src.quality_semantic_audit import independent_confusion, metrics
    from src.semantic_cable_binary import read_rgb

    _, model, loaded = make_model(config, model_dir)
    result["pretrained_load"] = loaded
    checkpoint = args.batch / f"mask2former/model-{checkpoint_step}.pt"
    train = read(checkpoint.parent / "result.json")
    verify_hashes(
        checkpoint.parent, {checkpoint.name: train["artifacts"][checkpoint.name]}
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    if payload["steps"] != checkpoint_step:
        raise ValueError("Checkpoint step mismatch")
    model.cuda().eval()
    result["checkpoint_step"] = checkpoint_step
    result["checkpoint_sha256"] = sha256(checkpoint)
    result["dinov3_state_sha256"] = module_state_sha256(model.backbone.dinov3)
    if result["dinov3_state_sha256"] != train["initial_dinov3_state_sha256"]:
        raise ValueError("Inference DINOv3 state differs from initialization")
    export = read(args.batch / "export/result.json")
    verify_hashes(args.batch / "export/images", export["image_hashes"])
    split = read(ROOT / config["split_path"])
    labeled = set(split["semantic_train"])
    fixed = set(
        read(ROOT / "experiments/runs/SEM-370-ALL300/config.yaml")["diagnostic_views"]
    )
    (output / "predictions").mkdir()
    (output / "probabilities").mkdir()
    total = np.zeros((5, 5), dtype=np.int64)
    rows = []
    times = []
    for index, name in enumerate(sorted(export["image_hashes"])):
        rgb = read_rgb(args.batch / "export/images" / name)
        torch.cuda.synchronize()
        started = time.monotonic()
        probability = native_predict(model, rgb)
        torch.cuda.synchronize()
        times.append(time.monotonic() - started)
        prediction = probability.argmax(0).astype(np.uint8)
        Image.fromarray(prediction).save(output / "predictions" / name)
        if name in fixed:
            np.savez_compressed(
                output / "probabilities" / (Path(name).stem + ".npz"),
                probabilities=probability,
            )
        if name in labeled:
            target = np.asarray(Image.open(SEM / "labels" / name))
            cm = independent_confusion(target, prediction)
            total += cm
            rows.append({"name": name, **metrics(cm)})
        if index % 25 == 0:
            print(
                json.dumps(
                    {
                        "checkpoint": checkpoint_step,
                        "inferred": index + 1,
                        "total": len(export["image_hashes"]),
                    }
                ),
                flush=True,
            )
    if len(rows) != 300:
        raise ValueError("Expected all 300 labeled evaluation views")
    result.update(
        metrics=metrics(total),
        per_view=rows,
        inference_seconds=times,
        inference_median_seconds=float(np.median(times[1:])),
        labeled_count=300,
        unlabeled_count=100,
        no_gt_trajectory_count=144,
        fixed_probability_count=len(fixed),
        dinov3_frozen_exact=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("preflight", "mask2former", "inference-3000", "inference-6000"),
        required=True,
    )
    args = parser.parse_args()
    args.artifact_root = args.artifact_root.resolve()
    args.method = "standard-3dgs"
    config_path = ROOT / "experiments/runs" / EID / "config.yaml"
    config = read(config_path)
    args.batch = ROOT / config["output_path"]
    args.rgb_run = ROOT / config["rgb_run"]
    rgb_eval = ROOT / config["rgb_evaluation"]
    model_dir = Path(os.environ[config["dinov3"]["model_dir_env"]]).resolve()
    binding_path = args.batch / "binding.json"
    binding = read(binding_path)
    export = read(args.batch / "export/result.json")
    output = args.batch / args.stage
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    result = {
        "schema_version": 1,
        "status": "waiting_gpu",
        "experiment_id": EID,
        "rgb_experiment_id": "RGB-140-STANDARD-ABSGRAD-ALL400",
        "stage": args.stage,
        "git_commit": config["git_commit"],
        "seed": config["seed"],
        "dataset_manifest": config["dataset_manifest"],
        "manifest_sha256": binding["manifest_sha256"],
        "split_id": config["split_id"],
        "split_sha256": binding["split_sha256"],
        "config_sha256": binding["config_sha256"],
        "binding_sha256": sha256(binding_path),
        "code_sha256": binding["code_sha256"],
        "quality_scope": "all300 training-view fit; unlabeled100 and trajectory144 qualitative only",
        "started_unix": time.time(),
    }

    def verify() -> None:
        if sha256(config_path) != binding["config_sha256"]:
            raise ValueError("Config changed after binding")
        verify_hashes(ROOT, binding["code_sha256"])
        verify_hashes(args.rgb_run, binding["rgb140_artifacts"])
        if sha256(args.rgb_run / "result.json") != binding["rgb140_result_sha256"]:
            raise ValueError("RGB140 result changed")
        if sha256(rgb_eval / "result.json") != binding["rgb140_evaluation_sha256"]:
            raise ValueError("RGB140 evaluation changed")
        verify_hashes(rgb_eval / "images", binding["rgb140_image_hashes"])
        verify_hashes(args.batch / "export/images", export["image_hashes"])
        verify_hashes(PARENT, binding["sem381_sha256"])
        verify_hashes(BASE, binding["sem378_sha256"])
        verify_hashes(SEM / "labels", binding["label_hashes"])
        if sha256(SEM / "label_manifest.json") != binding["label_manifest_sha256"]:
            raise ValueError("Label manifest changed")
        if (
            sha256(args.batch / "sampling-plan.jsonl")
            != binding["sampling_plan_sha256"]
        ):
            raise ValueError("Sampling plan changed")
        if validate_model_dir(model_dir) != binding["dinov3_source_sha256"]:
            raise ValueError("DINOv3 source changed")
        verify_hashes(args.artifact_root, binding["frozen18"])

    try:
        verify()
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            queued = time.monotonic()
            print(
                "Waiting for shared GPU lock; no process will be interrupted",
                flush=True,
            )
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
                print("Waiting for GPU work outside the shared lock", flush=True)
                time.sleep(30)
            result["queue_seconds"] = time.monotonic() - queued
            result["gpu_processes_before"] = apps
            _, torch, np = shared.setup(args)
            torch.set_num_threads(4)
            torch.cuda.reset_peak_memory_stats()
            result["runtime"] = {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "transformers": importlib.metadata.version("transformers"),
                "gpu": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
            }
            result["binding"] = {
                "config_sha256": binding["config_sha256"],
                "code_sha256": binding["code_sha256"],
                "split_sha256": binding["split_sha256"],
                "label_manifest_sha256": binding["label_manifest_sha256"],
                "export_result_sha256": sha256(args.batch / "export/result.json"),
                "dinov3_source_sha256": binding["dinov3_source_sha256"],
                "seed": config["seed"],
            }
            result["access"] = {
                "train": sorted(read(ROOT / config["split_path"])["semantic_train"]),
                "partition": "all300_training_diagnostic",
                "independent_holdout": False,
            }
            write_new_json(output / "started.json", result)
            if args.stage == "preflight":
                run_updates(args, output, result, config, model_dir, torch, np, steps=8)
                result.update(
                    quality_gate=False,
                    native_768_preflight=True,
                    first8_samples_match_sem381=True,
                )
            elif args.stage == "mask2former":
                if read(args.batch / "preflight/result.json")["status"] != "completed":
                    raise ValueError("Completed preflight required")
                run_updates(
                    args,
                    output,
                    result,
                    config,
                    model_dir,
                    torch,
                    np,
                    steps=config["iterations"],
                )
            else:
                if (
                    read(args.batch / "mask2former/result.json")["status"]
                    != "completed"
                ):
                    raise ValueError("Completed training required")
                checkpoint_step = int(args.stage.split("-")[1])
                infer(
                    args,
                    output,
                    result,
                    config,
                    model_dir,
                    torch,
                    np,
                    checkpoint_step=checkpoint_step,
                )
            result.update(
                status="completed",
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
        verify()
        result.update(
            frozen18_exact=True,
            rgb140_endpoint_unchanged=True,
            rgb544_hashes_unchanged=True,
            sem381_unchanged=True,
            sem378_unchanged=True,
        )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        try:
            import torch

            if torch.cuda.is_initialized():
                result.update(
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                )
        except Exception:
            pass
        write_new_json(output / "result.json", result)


if __name__ == "__main__":
    main()
