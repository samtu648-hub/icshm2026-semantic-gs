#!/usr/bin/env python3
"""Run SEM381 native-768 preflight, all300 training or 544-view inference."""

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
from src.quality_dinov3_mask2former import (
    load_dinov3_mask2former,
    module_state_sha256,
    validate_model_dir,
)
from src.quality_protocol import sha256, verify_hashes, write_new_json

EID = "SEM-381-DINOV3-VITB-SFP-ALL300"
BASE = ROOT / "outputs/quality/SEM-378/r1"
SEM = ROOT / "outputs/quality/all_data/SEM-370-r1"
RUNTIME = ROOT / "outputs/quality/mask2former/SEM-340"
UPSTREAM_CONFIG = "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def model_config():
    from src.quality_mask2former import config_for_model

    cfg = config_for_model(RUNTIME / "vendor/Mask2Former", UPSTREAM_CONFIG, False)
    cfg.defrost()
    cfg.DATASETS.TRAIN = ("sem341_rendered_training",)
    cfg.SOLVER.MAX_ITER = 6000
    cfg.SOLVER.POLY_LR_POWER = 0.9
    cfg.freeze()
    return cfg


def trainable_state_sha256(model) -> str:
    import hashlib
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if not value.requires_grad:
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_model(config: dict, model_dir: Path):
    cfg = model_config()
    model, loaded = load_dinov3_mask2former(
        cfg,
        RUNTIME / "weights/model_final_6b4a3a.pkl",
        model_dir,
        adapter_seed=config["sfp"]["adapter_seed"],
    )
    return cfg, model, loaded


def labels_and_names(batch: Path, config: dict):
    from src.quality_static_training import TrainingLabels

    split = read(ROOT / config["split_path"])
    labels = TrainingLabels(
        SEM / "labels",
        split["semantic_train"],
        read(SEM / "label_manifest.json"),
        [1320, 989],
    )
    assert len(labels.allowed) == 300
    return labels, sorted(labels.allowed)


def optimizer_audit(optimizer) -> dict:
    by_lr = {}
    for group in optimizer.param_groups:
        key = format(group["lr"], ".12g")
        by_lr[key] = by_lr.get(key, 0) + sum(p.numel() for p in group["params"])
    return {
        "parameter_count_by_initial_lr": by_lr,
        "groups": len(optimizer.param_groups),
    }


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
    from src.quality_dinov3_mask2former import module_state_sha256
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
    assert (output / "resolved_model.yaml").read_bytes() == (
        BASE / "mask2former/resolved_model.yaml"
    ).read_bytes()
    result["pretrained_load"] = loaded
    result["initial_trainable_state_sha256"] = trainable_state_sha256(model)
    result["initial_dinov3_state_sha256"] = module_state_sha256(model.backbone.dinov3)
    result["mask2former_recipe_matches_sem378"] = True
    if steps == 6000:
        preflight = read(args.batch / "preflight/result.json")
        assert (
            result["initial_trainable_state_sha256"]
            == preflight["initial_trainable_state_sha256"]
        )
        assert (
            result["initial_dinov3_state_sha256"]
            == preflight["initial_dinov3_state_sha256"]
        )
        result["initial_state_matches_preflight"] = True
    model.cuda().train()
    assert not model.backbone.dinov3.training
    assert not any(p.requires_grad for p in model.backbone.dinov3.parameters())
    optimizer = author_optimizer(model, cfg)
    rates = [group["lr"] for group in optimizer.param_groups]
    result["optimizer"] = optimizer_audit(optimizer)
    expected_lrs = {
        format(config["optimizer"]["decoder_learning_rate"], ".12g"),
        format(config["optimizer"]["sfp_learning_rate"], ".12g"),
    }
    assert set(result["optimizer"]["parameter_count_by_initial_lr"]) == expected_lrs
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    labels, names = labels_and_names(args.batch, config)

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
    assert len(plan) == 6000
    rng = random.Random(config["seed"])
    model.train()
    started = time.monotonic()
    result.update(trained_steps=0, amp_skips=0, updates=[] if steps < 6000 else None)
    journal_path = output / "training.jsonl"
    with journal_path.open("x") as journal:
        for step in range(1, steps + 1):
            rgb, target, trace = sampled_crop(
                rng, names, images, targets, config["crop"]
            )
            assert trace == plan[step - 1]
            ids, masks = class_masks(target)
            instances = Instances((config["crop"], config["crop"]))
            instances.gt_classes = torch.from_numpy(ids)
            instances.gt_masks = torch.from_numpy(masks)
            factor = poly_factor(step - 1)
            for group, rate in zip(optimizer.param_groups, rates):
                group["lr"] = rate * factor
            row = amp_update(
                model,
                optimizer,
                scaler,
                torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float(),
                instances,
                max_attempts=12,
                clip=config["optimizer"]["clip_global_norm"],
            )
            assert all(
                parameter.grad is None
                for parameter in model.backbone.dinov3.parameters()
            )
            row.update(
                step=step,
                sample=trace,
                lr_factor=factor,
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
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
            if steps == 6000 and step in config["stage_checkpoints"]:
                save_checkpoint(
                    model,
                    SimpleNamespace(binding=result["binding"], access=result["access"]),
                    step,
                    output / f"model-{step}.pt",
                )
    result["training_seconds"] = time.monotonic() - started
    result["final_dinov3_state_sha256"] = module_state_sha256(model.backbone.dinov3)
    assert result["final_dinov3_state_sha256"] == result["initial_dinov3_state_sha256"]
    result["dinov3_frozen_exact"] = True
    if steps == 6000:
        result["updates"] = None
        result["sampling6000_matches_sem378"] = True
        result["artifacts"] = {
            name: sha256(output / name)
            for name in (
                "model-3000.pt",
                "model-6000.pt",
                "resolved_model.yaml",
                "training.jsonl",
            )
        }
    del optimizer, scaler, images, targets, model
    torch.cuda.empty_cache()


def infer(
    args, output: Path, result: dict, config: dict, model_dir: Path, torch, np
) -> None:
    from PIL import Image
    from src.quality_mask2former import native_predict
    from src.quality_semantic_audit import independent_confusion, metrics
    from src.semantic_cable_binary import read_rgb

    _, model, loaded = make_model(config, model_dir)
    result["pretrained_load"] = loaded
    checkpoint = args.batch / "mask2former/model-6000.pt"
    train = read(checkpoint.parent / "result.json")
    verify_hashes(
        checkpoint.parent, {checkpoint.name: train["artifacts"][checkpoint.name]}
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    assert payload["steps"] == 6000
    model.cuda().eval()
    result["checkpoint_sha256"] = sha256(checkpoint)
    result["dinov3_state_sha256"] = module_state_sha256(model.backbone.dinov3)
    assert result["dinov3_state_sha256"] == train["initial_dinov3_state_sha256"]
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
                    {"inferred": index + 1, "total": len(export["image_hashes"])}
                ),
                flush=True,
            )
    assert len(rows) == 300
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
        "--stage", choices=("preflight", "mask2former", "inference"), required=True
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
        assert sha256(config_path) == binding["config_sha256"]
        assert sha256(binding_path) == result["binding_sha256"]
        verify_hashes(ROOT, binding["code_sha256"])
        verify_hashes(args.rgb_run, binding["rgb140_artifacts"])
        assert sha256(args.rgb_run / "result.json") == binding["rgb140_result_sha256"]
        assert sha256(rgb_eval / "result.json") == binding["rgb140_evaluation_sha256"]
        verify_hashes(rgb_eval / "images", binding["rgb140_image_hashes"])
        verify_hashes(args.batch / "export/images", export["image_hashes"])
        verify_hashes(BASE, binding["sem378_sha256"])
        verify_hashes(SEM / "labels", binding["label_hashes"])
        assert sha256(SEM / "label_manifest.json") == binding["label_manifest_sha256"]
        assert (
            sha256(args.batch / "sampling-plan.jsonl")
            == binding["sampling_plan_sha256"]
        )
        assert validate_model_dir(model_dir) == binding["dinov3_source_sha256"]
        assert (
            sha256(model_dir / "download_manifest.json")
            == binding["dinov3_source_manifest_sha256"]
        )
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
                    first8_samples_match_sem378=True,
                )
            elif args.stage == "mask2former":
                assert (
                    read(args.batch / "preflight/result.json")["status"] == "completed"
                )
                run_updates(
                    args, output, result, config, model_dir, torch, np, steps=6000
                )
            else:
                assert (
                    read(args.batch / "mask2former/result.json")["status"]
                    == "completed"
                )
                infer(args, output, result, config, model_dir, torch, np)
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
