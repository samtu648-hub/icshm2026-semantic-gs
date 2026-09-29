#!/usr/bin/env python3
"""Run the fixed SEM385 50 percent full-frame context-mix experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_sem384_dev270 as shared
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_seed_initialization import (
    tensor_manifest,
    validate_cross_seed_reference,
    validate_same_seed_pair,
)
from src.semantic_context_sampling import (
    TRACE_KEYS,
    FullFrameContextSampler,
    context_plan,
)


EID = "SEM-385-RGB150-DEV270-CONTEXT50"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
PARENT = ROOT / "outputs/quality/SEM-384/r1"
CODE_PATHS = (
    "scripts/run_sem385_context_mix.py",
    "scripts/run_sem384_dev270.py",
    "scripts/run_quality_all300.py",
    "src/semantic_context_sampling.py",
    "src/quality_mask2former.py",
    "src/quality_mask2former_training.py",
    "src/quality_static_training.py",
    "src/quality_semantic_audit.py",
    "src/quality_seed_initialization.py",
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def model_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def shared_identity():
    previous = shared.EID, shared.CONFIG, shared.CODE_PATHS
    shared.EID, shared.CONFIG, shared.CODE_PATHS = EID, CONFIG, CODE_PATHS
    try:
        yield
    finally:
        shared.EID, shared.CONFIG, shared.CODE_PATHS = previous


def prepare(config: dict[str, Any]) -> None:
    split_path, rgb_run, rgb_evaluation, label_root, batch = shared.registered_paths(
        config
    )
    if batch.exists():
        raise FileExistsError(f"SEM385 output already exists: {batch}")
    if config.get("parent_completion_required_for_prepare", True):
        parent_training = read(PARENT / "mask2former/result.json")
        parent_inference = read(PARENT / "inference/result.json")
        if (
            parent_training["status"] != "completed_sealed_6000"
            or parent_inference["status"] != "completed"
        ):
            raise ValueError("Completed sealed parent artifacts are required")
        verify_hashes(PARENT / "mask2former", parent_training["artifacts"])

    split = read(split_path)
    fold = split["primary_fold"]
    train_names = sorted(fold["semantic_train_labeled"])
    holdout_names = sorted(fold["semantic_validation_labeled"])
    if (
        len(train_names) != config["semantic_train_views"]
        or len(holdout_names) != config["semantic_holdout_views"]
        or set(train_names) & set(holdout_names)
    ):
        raise ValueError("Expected the registered disjoint semantic 270/30 roles")

    manifest_path = ROOT / config["label_manifest"]
    manifest = read(manifest_path)
    records = {record["image"]: record for record in manifest["records"]}
    label_hashes = {name: sha256(label_root / name) for name in train_names}
    if any(
        label_hashes[name] != records[name]["label_png_sha256"] for name in train_names
    ):
        raise ValueError("A semantic training label differs from the frozen manifest")

    parent_plan_path = ROOT / config["parent_sampling_plan"]
    if sha256(parent_plan_path) != config["parent_sampling_plan_sha256"]:
        raise ValueError("SEM384 parent sampling plan hash changed")
    parent_plan = read(parent_plan_path)
    if (
        parent_plan["train_names"] != train_names
        or parent_plan["holdout_names_excluded"] != holdout_names
        or parent_plan["steps"] != config["iterations"]
    ):
        raise ValueError("SEM384 parent sampling roles changed")
    samples = context_plan(parent_plan["samples"])
    mode_counts = Counter(str(row["mode"]) for row in samples)
    if mode_counts != {"native_crop": 3000, "full_frame_context": 3000}:
        raise ValueError("SEM385 context plan is not exactly 3000/3000")

    parent_export = read(PARENT / "export/result.json")
    verify_hashes(PARENT / "export/images", parent_export["image_hashes"])
    rgb_parent = read(rgb_run / "result.json")
    rgb_result = read(rgb_evaluation / "result.json")
    if (
        rgb_parent["status"] != "completed_dev30_training"
        or rgb_result["status"] != "completed"
    ):
        raise ValueError("Sealed RGB150 artifacts are required")
    training_rgb_names = sorted(fold["train_rgb_400"])
    if sorted(parent_export["image_hashes"]) != training_rgb_names:
        raise ValueError("SEM384 export does not contain the 370 RGB training role")

    batch.mkdir(parents=True)
    export = batch / "export"
    export.mkdir()
    relative_images = os.path.relpath(PARENT / "export/images", export)
    (export / "images").symlink_to(relative_images, target_is_directory=True)
    verify_hashes(export / "images", parent_export["image_hashes"])
    source_parent_export_key = config.get(
        "parent_export_binding_key", "source_sem384_export_result_sha256"
    )
    write_new_json(
        export / "result.json",
        {
            "schema_version": 1,
            "status": config.get(
                "export_reuse_status",
                "completed_reusing_verified_sem384_rgb_export",
            ),
            "experiment_id": EID,
            "rgb_experiment_id": rgb_parent["experiment_id"],
            "split_id": config["split_id"],
            "split_sha256": sha256(split_path),
            source_parent_export_key: sha256(PARENT / "export/result.json"),
            "source_rgb_result_sha256": sha256(rgb_run / "result.json"),
            "source_rgb_evaluation_sha256": sha256(rgb_evaluation / "result.json"),
            "image_hashes": parent_export["image_hashes"],
            "image_count": len(parent_export["image_hashes"]),
            "semantic_train": train_names,
            "semantic_holdout": holdout_names,
            "holdout_rgb_accessed_during_prepare": False,
            "holdout_labels_accessed_during_prepare": False,
        },
    )

    plan_path = batch / "sampling-plan.json"
    write_new_json(
        plan_path,
        {
            "schema_version": 1,
            "experiment_id": EID,
            "seed": config["seed"],
            "steps": config["iterations"],
            "crop": config["crop"],
            "context_schedule": config["context_schedule"],
            "context_resize": config["context_resize"],
            "parent_sampling_plan_sha256": sha256(parent_plan_path),
            "trace_keys": list(TRACE_KEYS),
            "train_names": train_names,
            "holdout_names_excluded": holdout_names,
            "samples": samples,
            "mode_counts": dict(sorted(mode_counts.items())),
            "parent_view_crop_flip_exact": True,
        },
    )
    write_new_json(
        batch / "prepare-result.json",
        {
            "schema_version": 1,
            "status": "completed_not_trained",
            "experiment_id": EID,
            "config_sha256": sha256(CONFIG),
            "split_sha256": sha256(split_path),
            "export_result_sha256": sha256(export / "result.json"),
            "sampling_plan_sha256": sha256(plan_path),
            "parent_sampling_plan_sha256": sha256(parent_plan_path),
            "label_manifest_sha256": sha256(manifest_path),
            "training_label_hashes": label_hashes,
            "semantic_train_count": len(train_names),
            "semantic_holdout_count": len(holdout_names),
            "holdout_label_hashes_read": 0,
            "mode_counts": dict(sorted(mode_counts.items())),
            "code_sha256": {path: sha256(ROOT / path) for path in CODE_PATHS},
        },
    )


def preflight(config: dict[str, Any], artifact_root: Path) -> None:
    split_path, _, _, label_root, batch = shared.registered_paths(config)
    output = batch / "preflight"
    output.mkdir(exist_ok=False)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "experiment_id": EID,
        "stage": "fresh_model_eight_update_preflight",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": sha256(CONFIG),
        "split_sha256": sha256(split_path),
        "seed": config["seed"],
        "trained_steps": 0,
        "preflight_checkpoint_retained": False,
        "code_sha256": {path: sha256(ROOT / path) for path in CODE_PATHS},
    }
    started = time.monotonic()
    try:
        prepare_result = read(batch / "prepare-result.json")
        plan_record = read(batch / "sampling-plan.json")
        if (
            prepare_result["status"] not in {"completed", "completed_not_trained"}
            or sha256(batch / "sampling-plan.json")
            != prepare_result["sampling_plan_sha256"]
        ):
            raise ValueError("Completed immutable preparation is required")
        export = read(batch / "export/result.json")
        verify_hashes(batch / "export/images", export["image_hashes"])
        verify_hashes(label_root, prepare_result["training_label_hashes"])
        split = read(split_path)
        train_names = sorted(split["primary_fold"]["semantic_train_labeled"])
        holdout_names = set(split["primary_fold"]["semantic_validation_labeled"])

        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            queued = time.monotonic()
            fcntl.flock(lock, fcntl.LOCK_EX)
            result["queue_seconds"] = time.monotonic() - queued
            applications = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            result["gpu_processes_before"] = applications
            if any(
                line.strip() and "ToDesk" not in line
                for line in applications.splitlines()
            ):
                raise RuntimeError(
                    "Other GPU compute process active outside the shared lock"
                )

            runtime, torch, np = shared.setup_runtime(artifact_root)
            from detectron2.structures import Instances
            from scripts.run_quality_all300 import (
                mask2former as _mask2former_import_gate,
            )
            from src.quality_mask2former import (
                author_optimizer,
                class_masks,
                config_for_model,
                load_official_model,
            )
            from src.quality_mask2former_training import (
                amp_update,
                poly_factor,
                sampled_crop,
            )
            from src.quality_static_training import TrainingLabels
            from src.semantic_cable_binary import read_rgb

            del _mask2former_import_gate
            torch.set_num_threads(4)

            def reset_model_rng(seed: int) -> None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            reset_model_rng(config["seed"])
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats()

            labels = TrainingLabels(
                label_root,
                train_names,
                read(ROOT / config["label_manifest"]),
                [1320, 989],
            )
            images = {
                name: read_rgb(batch / "export/images" / name) for name in train_names
            }
            targets = {name: labels.read(name) for name in train_names}
            model_config = config_for_model(
                runtime / "vendor/Mask2Former",
                "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
                False,
            )
            model_config.defrost()
            model_config.DATASETS.TRAIN = ("sem341_rendered_training",)
            model_config.SOLVER.MAX_ITER = config["iterations"]
            model_config.SOLVER.POLY_LR_POWER = 0.9
            model_config.freeze()
            resolved = model_config.dump()
            (output / "resolved_model.yaml").write_text(resolved, encoding="utf-8")
            recipe_reference = ROOT / config.get(
                "recipe_reference_semantic_run", config["parent_semantic_run"]
            )
            if (
                resolved.encode()
                != (recipe_reference / "mask2former/resolved_model.yaml").read_bytes()
            ):
                raise ValueError("Preflight Mask2Former recipe differs from reference")

            parent_training = read(recipe_reference / "mask2former/result.json")
            initialization = config.get("seed_aware_initialization")
            if initialization:
                reference_seed = int(initialization["reference_seed"])
                reset_model_rng(reference_seed)
                reference_model, reference_loaded = load_official_model(
                    model_config, runtime / "weights/model_final_6b4a3a.pkl"
                )
                reference_state = model_state_sha256(reference_model)
                reference_manifest = tensor_manifest(reference_model.state_dict())
                if reference_state != parent_training["initial_model_state_sha256"]:
                    raise ValueError(
                        "Reconstructed old-seed ADE state differs from reference"
                    )
                del reference_model
                gc.collect()

                reset_model_rng(config["seed"])
                model, loaded = load_official_model(
                    model_config, runtime / "weights/model_final_6b4a3a.pkl"
                )
                initial_state = model_state_sha256(model)
                current_manifest = tensor_manifest(model.state_dict())
                cross_seed = validate_cross_seed_reference(
                    current_manifest,
                    reference_manifest,
                    loaded,
                    reference_loaded,
                )
                paired = None
                paired_path = initialization.get("paired_control_preflight")
                if paired_path:
                    paired_result_path = ROOT / paired_path / "result.json"
                    paired_result = read(paired_result_path)
                    paired_contract = read(
                        ROOT / paired_path / "initialization-contract.json"
                    )
                    if paired_result["status"] != "completed_fresh_model_discarded":
                        raise ValueError(
                            "Completed paired control preflight is required"
                        )
                    paired = validate_same_seed_pair(
                        paired_contract["current_seed_state_manifest"],
                        current_manifest,
                        paired_contract["current_seed_pretrained_load"],
                        loaded,
                    )
                initialization_record = {
                    "schema_version": 1,
                    "experiment_id": EID,
                    "current_seed": config["seed"],
                    "reference_seed": reference_seed,
                    "current_seed_state_sha256": initial_state,
                    "reference_seed_state_sha256": reference_state,
                    "current_seed_state_manifest": current_manifest,
                    "reference_seed_state_manifest": reference_manifest,
                    "current_seed_pretrained_load": loaded,
                    "reference_seed_pretrained_load": reference_loaded,
                    "cross_seed_validation": cross_seed,
                    "paired_same_seed_validation": paired,
                    "paired_control_preflight": paired_path,
                }
                write_new_json(
                    output / "initialization-contract.json", initialization_record
                )
            else:
                model, loaded = load_official_model(
                    model_config, runtime / "weights/model_final_6b4a3a.pkl"
                )
                initial_state = model_state_sha256(model)
                if initial_state != parent_training["initial_model_state_sha256"]:
                    raise ValueError("Fresh preflight ADE state differs from reference")
            model.cuda()
            optimizer = author_optimizer(model, model_config)
            base_rates = [group["lr"] for group in optimizer.param_groups]
            scaler = torch.amp.GradScaler("cuda", init_scale=1024)
            context_fraction = float(config.get("context_fraction", 0.5))
            if context_fraction not in {0.0, 0.5}:
                raise ValueError("Only native-only or fixed 50% context is supported")
            sampler = (
                sampled_crop
                if context_fraction == 0.0
                else FullFrameContextSampler(sampled_crop)
            )
            rng = random.Random(config["seed"])
            model.train()
            rows, sample_shapes = [], []
            with (output / "training.jsonl").open("x", encoding="utf-8") as journal:
                for step in range(1, 9):
                    image, target, trace = sampler(
                        rng, train_names, images, targets, config["crop"]
                    )
                    ids, masks = class_masks(target)
                    instances = Instances(tuple(target.shape))
                    instances.gt_classes = torch.from_numpy(ids)
                    instances.gt_masks = torch.from_numpy(masks)
                    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
                    factor = poly_factor(step - 1, config["iterations"])
                    for group, rate in zip(optimizer.param_groups, base_rates):
                        group["lr"] = rate * factor
                    row = amp_update(
                        model,
                        optimizer,
                        scaler,
                        tensor,
                        instances,
                        max_attempts=12,
                        clip=0.01,
                    )
                    row.update(step=step, sample=trace, lr_factor=factor)
                    journal.write(json.dumps(row) + "\n")
                    rows.append(row)
                    sample_shapes.append([int(target.shape[0]), int(target.shape[1])])
                    result["trained_steps"] = step

            actual = [row["sample"] for row in rows]
            trace_keys = tuple(
                config.get(
                    "sampling_trace_keys",
                    TRACE_KEYS
                    if context_fraction == 0.5
                    else ("name", "top", "left", "flip"),
                )
            )
            expected = [
                {key: row[key] for key in trace_keys}
                for row in plan_record["samples"][:8]
            ]
            if actual != expected:
                raise ValueError(
                    "Preflight trace differs from the first eight frozen steps"
                )
            modes = (
                Counter({"native_crop": len(actual)})
                if context_fraction == 0.0
                else Counter(str(row["mode"]) for row in actual)
            )
            expected_modes = Counter(
                config.get(
                    "preflight_mode_counts",
                    {"native_crop": 8}
                    if context_fraction == 0.0
                    else {"native_crop": 4, "full_frame_context": 4},
                )
            )
            if modes != expected_modes:
                raise ValueError(
                    "Preflight input modes differ from the frozen contract"
                )
            if labels.accessed != set(train_names) or labels.accessed & holdout_names:
                raise ValueError("Preflight label access escaped the 270-view role")
            verify_hashes(batch / "export/images", export["image_hashes"])
            result.update(
                status="completed_fresh_model_discarded",
                mode_counts=dict(sorted(modes.items())),
                sample_shapes=sample_shapes,
                amp_skips=sum(len(row["skips"]) for row in rows),
                all_losses_finite=all(np.isfinite(row["loss"]) for row in rows),
                initial_model_state_sha256=initial_state,
                **(
                    {"seed_aware_initialization_validated": True}
                    if initialization
                    else {"initial_state_matches_reference": True}
                ),
                resolved_recipe_matches_reference=True,
                recipe_reference_semantic_run=str(recipe_reference.relative_to(ROOT)),
                training_labels_accessed=sorted(labels.accessed),
                holdout_labels_accessed=[],
                frozen_rgb_hashes_exact=True,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                pretrained_load=loaded,
                artifacts={
                    "resolved_model.yaml": sha256(output / "resolved_model.yaml"),
                    "training.jsonl": sha256(output / "training.jsonl"),
                    **(
                        {
                            "initialization-contract.json": sha256(
                                output / "initialization-contract.json"
                            )
                        }
                        if initialization
                        else {}
                    ),
                },
            )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


def train(config: dict[str, Any], artifact_root: Path) -> None:
    batch = ROOT / config["output_path"]
    preflight_result = read(batch / "preflight/result.json")
    context_fraction = float(config.get("context_fraction", 0.5))
    expected_modes = config.get(
        "preflight_mode_counts",
        {"native_crop": 8}
        if context_fraction == 0.0
        else {"full_frame_context": 4, "native_crop": 4},
    )
    if (
        preflight_result["status"] != "completed_fresh_model_discarded"
        or preflight_result["mode_counts"] != expected_modes
    ):
        raise ValueError("Successful fixed eight-update preflight is required")
    shared.setup_runtime(artifact_root)
    import src.quality_mask2former_training as training

    parent_sampler = training.sampled_crop
    if context_fraction == 0.5:
        training.sampled_crop = FullFrameContextSampler(parent_sampler)
    elif context_fraction != 0.0:
        raise ValueError("Only native-only or fixed 50% context is supported")
    try:
        with shared_identity():
            shared.train(config, artifact_root)
    finally:
        training.sampled_crop = parent_sampler


def inference(config: dict[str, Any], artifact_root: Path) -> None:
    with shared_identity():
        shared.inference(config, artifact_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("prepare", "preflight", "train", "inference"),
        required=True,
    )
    args = parser.parse_args()
    config = read(CONFIG)
    if config["experiment_id"] != EID:
        raise ValueError("SEM385 config identity changed")
    if tuple(config["sampling_trace_keys"]) != TRACE_KEYS:
        raise ValueError("SEM385 trace contract changed")
    artifact_root = args.artifact_root.resolve()
    if args.stage == "prepare":
        prepare(config)
    elif args.stage == "preflight":
        preflight(config, artifact_root)
    elif args.stage == "train":
        train(config, artifact_root)
    else:
        inference(config, artifact_root)


if __name__ == "__main__":
    main()
