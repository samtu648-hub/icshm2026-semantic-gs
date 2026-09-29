#!/usr/bin/env python3
"""Run SEM383 preflight, training and endpoint inference."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import backbone_comparison_stage as shared
from scripts import run_sem381 as common
from scripts import run_sem382 as control_runner
from src.quality_dinov3_mask2former import module_state_sha256, validate_model_dir
from src.quality_dinov3_spatial_sfp import load_dinov3_spatial_mask2former
from src.quality_protocol import sha256, verify_hashes, write_new_json

EID = "SEM-383-DINOV3-VITB-SPATIAL-SFP-ALL300"
CONTROL = ROOT / "outputs/quality/SEM-382/r1"
BASE = ROOT / "outputs/quality/SEM-378/r1"
SEM = ROOT / "outputs/quality/all_data/SEM-370-r1"
RUNTIME = ROOT / "outputs/quality/mask2former/SEM-340"
SPATIAL_PREFIXES = (
    "backbone.spatial_prior.",
    "backbone.spatial_projections.",
    "backbone.spatial_gates.",
)
COLLECT_SPATIAL_GRADIENTS = False
SPATIAL_GRADIENTS: dict[str, dict] = {}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def model_config():
    cfg = control_runner.model_config()
    return cfg


def _gradient_hook(name: str):
    def record(gradient):
        import torch

        row = SPATIAL_GRADIENTS.setdefault(
            name, {"calls": 0, "finite": True, "nonzero": False, "max_abs": 0.0}
        )
        row["calls"] += 1
        row["finite"] = row["finite"] and bool(torch.isfinite(gradient).all())
        if gradient.numel():
            maximum = float(gradient.detach().abs().max())
            row["max_abs"] = max(row["max_abs"], maximum)
            row["nonzero"] = row["nonzero"] or maximum > 0

    return record


def shared_trainable_state_sha256(model) -> str:
    import hashlib
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if not value.requires_grad or name.startswith(SPATIAL_PREFIXES):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_model(config: dict, model_dir: Path):
    cfg = model_config()
    model, loaded = load_dinov3_spatial_mask2former(
        cfg,
        RUNTIME / "weights/model_final_6b4a3a.pkl",
        model_dir,
        adapter_seed=config["seed"],
    )
    shared_hash = shared_trainable_state_sha256(model)
    if (
        shared_hash
        != config["expected_initial_state"]["shared_sem382_trainable_sha256"]
    ):
        raise ValueError("Shared SFP/decoder initialization differs from SEM382")
    loaded["shared_sem382_trainable_sha256"] = shared_hash
    if COLLECT_SPATIAL_GRADIENTS:
        SPATIAL_GRADIENTS.clear()
        for name, parameter in model.named_parameters():
            if name.startswith(SPATIAL_PREFIXES):
                parameter.register_hook(_gradient_hook(name))
    return cfg, model, loaded


def parameter_partition(name: str, requires_grad: bool) -> str:
    if name.startswith("backbone.dinov3."):
        return "dinov3"
    if name.startswith("backbone."):
        if ".norm" in name or "norms." in name:
            return "sfp_norm"
        return "sfp_adapter"
    if requires_grad:
        return "decoder"
    raise ValueError(f"Unexpected frozen parameter outside DINOv3: {name}")


def _install_candidate_overrides() -> None:
    control_runner.make_model = make_model
    control_runner.parameter_partition = parameter_partition


def main() -> None:
    global COLLECT_SPATIAL_GRADIENTS

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
        verify_hashes(CONTROL, binding["sem382_sha256"])
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
        if (
            read(args.batch / "input-diagnostic-control/result.json")["status"]
            != "completed"
        ):
            raise ValueError("Completed control input diagnostic is required")
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
            _install_candidate_overrides()
            if args.stage == "preflight":
                COLLECT_SPATIAL_GRADIENTS = True
                control_runner.run_updates(
                    args, output, result, config, model_dir, torch, np, steps=8
                )
                expected_names = {
                    name
                    for name in result["optimizer_contract"]["partitions"][
                        "sfp_adapter"
                    ]["names"]
                    + result["optimizer_contract"]["partitions"]["sfp_norm"]["names"]
                    if name.startswith(SPATIAL_PREFIXES)
                }
                if set(SPATIAL_GRADIENTS) != expected_names or not all(
                    row["finite"] and row["nonzero"]
                    for row in SPATIAL_GRADIENTS.values()
                ):
                    raise ValueError(
                        "Spatial adapter did not receive finite nonzero gradients"
                    )
                result.update(
                    quality_gate=False,
                    native_768_preflight=True,
                    first8_samples_match_sem382=True,
                    spatial_gradient_audit=SPATIAL_GRADIENTS,
                )
            elif args.stage == "mask2former":
                if read(args.batch / "preflight/result.json")["status"] != "completed":
                    raise ValueError("Completed preflight required")
                control_runner.run_updates(
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
                control_runner.infer(
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
            sem382_unchanged=True,
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
