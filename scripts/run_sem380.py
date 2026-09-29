#!/usr/bin/env python3
"""Run matched SEM380 engineering preflight, training, or inference."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import backbone_comparison_stage as shared
from src.quality_protocol import sha256, verify_hashes, write_new_json

EID = "SEM-380-RGB148-GT-REGION-RGB-ALL300"
BASE = ROOT / "outputs/quality/SEM-378/r1"
SEM = ROOT / "outputs/quality/all_data/SEM-370-r1"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def model_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


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
        "rgb_experiment_id": "RGB-148-STANDARD-GT-REGION-RGB",
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
        assert read(config_path)["experiment_id"] == EID
        assert sha256(config_path) == binding["config_sha256"]
        assert sha256(binding_path) == result["binding_sha256"]
        verify_hashes(ROOT, binding["code_sha256"])
        verify_hashes(args.rgb_run, binding["rgb148_artifacts"])
        assert sha256(args.rgb_run / "result.json") == binding["rgb148_result_sha256"]
        assert sha256(rgb_eval / "result.json") == binding["rgb148_evaluation_sha256"]
        verify_hashes(rgb_eval / "images", binding["rgb148_image_hashes"])
        verify_hashes(args.batch / "export/images", export["image_hashes"])
        verify_hashes(BASE, binding["sem378_sha256"])
        verify_hashes(SEM / "labels", binding["label_hashes"])
        assert sha256(SEM / "label_manifest.json") == binding["label_manifest_sha256"]
        assert (
            sha256(args.batch / "sampling-plan.jsonl")
            == binding["sampling_plan_sha256"]
        )
        assert (
            sha256(
                ROOT
                / "outputs/quality/mask2former/SEM-340/weights/model_final_6b4a3a.pkl"
            )
            == binding["pretrained_sha256"]
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
            runtime, torch, np = shared.setup(args)
            torch.set_num_threads(4)
            torch.cuda.reset_peak_memory_stats()
            result["runtime"] = {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "python": sys.version,
            }
            write_new_json(output / "started.json", result)

            if args.stage == "preflight":
                import src.quality_mask2former as api

                loader = api.load_official_model

                def capture(*call_args, **call_kwargs):
                    model, loaded = loader(*call_args, **call_kwargs)
                    result.setdefault(
                        "initial_model_state_sha256", model_state_sha256(model)
                    )
                    return model, loaded

                api.load_official_model = capture
                try:
                    shared.preflight(args, output, result, torch, np, runtime)
                finally:
                    api.load_official_model = loader
                assert (
                    result["initial_model_state_sha256"]
                    == binding["sem378_initial_model_state_sha256"]
                )
                expected = [
                    json.loads(line)["sample"]
                    for line in (BASE / "mask2former/training.jsonl")
                    .read_text()
                    .splitlines()[:8]
                ]
                assert [row["sample"] for row in result["updates"]] == expected
                from src.quality_mask2former import config_for_model

                resolved = config_for_model(
                    runtime / "vendor/Mask2Former",
                    "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
                    False,
                )
                resolved.defrost()
                resolved.DATASETS.TRAIN = ("sem341_rendered_training",)
                resolved.SOLVER.MAX_ITER = 6000
                resolved.SOLVER.POLY_LR_POWER = 0.9
                resolved.freeze()
                assert (
                    resolved.dump().encode()
                    == (BASE / "mask2former/resolved_model.yaml").read_bytes()
                )
                result.update(
                    initial_state_matches_sem378=True,
                    first8_samples_match_sem378=True,
                    resolved_recipe_matches_sem378=True,
                    independent_initialization=True,
                    quality_gate=False,
                )
            elif args.stage == "mask2former":
                shared.train_stage(args, output, result, torch, np)
                assert (
                    result["initial_model_state_sha256"]
                    == binding["sem378_initial_model_state_sha256"]
                )
                actual = [
                    json.loads(line)["sample"]
                    for line in (output / "training.jsonl").read_text().splitlines()
                ]
                expected = [
                    json.loads(line)["sample"]
                    for line in (BASE / "mask2former/training.jsonl")
                    .read_text()
                    .splitlines()
                ]
                assert actual == expected and len(actual) == 6000
                assert (output / "resolved_model.yaml").read_bytes() == (
                    BASE / "mask2former/resolved_model.yaml"
                ).read_bytes()
                result.update(
                    sampling6000_matches_sem378=True,
                    resolved_recipe_matches_sem378=True,
                    initial_state_matches_sem378=True,
                    independent_initialization=True,
                )
            else:
                assert (
                    read(args.batch / "mask2former/result.json")["status"]
                    == "completed"
                )
                shared.infer_stage(args, output, result, torch, np, runtime)

            result.update(
                status="completed",
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
        verify()
        result.update(
            frozen18_exact=True,
            rgb148_endpoint_unchanged=True,
            rgb544_hashes_unchanged=True,
            sem378_unchanged=True,
        )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


if __name__ == "__main__":
    main()
