#!/usr/bin/env python3
"""Run RGB167 preflight or the single authorized from-scratch 30k trajectory."""
from __future__ import annotations

from _artifact_paths import artifact_root_default, parse_artifact_root

import argparse
import ast
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_quality_rgb import materialize_scene, source_manifest
from src.quality_protocol import sha256, verify_hashes, write_new_json

EXPERIMENT_ID = "RGB-167-SCAFFOLD-GROWTH20K-ALL400"
CONFIG_PATH = ROOT / "experiments/runs" / EXPERIMENT_ID / "config.yaml"
PARENT = ROOT / "outputs/quality/scaffold_noappearance/full-r1"
DATA_PARENT = ROOT / "outputs/quality/all_data/RGB-120-r2"
OUTPUT_ROOT = ROOT / "outputs/quality/RGB-167"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_logged_namespace(path: Path) -> dict:
    line = next(line for line in path.read_text(errors="replace").splitlines() if "args: Namespace(" in line)
    expression = line[line.index("Namespace(") :]
    node = ast.parse(expression, mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "Namespace":
        raise ValueError("Unexpected upstream argument log")
    return {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}


def protected_snapshot(artifact_root: Path) -> dict[str, str]:
    parent_result = load(PARENT / "result.json")
    verify_hashes(PARENT, parent_result["artifacts"])
    frozen = {
        item["path"]: item["sha256"]
        for item in load(ROOT / "experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json")["artifacts"]
    }
    verify_hashes(artifact_root, frozen)
    snapshot = {f"RGB136/{path}": digest for path, digest in parent_result["artifacts"].items()}
    snapshot["RGB136/result.json"] = sha256(PARENT / "result.json")
    snapshot["SEM_FROZEN18/manifest.json"] = sha256(ROOT / "experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json")
    rgb166 = ROOT / "outputs/quality/RGB-166/full-r1"
    evaluation = load(rgb166 / "evaluation-r1/result.json")
    if evaluation.get("status") != "completed":
        raise ValueError("RGB166 endpoint evaluation is not complete")
    for iteration, digest in evaluation["endpoint_ply_sha256"].items():
        path = rgb166 / f"model/point_cloud/iteration_{iteration}/point_cloud.ply"
        if sha256(path) != digest:
            raise ValueError(f"RGB166 endpoint changed: {iteration}")
        snapshot[f"RGB166/{path.relative_to(rgb166)}"] = digest
    snapshot["RGB166/evaluation-r1/result.json"] = sha256(rgb166 / "evaluation-r1/result.json")
    snapshot["RGB166/result.json"] = sha256(rgb166 / "result.json")
    return snapshot


def build_env(artifact_root: Path, source: Path) -> dict[str, str]:
    cuda = artifact_root / "r1/toolchain/cuda-12.8"
    env = os.environ.copy()
    env.update(
        CUDA_HOME=str(cuda),
        PATH=str(cuda / "bin") + os.pathsep + env.get("PATH", ""),
        LD_LIBRARY_PATH=str(cuda / "lib64") + os.pathsep + env.get("LD_LIBRARY_PATH", ""),
        PYTHONPATH=os.pathsep.join(
            map(str, [source, source / "submodules/diff-gaussian-rasterization", source / "submodules/simple-knn", ROOT])
        ),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        TORCH_HOME=str(artifact_root / "r1/weights/torch"),
        WANDB_MODE="disabled",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def worker_command(
    python: Path,
    source: Path,
    run: Path,
    model_subdir: str,
    evidence_name: str,
    lifecycle_name: str,
    iterations: int,
    saves: list[int],
    start_checkpoint: Path | None = None,
    trusted_checkpoint_load: bool = False,
) -> list[str]:
    command = [
        str(python), str(ROOT / "scripts/quality_all400_worker.py"),
        "--source", str(source), "--access", str(run / "access.json"),
        "--evidence", str(run / evidence_name), "--seed", "0",
        "--scaffold-lifecycle-evidence", str(run / lifecycle_name), "--",
        "--source_path", str(run / "scene"), "--model_path", str(run / model_subdir),
        "--eval", "--resolution", "1", "--data_device", "cpu",
        "--iterations", str(iterations), "--appearance_dim", "0", "--update_until", "20000",
        "--test_iterations", str(iterations),
        "--save_iterations", *map(str, saves),
        "--checkpoint_iterations", *map(str, saves),
        "--quiet", "--skip_metric_eval",
    ]
    if start_checkpoint is not None:
        command.extend(["--start_checkpoint", str(start_checkpoint)])
    if trusted_checkpoint_load:
        command.insert(command.index("--") , "--trusted-checkpoint-load")
    return command


def verify_command(python: Path, artifact_root: Path, run: Path, model_subdir: str, iteration: int, output: Path) -> list[str]:
    return [
        str(python), str(ROOT / "scripts/verify_rgb167_checkpoint.py"),
        "--artifact-root", str(artifact_root), "--run", str(run),
        "--model-subdir", model_subdir, "--iteration", str(iteration), "--output", str(output),
    ]


def validate_parameters(log_path: Path, cfg: dict, expected_iterations: int) -> dict:
    args = parse_logged_namespace(log_path)
    expected = {
        "iterations": expected_iterations,
        "appearance_dim": 0,
        "n_offsets": cfg["model"]["n_offsets"],
        "voxel_size": cfg["model"]["voxel_size"],
        "update_from": cfg["training"]["update_from"],
        "update_interval": cfg["training"]["update_interval"],
        "update_until": cfg["training"]["update_until"],
        "min_opacity": cfg["training"]["min_opacity"],
        "success_threshold": cfg["training"]["success_threshold"],
        "densify_grad_threshold": cfg["training"]["densify_grad_threshold"],
        "lambda_dssim": cfg["training"]["lambda_dssim"],
    }
    for key in (
        "position_lr_init", "position_lr_final", "position_lr_delay_mult", "position_lr_max_steps",
        "offset_lr_init", "offset_lr_final", "offset_lr_delay_mult", "offset_lr_max_steps", "feature_lr",
        "opacity_lr", "scaling_lr", "rotation_lr", "mlp_opacity_lr_init", "mlp_opacity_lr_final",
        "mlp_opacity_lr_delay_mult", "mlp_opacity_lr_max_steps", "mlp_cov_lr_init", "mlp_cov_lr_final",
        "mlp_cov_lr_delay_mult", "mlp_cov_lr_max_steps", "mlp_color_lr_init", "mlp_color_lr_final",
        "mlp_color_lr_delay_mult", "mlp_color_lr_max_steps", "mlp_featurebank_lr_init",
        "mlp_featurebank_lr_final", "mlp_featurebank_lr_delay_mult", "mlp_featurebank_lr_max_steps",
        "appearance_lr_init", "appearance_lr_final", "appearance_lr_delay_mult", "appearance_lr_max_steps",
    ):
        expected[key] = cfg["training"][key]
    mismatches = {key: {"expected": value, "actual": args.get(key)} for key, value in expected.items() if args.get(key) != value}
    if mismatches:
        raise ValueError(f"RGB167 parameter mismatch: {mismatches}")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root_default())
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load(CONFIG_PATH)
    run = OUTPUT_ROOT / args.attempt
    run.mkdir(parents=True, exist_ok=False)
    source = args.artifact_root / "r1/third_party/scaffold-gs"
    python = args.artifact_root / "r1/envs/r1-cuda128/bin/python"
    env = build_env(args.artifact_root, source)
    materialize_scene(DATA_PARENT / "scene", run / "scene", "scaffold")
    shutil.copy2(DATA_PARENT / "access.json", run / "access.json")
    shutil.copy2(DATA_PARENT / "split.json", run / "split.json")
    code_paths = [
        "scripts/run_rgb167_scaffold_growth20k.py", "scripts/verify_rgb167_checkpoint.py",
        "scripts/quality_all400_worker.py", "scripts/run_quality_rgb.py", "src/quality_protocol.py",
    ]
    for name in code_paths:
        target = run / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    shutil.copy2(CONFIG_PATH, run / "code/config.json")

    parent_source = load(PARENT / "source_manifest.json")
    current_source = source_manifest(source)
    if current_source != parent_source:
        raise ValueError("Scaffold source changed since RGB136")
    write_new_json(run / "source_manifest.json", current_source)
    protected_before = protected_snapshot(args.artifact_root)
    write_new_json(run / "protected_before.json", protected_before)
    initial = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "queued",
        "smoke": args.smoke,
        "from_scratch": True,
        "start_checkpoint": None,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": sha256(CONFIG_PATH),
        "manifest_sha256": sha256(ROOT / cfg["dataset_manifest"]),
        "split_sha256": sha256(run / "split.json"),
        "code_sha256": {name: sha256(ROOT / name) for name in code_paths},
        "protected_before_sha256": sha256(run / "protected_before.json"),
        "started_unix": time.time(),
    }
    write_new_json(run / "queued.json", initial)
    result = dict(initial)
    started = time.monotonic()
    iterations = cfg["preflight"]["iterations"] if args.smoke else cfg["training"]["iterations"]
    saves = [iterations] if args.smoke else cfg["training"]["save_iterations"]
    command = worker_command(python, source, run, "model", "worker_train.json", "lifecycle.json", iterations, saves)
    result["command"] = command
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result["queue_seconds"] = time.monotonic() - started
            apps = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True
            )
            result["gpu_processes_before"] = apps
            active = [line for line in apps.splitlines() if line.strip() and "ToDesk" not in line]
            if active:
                raise RuntimeError("Uncoordinated GPU compute process: " + " | ".join(active))
            result["status"] = "running_preflight" if args.smoke else "running_from_scratch_30k"
            write_new_json(run / "started.json", result)
            with (run / "train.log").open("x") as log:
                subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            worker = load(run / "worker_train.json")
            if worker["status"] != "completed_native_execution" or len(worker["actual_train"]) != 400 or worker["actual_evaluation"]:
                raise ValueError("Native worker did not execute the registered all400 train-only protocol")
            if worker["initialization"]["canonical_ply_sha256"] != cfg["data"]["initial_ply_sha256"] or worker["initialization"]["common_point_count"] != 72292:
                raise ValueError("Initialization identity changed")
            parameters = validate_parameters(run / "train.log", cfg, iterations)
            verifications = []
            for endpoint in saves:
                output = run / f"verification_{endpoint}.json"
                verify = verify_command(python, args.artifact_root, run, "model", endpoint, output)
                subprocess.run(verify, cwd=ROOT, env=env, check=True)
                verifications.append(load(output))

            resume = None
            if args.smoke:
                resume_command = worker_command(
                    python, source, run, "resume_model", "worker_resume.json", "lifecycle_resume.json", 33, [33],
                    run / "model/chkpnt32.pth",
                    trusted_checkpoint_load=True,
                )
                with (run / "resume.log").open("x") as log:
                    subprocess.run(resume_command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                resume_worker = load(run / "worker_resume.json")
                if resume_worker["status"] != "completed_native_execution":
                    raise ValueError("Checkpoint resume worker failed")
                resume_output = run / "verification_resume_33.json"
                subprocess.run(
                    verify_command(python, args.artifact_root, run, "resume_model", 33, resume_output),
                    cwd=ROOT, env=env, check=True,
                )
                resume = {
                    "command": resume_command,
                    "worker": resume_worker,
                    "verification": load(resume_output),
                    "parameters": validate_parameters(run / "resume.log", cfg, 33),
                }

            protected_after = protected_snapshot(args.artifact_root)
            if protected_after != protected_before:
                raise ValueError("Protected RGB136/RGB166/frozen semantic artifacts changed")
            if source_manifest(source) != parent_source:
                raise ValueError("Scaffold source changed during RGB167")
            verify_hashes(ROOT, result["code_sha256"])
            result.update(
                status="completed_engineering_preflight" if args.smoke else "completed_from_scratch_30k",
                worker=worker,
                parameters=parameters,
                verifications=verifications,
                resume_test=resume,
                protected_exact=True,
                source_unchanged=True,
                lifecycle=load(run / "lifecycle.json"),
            )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(run / "result.json", result)


if __name__ == "__main__":
    main()
