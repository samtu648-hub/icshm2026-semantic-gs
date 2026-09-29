#!/usr/bin/env python3
"""Run the matched AbsGS max_radii2D lifecycle control and candidate."""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_rgb152_spatial as base
from src.quality_protocol import sha256, verify_hashes, write_new_json


SUITE_ID = "QUAL-021-ABSGS-SCREEN-PRUNE-FIX"
SUITE_CONFIG = ROOT / "experiments/runs" / SUITE_ID / "config.yaml"
ARM_IDS = (
    "RGB-161-ABSGS-SCREEN-PRUNE-CTRL",
    "RGB-162-ABSGS-SCREEN-PRUNE-FIX",
)
CODE_PATHS = (
    "scripts/backbone_explicit_split_worker.py",
    "scripts/run_rgb152_spatial.py",
    "scripts/run_absgs_screen_prune_lifecycle.py",
    "scripts/worker_absgs_screen_radius.py",
    "src/absgs_screen_radius_lifecycle.py",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge(left: dict, right: dict) -> dict:
    output = copy.deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = merge(output[key], value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def arm_registration(experiment_id: str) -> tuple[Path, dict]:
    path = ROOT / "experiments/runs" / experiment_id / "config.yaml"
    overlay = read(path)
    config = merge(read(ROOT / overlay["base_config"]), overlay)
    if config["experiment_id"] != experiment_id:
        raise ValueError(f"Arm identity mismatch: {experiment_id}")
    return path, config


def artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def prepare(artifact_root: Path) -> None:
    suite = read(SUITE_CONFIG)
    output = ROOT / suite["output_path"]
    output.mkdir(parents=True, exist_ok=False)
    result: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": SUITE_ID,
        "status": "starting",
        "suite_config_sha256": sha256(SUITE_CONFIG),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "code_sha256": {name: sha256(ROOT / name) for name in CODE_PATHS},
    }
    try:
        arms = {}
        for experiment_id in ARM_IDS:
            config_path, config = arm_registration(experiment_id)
            base.EID = experiment_id
            base.CONFIG = config_path
            base.CODE_PATHS = CODE_PATHS
            base.preflight(config, artifact_root)
            root = ROOT / config["output"]["root"]
            arms[experiment_id] = {
                "config_sha256": sha256(config_path),
                "preflight_sha256": sha256(root / "preflight-r3/result.json"),
                "access_sha256": sha256(root / "access.json"),
                "mode": config["screen_radius_lifecycle"],
            }
        if len({record["access_sha256"] for record in arms.values()}) != 1:
            raise ValueError("Paired arms have different access records")
        result.update(status="passed", arms=arms)
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_new_json(output / "prepare-result.json", result)


def train_arm(
    experiment_id: str, artifact_root: Path, run_name: str, iterations: int
) -> dict:
    config_path, config = arm_registration(experiment_id)
    split_path, scene, output_root, _, source = base.registered_paths(config)
    run = output_root / run_name
    preflight_path = output_root / "preflight-r3/result.json"
    if read(preflight_path)["status"] != "passed":
        raise ValueError(f"{experiment_id} preflight did not pass")
    run.mkdir(parents=True, exist_ok=False)
    environment, python = base.runtime_environment(artifact_root, source)
    access_path = output_root / "access.json"
    lifecycle_path = run / "screen-radius-lifecycle.json"
    command = [
        str(python),
        str(ROOT / "scripts/worker_absgs_screen_radius.py"),
        "--mode",
        config["screen_radius_lifecycle"],
        "--lifecycle-output",
        str(lifecycle_path),
        "--",
        "--source",
        str(source),
        "--access",
        str(access_path),
        "--evidence",
        str(run / "worker.json"),
        "--seed",
        str(config["seed"]),
        "--partition",
        str(config["split_id"]),
        "--expected-train",
        str(config["data"]["rgb_train_views"]),
        "--expected-evaluation",
        str(config["data"]["rgb_confirmation_views"]),
        "--expected-universe",
        str(config["data"]["scene_views"]),
        "--",
        "--source_path",
        str(scene),
        "--model_path",
        str(run / "model"),
        "--resolution",
        "1",
        "--data_device",
        "cpu",
        "--iterations",
        str(iterations),
        "--densify_grad_threshold",
        str(config["training"]["signed_clone_threshold"]),
        "--densify_grad_abs_threshold",
        str(config["training"]["absolute_split_threshold"]),
        "--test_iterations",
        "-1",
        "--save_iterations",
        str(iterations),
        "--checkpoint_iterations",
        str(iterations),
        "--eval",
        "--quiet",
    ]
    result: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "status": "starting",
        "run_name": run_name,
        "iterations": iterations,
        "command": command,
        "seed": config["seed"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": sha256(config_path),
        "split_sha256": sha256(split_path),
        "access_sha256": sha256(access_path),
        "preflight_sha256": sha256(preflight_path),
        "initialization_ply_sha256": sha256(scene / "sparse/0/points3D.ply"),
        "code_sha256": {name: sha256(ROOT / name) for name in CODE_PATHS},
        "free_bytes_start": shutil.disk_usage(run).free,
    }
    started = time.monotonic()
    try:
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
                raise RuntimeError("Other GPU process active outside shared lock")
            write_new_json(run / "started.json", result)
            with (run / "train.log").open("x") as log:
                process = subprocess.Popen(
                    command,
                    cwd=source,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                with (run / "resources.jsonl").open("x") as monitor:
                    while process.poll() is None:
                        gpu = subprocess.run(
                            [
                                "nvidia-smi",
                                "--query-gpu=memory.used,utilization.gpu,power.draw,clocks.sm",
                                "--format=csv,noheader,nounits",
                            ],
                            capture_output=True,
                            text=True,
                        )
                        monitor.write(
                            json.dumps(
                                {
                                    "seconds": time.monotonic() - started,
                                    "gpu": gpu.stdout.strip(),
                                    "free_bytes": shutil.disk_usage(run).free,
                                }
                            )
                            + "\n"
                        )
                        monitor.flush()
                        time.sleep(10)
                if process.returncode:
                    raise RuntimeError(f"Native process exited {process.returncode}")
        worker = read(run / "worker.json")
        access = read(access_path)
        if worker["actual_train"] != sorted(access["train"]):
            raise RuntimeError("Worker training cameras differ from access record")
        if worker["actual_evaluation"] != sorted(access["evaluation"]):
            raise RuntimeError("Worker evaluation cameras differ from access record")
        if worker["initialization"]["common_point_count"] != config["data"]["initial_point_count"]:
            raise RuntimeError("Worker loaded a different initialization")
        lifecycle = read(lifecycle_path)
        result.update(
            status="completed_spatial_training",
            worker=worker,
            lifecycle_summary={
                key: value for key, value in lifecycle.items() if key != "events"
            },
            artifacts=artifact_hashes(run),
            free_bytes_end=shutil.disk_usage(run).free,
        )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(run / "result.json", result)
    return result


def validate_smoke(run_name: str) -> dict:
    records = {}
    for experiment_id in ARM_IDS:
        _, config = arm_registration(experiment_id)
        root = ROOT / config["output"]["root"] / run_name
        training = read(root / "result.json")
        lifecycle = read(root / "screen-radius-lifecycle.json")
        if training["status"] != "completed_spatial_training":
            raise ValueError(f"{experiment_id} smoke did not complete")
        active = [event for event in lifecycle["events"] if event["max_screen_size"]]
        append_after = [
            call["oversized_after"]
            for event in active
            for call in event["append_calls"]
            if call["oversized_before"] > 0
        ]
        if not active or not append_after:
            raise ValueError(f"{experiment_id} did not exercise active screen pruning")
        records[experiment_id] = {
            "mode": lifecycle["mode"],
            "active_events": len(active),
            "append_observations": len(append_after),
            "append_after_nonzero": sum(value > 0 for value in append_after),
            "max_oversized_after_append": max(append_after),
            "training_seconds": training["seconds"],
        }
    control = records[ARM_IDS[0]]
    candidate = records[ARM_IDS[1]]
    if control["append_after_nonzero"] != 0:
        raise ValueError("Native control unexpectedly retained screen-radius history")
    if candidate["append_after_nonzero"] == 0:
        raise ValueError("Candidate failed to retain screen-radius history")
    return records


def run_smoke(artifact_root: Path) -> None:
    suite = read(SUITE_CONFIG)
    output = ROOT / suite["output_path"]
    attempts = []
    for experiment_id in ARM_IDS:
        _, config = arm_registration(experiment_id)
        attempts.extend(
            int(path.name.removeprefix("smoke-r"))
            for path in (ROOT / config["output"]["root"]).glob("smoke-r*")
            if path.name.removeprefix("smoke-r").isdigit()
        )
    attempt = max(attempts, default=0) + 1
    run_name = f"smoke-r{attempt}"
    result_path = output / f"smoke-result-r{attempt}.json"
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "starting",
        "run_name": run_name,
    }
    try:
        arms = {
            experiment_id: train_arm(
                experiment_id,
                artifact_root,
                run_name,
                int(suite["smoke_iterations"]),
            )["status"]
            for experiment_id in ARM_IDS
        }
        result.update(
            status="passed_causal_gate",
            arms=arms,
            evidence=validate_smoke(run_name),
        )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_new_json(result_path, result)


def run_full(artifact_root: Path) -> None:
    suite = read(SUITE_CONFIG)
    smoke_paths = sorted((ROOT / suite["output_path"]).glob("smoke-result-r*.json"))
    passed = [read(path) for path in smoke_paths if read(path)["status"] == "passed_causal_gate"]
    if not passed:
        raise ValueError("No smoke causal gate has passed")
    smoke = passed[-1]
    if smoke["status"] != "passed_causal_gate":
        raise ValueError("Smoke causal gate did not pass")
    for experiment_id in ARM_IDS:
        train_arm(
            experiment_id,
            artifact_root,
            "full-r1",
            int(suite["full_iterations"]),
        )


def evaluate_arms(artifact_root: Path) -> None:
    for experiment_id in ARM_IDS:
        config_path, config = arm_registration(experiment_id)
        base.EID = experiment_id
        base.CONFIG = config_path
        base.CODE_PATHS = CODE_PATHS
        base.render_and_evaluate(config, artifact_root)


def numeric_delta(candidate: dict, control: dict) -> dict:
    output = {}
    for key in candidate:
        if isinstance(candidate[key], dict) and isinstance(control.get(key), dict):
            output[key] = numeric_delta(candidate[key], control[key])
        elif isinstance(candidate[key], (int, float)) and isinstance(
            control.get(key), (int, float)
        ):
            output[key] = candidate[key] - control[key]
    return output


def compare() -> None:
    suite = read(SUITE_CONFIG)
    output = ROOT / suite["output_path"]
    evaluations = {}
    training = {}
    lifecycle = {}
    for experiment_id in ARM_IDS:
        _, config = arm_registration(experiment_id)
        run = ROOT / config["output"]["root"] / "full-r1"
        evaluations[experiment_id] = read(run / "evaluation/result.json")
        training[experiment_id] = read(run / "result.json")
        lifecycle[experiment_id] = read(run / "screen-radius-lifecycle.json")
        verify_hashes(run / "model", training[experiment_id]["artifacts"])
    control = evaluations[ARM_IDS[0]]
    candidate = evaluations[ARM_IDS[1]]
    result = {
        "schema_version": 1,
        "experiment_id": SUITE_ID,
        "status": "completed_matched_pair",
        "control": ARM_IDS[0],
        "candidate": ARM_IDS[1],
        "metric_delta_candidate_minus_control": numeric_delta(
            {
                "metrics": candidate["metrics"],
                "group_metrics": candidate["group_metrics"],
            },
            {
                "metrics": control["metrics"],
                "group_metrics": control["group_metrics"],
            },
        ),
        "training_seconds": {
            experiment_id: training[experiment_id]["seconds"]
            for experiment_id in ARM_IDS
        },
        "lifecycle_summary": {
            experiment_id: {
                key: value
                for key, value in lifecycle[experiment_id].items()
                if key != "events"
            }
            for experiment_id in ARM_IDS
        },
        "bridge_review_views": suite["bridge_review_views"],
        "interpretation": "Quality decision remains pending targeted bridge review; this file reports the matched numeric pair.",
    }
    write_new_json(output / "result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("prepare", "smoke", "full", "evaluate", "compare", "all"),
        required=True,
    )
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    if args.stage in {"prepare", "all"}:
        prepare(artifact_root)
    if args.stage in {"smoke", "all"}:
        run_smoke(artifact_root)
    if args.stage in {"full", "all"}:
        run_full(artifact_root)
    if args.stage in {"evaluate", "all"}:
        evaluate_arms(artifact_root)
    if args.stage in {"compare", "all"}:
        compare()


if __name__ == "__main__":
    main()
