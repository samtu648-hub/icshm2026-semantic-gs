#!/usr/bin/env python3
"""Run RGB171 preflight or full all400/50k bridge-repair training."""

from __future__ import annotations

from _artifact_paths import artifact_root_default, parse_artifact_root

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_rgb164_rgb165_improvedgs_pair import SOURCE_REL, build_env, source_record
from src.quality_protocol import sha256, write_new_json

DEFAULT_EXPERIMENT_ID = "RGB-171-IMPROVEDGS-BRIDGE-STRUCTURE-REPAIR"


def vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        while True:
            line = handle.readline().decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header" or not line:
                raise ValueError(f"PLY vertex declaration missing: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root_default())
    parser.add_argument("--stage", choices=("preflight", "full"), required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/quality/RGB-171")
    parser.add_argument(
        "--selection-mode",
        choices=(
            "native-intersection",
            "evidence-directed-fixed-budget",
            "responsibility-asymmetric-fixed-budget",
        ),
        default="native-intersection",
    )
    args = parser.parse_args()
    config_path = ROOT / "experiments/runs" / args.experiment_id / "config.yaml"
    config = json.loads(config_path.read_text())
    source = ROOT / SOURCE_REL
    if source_record(source)["commit"] != "20b5db343a87c2f9fb4b70c6e39ff1f042a8a47a":
        raise ValueError("ImprovedGS source commit changed")
    run = args.output_root / args.attempt
    run.mkdir(parents=True, exist_ok=False)
    scene_root = ROOT / "outputs/quality/all_data/RGB-120-r2"
    python = args.artifact_root / "r1/envs/r1-cuda128/bin/python"
    if args.stage == "preflight":
        iterations, repair_iteration, endpoints = 901, 700, [901]
    else:
        iterations = 50_000
        repair_iteration = int(config["training"]["repair_iteration"])
        endpoints = list(config["training"]["save_iterations"])
    command = [
        str(python), str(ROOT / "scripts/improvedgs_bridge_repair_worker.py"),
        "--source", str(source),
        "--access", str(scene_root / "access.json"),
        "--labels", str(ROOT / "outputs/quality/all_data/SEM-370-r1/labels"),
        "--evidence", str(run / "worker.json"),
        "--repair-record", str(run / "bridge-repair-evidence.json.gz"),
        "--seed", "0", "--repair-iteration", str(repair_iteration),
        "--top-count", "16", "--transverse-multiplier", "0.5",
        "--selection-mode", args.selection_mode, "--",
        "--source_path", str(scene_root / "scene"),
        "--model_path", str(run / "model"),
        "--resolution", "1", "--data_device", "cpu", "--eval", "false",
        "--iterations", str(iterations), "--training_method", "improvedgs",
        "--use_las", "true", "--use_eas", "true", "--use_rap", "true", "--use_mu", "true",
        "--budget", "1500000", "--edge_sample_cams", "10",
        "--densify_until_iter", "15000", "--test_iterations", "-1",
        "--save_iterations", *[str(value) for value in endpoints],
        "--empty_cache_interval", "200", "--quiet",
    ]
    protected = {
        "rgb170_result": sha256(ROOT / "outputs/quality/RGB-170/full-r1/result.json"),
        "rgb170_50k": sha256(ROOT / "outputs/quality/RGB-170/full-r1/model/point_cloud/iteration_50000/point_cloud.ply"),
        "rgb166_50k": sha256(ROOT / "outputs/quality/RGB-166/full-r1/model/point_cloud/iteration_50000/point_cloud.ply"),
        "split": sha256(scene_root / "split.json"),
        "initialization": sha256(scene_root / "scene/sparse/0/points3D.ply"),
        "diagnostic": sha256(ROOT / "outputs/quality/QUAL-022/diagnostic-r1/result.json"),
    }
    record: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "stage": args.stage,
        "status": "waiting_lock",
        "iterations": iterations,
        "repair_iteration": repair_iteration,
        "single_primary_variable": config["training"]["single_primary_variable"],
        "command": command,
        "config_sha256": sha256(config_path),
        "source": source_record(source),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "protected_before": protected,
        "free_bytes_start": shutil.disk_usage(run).free,
    }
    started = time.monotonic()
    try:
        write_new_json(run / "started.json", record)
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            active = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True,
            )
            other = [line for line in active.splitlines() if line.strip() and "ToDesk" not in line]
            if other:
                raise RuntimeError(f"Other GPU compute process active: {other}")
            record["status"] = "running"
            env = build_env(args.artifact_root, source)
            with (run / "train.log").open("x") as log:
                process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
                with (run / "resources.jsonl").open("x") as monitor:
                    while process.poll() is None:
                        gpu = subprocess.run(
                            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True,
                        )
                        monitor.write(json.dumps({
                            "seconds": time.monotonic() - started,
                            "gpu": gpu.stdout.strip(),
                            "free_bytes": shutil.disk_usage(run).free,
                        }) + "\n")
                        monitor.flush()
                        time.sleep(10)
            if process.returncode:
                raise RuntimeError(f"training exited {process.returncode}; see {run / 'train.log'}")
        worker = json.loads((run / "worker.json").read_text())
        if worker.get("status") != "completed_native_execution":
            raise RuntimeError("native bridge-repair worker did not complete")
        if len(worker["actual_train"]) != 400 or worker["actual_evaluation"]:
            raise RuntimeError("registered all400 split changed")
        if worker["repair"]["evidence_calls"] != 1 or worker["repair"]["repaired_parents"] <= 0:
            raise RuntimeError("bridge repair did not activate")
        if args.selection_mode in {
            "evidence-directed-fixed-budget",
            "responsibility-asymmetric-fixed-budget",
        }:
            accepted = int(worker["repair"]["evidence"]["accepted_gaussians"])
            repaired = int(worker["repair"]["repaired_parents"])
            native_budget = int(worker["repair"]["native_selected_parents"])
            if repaired != min(accepted, native_budget):
                raise RuntimeError("direct evidence targets were not all assigned fixed-budget split slots")
        endpoint_records = {}
        for iteration in endpoints:
            ply = run / f"model/point_cloud/iteration_{iteration}/point_cloud.ply"
            if not ply.is_file():
                raise FileNotFoundError(ply)
            endpoint_records[str(iteration)] = {
                "sha256": sha256(ply), "bytes": ply.stat().st_size, "gaussians": vertex_count(ply),
            }
        protected_after = {
            "rgb170_result": sha256(ROOT / "outputs/quality/RGB-170/full-r1/result.json"),
            "rgb170_50k": sha256(ROOT / "outputs/quality/RGB-170/full-r1/model/point_cloud/iteration_50000/point_cloud.ply"),
            "rgb166_50k": sha256(ROOT / "outputs/quality/RGB-166/full-r1/model/point_cloud/iteration_50000/point_cloud.ply"),
            "split": sha256(scene_root / "split.json"),
            "initialization": sha256(scene_root / "scene/sparse/0/points3D.ply"),
            "diagnostic": sha256(ROOT / "outputs/quality/QUAL-022/diagnostic-r1/result.json"),
        }
        if protected_after != protected:
            raise RuntimeError("Protected parent, split, initialization, or diagnostic changed")
        record.update(
            status="completed_training",
            worker=worker,
            endpoints=endpoint_records,
            protected_exact=True,
            peak_reserved_mib=worker["peak_reserved_mib"],
            repair=worker["repair"],
        )
    except BaseException as exc:
        record.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        record["free_bytes_end"] = shutil.disk_usage(run).free
        write_new_json(run / "result.json", record)


if __name__ == "__main__":
    main()
