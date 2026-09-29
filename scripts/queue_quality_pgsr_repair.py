#!/usr/bin/env python3
"""One-shot accepted repair smoke probe -> fresh full PGSR -> RGB audit/probe."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
from scripts.run_quality_stage_a import verify_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("python", "cuda", "artifact-root", "source", "extra-pythonpath", "smoke-run"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--attempt", required=True)
    args = p.parse_args()
    if not args.attempt.replace("-", "").isalnum():
        p.error("Invalid attempt ID")
    root, experiment = ROOT / "outputs/quality", "RGB-115-PGSR-SAFE-CYCLE"
    with (root / "stage_a_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = root / "queues" / args.attempt
        output.mkdir(parents=True, exist_ok=False)
        record = {"status": "running", "experiment_id": experiment, "events": [], "quality_selection": None,
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        code = ("scripts/queue_quality_pgsr_repair.py", "scripts/run_quality_rgb.py", "scripts/run_quality_stage_a.py",
                "scripts/quality_upstream_worker.py", "scripts/audit_quality_rgb.py", "scripts/probe_quality_adapter.py",
                "scripts/evaluate_rgb.py", "scripts/organize_r1_renders.py", "src/quality_numeric.py", "src/quality_protocol.py",
                "src/quality_pgsr_geometry.py", "src/quality_pgsr_prerequisite.py", "src/quality_backbone.py",
                "src/quality_native_gaussian.py", "src/quality_scaffold.py", "src/quality_cameras.py", "src/camera_adapter.py", "src/scaffold_semantic_sidecar.py")
        record["code_sha256"] = {n: sha256(ROOT / n) for n in code}
        record["input_hashes"] = {n: sha256(ROOT / n) for n in ("experiments/runs/QUAL-001-FOUNDATION/config.yaml",
            "experiments/runs/RGB-115-PGSR-SAFE-CYCLE/config.yaml", "experiments/runs/RGB-115-PGSR-SAFE-CYCLE/execution_config.yaml")}
        started = time.monotonic()
        write_new_json(output / "started.json", record)
        def event(name, **kw):
            item = {"index": len(record["events"]), "name": name, "unix": time.time(), **kw}
            write_new_json(output / f"event-{item['index']:03d}.json", item)
            record["events"].append(item)
            print(json.dumps(item), flush=True)
        source = args.source.resolve()
        env = dict(os.environ, CUDA_HOME=str(args.cuda), HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4")
        env["PYTHONPATH"] = os.pathsep.join(map(str, [source, source / "submodules/diff-plane-rasterization", source / "submodules/simple-knn", ROOT, args.extra_pythonpath.resolve()]))
        def execute(name, command, target, status):
            verify_hashes(ROOT, record["code_sha256"])
            verify_hashes(ROOT, record["input_hashes"])
            event(name + "_started", output=str(target))
            with (output / (name + ".log")).open("x") as log:
                subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            child = json.loads((target / "result.json").read_text())
            if child["status"] != status:
                raise ValueError("Child did not pass registered status gate")
            event(name + "_completed", result_sha256=sha256(target / "result.json"))
        try:
            protocol = json.loads((ROOT / "experiments/runs/QUAL-001-FOUNDATION/config.yaml").read_text())
            verify_run(args.smoke_run, mode="smoke", experiment_id=experiment,
                       common_ply_sha256=sha256(args.artifact_root / protocol["rgb"]["scene"] / "sparse/0/points3D.ply"))
            smoke_probe = output / "smoke-probe"
            execute("smoke_probe", [args.python, ROOT / "scripts/probe_quality_adapter.py", "--run", args.smoke_run,
                "--source", source, "--output", smoke_probe, "--artifact-root", args.artifact_root], smoke_probe, "passed_native_adapter_engineering_probe")
            run = root / "runs" / f"{experiment}-full-{args.attempt}"
            common = ["--artifact-root", args.artifact_root, "--source", source, "--python", args.python, "--cuda", args.cuda,
                      "--extra-pythonpath", args.extra_pythonpath]
            execute("full", [sys.executable, ROOT / "scripts/run_quality_rgb.py", "--experiment", experiment, *common,
                "--mode", "full", "--smoke-run", args.smoke_run, "--attempt", args.attempt], run, "completed_native_training_pending_independent_reload_and_evaluation")
            audit = root / "audits" / run.name
            execute("audit", [sys.executable, ROOT / "scripts/audit_quality_rgb.py", "--run", run, "--output", audit, *common], audit, "completed_rgb_reload_audit")
            probe = root / "probes" / run.name
            execute("full_probe", [args.python, ROOT / "scripts/probe_quality_adapter.py", "--run", run, "--source", source,
                "--output", probe, "--artifact-root", args.artifact_root], probe, "passed_native_adapter_engineering_probe")
            verify_hashes(ROOT, record["code_sha256"])
            record["status"] = "completed_repair_rgb_only_semantic_comparison_pending"
        except BaseException as exc:
            record.update(status="stopped_repair_queue_not_method_rejection", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            write_new_json(output / "result.json", record)
            print(json.dumps({"status": record["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
