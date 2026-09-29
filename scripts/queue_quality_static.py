#!/usr/bin/env python3
"""One-shot Scaffold static fitting after the existing SOF queue succeeds.

Uses local batch locks, not polling, a recurring automation, or notifications.
Never promotes a backbone, changes a recipe, retries a failure, or uploads data.
"""
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
from scripts.run_quality_static import CODE
from src.quality_protocol import sha256, verify_hashes, write_new_json


def require_sof_success(record):
    expected = {"RGB-112-SOF-NVS", "RGB-113-SOF-GEO"}
    if (record.get("status") != "completed_selected_arms_other_arms_not_certified"
            or set(record.get("selected_arms", [])) != expected
            or {e.get("experiment_id") for e in record.get("events", []) if e.get("name") == "arm_completed_not_promoted"} != expected):
        raise ValueError("SOF prerequisite has not completed both registered arms and acceptance stages")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wait-for", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--cuda", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--attempt", required=True)
    args = p.parse_args()
    if not args.attempt.replace("-", "").isalnum():
        p.error("Use an alphanumeric/hyphen attempt ID")
    root = ROOT / "outputs/quality"
    output = root / "queues" / args.attempt
    parent_started = args.wait_for / "started.json"
    with (root / "static_queue.lock").open("a") as admission:
        fcntl.flock(admission, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        record = {"status": "waiting_for_sof_batch_lock", "experiment_id": "SEM-330-QUALITY-STATIC",
                  "attempt": args.attempt, "parent_queue_started_sha256": sha256(parent_started),
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "code_sha256": {n: sha256(ROOT / n) for n in (*CODE, "scripts/queue_quality_static.py", "scripts/probe_quality_adapter.py")},
                  "input_hashes": {n: sha256(ROOT / n) for n in ("experiments/runs/SEM-330-QUALITY-STATIC/config.yaml", "experiments/runs/QUAL-001-FOUNDATION/config.yaml")},
                  "events": [], "quality_metrics": None, "other_semantic_experts_completed": False}
        started = time.monotonic()
        write_new_json(output / "started.json", record)
        def event(name, **values):
            item = {"index": len(record["events"]), "name": name, "unix": time.time(), **values}
            write_new_json(output / f"event-{item['index']:03d}.json", item)
            record["events"].append(item)
            print(json.dumps(item), flush=True)
        def unchanged():
            verify_hashes(ROOT, record["code_sha256"])
            verify_hashes(ROOT, record["input_hashes"])
            if sha256(parent_started) != record["parent_queue_started_sha256"]:
                raise ValueError("Parent queue identity changed while waiting")
        try:
            event("waiting_for_existing_sof_queue", parent=str(args.wait_for))
            # Block in this independent batch process, never occupy the GPU while
            # waiting or seize the gap between the predecessor's GPU stages.
            with (root / "stage_a_queue.lock").open("a") as predecessor:
                fcntl.flock(predecessor, fcntl.LOCK_EX)
                unchanged()
                require_sof_success(json.loads((args.wait_for / "result.json").read_text()))
                record["parent_queue_result_sha256"] = sha256(args.wait_for / "result.json")
                event("sof_complete_static_sequence_admitted")
                source = args.source.resolve()
                env = dict(os.environ, CUDA_HOME=str(args.cuda), HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4")
                env["PYTHONPATH"] = os.pathsep.join(map(str, (source, source / "submodules/diff-gaussian-rasterization", source / "submodules/simple-knn", ROOT)))
                env["LD_LIBRARY_PATH"] = str(args.cuda / "lib64") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
                run = root / "runs/RGB-110-SCAFFOLD-CTRL-full-r1"
                proof = root / "probes" / ("Scaffold-full-" + args.attempt)
                def execute(name, command, destination, status):
                    unchanged()
                    event(name + "_started", destination=str(destination))
                    with (output / (name + ".log")).open("x") as log:
                        subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    unchanged()
                    if json.loads((destination / "result.json").read_text()).get("status") != status:
                        raise ValueError("Child returned without the required acceptance status")
                    event(name + "_completed", result_sha256=sha256(destination / "result.json"))
                execute("refresh_native_probe", [args.python, ROOT / "scripts/probe_quality_adapter.py", "--run", run,
                    "--source", source, "--output", proof, "--artifact-root", args.artifact_root], proof, "passed_native_adapter_engineering_probe")
                common = [args.python, ROOT / "scripts/run_quality_static.py", "--run", run, "--audit", root / "audits/RGB-110-full-r1",
                          "--probe", proof, "--source", source, "--artifact-root", args.artifact_root, "--partition", "development"]
                smoke = root / "semantic" / ("SEM-330-Scaffold-development-smoke-" + args.attempt)
                full = root / "semantic" / ("SEM-330-Scaffold-development-full-" + args.attempt)
                execute("semantic_smoke", common + ["--mode", "smoke", "--output", smoke], smoke, "passed_static_semantic_smoke")
                execute("semantic_full", common + ["--mode", "full", "--smoke", smoke, "--output", full], full,
                        "completed_static_fit_pending_independent_evaluation")
                record["status"] = "completed_scaffold_static_fit_only_independent_evaluation_pending"
        except BaseException as exc:
            record.update(status="stopped_static_followup_not_method_rejection", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            write_new_json(output / "result.json", record)
            print(json.dumps({"status": record["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
