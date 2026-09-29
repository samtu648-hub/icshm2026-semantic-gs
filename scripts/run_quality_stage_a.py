#!/usr/bin/env python3
"""One-shot serial Stage A queue: full native training, reload, metrics, adapter.

No automatic retry, model promotion, remote resource purchase or data upload.
Already completed control evidence is checked, not rerun or overwritten. Each
event is a separate immutable JSON file, so a failure remains visible. This is a
local batch job, not a recurring notification/automation or a claim Stage B ran.
"""
from __future__ import annotations
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
from scripts.run_quality_rgb import source_manifest
from src.quality_protocol import sha256, verify_hashes, write_new_json

ARM_IDS = ("RGB-111-PGSR", "RGB-112-SOF-NVS", "RGB-113-SOF-GEO")


def selected_arms(requested):
    if not requested or len(requested) != len(set(requested)) or not set(requested) <= set(ARM_IDS):
        raise ValueError("Select each registered arm at most once")
    return [name for name in ARM_IDS if name in requested]


def verify_run(run, *, mode, experiment_id, common_ply_sha256):
    record = json.loads((run / "result.json").read_text())
    if record.get("experiment_id") != experiment_id or record.get("mode") != mode or record.get("status") != "completed_native_training_pending_independent_reload_and_evaluation":
        raise ValueError("Incomplete or mismatched native run prerequisite")
    verify_hashes(run, record["artifacts"])
    verify_hashes(ROOT, record["inputs"])
    config = ROOT / "experiments/runs" / experiment_id / "config.yaml"
    if sha256(config) != record["config_sha256"]:
        raise ValueError("Registered experiment configuration changed")
    if sha256(run / "model/input.ply") != common_ply_sha256:
        raise ValueError("Run did not retain the common initialization PLY")
    return record


def verify_audit(path, run, status):
    record = json.loads((path / "result.json").read_text())
    if record.get("status") != status or record.get("run_result_sha256") != sha256(run / "result.json"):
        raise ValueError("Audit/probe does not certify this exact run")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--quality-root", type=Path, default=ROOT / "outputs/quality")
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--cuda", type=Path, required=True)
    p.add_argument("--attempt", required=True)
    p.add_argument("--arms", nargs="+", choices=ARM_IDS, default=list(ARM_IDS),
                   help="Explicit independent arms for this attempt; omitted arms remain pending, never passed")
    args = p.parse_args()
    if not args.attempt.replace("-", "").isalnum():
        raise ValueError("Invalid attempt ID")
    root = args.quality_root.resolve()
    with (root / "stage_a_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = root / "queues" / args.attempt
        output.mkdir(parents=True, exist_ok=False)
        record = {"status": "running", "stage": "A_only", "attempt": args.attempt,
                  "selected_arms": selected_arms(args.arms),
                  "omitted_arms_not_certified": [n for n in ARM_IDS if n not in args.arms],
                  "started_unix": time.time(), "events": [], "quality_selection": None,
                  "later_semantics_features_review_validation_completed": False}
        started = time.monotonic()
        try:
            protocol_path = ROOT / "experiments/runs/QUAL-001-FOUNDATION/config.yaml"
            protocol = json.loads(protocol_path.read_text())
            common_hash = sha256(args.artifact_root / protocol["rgb"]["scene"] / "sparse/0/points3D.ply")
            # Snapshot actual execution dependencies, not unrelated future Stage B
            # code. Changing any of these still stops this immutable queue attempt.
            code_files = ["scripts/run_quality_stage_a.py", "scripts/run_quality_rgb.py", "scripts/quality_upstream_worker.py",
                          "scripts/audit_quality_rgb.py", "scripts/probe_quality_adapter.py", "scripts/evaluate_rgb.py",
                          "scripts/organize_r1_renders.py", "src/quality_protocol.py", "src/quality_numeric.py",
                          "src/quality_backbone.py", "src/quality_cameras.py", "src/quality_native_gaussian.py",
                          "src/quality_scaffold.py", "src/camera_adapter.py", "src/scaffold_semantic_sidecar.py"]
            record["code_sha256"] = {n: sha256(ROOT / n) for n in code_files}
            record["protocol_sha256"] = sha256(protocol_path)
            record["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            record["common_initialization_sha256"] = common_hash
            write_new_json(output / "started.json", record)

            def event(name, **info):
                item = {"index": len(record["events"]), "name": name, "unix": time.time(), **info}
                write_new_json(output / f"event-{item['index']:03d}.json", item)
                record["events"].append(item)
                print(json.dumps(item), flush=True)

            def unchanged():
                verify_hashes(ROOT, record["code_sha256"])
                if sha256(protocol_path) != record["protocol_sha256"]:
                    raise ValueError("Protocol changed while serial batch was active")

            def run_command(name, command, env=None):
                unchanged()
                event(name + "_started", command=list(map(str, command)))
                with (output / f"{name}.log").open("x") as log:
                    subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                event(name + "_completed")

            control = root / "runs/RGB-110-SCAFFOLD-CTRL-full-r1"
            verify_run(control, mode="full", experiment_id="RGB-110-SCAFFOLD-CTRL", common_ply_sha256=common_hash)
            verify_audit(root / "audits/RGB-110-full-r1", control, "completed_rgb_reload_audit")
            verify_audit(root / "probes/Scaffold-full-r1", control, "passed_native_adapter_engineering_probe")
            event("scaffold_control_reused", run_result_sha256=sha256(control / "result.json"))

            arms = [
                ("RGB-111-PGSR", "pgsr", "common-init-cpu-r3", "PGSR-common-init-cpu-r3", [root / "source/pytorch3d"], "cpu"),
                ("RGB-112-SOF-NVS", "sof", "r1", "SOF-NVS-smoke-r1", [root / "source/fused-ssim", root / "python-overlay"], None),
                ("RGB-113-SOF-GEO", "sof", "r1", None, [root / "source/fused-ssim", root / "python-overlay"], None),
            ]
            for experiment, backend, smoke_attempt, probe_name, extras, cache in arms:
                if experiment not in record["selected_arms"]:
                    continue
                source = root / "source" / backend
                smoke = root / "runs" / f"{experiment}-smoke-{smoke_attempt}"
                verify_run(smoke, mode="smoke", experiment_id=experiment, common_ply_sha256=common_hash)
                if source_manifest(source) != json.loads((smoke / "source_manifest.json").read_text()):
                    raise ValueError("Candidate source differs from its successful smoke")
                env = dict(os.environ, CUDA_HOME=str(args.cuda), HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
                extension = "diff-plane-rasterization" if backend == "pgsr" else "diff-gaussian-rasterization"
                env["PYTHONPATH"] = os.pathsep.join(map(str, [source, source / "submodules" / extension, source / "submodules/simple-knn", ROOT, *extras]))
                def probe(run, destination, name):
                    run_command(name, [args.python, ROOT / "scripts/probe_quality_adapter.py", "--run", run, "--source", source,
                                       "--output", destination, "--artifact-root", args.artifact_root], env)
                    verify_audit(destination, run, "passed_native_adapter_engineering_probe")
                if probe_name:
                    verify_audit(root / "probes" / probe_name, smoke, "passed_native_adapter_engineering_probe")
                else:
                    probe(smoke, output / (experiment + "-smoke-probe"), experiment + "-smoke-probe")

                run = root / "runs" / f"{experiment}-full-{args.attempt}"
                command = [sys.executable, ROOT / "scripts/run_quality_rgb.py", "--experiment", experiment, "--mode", "full",
                           "--attempt", args.attempt, "--quality-root", root, "--artifact-root", args.artifact_root,
                           "--source", source, "--python", args.python, "--cuda", args.cuda]
                for extra in extras:
                    command += ["--extra-pythonpath", extra]
                if cache:
                    command += ["--image-cache-device", cache]
                run_command(experiment + "-full", command)
                verify_run(run, mode="full", experiment_id=experiment, common_ply_sha256=common_hash)
                audit = root / "audits" / f"{experiment}-full-{args.attempt}"
                command = [sys.executable, ROOT / "scripts/audit_quality_rgb.py", "--run", run, "--output", audit,
                           "--source", source, "--artifact-root", args.artifact_root, "--python", args.python, "--cuda", args.cuda]
                for extra in extras:
                    command += ["--extra-pythonpath", extra]
                run_command(experiment + "-audit", command)
                verify_audit(audit, run, "completed_rgb_reload_audit")
                probe(run, root / "probes" / f"{experiment}-full-{args.attempt}", experiment + "-full-probe")
                event("arm_completed_not_promoted", experiment_id=experiment, run_result_sha256=sha256(run / "result.json"), audit=str(audit))
            unchanged()
            record["status"] = ("completed_stage_a_rgb_only_semantic_comparison_required" if not record["omitted_arms_not_certified"]
                                else "completed_selected_arms_other_arms_not_certified")
        except BaseException as exc:
            record.update(status="stopped_stage_a_queue_not_method_rejection", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            write_new_json(output / "result.json", record)
            print(json.dumps({"status": record["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
