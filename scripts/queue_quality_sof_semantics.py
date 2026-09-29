#!/usr/bin/env python3
"""Independent SOF static-semantic comparisons after the active GPU batch exits.

The predecessor serializes resources only: a failed PGSR trial does not reject
either separately accepted SOF parent. No implicit promotion or recipe changes.
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
from src.quality_protocol import sha256, verify_hashes, write_new_json
from scripts.run_quality_static import CODE
from scripts.run_quality_stage_a import verify_audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("python", "artifact-root", "source", "wait-for"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--extra-pythonpath", type=Path, action="append", default=[])
    p.add_argument("--attempt", required=True)
    args = p.parse_args()
    if not args.attempt.replace("-", "").isalnum():
        p.error("Invalid attempt ID")
    root = ROOT / "outputs/quality"
    with (root / "static_queue.lock").open("a") as admission:
        fcntl.flock(admission, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = root / "queues" / args.attempt
        output.mkdir(parents=True, exist_ok=False)
        code = (*CODE, "scripts/queue_quality_sof_semantics.py", "scripts/audit_quality_static.py", "src/quality_semantic_audit.py")
        record = {"status": "waiting_for_predecessor_resources", "experiment_id": "SEM-330-QUALITY-STATIC", "events": [],
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "code_sha256": {n: sha256(ROOT / n) for n in code},
                  "input_hashes": {n: sha256(ROOT / n) for n in ("experiments/runs/SEM-330-QUALITY-STATIC/config.yaml",
                    "experiments/runs/SEM-330-QUALITY-STATIC/audit_config.yaml", "experiments/runs/QUAL-001-FOUNDATION/config.yaml")},
                  "predecessor_started_sha256": sha256(args.wait_for / "started.json"), "quality_selection": None}
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
            if sha256(args.wait_for / "started.json") != record["predecessor_started_sha256"]:
                raise ValueError("Predecessor identity changed")
        env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4")
        def execute(name, command, destination, status):
            unchanged()
            event(name + "_started", output=str(destination))
            with (output / (name + ".log")).open("x") as log:
                subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            unchanged()
            if json.loads((destination / "result.json").read_text())["status"] != status:
                raise ValueError("Child did not pass requested status")
            event(name + "_completed", result_sha256=sha256(destination / "result.json"))
        try:
            event("waiting_for_predecessor_resources", predecessor=str(args.wait_for))
            with (root / "stage_a_queue.lock").open("a") as batch:
                fcntl.flock(batch, fcntl.LOCK_EX)
                unchanged()
                predecessor = json.loads((args.wait_for / "result.json").read_text())
                if predecessor["status"] not in ("completed_repair_rgb_only_semantic_comparison_pending", "stopped_repair_queue_not_method_rejection"):
                    raise ValueError("Predecessor has no recognized terminal result")
                event("independent_sof_comparisons_admitted", predecessor_status=predecessor["status"])
                extra = []
                for path in args.extra_pythonpath:
                    extra.extend(["--extra-pythonpath", path])
                for experiment in ("RGB-112-SOF-NVS", "RGB-113-SOF-GEO"):
                    rgb_name = experiment + "-full-sof-serial-20260909-r2"
                    rgb = root / "runs" / rgb_name
                    audit, probe = root / "audits" / rgb_name, root / "probes" / rgb_name
                    verify_audit(audit, rgb, "completed_rgb_reload_audit")
                    verify_audit(probe, rgb, "passed_native_adapter_engineering_probe")
                    common = [args.python, ROOT / "scripts/run_quality_static.py", "--run", rgb, "--audit", audit,
                        "--probe", probe, "--source", args.source, "--artifact-root", args.artifact_root, "--partition", "development", *extra]
                    smoke = root / "semantic" / f"SEM-330-{experiment}-smoke-{args.attempt}"
                    full = root / "semantic" / f"SEM-330-{experiment}-full-{args.attempt}"
                    execute(experiment + "_smoke", common + ["--mode", "smoke", "--output", smoke], smoke, "passed_static_semantic_smoke")
                    execute(experiment + "_full", common + ["--mode", "full", "--output", full, "--smoke", smoke], full, "completed_static_fit_pending_independent_evaluation")
                    result = root / "audits" / full.name
                    execute(experiment + "_audit", [args.python, ROOT / "scripts/audit_quality_static.py", "--run", full,
                        "--rgb-run", rgb, "--source", args.source, "--artifact-root", args.artifact_root, "--output", result, *extra],
                        result, "completed_independent_static_semantic_audit")
                record["status"] = "completed_two_sof_static_comparisons_2d_experts_pending"
        except BaseException as exc:
            record.update(status="stopped_sof_semantic_queue_not_method_rejection", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            write_new_json(output / "result.json", record)
            print(json.dumps({"status": record["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
