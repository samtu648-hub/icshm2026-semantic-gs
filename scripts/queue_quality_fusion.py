#!/usr/bin/env python3
"""One-shot dependency continuation: wait for fits, calibrate, stop for review."""
import argparse
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
from src.quality_protocol import sha256, verify_hashes, write_new_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("wait-for", "python", "artifact-root", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    with (ROOT / "outputs/quality/fusion_queue.lock").open("a") as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        code = ("scripts/queue_quality_fusion.py", "scripts/evaluate_quality_fusion.py", "src/quality_fusion.py",
            "src/quality_experts.py", "src/quality_deck_evidence.py", "src/quality_static_training.py", "src/quality_semantic_audit.py",
            "src/semantic_protected_fusion.py", "src/semantic2d.py", "src/quality_protocol.py")
        configs = ("experiments/runs/SEM-331-QUALITY-EXPERTS/fusion_config.yaml", "experiments/runs/SEM-332-DECK-EVIDENCE-OOF/config.yaml")
        record = {"status": "waiting_for_registered_fit_batch", "events": [], "automatic_promotion": False,
            "code_sha256": {n: sha256(ROOT / n) for n in code}, "config_sha256": {n: sha256(ROOT / n) for n in configs},
            "parent_started_sha256": sha256(a.wait_for / "started.json"),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        for name in (*code, *configs):
            dest = a.output / "code" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        write_new_json(a.output / "started.json", record)
        def event(name, **kw):
            e = {"index": len(record["events"]), "name": name, "unix": time.time(), **kw}
            write_new_json(a.output / f"event-{e['index']:03d}.json", e)
            record["events"].append(e)
            print(json.dumps(e), flush=True)
        def unchanged():
            verify_hashes(ROOT, record["code_sha256"])
            verify_hashes(ROOT, record["config_sha256"])
            if sha256(a.wait_for / "started.json") != record["parent_started_sha256"]:
                raise ValueError("Waiting for a different fit batch")
        try:
            event("waiting_without_gpu", parent=str(a.wait_for))
            # Blocking occurs in this independent local process, never in the
            # interactive agent or holding the GPU. No polling/recurring task.
            with (ROOT / "outputs/quality/stage_a_queue.lock").open("a") as predecessor:
                fcntl.flock(predecessor, fcntl.LOCK_EX)
                unchanged()
                parent = json.loads((a.wait_for / "result.json").read_text())
                if parent["status"] != "completed_four_arm_inputs_fusion_calibration_pending":
                    raise ValueError("Fit batch did not finish; calibration refused")
                record["parent_result_sha256"] = sha256(a.wait_for / "result.json")
                jobs = [(n, False) for n in ("Scaffold", "PGSR", "SOF-NVS", "SOF-GEO")]+[("Scaffold", True)]
                for arm, extra in jobs:
                    unchanged()
                    stage = "SEM332-Scaffold" if extra else "fusion-"+arm
                    dest = a.wait_for / stage
                    event("started", stage=stage, output=str(dest))
                    cmd = [a.python, ROOT / "scripts/evaluate_quality_fusion.py", "--batch", a.wait_for,
                        "--artifact-root", a.artifact_root, "--arm", arm, "--output", dest]
                    if extra:
                        cmd.append("--deck-evidence")
                    with (a.output / (stage+".log")).open("x") as log:
                        subprocess.run(list(map(str, cmd)), cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1"),
                            stdout=log, stderr=subprocess.STDOUT, check=True)
                    unchanged()
                    result = json.loads((dest / "result.json").read_text())
                    if result["status"] != "completed_matched_fusion_pending_human_review":
                        raise ValueError("Fusion did not complete registered evaluation")
                    event("completed", stage=stage, result_sha256=sha256(dest / "result.json"))
                record["status"] = "completed_matched_system_evidence_requires_human_promotion_review"
        except BaseException as exc:
            record.update(status="stopped_fusion_followup_not_method_rejection", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic()-started
            write_new_json(a.output / "result.json", record)
            print(json.dumps({"status": record["status"], "output": str(a.output)}), flush=True)


if __name__ == "__main__":
    main()
