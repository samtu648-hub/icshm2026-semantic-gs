#!/usr/bin/env python3
"""One-shot SEM341 CPU comparison dependency, released only by accepted threefold batch."""
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
from scripts.evaluate_quality_mask2former_oof import CODE, CONFIG
from src.quality_protocol import sha256, verify_hashes, write_new_json


def require_completed_batch(result):
    if (result["status"] != "completed_mask2former_threefold_acceptance_pending_comparison"
            or result["automatic_promotion"] or result["target_labels_decoded"]):
        raise ValueError("Training/independent replay batch did not complete")
    completed = [e["stage"] for e in result["events"] if e["state"] == "completed"]
    expected = [name for part in ("oof-0", "oof-1", "oof-2")
                for name in ("full-"+part, "fresh-"+part+"-1", "fresh-"+part+"-2", "audit-"+part)]
    if completed != expected:
        raise ValueError("Missing or repeated accepted stage")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--detach", action="store_true")
    a = p.parse_args()
    batch = a.batch.resolve()
    if not batch.is_relative_to(ROOT / "outputs/quality/mask2former"):
        p.error("Use the SEM341 isolated batch")
    cfg = json.loads((ROOT / CONFIG).read_text())
    started = json.loads((batch / "queue/started.json").read_text())
    if started["config_sha256"] != sha256(ROOT / CONFIG):
        raise ValueError("Active training queue configuration differs")
    verify_hashes(ROOT, started["code_sha256"])
    for name in ("comparison-queue", "comparison", "comparison-replay"):
        if (batch / name).exists():
            raise FileExistsError("Refusing to overwrite or retry existing stage: "+name)
    if a.detach:
        with (batch / "comparison-driver.log").open("x") as log:
            child = subprocess.Popen([sys.executable, Path(__file__).resolve(), "--batch", batch,
                "--artifact-root", a.artifact_root.resolve()], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, OMP_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1"))
        record = {"status": "launched_bounded_comparison_dependency", "pid": child.pid, "unix": time.time(),
                  "new_training": False, "automatic_promotion": False, "script_sha256": sha256(Path(__file__))}
        write_new_json(batch / "comparison-launch.json", record)
        print(json.dumps(record), flush=True)
        return
    with (ROOT / "outputs/quality/mask2former_comparison.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = batch / "comparison-queue"
        output.mkdir(exist_ok=False)
        start = time.monotonic()
        code = sorted(set(CODE + ("scripts/queue_quality_mask2former_comparison.py",)))
        record = {"status": "waiting_accepted_threefold_batch", "experiment_id": cfg["experiment_id"],
            "git_commit": cfg["git_commit"], "seed": cfg["seed"], "pid": os.getpid(), "automatic_promotion": False,
            "config_sha256": sha256(ROOT / CONFIG), "training_started_sha256": sha256(batch / "queue/started.json"),
            "code_sha256": {n: sha256(ROOT/n) for n in code}, "events": []}
        for name in (*code, CONFIG):
            dest = output / "code" / name
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, dest)
        write_new_json(output / "started.json", record)

        def guard():
            verify_hashes(ROOT, {CONFIG: record["config_sha256"], **record["code_sha256"]})
            verify_hashes(batch, {"queue/started.json": record["training_started_sha256"]})
            if shutil.disk_usage(batch).free < cfg["resource"]["minimum_free_gib"]*1024**3:
                raise OSError("100GiB reserve unavailable")

        def event(state, **kw):
            row = {"state": state, "unix": time.time(), **kw}
            write_new_json(output / f"event-{len(record['events']):03d}.json", row)
            record["events"].append(row); print(json.dumps(row), flush=True)

        try:
            guard(); event("waiting_training_queue_lock")
            # Blocking dependency in this detached local batch, not a polling loop or scheduled task.
            with (ROOT / "outputs/quality/mask2former_queue.lock").open("a") as dependency:
                fcntl.flock(dependency, fcntl.LOCK_EX)
                guard()
                result = json.loads((batch / "queue/result.json").read_text())
                require_completed_batch(result)
                for row in result["events"]:
                    if row["state"] == "completed":
                        verify_hashes(batch, {row["stage"]+"/result.json": row["result_sha256"]})
                record["training_result_sha256"] = sha256(batch / "queue/result.json")
                event("accepted_threefold_batch")
            for stage, expected in (("comparison", "completed_mask2former_comparison_pending_independent_replay"),
                                    ("comparison-replay", "completed_mask2former_comparison_independent_replay")):
                guard(); event("started", stage=stage)
                cmd = [sys.executable, ROOT / "scripts/evaluate_quality_mask2former_oof.py", "--batch", batch,
                       "--artifact-root", a.artifact_root.resolve(), "--output", batch / stage]
                if stage == "comparison-replay":
                    cmd += ["--replay", batch / "comparison"]
                with (output / (stage+".log")).open("x") as log:
                    subprocess.run(list(map(str, cmd)), cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, check=True, timeout=cfg["resource"]["maximum_seconds_per_stage"],
                        env=dict(os.environ, OMP_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1"))
                guard()
                actual = json.loads((batch / stage / "result.json").read_text())
                if actual["status"] != expected:
                    raise ValueError("Incomplete comparison/replay stage")
                event("completed", stage=stage, result_sha256=sha256(batch / stage / "result.json"))
            record["status"] = "completed_mask2former_quality_batch_requires_human_review"
        except BaseException as exc:
            record.update(status="stopped_mask2former_comparison_dependency_not_method_rejection", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic()-start
            write_new_json(output / "result.json", record)
            print(json.dumps({"status": record["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
