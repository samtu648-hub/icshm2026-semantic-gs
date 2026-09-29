#!/usr/bin/env python3
"""Immediate, bounded user-authorized queue; run under a user service for long jobs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--queue-name", default="queue")
    args = p.parse_args()
    root = args.output_root
    output = root / "r2/rgb_092" / args.queue_name
    output.mkdir(parents=True, exist_ok=False)
    # Preserve and explicitly close the interrupted engineering smoke record.
    previous = root / "r1/runs/RGB-092-TRANSIENT-SMOKE/execution.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        if old["status"] == "running":
            old.update(status="interrupted_exec_session_exit143", error="First exec session ended near its 180s lifetime; no CUDA traceback; no final checkpoint. Retry identical smoke in supervised long-lived user job.")
            previous.write_text(json.dumps(old, indent=2))
    stopped = threading.Event()
    def monitor():
        with (output / "gpu_samples.jsonl").open("w") as log:
            while not stopped.is_set():
                r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
                log.write(json.dumps({"time": time.time(), "gpu": r.stdout.strip(), "returncode": r.returncode}) + "\n")
                log.flush()
                stopped.wait(30)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    jobs = [("RGB-092-TRANSIENT", "smoke", ["--run-suffix=-RETRY1"]),
            ("RGB-091-CTRL", "full", []), ("RGB-092-TRANSIENT", "full", [])]
    status = {"status": "running", "started": time.time(), "jobs": []}
    try:
        for ident, mode, extra in jobs:
            status["active"] = ident + ":" + mode
            (output / "status.json").write_text(json.dumps(status, indent=2))
            command = [sys.executable, str(REPO / "scripts/run_rgb_transient.py"), "--config", str(REPO / f"experiments/runs/{ident}/config.yaml"),
                       "--output-root", str(root), "--mode", mode, "--dino-dir", str(root / "r2/rgb_092/dino"), *extra]
            with (output / f"{ident}-{mode}.log").open("w") as log:
                result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            status["jobs"].append({"experiment": ident, "mode": mode, "returncode": result.returncode, "finished": time.time()})
            if result.returncode:
                raise RuntimeError("Queue job failed: " + status["active"])
        status["status"] = "completed"
    except Exception as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        stopped.set()
        thread.join(timeout=5)
        status["ended"] = time.time()
        (output / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
