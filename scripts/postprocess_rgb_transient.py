#!/usr/bin/env python3
"""Wait for this exact training queue, then generate its registered audits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--queue-status", type=Path, required=True)
    args = p.parse_args()
    root = args.output_root
    output = root / "r2/rgb_092/postprocess"
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "waiting_for_registered_queue", "started": time.time(), "jobs": []}
    status_path = output / "status.json"
    status_path.write_text(json.dumps(record, indent=2))
    deadline = time.monotonic() + 3 * 3600
    try:
        while time.monotonic() < deadline:
            queue = json.loads(args.queue_status.read_text())
            if queue["status"] == "failed":
                raise RuntimeError("Training queue failed; no downstream audit allowed")
            if queue["status"] == "completed":
                break
            time.sleep(30)
        else:
            raise TimeoutError("Registered queue did not complete in 3 hours")
        source = root / "r1/third_party/scaffold-gs"
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": "4", "TORCH_HOME": str(root / "r1/weights/torch"),
          "PYTHONPATH": ":".join(map(str, [REPO, source / "submodules/diff-gaussian-rasterization", source / "submodules/simple-knn", source]))})
        for script in ["summarize_rgb_transient.py", "audit_rgb092_masks.py", "render_rgb_transient_flythrough.py"]:
            record.update(status="running", active=script)
            status_path.write_text(json.dumps(record, indent=2))
            with (output / (script + ".log")).open("w") as log:
                result = subprocess.run([sys.executable, str(REPO / "scripts" / script), "--output-root", str(root)],
                            cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            record["jobs"].append({"script": script, "returncode": result.returncode, "finished": time.time()})
            if result.returncode:
                raise RuntimeError("Audit failed: " + script)
        record["status"] = "completed_needs_visual_review"
    except Exception as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        record["ended"] = time.time()
        status_path.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
