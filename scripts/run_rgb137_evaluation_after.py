#!/usr/bin/env python3
"""Wait for one RGB137 training attempt, then run its matched evaluation."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    run = args.run.resolve()
    result = run / "result.json"
    deadline = time.monotonic() + args.timeout_seconds
    while not result.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"training result did not appear: {result}")
        time.sleep(15)
    payload = json.loads(result.read_text(encoding="utf-8"))
    if payload.get("status") != "completed_all400":
        raise RuntimeError(f"training did not complete: {payload.get('status')}")
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_rgb137.py"),
        "--artifact-root",
        str(args.artifact_root.resolve()),
        "--run",
        str(run),
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
