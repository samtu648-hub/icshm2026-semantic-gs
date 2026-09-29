#!/usr/bin/env python3
"""Run the queued RGB169 training, evaluation and trajectory stages sequentially."""
from __future__ import annotations

from _artifact_paths import artifact_root_default, parse_artifact_root

import fcntl
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_rgb164_rgb165_improvedgs_pair import SOURCE_REL, build_env


def main() -> None:
    artifact_root = parse_artifact_root(__doc__)
    python = artifact_root / "r1/envs/r1-cuda128/bin/python"
    source = ROOT / SOURCE_REL
    subprocess.run([sys.executable, str(ROOT / "scripts/run_rgb169_improvedgs_densify20k.py")], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/run_rgb169_evaluation.py")], cwd=ROOT, check=True)
    run = ROOT / "outputs/quality/RGB-169/full-r1"
    trajectory_output = run / "trajectory-r1"
    command = [
        str(python), str(ROOT / "scripts/render_rgb169_trajectory.py"),
        "--source", str(source),
        "--trajectory", str(artifact_root / "results/frozen_scaffold_sem001/flythrough_v1/cameras"),
        "--model", str(run / "model"),
        "--baseline-images", str(ROOT / "outputs/quality/RGB-166/full-r1/trajectory-r1/50000"),
        "--output", str(trajectory_output),
    ]
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(command, cwd=ROOT, env=build_env(artifact_root, source), check=True)
    summary = {
        "status": "completed_training_evaluation_trajectory",
        "training": json.loads((run / "result.json").read_text()),
        "evaluation": json.loads((run / "evaluation-r1/result.json").read_text()),
        "trajectory": json.loads((trajectory_output / "result.json").read_text()),
    }
    (run / "queue-result.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
