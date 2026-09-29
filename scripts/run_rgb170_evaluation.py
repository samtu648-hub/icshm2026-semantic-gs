#!/usr/bin/env python3
"""Run RGB170 evaluation after its queued training completes."""
from __future__ import annotations

from _artifact_paths import artifact_root_default, parse_artifact_root

import fcntl
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_rgb164_rgb165_improvedgs_pair import ROOT, SOURCE_REL, build_env


def main() -> None:
    artifact_root = parse_artifact_root(__doc__)
    source = ROOT / SOURCE_REL
    run = ROOT / "outputs/quality/RGB-170/full-r1"
    training = json.loads((run / "result.json").read_text())
    if training.get("status") != "completed_training":
        raise RuntimeError(f"RGB170 training is not complete: {training.get('status')}")
    output = run / "evaluation-r1"
    if output.exists():
        raise FileExistsError(output)
    command = [
        str(artifact_root / "r1/envs/r1-cuda128/bin/python"),
        str(ROOT / "scripts/evaluate_rgb170_budget15m.py"),
        "--source", str(source),
        "--scene", str(ROOT / "outputs/quality/all_data/RGB-120-r2/scene"),
        "--model", str(run / "model"),
        "--labels", str(ROOT / "outputs/quality/all_data/SEM-370-r1/labels"),
        "--baseline-images", str(ROOT / "outputs/quality/RGB-166/full-r1/evaluation-r1/images/50000"),
        "--output", str(output),
    ]
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(command, cwd=ROOT, env=build_env(artifact_root, source), check=True)


if __name__ == "__main__":
    main()
