#!/usr/bin/env python3
"""Run RGB166 endpoint evaluation after the background training service completes."""
from __future__ import annotations

from _artifact_paths import artifact_root_default, parse_artifact_root

import fcntl
import json
import subprocess
from pathlib import Path

from run_rgb164_rgb165_improvedgs_pair import ROOT, SOURCE_REL, build_env


def main() -> None:
    artifact_root = parse_artifact_root(__doc__)
    source = ROOT / SOURCE_REL
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        training_summary = ROOT / "outputs/quality/RGB-166/horizon-summary-r1.json"
        if not training_summary.is_file():
            raise FileNotFoundError(training_summary)
        training = json.loads(training_summary.read_text())
        if training.get("status") != "completed_training":
            raise RuntimeError(f"RGB166 training did not complete: {training.get('status')}")
        output = ROOT / "outputs/quality/RGB-166/full-r1/evaluation-r1"
        if output.exists():
            raise FileExistsError(output)
        command = [
            str(artifact_root / "r1/envs/r1-cuda128/bin/python"),
            str(ROOT / "scripts/evaluate_rgb166_horizon.py"),
            "--source", str(source),
            "--scene", str(ROOT / "outputs/quality/all_data/RGB-120-r2/scene"),
            "--model", str(ROOT / "outputs/quality/RGB-166/full-r1/model"),
            "--labels", str(ROOT / "outputs/quality/all_data/SEM-370-r1/labels"),
            "--parent-images", str(ROOT / "outputs/quality/RGB-165/full-r2/evaluation-r2/images/RGB165/30000"),
            "--output", str(output),
        ]
        subprocess.run(command, cwd=ROOT, env=build_env(artifact_root, source), check=True)


if __name__ == "__main__":
    main()
