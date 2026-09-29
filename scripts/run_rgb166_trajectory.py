#!/usr/bin/env python3
"""Launch the fixed RGB166 trajectory render in the pinned CUDA environment."""
from __future__ import annotations

from _artifact_paths import artifact_root_default, parse_artifact_root

import fcntl
import subprocess
from pathlib import Path

from run_rgb164_rgb165_improvedgs_pair import ROOT, SOURCE_REL, build_env


def main() -> None:
    artifact_root = parse_artifact_root(__doc__)
    source = ROOT / SOURCE_REL
    output = ROOT / "outputs/quality/RGB-166/full-r1/trajectory-r1"
    if output.exists():
        raise FileExistsError(output)
    command = [
        str(artifact_root / "r1/envs/r1-cuda128/bin/python"),
        str(ROOT / "scripts/render_rgb166_trajectory.py"),
        "--source", str(source),
        "--trajectory", str(artifact_root / "results/frozen_scaffold_sem001/flythrough_v1/cameras"),
        "--model", str(ROOT / "outputs/quality/RGB-166/full-r1/model"),
        "--parent-images", str(ROOT / "outputs/quality/RGB-165/full-r2/trajectory-r1/RGB165"),
        "--output", str(output),
    ]
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(command, cwd=ROOT, env=build_env(artifact_root, source), check=True)


if __name__ == "__main__":
    main()
