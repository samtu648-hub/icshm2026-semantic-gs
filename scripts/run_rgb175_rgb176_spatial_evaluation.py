#!/usr/bin/env python3
"""Run QUAL026 full-image, bridge, difficult-group, and trajectory evaluation."""
from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_rgb164_rgb165_improvedgs_pair import SOURCE_REL, build_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / SOURCE_REL
    python = args.artifact_root / "r1/envs/r1-cuda128/bin/python"
    output = ROOT / "outputs/quality/QUAL-026/r1"
    if output.exists():
        raise FileExistsError(output)
    commands = [
        [str(python), str(ROOT / "scripts/evaluate_rgb175_rgb176_spatial_pair.py")],
        [str(python), str(ROOT / "scripts/render_rgb175_rgb176_trajectory.py"),
         "--trajectory", str(args.artifact_root / "results/frozen_scaffold_sem001/flythrough_v1/cameras")],
    ]
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True,
        )
        other = [line for line in active.splitlines() if line.strip() and "ToDesk" not in line]
        if other:
            raise RuntimeError(f"Other GPU compute process active: {other}")
        environment = build_env(args.artifact_root, source)
        for command in commands:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
