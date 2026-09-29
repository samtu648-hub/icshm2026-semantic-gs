#!/usr/bin/env python3
"""Serialize RGB167 evaluation with the repository GPU lock."""
from __future__ import annotations

from _artifact_paths import artifact_root_default, parse_artifact_root

import fcntl
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    artifact_root = parse_artifact_root(__doc__)
    source = artifact_root / "r1/third_party/scaffold-gs"
    cuda = artifact_root / "r1/toolchain/cuda-12.8"
    python = artifact_root / "r1/envs/r1-cuda128/bin/python"
    run = ROOT / "outputs/quality/RGB-167/full-r1"
    output = run / "evaluation-r1"
    env = os.environ.copy()
    env.update(
        CUDA_HOME=str(cuda),
        PATH=str(cuda / "bin") + os.pathsep + env.get("PATH", ""),
        LD_LIBRARY_PATH=str(cuda / "lib64") + os.pathsep + env.get("LD_LIBRARY_PATH", ""),
        PYTHONPATH=os.pathsep.join(map(str, [source, source / "submodules/diff-gaussian-rasterization", source / "submodules/simple-knn", ROOT])),
        TORCH_HOME=str(artifact_root / "r1/weights/torch"),
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1",
    )
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True
        )
        active = [line for line in apps.splitlines() if line.strip() and "ToDesk" not in line]
        if active:
            raise RuntimeError("Other GPU process active: " + " | ".join(active))
        command = [
            str(python), str(ROOT / "scripts/evaluate_rgb167_scaffold_growth20k.py"),
            "--artifact-root", str(artifact_root), "--run", str(run), "--output", str(output),
        ]
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
