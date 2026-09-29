#!/usr/bin/env python3
"""Run the complete RGB174 reload, all400, trajectory and speed evaluation."""

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
    source = ROOT / SOURCE_REL
    run = ROOT / "outputs/quality/RGB-174/full-r1"
    training = json.loads((run / "result.json").read_text())
    if training.get("status") != "completed_training":
        raise RuntimeError(f"RGB174 training is not complete: {training.get('status')}")
    ply = run / "model/point_cloud/iteration_50000/point_cloud.ply"
    python = artifact_root / "r1/envs/r1-cuda128/bin/python"
    scene = ROOT / "outputs/quality/all_data/RGB-120-r2/scene"
    trajectory = artifact_root / "results/frozen_scaffold_sem001/flythrough_v1/cameras"
    commands = []
    if not (run / "reload-audit.json").exists():
        commands.append([
            str(python), str(ROOT / "scripts/audit_rgb171_preflight.py"),
            "--source", str(source), "--scene", str(scene), "--ply", str(ply),
            "--output", str(run / "reload-audit.json"),
        ])
    if not (run / "evaluation-r1/result.json").exists():
        commands.append([
            str(python), str(ROOT / "scripts/evaluate_rgb171_bridge_repair.py"),
            "--source", str(source), "--scene", str(scene),
            "--labels", str(ROOT / "outputs/quality/all_data/SEM-370-r1/labels"),
            "--ply", str(ply),
            "--rgb166-images", str(ROOT / "outputs/quality/RGB-166/full-r1/evaluation-r1/images/50000"),
            "--rgb170-images", str(ROOT / "outputs/quality/RGB-170/full-r1/evaluation-r1/images/50000"),
            "--training-result", str(run / "result.json"),
            "--output", str(run / "evaluation-r1"),
            "--candidate-key", "RGB174_50000", "--candidate-display", "RGB174 50k",
        ])
    if not (run / "trajectory-r1/result.json").exists():
        partial = run / "trajectory-r1"
        if partial.exists():
            raise FileExistsError(f"Preserve or relocate incomplete trajectory output before resuming: {partial}")
        commands.append([
            str(python), str(ROOT / "scripts/render_rgb171_trajectory.py"),
            "--source", str(source),
            "--trajectory", str(trajectory),
            "--ply", str(ply),
            "--rgb166-images", str(ROOT / "outputs/quality/RGB-166/full-r1/trajectory-r1/50000"),
            "--rgb170-images", str(ROOT / "outputs/quality/RGB-170/full-r1/trajectory-r1/50000"),
            "--output", str(run / "trajectory-r1"),
            "--candidate-key", "RGB174_50000", "--candidate-display", "RGB174 50k",
        ])
    if not (run / "render-benchmark.json").exists():
        commands.append([
            str(python), str(ROOT / "scripts/benchmark_rgb171_render.py"),
            "--source", str(source), "--scene", str(scene),
            "--rgb166", str(ROOT / "outputs/quality/RGB-166/full-r1/model/point_cloud/iteration_50000/point_cloud.ply"),
            "--rgb170", str(ROOT / "outputs/quality/RGB-170/full-r1/model/point_cloud/iteration_50000/point_cloud.ply"),
            "--rgb171", str(ply),
            "--output", str(run / "render-benchmark.json"),
            "--candidate-key", "RGB174_50000",
        ])
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True,
        )
        other = [line for line in active.splitlines() if line.strip() and "ToDesk" not in line]
        if other:
            raise RuntimeError(f"Other GPU compute process active: {other}")
        env = build_env(artifact_root, source)
        for command in commands:
            subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
