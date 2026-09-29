#!/usr/bin/env python3
"""Run the frozen RGB147-149 full-training arms serially."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, write_new_json


CONFIGS = (
    "experiments/runs/RGB-147-STANDARD-GT-SUPERVISION-CTRL/config.yaml",
    "experiments/runs/RGB-148-STANDARD-GT-REGION-RGB/config.yaml",
    "experiments/runs/RGB-149-STANDARD-GT-REGION-GRADIENT/config.yaml",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/quality/rgb147/queue-r1"
    )
    parser.add_argument("--minimum-free-gib", type=float, default=25.0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    python = args.artifact_root.resolve() / "r1/envs/r1-cuda128/bin/python"
    minimum_free = int(args.minimum_free_gib * 2**30)
    started = time.monotonic()
    result = {
        "schema_version": 1,
        "status": "running",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "preflight_sha256": sha256(
            ROOT / "outputs/quality/rgb147/preflight-r4/result.json"
        ),
        "runs": [],
    }
    try:
        for config_name in CONFIGS:
            free_before = shutil.disk_usage(ROOT).free
            if free_before < minimum_free:
                raise RuntimeError(
                    f"Only {free_before / 2**30:.1f} GiB free before {config_name}"
                )
            config_path = ROOT / config_name
            config = json.loads(config_path.read_text())
            output = ROOT / config["output"]
            command = [
                str(python),
                str(ROOT / "scripts/run_standard_rgb_supervision.py"),
                "--artifact-root",
                str(args.artifact_root.resolve()),
                "--config",
                str(config_path),
                "--output",
                str(output),
            ]
            launch = args.output / f"{config['experiment_id']}.log"
            with launch.open("x") as stream:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            completed = json.loads((output / "completed.json").read_text())
            if completed["status"] != "completed_all400":
                raise RuntimeError(f"Incomplete arm: {config['experiment_id']}")
            result["runs"].append(
                {
                    "experiment_id": config["experiment_id"],
                    "output": config["output"],
                    "config_sha256": sha256(config_path),
                    "completed_sha256": sha256(output / "completed.json"),
                    "camera_sequence_sha256": sha256(output / "camera_sequence.txt"),
                    "wall_seconds": completed["training_wall_seconds_excluding_lock"],
                    "free_bytes_before": free_before,
                    "free_bytes_after": shutil.disk_usage(ROOT).free,
                }
            )
        camera_hashes = {run["camera_sequence_sha256"] for run in result["runs"]}
        if len(camera_hashes) != 1:
            raise ValueError("Full A/B/C camera sequences differ")
        result.update(status="completed_all400_triplet", camera_sequences_exact=True)
    except BaseException as error:
        result.update(
            status="failed_preserved", error=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        result.update(
            seconds=time.monotonic() - started,
            free_bytes_end=shutil.disk_usage(ROOT).free,
        )
        write_new_json(args.output / "result.json", result)
        print(json.dumps({"status": result["status"], "runs": result["runs"]}))


if __name__ == "__main__":
    main()
