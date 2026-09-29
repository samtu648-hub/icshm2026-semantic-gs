#!/usr/bin/env python3
"""Run the registered SEM382 stages sequentially with preserved logs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/quality/SEM-382/r1"


def save(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n")


def completed(path: Path) -> bool:
    return path.exists() and json.loads(path.read_text()).get("status", "").startswith(
        "completed"
    )


def next_log(name: str) -> Path:
    path = OUT / f"{name}.log"
    if not path.exists():
        return path
    attempt = 2
    while (OUT / f"{name}-attempt{attempt}.log").exists():
        attempt += 1
    return OUT / f"{name}-attempt{attempt}.log"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    args.artifact_root = args.artifact_root.resolve()
    if not completed(OUT / "prepare/result.json"):
        raise ValueError("Run prepare_sem382.py first")
    if "DINOV3_VITB16_DIR" not in os.environ:
        raise ValueError("DINOV3_VITB16_DIR is required")
    path = OUT / "driver.json"
    recovery_path = OUT / "recovery.json"
    record = {
        "schema_version": 1,
        "status": "running",
        "experiment_id": "SEM-382-DINOV3-VITB-SFP-LR1E4-ALL300",
        "started_unix": time.time(),
        "stages": [],
    }
    stages = [
        (
            "preflight",
            [
                "scripts/run_sem382.py",
                "--artifact-root",
                str(args.artifact_root),
                "--stage",
                "preflight",
            ],
            OUT / "preflight/result.json",
        ),
        (
            "training",
            [
                "scripts/run_sem382.py",
                "--artifact-root",
                str(args.artifact_root),
                "--stage",
                "mask2former",
            ],
            OUT / "mask2former/result.json",
        ),
        (
            "inference-3000",
            [
                "scripts/run_sem382.py",
                "--artifact-root",
                str(args.artifact_root),
                "--stage",
                "inference-3000",
            ],
            OUT / "inference-3000/result.json",
        ),
        (
            "inference-6000",
            [
                "scripts/run_sem382.py",
                "--artifact-root",
                str(args.artifact_root),
                "--stage",
                "inference-6000",
            ],
            OUT / "inference-6000/result.json",
        ),
        (
            "verification",
            ["scripts/verify_sem382.py", "--artifact-root", str(args.artifact_root)],
            OUT / "verification.json",
        ),
        ("audit", ["scripts/audit_sem382.py"], OUT / "audit.json"),
        (
            "comparison",
            ["scripts/build_sem382_comparison.py"],
            OUT / "comparison/result.json",
        ),
    ]
    try:
        for name, command, result_path in stages:
            if completed(result_path):
                record["stages"].append({"name": name, "status": "already_completed"})
                continue
            record.update(stage=name, status=f"running_{name}")
            save(path, record)
            stage_started = time.monotonic()
            log_path = next_log(name)
            with log_path.open("x") as log:
                subprocess.run(
                    [sys.executable, *command],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            if not completed(result_path):
                raise ValueError(
                    f"Stage did not produce a completed result: {result_path}"
                )
            record["stages"].append(
                {
                    "name": name,
                    "status": "completed",
                    "seconds": time.monotonic() - stage_started,
                    "log": log_path.name,
                }
            )
            done = {row["name"] for row in record["stages"]}
            save(
                recovery_path,
                {
                    "status": "recoverable_at_stage_boundary",
                    "completed_stages": sorted(done),
                    "next_stage": next(
                        (stage[0] for stage in stages if stage[0] not in done), None
                    ),
                    "note": "3000/6000 are analysis/primary endpoints; no mid-training resume or quality-based checkpoint selection",
                },
            )
        record["status"] = "artifacts_ready_pending_visual_review"
    except BaseException as exc:
        record.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["finished_unix"] = time.time()
        save(path, record)


if __name__ == "__main__":
    main()
