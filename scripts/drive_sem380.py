#!/usr/bin/env python3
"""Durable sequential driver for the authorized SEM380 system comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/quality/SEM-380/r2"


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
    assert completed(OUT / "prepare/result.json"), "Run prepare_sem380.py first"
    path = OUT / "driver.json"
    recovery_path = OUT / "recovery.json"
    record = {
        "schema_version": 1,
        "status": "running",
        "experiment_id": "SEM-380-RGB148-GT-REGION-RGB-ALL300",
        "started_unix": time.time(),
        "stages": [],
    }
    stages = [
        (
            "preflight",
            [
                "scripts/run_sem380.py",
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
                "scripts/run_sem380.py",
                "--artifact-root",
                str(args.artifact_root),
                "--stage",
                "mask2former",
            ],
            OUT / "mask2former/result.json",
        ),
        (
            "inference",
            [
                "scripts/run_sem380.py",
                "--artifact-root",
                str(args.artifact_root),
                "--stage",
                "inference",
            ],
            OUT / "inference/result.json",
        ),
        (
            "verification",
            [
                "scripts/verify_sem380.py",
                "--artifact-root",
                str(args.artifact_root),
            ],
            OUT / "verification.json",
        ),
        ("audit", ["scripts/audit_sem380.py"], OUT / "audit.json"),
        (
            "comparison",
            ["scripts/build_sem380_comparison.py"],
            OUT / "comparison/result.json",
        ),
    ]
    try:
        for name, command, result_path in stages:
            if completed(result_path):
                record["stages"].append({"name": name, "status": "already_completed"})
                continue
            record["stage"] = name
            record["status"] = f"running_{name}"
            save(path, record)
            started = time.monotonic()
            log_path = next_log(name)
            with log_path.open("x") as log:
                subprocess.run(
                    [sys.executable, *command],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            assert completed(result_path), result_path
            record["stages"].append(
                {
                    "name": name,
                    "status": "completed",
                    "seconds": time.monotonic() - started,
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
                    "note": "3000/6000 checkpoints preserve endpoints; optimizer/RNG are not resumable mid-stage",
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
