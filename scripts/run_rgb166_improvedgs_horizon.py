#!/usr/bin/env python3
"""Run the RGB165 ImprovedGS recipe to 50k with fixed structural-review endpoints."""
from __future__ import annotations

from _artifact_paths import artifact_root_default, parse_artifact_root

import argparse
import fcntl
import json
import subprocess
from pathlib import Path

from run_rgb164_rgb165_improvedgs_pair import ROOT, SOURCE_REL, run_arm, source_record
from src.quality_protocol import write_new_json

EXPERIMENT_ID = "RGB-166-IMPROVEDGS-FULL-ALL400-50K"
OUTPUT_KEY = "RGB-166"
COMPONENTS = {"las": True, "eas": True, "rap": True, "mu": True}
ENDPOINTS = [30_000, 35_000, 40_000, 45_000, 50_000]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root_default())
    parser.add_argument("--attempt", default="full-r1")
    args = parser.parse_args()

    source = ROOT / SOURCE_REL
    if source_record(source)["commit"] != "20b5db343a87c2f9fb4b70c6e39ff1f042a8a47a":
        raise ValueError("ImprovedGS source commit changed")

    summary_path = ROOT / "outputs/quality/RGB-166/horizon-summary-r1.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    record: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "starting",
        "parent_experiment": "RGB-165-IMPROVEDGS-FULL-ALL400",
        "single_primary_variable": "training horizon 30k to 50k",
        "save_iterations": ENDPOINTS,
        "evaluation_scope": {
            "global": "all 400 released photographs",
            "labeled_bridge_classes": "all 300 labeled views",
            "known_difficult_groups": ["fixed legacy samples", "159-160", "235-245", "385-392"],
            "generalization": "fixed novel trajectory and spatial holdout evidence",
        },
    }
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            apps = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                text=True,
            )
            active = [line for line in apps.splitlines() if line.strip() and "ToDesk" not in line]
            if active:
                raise RuntimeError("Other GPU compute process active:\n" + "\n".join(active))
            result = run_arm(
                artifact_root=args.artifact_root,
                source=source,
                stage="full",
                iterations=50_000,
                experiment_id=EXPERIMENT_ID,
                output_key=OUTPUT_KEY,
                components=COMPONENTS,
                attempt=args.attempt,
                save_iterations=ENDPOINTS,
                runner_path=Path(__file__).resolve(),
            )
        record.update(status="completed_training", result=result)
    except BaseException as exc:
        record.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_new_json(summary_path, record)


if __name__ == "__main__":
    main()
