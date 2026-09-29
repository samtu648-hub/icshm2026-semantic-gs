#!/usr/bin/env python3
"""Verify saved SEM-161 predictions/RGB; export historical metrics, not a new eval."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.semantic2d import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    root = args.output_root
    spec_path = REPO / "experiments/runs/SEM-180-RGB-INPUT-AUDIT/frozen_fallback.json"
    spec = json.loads(spec_path.read_text())
    for key in ["result", "prediction_manifest", "checkpoint"]:
        if sha256(root / spec[key]["path"]) != spec[key]["sha256"]:
            raise ValueError("Frozen SEM-161 changed: " + key)
    manifest_path = root / spec["prediction_manifest"]["path"]
    records = json.loads(manifest_path.read_text())["predictions"]
    if len(records) != 74 or len({r["name"] for r in records}) != 74:
        raise ValueError("Frozen SEM-161 view set changed")
    for row in records:
        if sha256(manifest_path.parent / row["prediction"]) != row["prediction_sha256"]:
            raise ValueError("Frozen semantic pixels changed: " + row["name"])
        if sha256(root / spec["frozen_rgb_directory"] / row["name"]) != row["frozen_rgb_sha256"]:
            raise ValueError("Frozen RGB pixels changed: " + row["name"])
    old = json.loads((root / spec["result"]["path"]).read_text())
    output = root / "r2/rgb_092/fallback_verification/full-20260906-r1"
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "verified_frozen_SEM161_and_RGB_inputs", "prediction_count": 74,
              "semantic_png_bytes_identical": True, "rgb_png_bytes_identical": True,
              "checkpoint_unchanged": True, "labels_opened": 0, "new_predictions": 0,
              "historical_metrics_not_new_evaluation": old["metrics"],
              "expected_artifacts": spec, "spec_sha256": sha256(spec_path)}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ["expected_artifacts", "historical_metrics_not_new_evaluation"]}))


if __name__ == "__main__":
    main()
