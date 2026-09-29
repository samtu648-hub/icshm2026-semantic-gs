#!/usr/bin/env python3
"""Audit preserved GEO-025 outputs using the preregistered GEO-026 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.geometry_prior.vggt_omega import (  # noqa: E402
    CHECKPOINT_SHA256,
    SOURCE_COMMIT,
    training_names_from_split,
)
from src.geometry_prior.vggt_p0_float64_audit import (  # noqa: E402
    CANONICAL_DEPTH_ATOL,
    CANONICAL_DEPTH_RTOL,
    CANONICAL_PIXEL_LIMIT,
    HISTORICAL_STORED_PIXEL_LIMIT,
    STORED_WORLD_ATOL,
    STORED_WORLD_RTOL,
    audit_npz_float64,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/runs/GEO-026-P0-F64/config.yaml",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "experiments/splits/rgb_r1_spatial_v1.json",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_hashes = config["source_artifacts_sha256"]
    source_files = {
        "progress.json": args.source_run / "progress.json",
        "window_manifest.json": args.source_run / "window_manifest.json",
        "failure.json": args.source_run / "failure.json",
        "coordinate_failure_diagnosis.json": args.source_run
        / "coordinate_failure_diagnosis.json",
    }
    observed_hashes = {name: sha256(path) for name, path in source_files.items()}
    if observed_hashes != expected_hashes:
        raise ValueError(
            f"Preserved source hashes changed: observed={observed_hashes} expected={expected_hashes}"
        )

    progress = json.loads(source_files["progress.json"].read_text(encoding="utf-8"))
    manifest = json.loads(source_files["window_manifest.json"].read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    records = progress.get("records")
    if (
        progress.get("experiment_id") != "GEO-025-P0"
        or progress.get("completed_windows") != 303
        or progress.get("total_windows") != 303
        or progress.get("record_count") != 7575
        or not isinstance(records, list)
        or len(records) != 7575
    ):
        raise ValueError("Preserved progress does not match the registered 303/7575 source")
    if (
        manifest.get("window_count") != 303
        or manifest.get("scope") != "full_training"
        or manifest.get("split_id") != "rgb_r1_spatial_v1"
        or manifest.get("source_commit") != SOURCE_COMMIT
        or manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256
    ):
        raise ValueError("Preserved window manifest does not match the fixed source")

    training_names = set(training_names_from_split(split))
    expected_pairs = {
        (window["window_manifest_id"], name)
        for window in manifest["windows"]
        for name in window["images"]
    }
    actual_pairs: set[tuple[str, str]] = set()
    worst: list[dict[str, Any]] = []
    totals = {
        "valid_count": 0,
        "stored_pixels_over_historical_1e_3": 0,
        "canonical_pixel_error_max": 0.0,
        "canonical_depth_error_max": 0.0,
        "stored_world_error_max": 0.0,
        "stored_pixel_error_max": 0.0,
        "stored_depth_error_max": 0.0,
        "minimum_valid_depth": float("inf"),
        "valid_fraction_min": 1.0,
    }
    for index, record in enumerate(records, 1):
        pair = (record.get("window_manifest_id"), record.get("image_name"))
        if pair in actual_pairs:
            raise ValueError(f"Duplicate geometry record: {pair}")
        actual_pairs.add(pair)
        if pair[1] not in training_names:
            raise ValueError(f"Non-training image in source records: {pair[1]}")
        if (
            record.get("source_commit") != SOURCE_COMMIT
            or record.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or record.get("coordinate_frame") != "official_colmap_reference"
            or record.get("world_point_pose") != "official_cam2world_P0_only"
        ):
            raise ValueError(f"Record provenance mismatch: {pair}")
        relative = Path(record["output_npz"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe output path: {relative}")
        width, height = record["input_resolution"]
        metrics = audit_npz_float64(
            args.source_run / relative,
            expected_resolution_wh=(int(width), int(height)),
        )
        totals["valid_count"] += metrics["valid_count"]
        totals["stored_pixels_over_historical_1e_3"] += metrics[
            "stored_pixels_over_historical_1e_3"
        ]
        for name in (
            "canonical_pixel_error_max",
            "canonical_depth_error_max",
            "stored_world_error_max",
            "stored_pixel_error_max",
            "stored_depth_error_max",
        ):
            totals[name] = max(totals[name], metrics[name])
        totals["minimum_valid_depth"] = min(
            totals["minimum_valid_depth"], metrics["minimum_valid_depth"]
        )
        totals["valid_fraction_min"] = min(
            totals["valid_fraction_min"], metrics["valid_fraction"]
        )
        worst.append(
            {
                "window_manifest_id": pair[0],
                "image_name": pair[1],
                **metrics,
            }
        )
        worst = sorted(
            worst, key=lambda item: item["stored_pixel_error_max"], reverse=True
        )[:20]
        if index % 25 == 0 or index == len(records):
            atomic_json(
                args.output_dir / "progress.json",
                {
                    "status": "running",
                    "experiment_id": "GEO-026-P0-F64",
                    "completed_records": index,
                    "total_records": len(records),
                    "elapsed_seconds": time.monotonic() - started,
                    "metrics_so_far": totals,
                },
            )
    if actual_pairs != expected_pairs:
        raise ValueError(
            f"Record coverage mismatch: missing={len(expected_pairs-actual_pairs)} extra={len(actual_pairs-expected_pairs)}"
        )
    selected_images = {name for _, name in actual_pairs}
    if selected_images != training_names:
        raise ValueError("Source records do not cover exactly the 303 training images")

    result = {
        "schema_version": 1,
        "experiment_id": "GEO-026-P0-F64",
        "status": "completed_passed_for_fusion",
        "source_experiment": "GEO-025-P0",
        "source_artifacts_sha256": observed_hashes,
        "config_sha256": sha256(args.config),
        "split_sha256": sha256(args.split),
        "contract": {
            "canonical_representation": "float32 depth plus float64 PINHOLE intrinsics and official cam2world",
            "canonical_world_computation": "float64; no serialized float32 world_points in the hard-gate path",
            "canonical_pixel_limit": CANONICAL_PIXEL_LIMIT,
            "canonical_depth_rtol": CANONICAL_DEPTH_RTOL,
            "canonical_depth_atol": CANONICAL_DEPTH_ATOL,
            "stored_world_fidelity_rtol": STORED_WORLD_RTOL,
            "stored_world_fidelity_atol": STORED_WORLD_ATOL,
            "historical_stored_pixel_limit_diagnostic_only": HISTORICAL_STORED_PIXEL_LIMIT,
            "fusion_consumes": "depth, confidence, valid_mask, float64 intrinsics and float64 official cam2world",
            "serialized_world_points_role": "fidelity diagnostic only",
        },
        "coverage": {
            "record_count": len(records),
            "window_count": manifest["window_count"],
            "unique_training_images": len(selected_images),
            "expected_pairs_exact": True,
            "split_isolation": "primary_fold.train_rgb_400_only",
        },
        "metrics": totals,
        "worst_20_serialized_world_roundtrips": worst,
        "elapsed_seconds": time.monotonic() - started,
        "model_execution": "not_run_reuses_preserved_GEO_025_outputs",
        "gpu_used": False,
    }
    atomic_json(args.output_dir / "result.json", result)
    atomic_json(
        args.output_dir / "accepted_records.json",
        {
            "schema_version": 1,
            "experiment_id": "GEO-026-P0-F64",
            "status": "accepted_for_fusion",
            "source_run": "external_gitignored:GEO-025-P0/full_v25_bf16_attempt1",
            "source_progress_sha256": observed_hashes["progress.json"],
            "records": records,
        },
    )
    atomic_json(
        args.output_dir / "progress.json",
        {
            "status": "completed",
            "experiment_id": "GEO-026-P0-F64",
            "completed_records": len(records),
            "total_records": len(records),
            "result": "result.json",
            "accepted_records": "accepted_records.json",
        },
    )
    print(json.dumps({"status": result["status"], "result": str(args.output_dir / "result.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
