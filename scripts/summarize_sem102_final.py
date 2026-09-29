#!/usr/bin/env python3
"""Combine frozen SEM-102 audit evidence without rerunning any model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem080_structural_diag as s80

from src.semantic_rendered_final import (
    final_retention_gate,
    validate_final_training_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-rendered-root", type=Path, required=True)
    parser.add_argument("--validation-audit-result", type=Path, required=True)
    parser.add_argument("--external-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def external_camera_contract(
    evidence: dict[str, Any], *, checkpoint_sha256: str, expected_size: list[int]
) -> bool:
    width, height = (int(value) for value in expected_size)
    return bool(
        evidence.get("experiment_id") == "SEM-102-RENDERED-SEG-FINAL"
        and evidence.get("status") == "completed_official_external_camera_semantic"
        and evidence.get("final_checkpoint_sha256") == checkpoint_sha256
        and evidence.get("input_camera_model") == "SIMPLE_RADIAL"
        and evidence.get("input_pose_convention") == "CAMERA_FROM_WORLD"
        and evidence.get("output_grid") == "official_SIMPLE_RADIAL"
        and evidence.get("rgb_shape") == [height, width, 3]
        and evidence.get("semantic_logits_shape") == [5, height, width]
        and evidence.get("semantic_png_shape") == [height, width]
        and evidence.get("semantic_png_dtype") == "uint8"
        and evidence.get("semantic_png_min", -1) >= 0
        and evidence.get("semantic_png_max", 5) < 5
        and evidence.get("probability_simplex") is True
        and evidence.get("uses_reference_rgb_or_labels") is False
    )


def main() -> int:
    args = parse_args()
    config = s80.load_json(args.config)
    require(
        config.get("experiment_id") == "SEM-102-RENDERED-SEG-FINAL",
        "SEM-102 final summary received another config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-102 final summary requires its registered-not-run config",
    )
    require(args.final_checkpoint.is_file(), "SEM-102 final checkpoint is missing")
    checkpoint_sha256 = s80.sha256(args.final_checkpoint)
    final_result = s80.load_json(args.final_result)
    config_sha256 = s80.sha256(args.config)
    validate_final_training_result(
        final_result,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
    )
    render_manifest_path = (
        args.validation_rendered_root / "rendered_validation_manifest.json"
    )
    require(
        render_manifest_path.is_file(), "SEM-102 validation render manifest is missing"
    )
    render_manifest = s80.load_json(render_manifest_path)
    validation_audit = s80.load_json(args.validation_audit_result)
    require(
        validation_audit.get("experiment_id") == config["experiment_id"]
        and validation_audit.get("status")
        == "completed_once_only_74_view_audit_pending_external_gate",
        "SEM-102 validation audit result is invalid",
    )
    require(
        validation_audit.get("config_sha256") == config_sha256
        and validation_audit.get("final_checkpoint_sha256") == checkpoint_sha256
        and validation_audit.get("validation_render_manifest_sha256")
        == s80.sha256(render_manifest_path),
        "SEM-102 validation audit does not match frozen inputs",
    )
    access = validation_audit.get("validation_access", {})
    require(
        access.get("validation_labels_opened_after_all_predictions") is True
        and access.get("released_or_reference_rgb_opened") is False
        and access.get("unlabeled_images_opened") is False
        and access.get("hidden_test_rgb_or_labels_opened") is False,
        "SEM-102 validation audit access boundary failed",
    )
    isolation = render_manifest.get("rgb_isolation", {})
    rgb_isolation_passed = bool(
        isolation.get("rgb_artifact_hashes_equal")
        and isolation.get("rgb_png_hashes_equal")
        and isolation.get("rgb_max_absolute_pixel_difference")
        == config["formal_retention_gates"]["rgb_max_absolute_pixel_difference"]
    )
    external = s80.load_json(args.external_evidence)
    external_passed = external_camera_contract(
        external,
        checkpoint_sha256=checkpoint_sha256,
        expected_size=list(config["external_camera"]["output_size"]),
    )
    metric_gate = final_retention_gate(
        validation_audit["metrics"],
        config["baseline_sem001"],
        config["formal_retention_gates"],
    )
    engineering_checks = {
        "checkpoint_reload": final_result.get("checkpoint_reload")
        == config["formal_retention_gates"]["checkpoint_reload"],
        "frozen_rgb_isolation": rgb_isolation_passed,
        "official_external_camera": external_passed,
    }
    passed = bool(metric_gate["passed"] and all(engineering_checks.values()))
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "Refusing to overwrite SEM-102 final summary",
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_formal_retention_gate_passed"
        if passed
        else "completed_formal_retention_gate_failed",
        "config_sha256": config_sha256,
        "final_training_result_sha256": s80.sha256(args.final_result),
        "final_checkpoint_sha256": checkpoint_sha256,
        "validation_render_manifest_sha256": s80.sha256(render_manifest_path),
        "validation_audit_result_sha256": s80.sha256(args.validation_audit_result),
        "external_evidence_sha256": s80.sha256(args.external_evidence),
        "metrics": validation_audit["metrics"],
        "per_view": validation_audit["per_view"],
        "panels": validation_audit["panels"],
        "metric_retention_gate": metric_gate,
        "engineering_checks": engineering_checks,
        "formal_replacement_allowed": passed,
        "fallback": config["fallback"],
        "next_step": "A passed result may be reviewed as a rendered-domain semantic candidate; a failed result remains show-only. Neither outcome authorises retraining or another 74-view audit.",
    }
    s80.write_json(args.output_dir / "result.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
