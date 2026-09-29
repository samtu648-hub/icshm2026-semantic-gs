#!/usr/bin/env python3
"""Independently audit the sealed SEM427 directional-semantic preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256  # noqa: E402
from src.quality_view_conditioned_semantics import directional_logit_sha256  # noqa: E402


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT / "outputs/quality/SEM-427/preflight-r1/result.json",
    )
    args = parser.parse_args()
    result_path = args.result.resolve()
    output = result_path.parent
    result = read(result_path)
    config_path = ROOT / "experiments/runs/SEM-427-RGB152-DIR5-PF/config.yaml"
    config = read(config_path)
    binding = read(output / "binding.json")
    split = read(ROOT / config["split_path"])["primary_fold"]
    training = set(split["semantic_train_labeled"])
    confirmation = set(split["semantic_validation_labeled"])
    buffer_names = set(split["buffer_excluded_all_cameras"])
    parent_plan = read(ROOT / config["parent_sampling_plan"])["samples"][:8]
    local_plan = read(output / "sampling-plan.json")["samples"]
    updates = result.get("updates", [])
    gates = result.get("gate_pass", {})

    checks = {
        "status": result.get("status") == "completed_passed",
        "identity": result.get("experiment_id") == config["experiment_id"] == binding["experiment_id"],
        "config_hash": sha256(config_path) == binding["config_sha256"],
        "split_hash": sha256(ROOT / config["split_path"]) == binding["split_sha256"],
        "parent_plan_hash": sha256(ROOT / config["parent_sampling_plan"]) == config["parent_sampling_plan_sha256"] == binding["sampling_plan_sha256"],
        "parent_audit_hash": sha256(ROOT / config["parent_audit"]) == config["parent_audit_sha256"] == binding["parent_audit_sha256"],
        "local_plan_hash": sha256(output / "sampling-plan.json") == binding["local_sampling_plan_sha256"],
        "matched_first_eight_samples": local_plan == parent_plan,
        "role_counts": (len(training), len(confirmation), len(buffer_names)) == (256, 30, 17),
        "roles_disjoint": not (training & confirmation or training & buffer_names or confirmation & buffer_names),
        "access_isolation": all(result.get(key) == 0 for key in (
            "confirmation_rgb_accesses", "confirmation_label_accesses",
            "buffer_rgb_accesses", "buffer_label_accesses",
        )),
        "training_only_samples": len(updates) == 8 and all(row["sample"]["name"] in training for row in updates),
        "finite_losses": all(np.isfinite(row["loss"]) for row in updates),
        "bias_gradients": all(np.isfinite(row["bias_gradient_norm"]) and row["bias_gradient_norm"] > 0 for row in updates),
        "direction_gradients": all(np.isfinite(row["direction_gradient_norm"]) and row["direction_gradient_norm"] > 0 for row in updates),
        "probability_conservation": result.get("probability_sum_max_error", 1) <= config["preflight"]["probability_sum_max_error"],
        "cross_camera_response": result.get("cross_camera_probe", {}).get("max_absolute_probability_delta", 0) > config["preflight"]["minimum_cross_camera_probability_delta"],
        "checkpoint_replay": result.get("checkpoint_replay_exact") is True,
        "rgb_isolation": result.get("rgb_state_hashes_equal") is True and result.get("fresh_rgb_exact") is True and result.get("frozen_rgb_max_absolute_pixel_difference") == 0,
        "resource_gate": result.get("peak_reserved_mib", 1e9) < config["preflight"]["maximum_peak_reserved_mib"],
        "all_runtime_gates": bool(gates) and all(gates.values()),
        "code_hashes": all(sha256(ROOT / path) == digest for path, digest in binding["code_sha256"].items()),
        "training_label_hashes": all(sha256(ROOT / config["label_root"] / name) == digest for name, digest in binding["training_label_hashes"].items()) and len(binding["training_label_hashes"]) == 256,
    }
    checkpoint = output / "directional-logits-step8.pt"
    checks["checkpoint_hash"] = sha256(checkpoint) == result["checkpoint_sha256"]
    torch = __import__("torch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    coefficients = payload["directional_coefficients"]
    checks["checkpoint_schema"] = (
        payload["experiment_id"] == config["experiment_id"]
        and payload["step"] == 8
        and payload["binding_sha256"] == result["binding_sha256"]
        and tuple(coefficients.shape) == (result["point_count"], 5, 4)
        and coefficients.numel() == result["trainable_parameter_count"]
        and torch.isfinite(coefficients).all().item()
    )
    checks["checkpoint_state_hash"] = directional_logit_sha256(coefficients) == result["final_directional_logit_sha256"]
    image = np.asarray(Image.open(output / "training-probe-semantic.png"), dtype=np.uint8)
    checks["semantic_png"] = image.shape == (768, 768) and int(image.max()) < 5
    counts = np.asarray(result["class_pixel_counts"], dtype=np.float64)
    expected = np.clip(np.median(counts) / counts, 0.05, 20.0).astype(np.float32)
    checks["class_weights"] = np.array_equal(expected, np.asarray(result["class_weights"], dtype=np.float32))

    audit = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "result_sha256": sha256(result_path),
        "quality_claim": None,
        "authorized_next_step": "Register a full matched spatial comparison under a new ID" if all(checks.values()) else None,
    }
    audit_path = output / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
