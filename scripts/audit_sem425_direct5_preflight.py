#!/usr/bin/env python3
"""Independently audit the sealed SEM-425 engineering preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_direct_semantics import direct_logit_sha256  # noqa: E402
from src.quality_protocol import sha256  # noqa: E402


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT / "outputs/quality/SEM-425/preflight-r1/result.json",
    )
    args = parser.parse_args()
    result_path = args.result.resolve()
    output = result_path.parent
    result = read(result_path)
    config_path = ROOT / "experiments/runs/SEM-425-RGB152-DIRECT5-PF/config.yaml"
    config = read(config_path)
    split = read(ROOT / config["split_path"])
    fold = split["primary_fold"]
    training = set(fold["semantic_train_labeled"])
    confirmation = set(fold["semantic_validation_labeled"])
    buffer_names = set(fold["buffer_excluded_all_cameras"])

    checks = {
        "status": result["status"] == "completed_engineering_preflight",
        "identity": result["experiment_id"] == config["experiment_id"],
        "config_hash": sha256(config_path) == result["config_sha256"],
        "split_hash": sha256(ROOT / config["split_path"]) == result["split_sha256"],
        "role_counts": (
            len(training), len(confirmation), len(buffer_names)
        ) == (256, 30, 17),
        "roles_disjoint": not (
            training & confirmation or training & buffer_names or confirmation & buffer_names
        ),
        "access_isolation": all(
            result[key] == 0
            for key in (
                "confirmation_rgb_accesses",
                "confirmation_label_accesses",
                "buffer_rgb_accesses",
                "buffer_label_accesses",
            )
        ),
        "training_label_count": result["training_label_hashes_read"] == 256,
        "successful_updates": result["successful_updates"] == 8,
        "sample_role": all(
            row["sample"]["name"] in training for row in result["update_trace"]
        ),
        "finite_positive_gradients": all(
            np.isfinite(row["gradient_norm"]) and row["gradient_norm"] > 0
            for row in result["update_trace"]
        ),
        "finite_losses": all(np.isfinite(row["loss"]) for row in result["update_trace"]),
        "probability_conservation": result["probability_sum_max_error"] <= 5e-7,
        "checkpoint_replay": result["checkpoint_state_replay_exact"] and result["checkpoint_render_replay_exact"],
        "rgb_isolation": (
            result["rgb_state_hashes_equal"]
            and result["fresh_rgb_exact"]
            and result["frozen_rgb_max_absolute_pixel_difference"] == 0
        ),
        "resource_gate": result["peak_reserved_bytes"] < 12 * 1024**3,
        "code_hashes": all(sha256(ROOT / path) == digest for path, digest in result["code_sha256"].items()),
    }

    checkpoint = output / "direct-logits-step8.pt"
    checks["checkpoint_hash"] = sha256(checkpoint) == result["checkpoint_sha256"]
    torch = __import__("torch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    logits = payload["direct_logits"]
    checks["checkpoint_schema"] = (
        payload["experiment_id"] == config["experiment_id"]
        and payload["step"] == 8
        and tuple(logits.shape) == (result["point_count"], 5)
        and logits.numel() == result["trainable_parameter_count"]
        and torch.isfinite(logits).all().item()
    )
    checks["checkpoint_state_hash"] = direct_logit_sha256(logits) == result["final_direct_logit_sha256"]

    image = np.asarray(Image.open(output / "training-probe-semantic.png"), dtype=np.uint8)
    checks["semantic_png"] = image.shape == (768, 768) and int(image.max()) < 5

    counts = np.asarray(result["class_pixel_counts"], dtype=np.float64)
    expected_weights = np.clip(np.median(counts) / counts, 0.05, 20.0).astype(np.float32)
    checks["class_weights"] = np.array_equal(
        expected_weights, np.asarray(result["class_weights"], dtype=np.float32)
    )

    audit = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "result_sha256": sha256(result_path),
        "quality_claim": None,
        "authorized_next_step": (
            "A separately registered direct-five-logit spatial training comparison"
            if all(checks.values())
            else None
        ),
    }
    audit_path = output / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
