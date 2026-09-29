#!/usr/bin/env python3
"""Run the corrected second-seed native/context confirmation pair."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_sem385_context_mix as base
import run_sem386_context_memory as memory
from src.quality_protocol import sha256, verify_hashes, write_new_json


ARMS = {
    "control": "SEM-390-RGB150-DEV270-SEED20260919-MEMORY",
    "candidate": "SEM-391-RGB150-DEV270-CONTEXT50-SEED20260919-MEMORY",
}
SCRIPT = "scripts/run_sem390_sem391_confirmation.py"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def config_path(arm: str) -> Path:
    return ROOT / "experiments/runs" / ARMS[arm] / "config.yaml"


@contextmanager
def bound_arm(path: Path, config: dict):
    previous = base.EID, base.CONFIG, base.PARENT, base.CODE_PATHS
    base.EID = config["experiment_id"]
    base.CONFIG = path
    base.PARENT = ROOT / config["parent_semantic_run"]
    base.CODE_PATHS = tuple(
        dict.fromkeys(
            (
                *previous[3],
                "scripts/run_sem386_context_memory.py",
                "scripts/run_sem390_sem391_confirmation.py",
                SCRIPT,
            )
        )
    )
    try:
        yield
    finally:
        base.EID, base.CONFIG, base.PARENT, base.CODE_PATHS = previous


def write_pairing_audit() -> None:
    control_config = read(config_path("control"))
    candidate_config = read(config_path("candidate"))
    control_root = ROOT / control_config["output_path"]
    candidate_root = ROOT / candidate_config["output_path"]
    control = read(control_root / "sampling-plan.json")
    candidate = read(candidate_root / "sampling-plan.json")
    if len(control["samples"]) != 6000 or len(candidate["samples"]) != 6000:
        raise ValueError("Paired plans must both contain 6000 updates")
    for source, derived in zip(control["samples"], candidate["samples"]):
        for key in ("step", "name", "top", "left", "flip"):
            if source[key] != derived[key]:
                raise ValueError(f"Paired sampling differs at step {source['step']}")
        expected_mode = "native_crop" if source["step"] % 2 else "full_frame_context"
        expected_shape = [768, 768] if source["step"] % 2 else [768, 1025]
        if (
            derived["mode"] != expected_mode
            or [
                derived["input_height"],
                derived["input_width"],
            ]
            != expected_shape
        ):
            raise ValueError(
                f"Candidate mode or shape differs at step {source['step']}"
            )

    control_prepare = read(control_root / "prepare-result.json")
    candidate_prepare = read(candidate_root / "prepare-result.json")
    control_export = read(control_root / "export/result.json")
    candidate_export = read(candidate_root / "export/result.json")
    if control_export["image_hashes"] != candidate_export["image_hashes"]:
        raise ValueError("Paired RGB exports differ")
    if any(
        record["holdout_label_hashes_read"] != 0
        for record in (control_prepare, candidate_prepare)
    ) or any(
        record["holdout_rgb_accessed_during_prepare"]
        or record["holdout_labels_accessed_during_prepare"]
        for record in (control_export, candidate_export)
    ):
        raise ValueError("Holdout access occurred during pair preparation")

    write_new_json(
        candidate_root / "pairing-audit.json",
        {
            "schema_version": 1,
            "status": "completed",
            "control_experiment_id": ARMS["control"],
            "candidate_experiment_id": ARMS["candidate"],
            "steps": 6000,
            "view_crop_flip_exact": True,
            "odd_native_updates": 3000,
            "even_context_updates": 3000,
            "native_shape": [768, 768],
            "context_shape": [768, 1025],
            "rgb_export_hashes_exact": True,
            "holdout_rgb_reads": 0,
            "holdout_label_reads": 0,
            "control_sampling_plan_sha256": sha256(control_root / "sampling-plan.json"),
            "candidate_sampling_plan_sha256": sha256(
                candidate_root / "sampling-plan.json"
            ),
            "control_export_result_sha256": sha256(control_root / "export/result.json"),
            "candidate_export_result_sha256": sha256(
                candidate_root / "export/result.json"
            ),
        },
    )


def require_both_preflights() -> None:
    for arm in ARMS:
        config = read(config_path(arm))
        root = ROOT / config["output_path"] / "preflight"
        result = read(root / "result.json")
        if (
            result["status"] != "completed_fresh_model_discarded"
            or result["trained_steps"] != 8
            or not result["seed_aware_initialization_validated"]
        ):
            raise ValueError(f"Successful seed-aware {arm} preflight is required")
        verify_hashes(root, result["artifacts"])


def require_both_checkpoints() -> None:
    for arm in ARMS:
        config = read(config_path(arm))
        root = ROOT / config["output_path"] / "mask2former"
        result = read(root / "result.json")
        if (
            result["status"] != "completed_sealed_6000"
            or result["trained_steps"] != 6000
            or not result["seed_aware_initialization_validated"]
        ):
            raise ValueError(f"Sealed seed-aware {arm} checkpoint is required")
        verify_hashes(root, result["artifacts"])
        if sha256(root / "model-6000.pt") != result["artifacts"]["model-6000.pt"]:
            raise ValueError(f"Sealed {arm} 6000-step checkpoint hash differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument(
        "--stage",
        choices=("prepare", "preflight", "train", "inference"),
        required=True,
    )
    args = parser.parse_args()
    eid = ARMS[args.arm]
    path = config_path(args.arm)
    config = read(path)
    initialization = config.get("seed_aware_initialization", {})
    if (
        config["experiment_id"] != eid
        or config["seed"] != 20260919
        or initialization.get("reference_seed") != 20260805
        or not config["memory_policy"]["swin_activation_recomputation"]
    ):
        raise ValueError("Confirmation identity or initialization contract changed")
    expected_fraction = 0.0 if args.arm == "control" else 0.5
    if float(config["context_fraction"]) != expected_fraction:
        raise ValueError("Confirmation arm context fraction changed")

    artifact_root = args.artifact_root.resolve()
    with bound_arm(path, config):
        if args.stage == "prepare":
            if args.arm == "control":
                with base.shared_identity():
                    base.shared.prepare(config)
            else:
                base.prepare(config)
                write_pairing_audit()
        elif args.stage == "inference":
            require_both_checkpoints()
            base.inference(config, artifact_root)
        else:
            if args.stage == "train":
                require_both_preflights()
            with memory.memory_policy(config, artifact_root, args.stage):
                getattr(base, args.stage)(config, artifact_root)


if __name__ == "__main__":
    main()
