#!/usr/bin/env python3
"""Eight-step engineering preflight for matched render/render and render/photo training."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_sem396_spatial import setup_runtime
from src.paired_domain_training import paired_mask2former
from src.quality_protocol import sha256, verify_hashes, write_new_json


EID = "SEM-431-DUAL-DOMAIN-PREFLIGHT"
CONFIG = ROOT / "experiments/runs/SEM-431-DUAL-DOMAIN-PREFLIGHT/config.yaml"
OUTPUT = ROOT / "outputs/quality/SEM-431/preflight-r1"
SEM396 = ROOT / "outputs/quality/SEM-396/r1"
CODE_PATHS = (
    "scripts/run_sem431_dual_domain_preflight.py",
    "src/paired_domain_training.py",
    "src/quality_mask2former.py",
    "src/quality_mask2former_training.py",
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def state_sha256(state) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_arm(config, arm, artifact_root, labels, torch, np):
    import src.quality_mask2former as model_api

    arm_dir = OUTPUT / arm
    arm_dir.mkdir()
    result = {
        "schema_version": 1,
        "status": "starting",
        "experiment_id": EID,
        "arm": arm,
        "seed": config["seed"],
        "trained_steps": 0,
        "binding": {
            "config_sha256": sha256(CONFIG),
            "parent_export_sha256": sha256(SEM396 / "export/result.json"),
            "parent_prepare_sha256": sha256(SEM396 / "prepare-result.json"),
            "seed": config["seed"],
        },
        "access": {
            "train": sorted(labels.allowed),
            "evaluation": [],
            "labels_allowed_during_fit": sorted(labels.allowed),
            "holdout_rgb_allowed_during_fit": False,
            "holdout_labels_allowed_during_fit": False,
        },
    }
    original_loader = model_api.load_official_model

    def capture_initial(*args, **kwargs):
        model, loaded = original_loader(*args, **kwargs)
        result["initial_model_state_sha256"] = state_sha256(model.state_dict())
        return model, loaded

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    model_api.load_official_model = capture_initial
    started = time.monotonic()
    try:
        paired_mask2former(
            SimpleNamespace(batch=SEM396),
            arm_dir,
            result,
            labels,
            {
                "seed": config["seed"],
                "dual_domain_arm": arm,
                "paired_photo_root": str(ROOT / config["photo_root"]),
                "paired_steps": config["steps_per_arm"],
                "paired_crop": config["crop"],
                "checkpoint_steps": [],
                "diagnostic_views": [],
            },
        )
        rows = [json.loads(line) for line in (arm_dir / "training.jsonl").read_text().splitlines()]
        if len(rows) != config["steps_per_arm"] or any(row["domain_arm"] != arm for row in rows):
            raise ValueError("Preflight trace length or arm identity changed")
        result.update(
            status="completed_engineering_preflight_arm",
            rows=rows,
            sample_trace=[row["sample"] for row in rows],
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            all_losses_finite=all(np.isfinite(row["loss"]) for row in rows),
            all_grad_norms_finite=all(np.isfinite(row["grad_norm"]) for row in rows),
        )
    finally:
        model_api.load_official_model = original_loader
        result["seconds"] = time.monotonic() - started
        write_new_json(arm_dir / "result.json", result)
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    config = read(CONFIG)
    if config["experiment_id"] != EID or OUTPUT.exists():
        raise ValueError("SEM431 config identity changed or output already exists")
    split = read(ROOT / config["split_path"])
    train_names = sorted(split["primary_fold"]["semantic_train_labeled"])
    holdout_names = set(split["primary_fold"]["semantic_validation_labeled"])
    if len(train_names) != 256 or set(train_names) & holdout_names:
        raise ValueError("Expected frozen v1 training and disjoint confirmation roles")
    parent = read(SEM396 / "mask2former/result.json")
    export = read(SEM396 / "export/result.json")
    prepare = read(SEM396 / "prepare-result.json")
    if parent["status"] != "completed_sealed_6000":
        raise ValueError("Sealed SEM396 parent required")
    verify_hashes(SEM396 / "export/images", export["image_hashes"])
    label_root = ROOT / config["label_root"]
    verify_hashes(label_root, prepare["training_label_hashes"])
    OUTPUT.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "status": "starting",
        "experiment_id": EID,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": sha256(CONFIG),
        "split_sha256": sha256(ROOT / config["split_path"]),
        "code_sha256": {path: sha256(ROOT / path) for path in CODE_PATHS},
        "confirmation_rgb_accessed": False,
        "confirmation_labels_accessed": False,
        "buffer_accessed": False,
    }
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            runtime, torch, np = setup_runtime(args.artifact_root.resolve())
            from src.quality_static_training import TrainingLabels

            torch.set_num_threads(4)
            manifest = read(ROOT / config["label_manifest"])
            control_labels = TrainingLabels(label_root, train_names, manifest, [1320, 989])
            control = run_arm(config, "render_render", args.artifact_root, control_labels, torch, np)
            if control_labels.accessed != set(train_names):
                raise ValueError("Control did not remain inside training labels")
            candidate_labels = TrainingLabels(label_root, train_names, manifest, [1320, 989])
            candidate = run_arm(config, "render_photo", args.artifact_root, candidate_labels, torch, np)
            if candidate_labels.accessed != set(train_names):
                raise ValueError("Candidate did not remain inside training labels")
            gates = {
                "initial_state_exact": control["initial_model_state_sha256"] == candidate["initial_model_state_sha256"],
                "sample_trace_exact": control["sample_trace"] == candidate["sample_trace"],
                "resolved_model_exact": (OUTPUT / "render_render/resolved_model.yaml").read_bytes() == (OUTPUT / "render_photo/resolved_model.yaml").read_bytes(),
                "control_finite": control["all_losses_finite"] and control["all_grad_norms_finite"],
                "candidate_finite": candidate["all_losses_finite"] and candidate["all_grad_norms_finite"],
                "memory_within_limit": max(control["peak_reserved_mib"], candidate["peak_reserved_mib"]) <= config["maximum_peak_reserved_mib"],
                "confirmation_unread": True,
            }
            result.update(
                status="completed_preflight_passed" if all(gates.values()) else "completed_preflight_failed",
                gates=gates,
                control_result_sha256=sha256(OUTPUT / "render_render/result.json"),
                candidate_result_sha256=sha256(OUTPUT / "render_photo/result.json"),
                control_peak_reserved_mib=control["peak_reserved_mib"],
                candidate_peak_reserved_mib=candidate["peak_reserved_mib"],
                initial_model_state_sha256=control["initial_model_state_sha256"],
                sample_trace=control["sample_trace"],
                full_training_authorized=all(gates.values()),
            )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(OUTPUT / "result.json", result)


if __name__ == "__main__":
    main()

