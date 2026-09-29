#!/usr/bin/env python3
"""Exercise trained SEM-280 five-class arms on official native/small cameras."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_sem280_foundation_boundary_oof import array_hash, initialize
from src.semantic_foundation_boundary import fuse_foundation
from src.semantic2d import read_json, sha256, write_json
from src.submission_sem161 import sliding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "data-root", "bundle", "run-dir", "reference-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/runs/SEM-280-FOUNDATION-BOUNDARY-OOF/config.yaml",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    config = read_json(args.config)
    rows = []
    started = time.perf_counter()
    environment = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    for size, camera in (
        ("native", args.data_root / "camera_parameters/cameras.txt"),
        ("small", args.reference_dir / "small_cameras.txt"),
    ):
        rendered = output_dir / size
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/render_submission_sem161.py"),
                "--bundle",
                str(args.bundle),
                "--cameras",
                str(camera),
                "--images",
                str(args.data_root / "camera_parameters/images.txt"),
                "--image-name",
                "140.png",
                "--output-dir",
                str(rendered),
            ],
            env=environment,
            cwd=ROOT,
            check=True,
        )
        rgb = np.asarray(Image.open(rendered / "rgb/140.png"))
        base = np.asarray(Image.open(rendered / "semantic/140.png"))
        reference = args.reference_dir / ("reload1" if size == "native" else "small")
        if not np.array_equal(rgb, np.asarray(Image.open(reference / "rgb/140.png"))):
            raise ValueError("Frozen RGB replay differs")
        if not np.array_equal(base, np.asarray(Image.open(reference / "semantic/140.png"))):
            raise ValueError("Frozen semantic replay differs")
        for arm in config["arm_names"]:
            checkpoint = args.run_dir / f"fold0_{arm}/model.pth"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            metadata = payload["metadata"]
            record = read_json(args.run_dir / f"fold0_{arm}/training_result.json")
            if (
                metadata["config_sha256"] != sha256(args.config)
                or metadata["fold"] != 0
                or metadata["arm"] != arm
                or sha256(checkpoint) != record["sha256"]
            ):
                raise ValueError("Wrong trained branch")
            model = initialize(args.artifact_root / config["parents"]["model_config"], config["seed"])
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.cuda().eval().requires_grad_(False)
            before = array_hash(rgb)
            torch.cuda.reset_peak_memory_stats()
            probability = sliding(model, rgb, crop=960, stride=480, classes=5)[4]
            disabled = fuse_foundation(base, probability, strength=0)
            prediction = fuse_foundation(base, probability)
            target = rendered / f"{arm}_140.png"
            Image.fromarray(prediction).save(target)
            checks = {
                "rgb_frozen_replay_exact": True,
                "baseline_frozen_replay_exact": True,
                "disabled_exact": bool(np.array_equal(disabled, base)),
                "protected_masks_exact": all(
                    np.array_equal(base == class_id, prediction == class_id)
                    for class_id in (1, 2, 3)
                ),
                "rgb_input_unchanged": array_hash(rgb) == before,
            }
            if not all(checks.values()):
                raise ValueError("External trained branch gate failed")
            rows.append(
                {
                    "size": size,
                    "arm": arm,
                    "shape": list(base.shape),
                    "camera": "140.png",
                    "pose": "COLMAP_CAMERA_FROM_WORLD",
                    "camera_model": "SIMPLE_RADIAL",
                    "checks": checks,
                    "changed_pixels": int((prediction != base).sum()),
                    "added_foundation": int(((base == 0) & (prediction == 4)).sum()),
                    "removed_foundation": int(((base == 4) & (prediction == 0)).sum()),
                    "checkpoint_sha256": sha256(checkpoint),
                    "prediction_sha256": sha256(target),
                    "rgb_sha256": sha256(rendered / "rgb/140.png"),
                    "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                }
            )
            del model, payload
            torch.cuda.empty_cache()
    result = {
        "experiment_id": config["experiment_id"],
        "status": "passed",
        "scope": "Trained fold0 two-arm engineering on one official camera at two sizes; no hidden/final quality claim",
        "target_labels_read": [],
        "reference_rgb_inputs": [],
        "checks": rows,
        "seconds": time.perf_counter() - started,
        "code_sha256": sha256(Path(__file__)),
    }
    write_json(output_dir / "result.json", result)
    print({"status": "passed", "probes": len(rows), "seconds": result["seconds"]}, flush=True)


if __name__ == "__main__":
    main()
