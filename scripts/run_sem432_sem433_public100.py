#!/usr/bin/env python3
"""Render-only SEM432/433 inference on the 100 released unlabeled RGB140 views."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_sem396_spatial import setup_runtime
from src.quality_protocol import sha256, write_new_json


RUNS = {
    "control": ROOT / "outputs/quality/SEM-432/r1",
    "candidate": ROOT / "outputs/quality/SEM-433/r1",
}
RGB_ROOT = ROOT / "outputs/quality/rgb140/full-r2/evaluation/images"
OUTPUT = ROOT / "outputs/quality/QUAL-019/public100"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    names = [f"{index:03d}.png" for index in range(301, 401)]
    if any(not (RGB_ROOT / name).is_file() for name in names):
        raise FileNotFoundError("RGB140 public100 render missing")
    OUTPUT.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "status": "starting",
        "experiment_id": "QUAL-019-PAIRED-PHOTO-DOMAIN",
        "names": names,
        "input": "sealed RGB140 render only",
        "labels_accessed": 0,
        "target_photos_accessed": 0,
        "arms": {},
    }
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            runtime, torch, np = setup_runtime(args.artifact_root.resolve())
            from PIL import Image
            from src.quality_mask2former import config_for_model, load_official_model, native_predict
            from src.semantic_cable_binary import read_rgb

            cfg = config_for_model(
                runtime / "vendor/Mask2Former",
                "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
                False,
            )
            for arm, run in RUNS.items():
                train = read(run / "mask2former/result.json")
                checkpoint = run / "mask2former/model-6000.pt"
                before = sha256(checkpoint)
                if train["status"] != "completed_sealed_6000" or before != train["artifacts"]["model-6000.pt"]:
                    raise ValueError(f"Unsealed {arm} checkpoint")
                destination = OUTPUT / arm
                destination.mkdir()
                model, _ = load_official_model(cfg, runtime / "weights/model_final_6b4a3a.pkl")
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
                model.cuda().eval()
                hashes = {}
                arm_started = time.monotonic()
                for index, name in enumerate(names):
                    prediction = native_predict(model, read_rgb(RGB_ROOT / name)).argmax(0).astype(np.uint8)
                    Image.fromarray(prediction).save(destination / name)
                    hashes[name] = sha256(destination / name)
                    if index % 25 == 0:
                        print(json.dumps({"arm": arm, "inferred": index + 1, "total": 100}), flush=True)
                if sha256(checkpoint) != before:
                    raise ValueError(f"{arm} checkpoint changed during public100 inference")
                result["arms"][arm] = {
                    "checkpoint_sha256": before,
                    "prediction_hashes": hashes,
                    "seconds": time.monotonic() - arm_started,
                }
                del model
                torch.cuda.empty_cache()
            result["status"] = "completed_no_label_public100"
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(OUTPUT / "result.json", result)


if __name__ == "__main__":
    main()

