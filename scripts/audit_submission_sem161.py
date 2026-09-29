#!/usr/bin/env python3
"""Audit bundle bytes and external-camera behavior; never open target labels."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from package_submission_baseline import sha256


def runtime_versions():
    import cv2
    versions = {package: importlib.metadata.version(package) for package in ("torch", "transformers", "numpy", "Pillow")}
    versions["cv2"] = cv2.__version__
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "data-root", "reference-external", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.bundle/"metadata/artifact_manifest.json").read_text())
    for entry in manifest["files"]:
        assert sha256(args.bundle/entry["path"]) == entry["sha256"], entry["path"]
    cameras = args.data_root/"camera_parameters/cameras.txt"
    images = args.data_root/"camera_parameters/images.txt"
    command = [sys.executable, str(args.bundle/"code/scripts/render_submission_sem161.py"), "--bundle", str(args.bundle), "--images", str(images), "--image-name", "140.png", "--write-fallbacks"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for attempt in ("reload1", "reload2"):
        subprocess.run([*command,"--cameras",str(cameras),"--output-dir",str(args.output_dir/attempt)], cwd="/tmp", env=env, check=True)
    exact = {}
    for key in ("rgb", "semantic"):
        first = np.asarray(Image.open(args.output_dir/"reload1"/key/"140.png"))
        second = np.asarray(Image.open(args.output_dir/"reload2"/key/"140.png"))
        reference = np.asarray(Image.open(args.reference_external/(key+".png")))
        exact[key] = {"repeat_exact": bool(np.array_equal(first, second)), "frozen_exact": bool(np.array_equal(first, reference))}
        assert all(exact[key].values())
    small = args.output_dir/"small_cameras.txt"
    small.write_text("1 SIMPLE_RADIAL 320 240 300 160 120 0\n")
    subprocess.run([*command,"--cameras",str(small),"--output-dir",str(args.output_dir/"small")], cwd="/tmp", env=env, check=True)
    assert np.asarray(Image.open(args.output_dir/"small/semantic/140.png")).shape == (240,320)
    overwrite = subprocess.run([*command,"--cameras",str(small),"--output-dir",str(args.output_dir/"small")], cwd="/tmp", env=env, capture_output=True, text=True)
    assert overwrite.returncode != 0 and "Refusing to overwrite" in overwrite.stderr
    # Verify the code/model package remained immutable after fresh-process loading.
    assert all(sha256(args.bundle/e["path"]) == e["sha256"] for e in manifest["files"])
    report = {"experiment_id":"SUBMIT-030-SEM161-BUNDLE", "status":"completed_local_interface_audit", "files_verified":len(manifest["files"]), "bundle_manifest_sha256":sha256(args.bundle/"metadata/artifact_manifest.json"), "external_140":exact, "fresh_process_reloads":2, "small_camera_shape":[240,320], "overwrite_rejected":True, "uses_reference_rgb_or_labels_for_inference":False, "audit_reference":"previously frozen external output only, not ground truth", "official_hidden_evaluation_performed":False, "environment":runtime_versions(), "render_manifests":{name:sha256(args.output_dir/name/"manifest.json") for name in ("reload1","reload2","small")}}
    (args.output_dir/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
