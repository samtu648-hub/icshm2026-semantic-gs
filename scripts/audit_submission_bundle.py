#!/usr/bin/env python3
"""Verify frozen bundle hashes and optional RGB/semantic render outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_bundle(bundle: Path) -> dict[str, Any]:
    manifest = json.loads((bundle / "metadata/artifact_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["files"]:
        path = bundle / entry["path"]
        if not path.is_file():
            failures.append(f"missing:{entry['path']}")
        elif path.stat().st_size != entry["bytes"]:
            failures.append(f"size:{entry['path']}")
        elif sha256(path) != entry["sha256"]:
            failures.append(f"sha256:{entry['path']}")
    lock = json.loads((bundle / "code/configs/submission/sem001_baseline_v1.json").read_text(encoding="utf-8"))
    for entry in lock["artifacts"]:
        path = bundle / entry["bundle_path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            failures.append(f"locked_artifact:{entry['bundle_path']}")
    if failures:
        raise RuntimeError("Bundle audit failed: " + ", ".join(failures))
    return {"manifest_files": len(manifest["files"]), "locked_artifacts": len(lock["artifacts"]), "bundle": "passed"}


def audit_renders(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "submission_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["outputs"]:
        rgb_path = root / entry["rgb"]["path"]
        semantic_path = root / entry["semantic"]["path"]
        if sha256(rgb_path) != entry["rgb"]["sha256"] or sha256(semantic_path) != entry["semantic"]["sha256"]:
            failures.append(f"output_hash:{entry['pose_name']}")
            continue
        rgb = np.asarray(Image.open(rgb_path))
        semantic_image = Image.open(semantic_path)
        semantic = np.asarray(semantic_image)
        expected_rgb = (entry["height"], entry["width"], 3)
        expected_semantic = (entry["height"], entry["width"])
        if rgb.shape != expected_rgb or rgb.dtype != np.uint8:
            failures.append(f"rgb_contract:{entry['pose_name']}")
        if semantic.shape != expected_semantic or semantic.dtype != np.uint8 or semantic_image.mode != "L":
            failures.append(f"semantic_contract:{entry['pose_name']}")
        if semantic.size and (int(semantic.min()) < 0 or int(semantic.max()) > 4):
            failures.append(f"semantic_ids:{entry['pose_name']}")
        if "logits" in entry:
            logits_path = root / entry["logits"]["path"]
            logits = np.load(logits_path, mmap_mode="r")
            if sha256(logits_path) != entry["logits"]["sha256"] or logits.shape != (5, entry["height"], entry["width"]):
                failures.append(f"logits_contract:{entry['pose_name']}")
    if manifest.get("uses_reference_rgb_or_labels") is not False:
        failures.append("reference_data_contract")
    if failures:
        raise RuntimeError("Render audit failed: " + ", ".join(failures))
    return {"rendered_views": len(manifest["outputs"]), "renders": "passed"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--renders", type=Path)
    args = parser.parse_args()
    result = audit_bundle(args.bundle)
    if args.renders is not None:
        result.update(audit_renders(args.renders))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
