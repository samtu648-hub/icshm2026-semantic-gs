#!/usr/bin/env python3
"""Archive exact stage outputs and attribute changes, not guessed geometry.

The input JSON binds experiment/config/seed/data/split and a common pixel_space.
Paths resolve relative to that JSON. Each stage is {name, labels, sha256}; optional
target, roi, real_rgb, reconstructed_rgb and probability entries are also hashed.
Original images and probability arrays are archived without resampling.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    from src.quality_diagnostics import diagnose_stages
    manifest = json.loads(args.manifest.read_text())
    required = {"experiment_id", "git_commit", "config_sha256", "seed", "dataset_manifest_sha256", "split_sha256", "pixel_space", "view", "stages"}
    if not required <= manifest.keys() or manifest["pixel_space"] not in ("PINHOLE", "SIMPLE_RADIAL"):
        raise ValueError("Incomplete diagnostic provenance or pixel space")
    if args.output.exists():
        raise FileExistsError("Diagnostic evidence already exists")
    args.output.mkdir(parents=True)
    record = {"status": "running", "manifest_sha256": sha256(args.manifest)}
    try:
        archived = {}
        def load(entry, role, kind="label"):
            path = (args.manifest.parent / entry["path"]).resolve()
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Changed {role} input")
            filename = f"input_{len(archived):03d}" + path.suffix
            shutil.copy2(path, args.output / filename)
            archived[role] = {"file": filename, "sha256": entry["sha256"]}
            return np.load(path, allow_pickle=False) if kind == "probability" else np.asarray(Image.open(path))
        stages = {}
        for entry in manifest["stages"]:
            if entry["name"] in stages:
                raise ValueError("Duplicate stage name")
            stages[entry["name"]] = load(entry, "stage/" + entry["name"])
        target = load(manifest["target"], "target") if "target" in manifest else None
        roi = load(manifest["roi"], "roi") != 0 if "roi" in manifest else None
        probabilities = {e["name"]: load(e, "probability/" + e["name"], "probability") for e in manifest.get("probabilities", [])}
        same_photo = "real_rgb" in manifest and manifest.get("real_rgb_camera_matches", False)
        result, masks = diagnose_stages(stages, target=target, roi=roi, probabilities=probabilities,
                                       has_same_camera_photo=same_photo, interpolation_view=manifest.get("interpolation_view", False))
        shape = next(iter(stages.values())).shape
        for role in ("real_rgb", "reconstructed_rgb"):
            if role in manifest:
                value = load(manifest[role], role)
                if value.shape != shape + (3,):
                    raise ValueError("RGB and semantic inputs must share the declared pixel grid")
        mask_names = {}
        for index, (name, mask) in enumerate(masks.items()):
            filename = f"change_{index:02d}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(args.output / filename)
            mask_names[name] = filename
        record.update(status="completed_pixel_stage_diagnostics", provenance={k: manifest[k] for k in required - {"stages"}},
                      result=result, inputs=archived, change_masks=mask_names, geometry_truth_claim=None)
    except BaseException as exc:
        record.update(status="diagnostics_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["code_sha256"] = {n: sha256(ROOT / n) for n in ("scripts/diagnose_quality_stages.py", "src/quality_diagnostics.py", "src/semantic2d.py")}
        write_new_json(args.output / "result.json", record)
        print(json.dumps({"status": record["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
