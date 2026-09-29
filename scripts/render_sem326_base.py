#!/usr/bin/env python3
"""SEM326 probe: unchanged frozen bundle render, additionally export PINHOLE RGB."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import render_submission as base
from src.submission_sem161 import load_segmenter, sliding, fuse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "cameras", "images", "output-dir"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--image-name", action="append")
    parser.add_argument("--write-fallbacks", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite existing inference output")
    bundle = args.bundle.resolve()
    lock = json.loads((bundle / "metadata/sem161_lock.json").read_text())
    base._verify_locked_artifacts(bundle, lock)
    config = json.loads((bundle / "metadata/sem001_config.json").read_text())
    poses = base.select_external_poses(base.list_external_poses(args.images), args.image_name)
    if len({Path(name).stem.casefold() for name, _ in poses}) != len(poses):
        raise ValueError("Output stem collision")
    rgbroot = bundle / "model/rgb"
    source = bundle / "third_party/scaffold-gs"
    base.configure_scaffold_import(source)
    model = base.load_frozen_scaffold(source, rgbroot / "point_cloud/iteration_30000/point_cloud.ply", base.ScaffoldSettings(train_camera_count=303, n_offsets=10))
    semantic = base._load_sidecar(bundle / "model/semantic/semantic_sidecar.pth", len(model.get_anchor), config, base.frozen_rgb_hashes(rgbroot, 30000))
    networks = [load_segmenter(bundle / "model/segformer", bundle / row[0], row[1], row[2]) for row in (
        ("model/segformer/five.pth", 5, "SEM-102-RENDERED-SEG-FINAL"),
        ("model/segformer/cable.pth", 2, "SEM-141-CABLE-BINARY-FINAL"),
        ("model/segformer/deck.pth", 2, "SEM-161-DECK-BINARY-FINAL"))]
    import torch
    pipe = base.pipeline()
    identity = base.Sim3(1., np.eye(3), np.zeros(3))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    records = []
    for name, camera_id in poses:
        camera = base.read_simple_radial_camera(args.cameras, camera_id)[1]
        pose = base.read_pose(args.images, name, camera_id)[1]
        view = base.make_view(camera, base.official_to_model_camera(pose, identity), 0)
        with torch.inference_mode():
            pinhole = base._tensor_rgb_u8(base.render_frozen_rgb(view, model, pipe))
            semantic_pinhole = base.render_semantic_probabilities(view, model, semantic, pipe).cpu().numpy().astype(np.float32)
        rgb = np.rint(base.redistort_rgb(pinhole.astype(np.float32)/255., camera).clip(0,1)*255).astype(np.uint8)
        prediction = base.redistort_semantic_probabilities(semantic_pinhole, camera).argmax(0).astype(np.uint8)
        five = base.redistort_semantic_probabilities(sliding(networks[0], pinhole, crop=768, stride=512, classes=5), camera)
        cable = sliding(networks[1], rgb, crop=960, stride=480, classes=2)[1]
        deck = sliding(networks[2], rgb, crop=960, stride=480, classes=2)[1]
        stages = fuse(prediction, five, cable, deck)
        outputs = {"rgb": rgb, "semantic": stages["SEM-161"], "pinhole_rgb": pinhole}
        if args.write_fallbacks:
            outputs.update(stages)
        record = {"pose": name, "camera_id": camera_id, "width": camera.width, "height": camera.height, "outputs": {}}
        for stage, data in outputs.items():
            path = args.output_dir / stage / (Path(name).stem+".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(data).save(path)
            record["outputs"][stage] = {"path": str(path.relative_to(args.output_dir)), "sha256": base.sha256(path)}
        records.append(record)
        print(json.dumps({"rendered": name, "count": len(records)}), flush=True)
    torch.cuda.synchronize()
    result = {"experiment_id": "SUBMIT-030-SEM161-BUNDLE", "bundle_lock_sha256": base.sha256(bundle / "metadata/sem161_lock.json"), "uses_reference_rgb_or_labels": False, "input_camera_model": "SIMPLE_RADIAL", "pose_convention": "COLMAP_CAMERA_FROM_WORLD", "cameras_sha256": base.sha256(args.cameras), "images_sha256": base.sha256(args.images), "seconds": time.perf_counter()-start, "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20, "outputs": records}
    (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
