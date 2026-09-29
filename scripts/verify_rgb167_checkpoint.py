#!/usr/bin/env python3
"""Fresh-process finite/checkpoint/model/UID verification for RGB167."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_cameras import read_camera_requests
from src.quality_native_gaussian import make_native_view
from src.quality_protocol import sha256, write_new_json
from src.quality_scaffold import ScaffoldQualityRenderer
from src.scaffold_semantic_sidecar import render_frozen_rgb


def finite_tree(value, torch, counters: dict[str, int]) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError("Non-finite tensor in recoverable checkpoint")
        counters["tensor_count"] += 1
        counters["elements"] += value.numel()
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child, torch, counters)
    elif isinstance(value, (tuple, list)):
        for child in value:
            finite_tree(child, torch, counters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model-subdir", default="model")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    model_root = args.run / args.model_subdir
    checkpoint = model_root / f"chkpnt{args.iteration}.pth"
    state, saved_iteration = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved_iteration != args.iteration:
        raise ValueError(f"Checkpoint iteration {saved_iteration} != {args.iteration}")
    if state["embedding_appearance"] is not None or state["mlp_feature_bank"] is not None:
        raise ValueError("RGB167 must not contain appearance or feature-bank state")
    groups = [group["name"] for group in state["optimizer"]["param_groups"]]
    if "embedding_appearance" in groups:
        raise ValueError("Appearance optimizer group unexpectedly restored")
    counters = {"tensor_count": 0, "elements": 0}
    finite_tree(state, torch, counters)

    renderer = ScaffoldQualityRenderer(
        args.artifact_root / "r1/third_party/scaffold-gs",
        model_root,
        args.iteration,
        train_camera_count=400,
    )
    if renderer.model.appearance_dim != 0 or renderer.model.embedding_appearance is not None:
        raise ValueError("Saved model is not UID independent")
    signature = renderer.rgb_signature()
    cameras = read_camera_requests(args.run / "scene/sparse/0")
    cases = []
    with torch.inference_mode():
        for name in ("159.png", "160.png", "174.png"):
            reference = None
            for uid in (0, 158, 399, 999):
                view = make_native_view(cameras[name])
                view.uid = uid
                image = render_frozen_rgb(view, renderer.model, renderer.pipe)
                if reference is None:
                    reference = image.clone()
                elif not torch.equal(reference, image):
                    raise ValueError(f"UID changed RGB for {name} at UID {uid}")
                cases.append({"view": name, "uid": uid, "exact": True})
    if signature != renderer.rgb_signature():
        raise ValueError("Verification modified saved model state")

    ply = model_root / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    result = {
        "status": "passed_finite_checkpoint_model_reload_and_uid_independence",
        "iteration": args.iteration,
        "checkpoint_sha256": sha256(checkpoint),
        "point_cloud_sha256": sha256(ply),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "model_endpoint_bytes": sum(path.stat().st_size for path in ply.parent.iterdir() if path.is_file()),
        "anchors": len(state["anchor"]),
        "offsets_per_anchor": state["offset"].shape[1],
        "appearance_absent": True,
        "optimizer_groups": groups,
        "signature": signature,
        "uid_cases": cases,
        **counters,
    }
    write_new_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "iteration", "anchors", "checkpoint_bytes", "model_endpoint_bytes")}))


if __name__ == "__main__":
    main()
