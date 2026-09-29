#!/usr/bin/env python3
"""Exercise a frozen-RGB, geometry-attached five-class semantic sidecar.

This is deliberately an R1 interface smoke test, not semantic training or an
accuracy claim.  It writes logits attached to fixed Gaussian/anchor vertices,
performs one optimization step on those logits only, renders a deterministic
z-buffer preview from an external official CAMERA_FROM_WORLD pose, and proves
that both the RGB evidence image and the frozen PLY remain byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from render_external_colmap_camera import read_pose, read_simple_radial_camera

CLASS_COUNT = 5
MAX_RENDERED_ENTITIES = 150_000


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("standard-3dgs", "mip-splatting", "scaffold-gs"), required=True)
    parser.add_argument("--point-cloud", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--rgb-reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--logits-output", type=Path, required=True)
    parser.add_argument("--class-output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    return parser.parse_args()


def load_points(path: Path) -> np.ndarray:
    from plyfile import PlyData

    vertices = PlyData.read(path)["vertex"].data
    return np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32, copy=False)


def train_and_reload(entity_count: int, checkpoint: Path) -> np.ndarray:
    """Optimize only a sidecar tensor; frozen RGB model tensors are never loaded or changed."""
    import torch

    torch.manual_seed(20260730)
    logits = torch.nn.Parameter(torch.zeros((entity_count, CLASS_COUNT), dtype=torch.float32))
    targets = torch.arange(entity_count, dtype=torch.long) % CLASS_COUNT
    optimizer = torch.optim.Adam([logits], lr=0.01)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "candidate_semantic_storage": "per_anchor_logits" if entity_count < 500_000 else "per_gaussian_logits",
            "class_count": CLASS_COUNT,
            "entity_count": entity_count,
            "semantic_logits": logits.detach().cpu(),
        },
        checkpoint,
    )
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored_logits = restored["semantic_logits"].numpy()
    if restored_logits.shape != (entity_count, CLASS_COUNT) or not np.array_equal(
        restored_logits, logits.detach().cpu().numpy()
    ):
        raise RuntimeError("Semantic sidecar checkpoint reload mismatch")
    return restored_logits


def render_logits(points: np.ndarray, logits: np.ndarray, camera, pose) -> np.ndarray:
    """Render a sparse deterministic logits preview from fixed 3D entities.

    The projection uses the official distorted SIMPLE_RADIAL grid directly. It
    is intentionally lightweight: the later R2 semantic renderer replaces
    this z-buffer smoke renderer without changing frozen RGB state.
    """
    if len(points) > MAX_RENDERED_ENTITIES:
        selected = np.linspace(0, len(points) - 1, MAX_RENDERED_ENTITIES, dtype=np.int64)
        points, logits = points[selected], logits[selected]
    camera_points = points @ pose.rotation.T + pose.translation
    depth = camera_points[:, 2]
    valid = depth > 1e-6
    normalized = camera_points[valid, :2] / depth[valid, None]
    radius2 = np.sum(normalized * normalized, axis=1)
    distorted = normalized * (1.0 + camera.radial_k * radius2)[:, None]
    u = np.rint(camera.focal * distorted[:, 0] + camera.cx).astype(np.int64)
    v = np.rint(camera.focal * distorted[:, 1] + camera.cy).astype(np.int64)
    valid_pixels = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    u, v = u[valid_pixels], v[valid_pixels]
    depth = depth[valid][valid_pixels]
    entity_logits = logits[valid][valid_pixels]

    output = np.full((CLASS_COUNT, camera.height * camera.width), -8.0, dtype=np.float32)
    output[0, :] = 8.0  # background where no frozen entity projects.
    if len(u):
        linear = v * camera.width + u
        far_to_near = np.argsort(depth)[::-1]
        linear, entity_logits = linear[far_to_near], entity_logits[far_to_near]
        # Repeated writes deliberately leave the nearest (last) projected entity.
        output[:, linear] = entity_logits.T
    return output.reshape(CLASS_COUNT, camera.height, camera.width)


def main() -> int:
    args = parse_args()
    if not args.point_cloud.is_file() or not args.rgb_reference.is_file():
        raise FileNotFoundError("point cloud and RGB reference must exist")
    rgb_before, ply_before = digest(args.rgb_reference), digest(args.point_cloud)
    camera_id, camera = read_simple_radial_camera(args.cameras, None)
    selected_name, pose = read_pose(args.images, args.image_name, camera_id)
    points = load_points(args.point_cloud)
    logits = train_and_reload(len(points), args.checkpoint)
    rendered = render_logits(points, logits, camera, pose)
    args.logits_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.logits_output, rendered)
    labels = rendered.argmax(axis=0).astype(np.uint8)
    Image.fromarray(labels, mode="L").save(args.class_output)
    rgb_after, ply_after = digest(args.rgb_reference), digest(args.point_cloud)
    evidence = {
        "schema_version": 1,
        "candidate": args.candidate,
        "semantic_storage": "per_anchor_logits" if args.candidate == "scaffold-gs" else "per_gaussian_logits",
        "class_count": CLASS_COUNT,
        "logits_shape": list(rendered.shape),
        "semantic_class_png_shape": [camera.height, camera.width],
        "selected_pose_name": selected_name,
        "input_pose_convention": "CAMERA_FROM_WORLD",
        "uses_reference_rgb_or_labels_for_semantics": False,
        "semantic_optimizer_parameters": "sidecar semantic_logits only",
        "semantic_checkpoint_reloaded": True,
        "rgb_reference_sha256_before": rgb_before,
        "rgb_reference_sha256_after": rgb_after,
        "rgb_max_absolute_pixel_difference": 0,
        "frozen_ply_sha256_before": ply_before,
        "frozen_ply_sha256_after": ply_after,
        "frozen_geometry_color_opacity_unchanged": ply_before == ply_after,
    }
    if rgb_before != rgb_after or ply_before != ply_after:
        raise RuntimeError("Frozen RGB evidence or PLY changed during semantic smoke test")
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
