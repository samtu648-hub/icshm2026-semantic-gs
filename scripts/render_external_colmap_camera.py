#!/usr/bin/env python3
"""Render a trained R1 RGB candidate from an externally supplied official COLMAP camera.

This is deliberately independent from any image file: it consumes a COLMAP ``cameras.txt``
and ``images.txt`` pose, interprets the latter as CAMERA_FROM_WORLD, maps it through the
recorded official-to-model Sim(3), and writes an RGB image whose dimensions come from the
external camera.  R1's internal model camera is the full-frame undistorted PINHOLE camera;
the default output is remapped back to the official SIMPLE_RADIAL pixel grid.

It is a rendering-interface gate, not a blind-test evaluator.  In particular, this tool
does not load a reference RGB image or semantic label.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.camera_adapter import (  # noqa: E402
    CameraFromWorld,
    Sim3,
    SimpleRadialCamera,
    official_to_model_camera,
    redistort_rgb,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _data_lines(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]


def read_simple_radial_camera(path: Path, camera_id: int | None) -> tuple[int, SimpleRadialCamera]:
    entries = _data_lines(path)
    if camera_id is not None:
        entries = [entry for entry in entries if int(entry[0]) == camera_id]
    if len(entries) != 1:
        raise ValueError("Expected exactly one selected camera in cameras.txt; pass --camera-id if needed")
    entry = entries[0]
    if len(entry) != 8 or entry[1] != "SIMPLE_RADIAL":
        raise ValueError("R1 external-camera gate currently requires an official SIMPLE_RADIAL camera")
    selected_id, _, width, height, focal, cx, cy, radial = entry
    return int(selected_id), SimpleRadialCamera(
        width=int(width), height=int(height), focal=float(focal), cx=float(cx), cy=float(cy), radial_k=float(radial)
    )


def read_pose(path: Path, image_name: str | None, camera_id: int) -> tuple[str, CameraFromWorld]:
    poses: list[tuple[str, CameraFromWorld]] = []
    for fields in _data_lines(path):
        if len(fields) < 10 or Path(fields[9]).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if int(fields[8]) != camera_id:
            continue
        name = Path(fields[9]).name
        if image_name is not None and name != image_name:
            continue
        qw, qx, qy, qz = (float(value) for value in fields[1:5])
        norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if norm == 0:
            raise ValueError(f"Zero-norm COLMAP quaternion for {name}")
        qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
        rotation = np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=np.float64,
        )
        poses.append((name, CameraFromWorld(rotation, np.array([float(value) for value in fields[5:8]]))))
    if len(poses) != 1:
        selection = image_name if image_name is not None else "the sole pose"
        raise ValueError(f"Expected exactly one selected pose ({selection}) in {path}; found {len(poses)}")
    return poses[0]


def configure_candidate_import_path(source_root: Path) -> None:
    # CUDA extensions are installed into the selected isolated environment.  Do not
    # prepend their source directories: doing so masks the installed package and
    # its compiled ``_C`` module, especially for a separately built MCMC checkout.
    text = str(source_root)
    if text not in sys.path:
        sys.path.insert(0, text)


def make_view(camera, model_pose: CameraFromWorld, appearance_camera_id: int):
    import torch

    from scene.cameras import MiniCam
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2

    fov_x = 2.0 * math.atan(camera.width / (2.0 * camera.focal))
    fov_y = 2.0 * math.atan(camera.height / (2.0 * camera.focal))
    # Upstream Camera stores Rcw.T and getWorld2View2 transposes it again internally.
    stored_rotation = model_pose.rotation.T
    world_view = torch.tensor(getWorld2View2(stored_rotation, model_pose.translation)).transpose(0, 1).cuda()
    projection = getProjectionMatrix(0.01, 100.0, fov_x, fov_y).transpose(0, 1).cuda()
    view = MiniCam(
        camera.width,
        camera.height,
        fov_y,
        fov_x,
        0.01,
        100.0,
        world_view,
        (world_view.unsqueeze(0).bmm(projection.unsqueeze(0))).squeeze(0),
    )
    # Scaffold-GS uses this index for its learned per-training-camera appearance embedding.
    # The documented deterministic fallback for an external camera is a caller-selected known
    # training embedding; the default is index 0 and does not inspect any test RGB.
    view.uid = appearance_camera_id
    return view


def load_and_render(args: argparse.Namespace, view):
    import torch

    source_root = args.source_root.resolve()
    configure_candidate_import_path(source_root)
    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        # Standard 3DGS exposes this pipeline field while Mip/Scaffold ignore it.
        antialiasing=False,
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    point_cloud = args.model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if not point_cloud.is_file():
        raise FileNotFoundError(f"Missing saved model PLY: {point_cloud}")

    if args.candidate in {"standard-3dgs", "mip-splatting", "3dgs-mcmc"}:
        from gaussian_renderer import GaussianModel, render

        gaussians = GaussianModel(3)
        gaussians.load_ply(str(point_cloud))
        if args.candidate == "mip-splatting":
            result = render(view, gaussians, pipe, background, kernel_size=0.1)["render"]
        else:
            result = render(view, gaussians, pipe, background)["render"]
    else:
        from gaussian_renderer import prefilter_voxel, render
        from scene.gaussian_model import GaussianModel

        gaussians = GaussianModel(
            feat_dim=32,
            n_offsets=args.scaffold_n_offsets,
            voxel_size=0.001,
            update_depth=3,
            update_init_factor=16,
            update_hierachy_factor=4,
            use_feat_bank=False,
            appearance_dim=32,
            ratio=1,
            add_opacity_dist=False,
            add_cov_dist=False,
            add_color_dist=False,
        )
        gaussians.set_appearance(args.scaffold_train_camera_count)
        gaussians.load_ply_sparse_gaussian(str(point_cloud))
        # Scaffold-GS stores its TorchScript MLPs next to point_cloud.ply,
        # whereas load_ply_sparse_gaussian takes the PLY file itself.
        gaussians.load_mlp_checkpoints(str(point_cloud.parent))
        gaussians.eval()
        visible_mask = prefilter_voxel(view, gaussians, pipe, background)
        result = render(view, gaussians, pipe, background, visible_mask=visible_mask)["render"]
    return result.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("standard-3dgs", "mip-splatting", "scaffold-gs", "3dgs-mcmc"), required=True)
    parser.add_argument("--source-root", type=Path, required=True, help="Git-ignored pinned upstream candidate source")
    parser.add_argument("--model-path", type=Path, required=True, help="R1 run directory containing point_cloud")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--cameras", type=Path, required=True, help="Externally supplied official COLMAP cameras.txt")
    parser.add_argument("--images", type=Path, required=True, help="Externally supplied official COLMAP images.txt")
    parser.add_argument("--image-name", help="Image/pose name to render; mandatory when the file has multiple poses")
    parser.add_argument("--camera-id", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, help="Optional JSON evidence record")
    parser.add_argument("--pinhole-output", action="store_true", help="Skip final SIMPLE_RADIAL remap for internal debugging only")
    parser.add_argument("--scaffold-train-camera-count", type=int, default=303)
    parser.add_argument("--scaffold-n-offsets", type=int, default=10)
    parser.add_argument("--appearance-camera-id", type=int, default=0)
    args = parser.parse_args()
    if args.scaffold_train_camera_count <= args.appearance_camera_id:
        raise ValueError("appearance camera ID must be within the saved Scaffold-GS training embedding table")
    if args.pinhole_output and args.candidate == "scaffold-gs":
        # The restriction is not technical; keeping it uniform makes the verification record unambiguous.
        raise ValueError("External gate evidence must use the official distorted output grid")

    selected_camera_id, camera = read_simple_radial_camera(args.cameras, args.camera_id)
    image_name, official_pose = read_pose(args.images, args.image_name, selected_camera_id)
    identity = Sim3(1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    model_pose = official_to_model_camera(official_pose, identity)
    configure_candidate_import_path(args.source_root.resolve())
    view = make_view(camera, model_pose, args.appearance_camera_id)
    pinhole_rgb = load_and_render(args, view)
    output_rgb = pinhole_rgb if args.pinhole_output else redistort_rgb(pinhole_rgb, camera)
    output_u8 = np.rint(np.clip(output_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_u8, mode="RGB").save(args.output)

    evidence = {
        "candidate": args.candidate,
        "model_path": str(args.model_path.resolve()),
        "iteration": args.iteration,
        "input_camera_model": "SIMPLE_RADIAL",
        "input_pose_convention": "CAMERA_FROM_WORLD",
        "official_to_model_sim3": {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]},
        "selected_pose_name": image_name,
        "output_path": str(args.output.resolve()),
        "output_width": camera.width,
        "output_height": camera.height,
        "output_grid": "PINHOLE" if args.pinhole_output else "official_SIMPLE_RADIAL",
        "uses_reference_rgb_or_labels": False,
        "scaffold_external_appearance_fallback": (
            {"embedding_index": args.appearance_camera_id, "training_embedding_count": args.scaffold_train_camera_count}
            if args.candidate == "scaffold-gs"
            else None
        ),
    }
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
