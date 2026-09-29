#!/usr/bin/env python3
"""Build fixed RGB170-residual EAS reliability maps with bridge hard protection."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, convolve

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from scripts.evaluate_rgb166_horizon import BRIDGE_ROIS, DIFFICULT_GROUPS
from src.improvedgs_bridge_protected_eas import build_reliability_map, quantize_reliability
from src.quality_cameras import read_camera_requests
from src.quality_protocol import sha256, write_new_json

EXPERIMENT_ID = "QUAL-025-BRIDGE-PROTECTED-EAS-DIAG"
CONFIG = ROOT / "experiments/runs" / EXPERIMENT_ID / "config.yaml"
SCENE = ROOT / "outputs/quality/all_data/RGB-120-r2/scene/images"
LABELS = ROOT / "outputs/quality/all_data/SEM-370-r1/labels"
PARENT = ROOT / "outputs/quality/RGB-170/full-r1/evaluation-r1/images/50000"
PARENT_PLY = ROOT / "outputs/quality/RGB-170/full-r1/model/point_cloud/iteration_50000/point_cloud.ply"
CONTRIBUTORS = ROOT / "outputs/quality/QUAL-022/diagnostic-r1/contributors.jsonl.gz"
REVIEW_NAMES = [
    "159.png", "160.png", "235.png", "240.png", "245.png",
    "385.png", "386.png", "387.png", "388.png", "389.png", "390.png", "391.png", "392.png",
]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def exact_edge_map(image8: np.ndarray) -> np.ndarray:
    """NumPy equivalent of the pinned ImprovedGS positive Laplacian edge map."""
    rgb = image8.astype(np.float32)
    gray = np.rint((299.0 * rgb[..., 0] + 587.0 * rgb[..., 1] + 114.0 * rgb[..., 2]) / 1000.0)
    kernel = np.asarray([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]], dtype=np.float32)
    response = np.clip(convolve(gray, kernel, mode="constant", cval=0.0), 0.0, 255.0) / 255.0
    low, high = float(response.min()), float(response.max())
    return np.zeros_like(response) if high <= low else (response - low) / (high - low)


def read_ply_xyz(path: Path, selected_ids: np.ndarray) -> np.ndarray:
    """Read selected XYZ rows from the binary-little-endian all-float PLY."""
    properties: list[str] = []
    vertex_count = None
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("unterminated PLY header")
            text = line.decode("ascii").strip()
            if text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])
            elif text.startswith("property float "):
                properties.append(text.split()[-1])
            elif text.startswith("property "):
                raise ValueError("QUAL025 expects the RGB170 all-float PLY layout")
            elif text == "end_header":
                offset = handle.tell()
                break
    if vertex_count is None or properties[:3] != ["x", "y", "z"]:
        raise ValueError("unexpected RGB170 PLY header")
    if int(selected_ids.max(initial=-1)) >= vertex_count:
        raise ValueError("contributor Gaussian ID exceeds PLY vertex count")
    dtype = np.dtype([(name, "<f4") for name in properties])
    vertices = np.memmap(path, mode="r", offset=offset, dtype=dtype, shape=(vertex_count,))
    return np.column_stack([vertices[axis][selected_ids] for axis in ("x", "y", "z")]).astype(np.float64)


def confirmed_bridge_centers(policy: dict[str, object]) -> tuple[np.ndarray, dict[str, int]]:
    minimum_views = int(policy["minimum_contributor_views"])
    by_class: dict[str, set[int]] = {name: set() for name in ("deck", "stay_cable", "tower", "foundation")}
    with gzip.open(CONTRIBUTORS, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            class_name = row.get("class")
            if (
                class_name in by_class
                and row.get("kind") == "true_like"
                and int(row.get("cross_view_top_count", 0)) >= minimum_views
            ):
                by_class[class_name].add(int(row["gaussian_id"]))
    selected = np.asarray(sorted(set().union(*by_class.values())), dtype=np.int64)
    if not len(selected):
        raise ValueError("No QUAL022 cross-view bridge contributors passed the frozen rule")
    return selected, {key: len(value) for key, value in by_class.items()}


def project_center_protection(xyz: np.ndarray, camera: object, shape: tuple[int, int], dilation: int) -> np.ndarray:
    rotation = np.asarray(camera.pose.rotation, dtype=np.float64)
    translation = np.asarray(camera.pose.translation, dtype=np.float64)
    camera_xyz = xyz @ rotation.T + translation
    z = camera_xyz[:, 2]
    safe_z = np.where(z > 1e-12, z, np.nan)
    matrix = np.asarray(camera.matrix, dtype=np.float64)
    u = matrix[0, 0] * camera_xyz[:, 0] / safe_z + matrix[0, 2]
    v = matrix[1, 1] * camera_xyz[:, 1] / safe_z + matrix[1, 2]
    x = np.rint(np.nan_to_num(u, nan=-1.0)).astype(np.int64)
    y = np.rint(np.nan_to_num(v, nan=-1.0)).astype(np.int64)
    valid = (z > 0) & np.isfinite(u) & np.isfinite(v) & (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
    centers = np.zeros(shape, dtype=bool)
    centers[y[valid], x[valid]] = True
    return binary_dilation(centers, iterations=int(dilation))


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = (
        "residual_q_low", "residual_q_high", "residual_mean", "reliability_mean",
        "minimum_fraction", "bridge_fraction", "protected_fraction", "extra_protected_fraction", "edge_mass_retention",
        "bridge_edge_mass_retention", "background_edge_mass_retention",
    )
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def review_page(name: str, photo8: np.ndarray, parent8: np.ndarray, reliability8: np.ndarray, output: Path) -> None:
    residual = np.mean(np.abs(photo8.astype(np.float32) - parent8.astype(np.float32)), axis=2)
    residual = np.clip(residual / max(float(np.quantile(residual, 0.98)), 1.0), 0.0, 1.0)
    heat = np.stack((residual, np.sqrt(residual), 1.0 - residual), axis=2)
    heat8 = np.rint(heat * 255).astype(np.uint8)
    reliability_rgb = np.stack((255 - reliability8, reliability8, np.full_like(reliability8, 64)), axis=2)
    cells = [
        ("photo", Image.fromarray(photo8)),
        ("RGB170 50k", Image.fromarray(parent8)),
        ("RGB170 residual (red=high)", Image.fromarray(heat8)),
        ("EAS reliability (green=protected/reliable)", Image.fromarray(reliability_rgb)),
    ]
    crop = BRIDGE_ROIS.get(name)
    rendered: list[tuple[str, Image.Image]] = []
    for label, image in cells:
        if crop:
            image = image.crop(tuple(crop))
        image.thumbnail((500, 380))
        rendered.append((label, image.copy()))
    canvas = Image.new("RGB", (520 * len(rendered), 425), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(rendered):
        x = index * 520
        draw.text((x + 8, 8), f"{name} · {label}", fill="black")
        canvas.paste(image, (x + 8, 34))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/quality/QUAL-025/diagnostic-r2")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reliability_root = args.output / "reliability"
    reliability_root.mkdir()
    started = time.monotonic()
    cfg = json.loads(CONFIG.read_text())
    policy = cfg["policy"]
    names = sorted(path.name for path in SCENE.glob("*.png"))
    label_names = {path.name for path in LABELS.glob("*.png")}
    if len(names) != 400 or len(label_names) != 300:
        raise ValueError("Expected 400 photos and 300 labels")
    if any(not (PARENT / name).is_file() for name in names):
        raise FileNotFoundError("RGB170 all400 render set is incomplete")
    selected_ids, selected_by_class = confirmed_bridge_centers(policy)
    selected_xyz = read_ply_xyz(PARENT_PLY, selected_ids)
    cameras = read_camera_requests(ROOT / "outputs/quality/all_data/RGB-120-r2/scene/sparse/0")
    if set(cameras) != set(names):
        raise ValueError("Official all400 camera universe changed")
    projected_root = args.output / "projected-bridge-protection"
    projected_root.mkdir()

    protected_before = {
        "rgb170_ply": sha256(PARENT_PLY),
        "rgb170_evaluation": sha256(ROOT / "outputs/quality/RGB-170/full-r1/evaluation-r1/result.json"),
        "split": sha256(ROOT / "outputs/quality/all_data/RGB-120-r2/split.json"),
        "qual022_contributors": sha256(CONTRIBUTORS),
    }
    rows: list[dict[str, float | str | bool]] = []
    manifest: list[dict[str, str]] = []
    for index, name in enumerate(names):
        photo8 = load_rgb(SCENE / name)
        parent8 = load_rgb(PARENT / name)
        semantic_path = LABELS / name
        semantic = np.asarray(Image.open(semantic_path), dtype=np.uint8) if semantic_path.is_file() else None
        projected = project_center_protection(
            selected_xyz,
            cameras[name],
            photo8.shape[:2],
            int(policy["projected_bridge_dilation_pixels"]),
        ) if semantic is None else None
        if projected is not None:
            Image.fromarray(projected.astype(np.uint8) * 255, mode="L").save(projected_root / name, compress_level=6)
        reliability, stats = build_reliability_map(
            photo8,
            parent8,
            semantic,
            extra_protected_mask=projected,
            residual_sigma=float(policy["residual_sigma"]),
            lower_quantile=float(policy["lower_quantile"]),
            upper_quantile=float(policy["upper_quantile"]),
            minimum_reliability=float(policy["minimum_reliability"]),
            bridge_dilation_pixels=int(policy["bridge_dilation_pixels"]),
        )
        reliability8 = quantize_reliability(reliability)
        target = reliability_root / name
        Image.fromarray(reliability8, mode="L").save(target, compress_level=6)
        edge = exact_edge_map(photo8)
        bridge = semantic > 0 if semantic is not None else np.zeros(edge.shape, dtype=bool)
        background = ~bridge
        edge_total = float(edge.sum())
        bridge_edge = float(edge[bridge].sum())
        background_edge = float(edge[background].sum())
        stats.update(
            name=name,
            labeled=semantic is not None,
            edge_mass_retention=float((edge * reliability).sum() / max(edge_total, 1e-12)),
            bridge_edge_mass_retention=float((edge[bridge] * reliability[bridge]).sum() / max(bridge_edge, 1e-12)) if bridge.any() else 1.0,
            background_edge_mass_retention=float((edge[background] * reliability[background]).sum() / max(background_edge, 1e-12)),
        )
        rows.append(stats)
        manifest.append({"name": name, "sha256": sha256(target)})
        if name in REVIEW_NAMES:
            review_page(name, photo8, parent8, reliability8, args.output / "review" / f"{name[:-4]}.jpg")
        if index % 50 == 0:
            print(json.dumps({"processed": index + 1, "name": name}), flush=True)

    typed_rows = [dict(row) for row in rows]
    by_name = {str(row["name"]): row for row in typed_rows}
    groups = {
        "all400": names,
        "labeled300": sorted(label_names),
        "unlabeled100": [name for name in names if name not in label_names],
        **DIFFICULT_GROUPS,
    }
    summary = {group: aggregate([by_name[name] for name in selected]) for group, selected in groups.items()}
    protected_after = {
        "rgb170_ply": sha256(PARENT_PLY),
        "rgb170_evaluation": sha256(ROOT / "outputs/quality/RGB-170/full-r1/evaluation-r1/result.json"),
        "split": sha256(ROOT / "outputs/quality/all_data/RGB-120-r2/split.json"),
        "qual022_contributors": sha256(CONTRIBUTORS),
    }
    if protected_after != protected_before:
        raise RuntimeError("Protected parent or split changed during diagnostic")
    write_new_json(args.output / "mask-manifest.json", {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "count": len(manifest),
        "files": manifest,
    })
    write_new_json(args.output / "per-view.json", typed_rows)
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_read_only",
        "read_only": True,
        "policy": policy,
        "config_sha256": sha256(CONFIG),
        "code_sha256": {
            "driver": sha256(Path(__file__)),
            "helper": sha256(ROOT / "src/improvedgs_bridge_protected_eas.py"),
        },
        "inputs": protected_before,
        "protected_exact": True,
        "views": {"all": 400, "labeled": 300, "unlabeled": 100},
        "projected_bridge_protection": {
            "selected_gaussians": int(len(selected_ids)),
            "selected_by_class_nonexclusive": selected_by_class,
            "source": "QUAL022 true_like contributors with frozen minimum cross-view count; official-camera center projection",
        },
        "summary": summary,
        "mask_manifest_sha256": sha256(args.output / "mask-manifest.json"),
        "review_views": REVIEW_NAMES,
        "seconds": time.monotonic() - started,
        "interpretation": "Fixed parent-residual reliability only changes ImprovedGS EAS ranking. Labels protect labeled 2D bridge evidence; unlabeled protection projects existing QUAL022-confirmed 3D centers through official cameras. No 3D geometry is created or moved.",
    }
    write_new_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
