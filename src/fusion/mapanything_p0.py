"""Conservative P0 multi-view fusion for MapAnything geometry outputs.

This module is deliberately model-agnostic after NPZ creation: it consumes the
versioned ``geometry_prior_npz_v2`` records, verifies reprojection agreement in
the official coordinate frame, and exports only XYZ plus original RGB.  It
does not alter Scaffold-GS source code, features, or RGB training losses.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PLY_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


@dataclass
class GeometryView:
    name: str
    depth: np.ndarray
    confidence: np.ndarray
    valid_mask: np.ndarray
    intrinsics: np.ndarray
    cam2world: np.ndarray
    source_npz: Path


def load_best_views(run_dir: Path) -> dict[str, GeometryView]:
    """Pick the most supported context prediction per image deterministically."""
    manifest = json.loads((run_dir / "window_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model_execution") != "completed":
        raise ValueError("MapAnything run manifest is not completed")
    selected: dict[str, GeometryView] = {}
    selected_keys: dict[str, tuple[float, float, str]] = {}
    for record in manifest.get("records", []):
        path = run_dir / record["output_npz"]
        with np.load(path) as archive:
            required = {"depth", "confidence", "valid_mask", "pinhole_intrinsics", "official_cam2world"}
            if not required.issubset(archive.files):
                raise ValueError(f"{path} does not satisfy geometry_prior_npz_v2 fields")
            view = GeometryView(
                name=record["image_name"],
                depth=np.asarray(archive["depth"], dtype=np.float32),
                confidence=np.asarray(archive["confidence"], dtype=np.float32),
                valid_mask=np.asarray(archive["valid_mask"], dtype=bool),
                intrinsics=np.asarray(archive["pinhole_intrinsics"], dtype=np.float64),
                cam2world=np.asarray(archive["official_cam2world"], dtype=np.float64),
                source_npz=path,
            )
        if view.depth.shape != view.confidence.shape or view.depth.shape != view.valid_mask.shape:
            raise ValueError(f"Inconsistent depth arrays in {path}")
        # Same model calibration applies to all overlapping contexts. Prefer the
        # map with the greatest valid, finite confidence support; file path is
        # only a deterministic tie-breaker. Select while streaming so a
        # 25-context run retains one view per image instead of all 7,575 arrays.
        key = (
            float(np.mean(view.confidence[view.valid_mask]))
            if view.valid_mask.any()
            else float("-inf"),
            float(view.valid_mask.mean()),
            str(view.source_npz),
        )
        if view.name not in selected_keys or key > selected_keys[view.name]:
            selected[view.name] = view
            selected_keys[view.name] = key
    return selected


def _normalised_confidence(view: GeometryView) -> np.ndarray:
    values = view.confidence[view.valid_mask & np.isfinite(view.confidence)]
    if values.size < 2:
        return np.zeros_like(view.confidence, dtype=np.float32)
    low, high = np.quantile(values, (0.05, 0.95))
    return np.clip((view.confidence - low) / max(float(high - low), 1e-8), 0.0, 1.0).astype(np.float32)


def _project_world(points: np.ndarray, target: GeometryView) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation_c2w = target.cam2world[:3, :3]
    center = target.cam2world[:3, 3]
    camera = (points - center) @ rotation_c2w
    z = camera[:, 2]
    u = target.intrinsics[0, 0] * camera[:, 0] / np.maximum(z, 1e-12) + target.intrinsics[0, 2]
    v = target.intrinsics[1, 1] * camera[:, 1] / np.maximum(z, 1e-12) + target.intrinsics[1, 2]
    return u, v, z


def _source_points(view: GeometryView, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if stride < 1:
        raise ValueError("stride must be positive")
    y, x = np.mgrid[0 : view.depth.shape[0] : stride, 0 : view.depth.shape[1] : stride]
    depth = view.depth[y, x]
    valid = view.valid_mask[y, x] & np.isfinite(depth) & (depth > 1e-6)
    x, y, depth = x[valid].astype(np.float32), y[valid].astype(np.float32), depth[valid]
    camera = np.column_stack(
        (
            (x - view.intrinsics[0, 2]) * depth / view.intrinsics[0, 0],
            (y - view.intrinsics[1, 2]) * depth / view.intrinsics[1, 1],
            depth,
        )
    )
    points = camera @ view.cam2world[:3, :3].T + view.cam2world[:3, 3]
    raw_confidence = view.confidence[y.astype(np.int64), x.astype(np.int64)]
    confidence = _normalised_confidence(view)[y.astype(np.int64), x.astype(np.int64)]
    # Finite-difference normals are diagnostics for fusion and PLY review. The
    # first-round Scaffold initialization intentionally consumes only XYZ/RGB.
    x_right = np.minimum(x.astype(np.int64) + 1, view.depth.shape[1] - 1)
    y_down = np.minimum(y.astype(np.int64) + 1, view.depth.shape[0] - 1)
    right_depth = view.depth[y.astype(np.int64), x_right]
    down_depth = view.depth[y_down, x.astype(np.int64)]
    def backproject(px: np.ndarray, py: np.ndarray, z: np.ndarray) -> np.ndarray:
        camera_points = np.column_stack(((px - view.intrinsics[0, 2]) * z / view.intrinsics[0, 0], (py - view.intrinsics[1, 2]) * z / view.intrinsics[1, 1], z))
        return camera_points @ view.cam2world[:3, :3].T + view.cam2world[:3, 3]
    normal = np.cross(backproject(x_right, y.astype(np.int64), right_depth) - points, backproject(x.astype(np.int64), y_down, down_depth) - points)
    norm = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm > 1e-8).astype(np.float32)
    return points.astype(np.float32), x.astype(np.int64), y.astype(np.int64), raw_confidence.astype(np.float32), confidence, normal


def _strong_neighbours(graph: dict, name: str, available: set[str], count: int) -> list[str]:
    selected = []
    for edge in graph.get("adjacency", {}).get(name, []):
        target = edge["target"] if edge.get("source") == name else edge.get("source")
        if target in available and target != name:
            selected.append((int(edge.get("shared_point3d", 0)), target))
    selected.sort(key=lambda item: (-item[0], item[1]))
    return [target for _, target in selected[:count]]


def _line_response(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import cv2

    resized = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 100, 200) > 0


def fuse_mapanything_p0(
    views: dict[str, GeometryView],
    graph: dict,
    image_root: Path,
    *,
    stride: int,
    consistency_neighbours: int,
    relative_depth_tolerance: float,
    ordinary_min_support: int,
    line_min_support: int,
) -> dict[str, np.ndarray]:
    """Keep P0 points supported by independent strong co-visible views."""
    import cv2

    all_xyz: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    all_score: list[np.ndarray] = []
    all_support: list[np.ndarray] = []
    all_line: list[np.ndarray] = []
    all_normal: list[np.ndarray] = []
    all_source: list[np.ndarray] = []
    all_raw_confidence: list[np.ndarray] = []
    available = set(views)
    source_names = sorted(views)
    for source_index, name in enumerate(source_names):
        source = views[name]
        rgb_bgr = cv2.imread(str(image_root / name), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise FileNotFoundError(f"Cannot read original RGB for {name}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        points, x, y, raw_confidence, confidence, normal = _source_points(source, stride)
        if not len(points):
            continue
        support = np.ones(len(points), dtype=np.int16)
        for target_name in _strong_neighbours(graph, name, available, consistency_neighbours):
            target = views[target_name]
            u, v, z = _project_world(points, target)
            px, py = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
            inside = (z > 1e-6) & (px >= 0) & (px < target.depth.shape[1]) & (py >= 0) & (py < target.depth.shape[0])
            sampled = np.zeros(len(points), dtype=np.float32)
            target_valid = np.zeros(len(points), dtype=bool)
            valid_indices = np.flatnonzero(inside)
            sampled[valid_indices] = target.depth[py[valid_indices], px[valid_indices]]
            target_valid[valid_indices] = target.valid_mask[py[valid_indices], px[valid_indices]]
            consistent = inside & target_valid & np.isfinite(sampled) & (sampled > 1e-6)
            consistent &= np.abs(sampled - z) / np.maximum(np.maximum(sampled, z), 1e-6) <= relative_depth_tolerance
            support += consistent.astype(np.int16)
        lines = _line_response(rgb, source.depth.shape)[y, x]
        keep = (support >= ordinary_min_support) | (lines & (support >= line_min_support))
        if not keep.any():
            continue
        rgb_small = cv2.resize(rgb, (source.depth.shape[1], source.depth.shape[0]), interpolation=cv2.INTER_AREA)
        all_xyz.append(points[keep])
        all_rgb.append(rgb_small[y[keep], x[keep]])
        all_score.append(confidence[keep])
        all_support.append(support[keep])
        all_line.append(lines[keep])
        all_normal.append(normal[keep])
        all_source.append(np.full(int(keep.sum()), source_index, dtype=np.int16))
        all_raw_confidence.append(raw_confidence[keep])
    if not all_xyz:
        raise ValueError("No points passed the P0 multi-view support gate")
    return {
        "xyz": np.concatenate(all_xyz).astype(np.float32),
        "rgb": np.concatenate(all_rgb).astype(np.uint8),
        "calibrated_confidence": np.concatenate(all_score).astype(np.float32),
        "raw_confidence": np.concatenate(all_raw_confidence).astype(np.float32),
        "support_views": np.concatenate(all_support).astype(np.int16),
        "line_structure": np.concatenate(all_line).astype(np.uint8),
        "normal": np.concatenate(all_normal).astype(np.float32),
        "source_image_index": np.concatenate(all_source).astype(np.int16),
        "source_images": np.asarray(source_names),
    }


def voxel_downsample(points: dict[str, np.ndarray], r0: float, *, maximum_points: int) -> dict[str, np.ndarray]:
    """Use v2 r0 voxels, preserving the highest calibrated-confidence sample."""
    if r0 <= 0 or not np.isfinite(r0):
        raise ValueError("r0 must be finite and positive")
    xyz, score = points["xyz"], points["calibrated_confidence"]
    size = np.where(points["line_structure"].astype(bool), r0 / 2.0, r0)
    keys = np.floor(xyz / size[:, None]).astype(np.int64)
    # Structured sorting avoids non-deterministic set/dict iteration.
    order = np.lexsort((-score, keys[:, 2], keys[:, 1], keys[:, 0]))
    ordered_keys = keys[order]
    take = np.ones(len(order), dtype=bool)
    take[1:] = np.any(ordered_keys[1:] != ordered_keys[:-1], axis=1)
    kept = order[take]
    if len(kept) > maximum_points:
        kept = kept[np.argsort(-score[kept], kind="stable")[:maximum_points]]
    return {name: (value if name == "source_images" else value[kept]) for name, value in points.items()}


def write_scaffold_init_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write the exact x/y/z/nx/ny/nz/rgb PLY layout Scaffold reads."""
    if xyz.shape != (len(rgb), 3) or rgb.shape[1:] != (3,):
        raise ValueError("XYZ and RGB must be Nx3")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    output = np.zeros(len(xyz), dtype=PLY_VERTEX_DTYPE)
    output["x"], output["y"], output["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    output["red"], output["green"], output["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(output.tobytes())


def read_binary_ply_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read XYZ/RGB from the fixed binary PLY layout Scaffold consumes."""
    with path.open("rb") as handle:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            block = handle.read(1)
            if not block or len(header) > 65536:
                raise ValueError("Invalid or oversized PLY header")
            header.extend(block)
        lines = header.decode("ascii").splitlines()
        count = next((int(line.split()[2]) for line in lines if line.startswith("element vertex ")), None)
        if count is None or "format binary_little_endian 1.0" not in lines:
            raise ValueError("Expected binary little-endian vertex PLY")
        data = np.frombuffer(
            handle.read(count * PLY_VERTEX_DTYPE.itemsize), dtype=PLY_VERTEX_DTYPE, count=count
        )
    if len(data) != count:
        raise ValueError("Truncated PLY vertex data")
    xyz = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float32)
    rgb = np.column_stack((data["red"], data["green"], data["blue"])).astype(np.uint8)
    return xyz, rgb


def read_binary_ply_xyz(path: Path) -> np.ndarray:
    """Read the fixed base initialization PLY layout needed to calculate r0."""
    return read_binary_ply_xyz_rgb(path)[0]


def median_five_nn_distance(xyz: np.ndarray) -> float:
    """Use scipy only in the isolated geometry environment for r0."""
    from scipy.spatial import cKDTree

    if len(xyz) < 6:
        raise ValueError("Need six points to define median 5-NN distance")
    distances, _ = cKDTree(xyz).query(xyz, k=6, workers=-1)
    return float(np.median(distances[:, 5]))
