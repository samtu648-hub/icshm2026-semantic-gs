"""One-event multiview tangent repair for native ImprovedGS LAS children."""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, sobel

from src.bridge_gaussian_support import project_gaussians, quantiles


def tangent_constraint(point_world: np.ndarray, camera: Any, tangent_angle: float) -> np.ndarray:
    """Return one normalized plane constraint for a projected 3D tangent.

    A valid world-space direction ``d`` satisfies ``constraint @ d == 0``
    when its image projection is parallel to the observed photo tangent.
    """
    point = np.asarray(point_world, dtype=np.float64)
    rotation = np.asarray(camera.pose.rotation, dtype=np.float64)
    translation = np.asarray(camera.pose.translation, dtype=np.float64)
    camera_point = rotation @ point + translation
    if camera_point[2] <= 1e-8:
        raise ValueError("point is behind the camera")
    matrix = np.asarray(camera.matrix, dtype=np.float64)
    fx, fy = matrix[0, 0], matrix[1, 1]
    x, y, z = camera_point
    jacobian_camera = np.array(
        [[fx / z, 0.0, -fx * x / (z * z)], [0.0, fy / z, -fy * y / (z * z)]],
        dtype=np.float64,
    )
    normal = np.array([-math.sin(tangent_angle), math.cos(tangent_angle)], dtype=np.float64)
    row = normal @ jacobian_camera @ rotation
    norm = float(np.linalg.norm(row))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("degenerate tangent constraint")
    return row / norm


def solve_tangent_constraints(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Solve the 3D tangent as the null direction of multiple view planes."""
    constraints = np.asarray(rows, dtype=np.float64)
    if constraints.ndim != 2 or constraints.shape[1] != 3 or len(constraints) < 2:
        raise ValueError("at least two 3D constraints are required")
    normal = constraints.T @ constraints
    eigenvalues, eigenvectors = np.linalg.eigh(normal)
    direction = eigenvectors[:, 0]
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    trace = max(float(eigenvalues.sum()), 1e-12)
    residual_fraction = float(eigenvalues[0] / trace)
    eigengap_ratio = float(eigenvalues[1] / max(eigenvalues[0], 1e-12))
    return direction, eigenvalues, residual_fraction, eigengap_ratio


def rotation_matrix_to_quaternion(matrix: Any) -> Any:
    """Convert proper rotation matrices to normalized WXYZ quaternions."""
    import torch

    value = matrix
    if value.ndim != 3 or value.shape[1:] != (3, 3):
        raise ValueError("matrix must have shape [N,3,3]")
    r00, r11, r22 = value[:, 0, 0], value[:, 1, 1], value[:, 2, 2]
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + r00 + r11 + r22, min=0.0))
    qx = 0.5 * torch.copysign(torch.sqrt(torch.clamp(1.0 + r00 - r11 - r22, min=0.0)), value[:, 2, 1] - value[:, 1, 2])
    qy = 0.5 * torch.copysign(torch.sqrt(torch.clamp(1.0 - r00 + r11 - r22, min=0.0)), value[:, 0, 2] - value[:, 2, 0])
    qz = 0.5 * torch.copysign(torch.sqrt(torch.clamp(1.0 - r00 - r11 + r22, min=0.0)), value[:, 1, 0] - value[:, 0, 1])
    quaternion = torch.stack((qw, qx, qy, qz), dim=1)
    return torch.nn.functional.normalize(quaternion, dim=1)


def align_rotation_to_tangent(parent_rotation: Any, max_indices: Any, tangent: Any) -> Any:
    """Align each parent's longest local axis to a world-space tangent."""
    import torch

    tangent = torch.nn.functional.normalize(tangent, dim=1)
    output = torch.empty_like(parent_rotation)
    for longest in range(3):
        chosen = max_indices == longest
        if not torch.any(chosen):
            continue
        t = tangent[chosen]
        parent = parent_rotation[chosen]
        anchor_index = (longest + 1) % 3
        anchor = parent[:, :, anchor_index]
        perpendicular = anchor - (anchor * t).sum(dim=1, keepdim=True) * t
        small = perpendicular.norm(dim=1) < 1e-6
        if torch.any(small):
            basis_index = torch.argmin(torch.abs(t[small]), dim=1)
            fallback = torch.nn.functional.one_hot(basis_index, num_classes=3).to(t.dtype)
            fallback = fallback - (fallback * t[small]).sum(dim=1, keepdim=True) * t[small]
            perpendicular[small] = fallback
        perpendicular = torch.nn.functional.normalize(perpendicular, dim=1)
        matrix = torch.empty_like(parent)
        if longest == 0:
            matrix[:, :, 0] = t
            matrix[:, :, 1] = perpendicular
            matrix[:, :, 2] = torch.cross(t, perpendicular, dim=1)
        elif longest == 1:
            matrix[:, :, 1] = t
            matrix[:, :, 0] = perpendicular
            matrix[:, :, 2] = torch.cross(perpendicular, t, dim=1)
        else:
            matrix[:, :, 2] = t
            matrix[:, :, 0] = perpendicular
            matrix[:, :, 1] = torch.cross(t, perpendicular, dim=1)
        output[chosen] = matrix
    return output


def directional_children(
    parent_xyz: Any,
    parent_scales: Any,
    parent_rotation: Any,
    tangent: Any,
    *,
    split_distance: float,
    transverse_multiplier: float,
) -> tuple[Any, Any, Any]:
    """Create tangent-aligned positions, scales, and WXYZ rotations."""
    import torch

    if not 0.0 < transverse_multiplier <= 1.0:
        raise ValueError("transverse_multiplier must be in (0,1]")
    max_values, max_indices_2d = parent_scales.max(dim=1, keepdim=True)
    max_indices = max_indices_2d[:, 0]
    aligned = align_rotation_to_tangent(parent_rotation, max_indices, tangent)
    unit = torch.nn.functional.normalize(tangent, dim=1)
    offset = unit * max_values * (3.0 * float(split_distance))
    child_xyz = torch.cat((parent_xyz + offset, parent_xyz - offset), dim=0)

    rate_w = max(1.0 - float(split_distance), 1e-6)
    rate_h = math.sqrt(max(1.0 - float(split_distance) ** 2, 1e-6))
    child_scale = parent_scales * rate_h * float(transverse_multiplier)
    child_scale = child_scale.scatter(1, max_indices_2d, max_values * rate_w)
    child_scales = child_scale.repeat(2, 1)
    child_rotation = rotation_matrix_to_quaternion(aligned).repeat(2, 1)
    return child_xyz, child_scales, child_rotation


def _photo_tangent_and_strength(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = image.astype(np.float32) / 255.0
    gray = 0.299 * value[..., 0] + 0.587 * value[..., 1] + 0.114 * value[..., 2]
    gx, gy = sobel(gray, axis=1), sobel(gray, axis=0)
    return np.arctan2(gy, gx) + math.pi / 2.0, np.hypot(gx, gy)


def build_multiview_tangent_evidence(
    *,
    model: Any,
    render: Callable[..., dict[str, Any]],
    make_native_view: Callable[[Any], Any],
    pipe: Any,
    background: Any,
    cameras: dict[str, Any],
    image_root: Path,
    label_root: Path,
    output_path: Path,
    top_count: int = 16,
    ring_pixels: int = 6,
    minimum_views: int = 3,
    minimum_density_ratio: float = 1.5,
    maximum_residual_fraction: float = 0.05,
    minimum_eigengap_ratio: float = 4.0,
    minimum_photo_gradient: float = 0.03,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Find class-consistent bridge contributors and solve their 3D tangents."""
    import torch
    from utils.general_utils import build_rotation

    class_ids = (1, 2, 3)
    names = sorted(path.name for path in label_root.glob("*.png"))
    if len(names) != 300:
        raise ValueError("bridge repair requires all 300 registered masks")
    observations: dict[tuple[int, int], list[tuple[str, np.ndarray, float]]] = defaultdict(list)
    evaluated = 0
    for view_index, name in enumerate(names):
        image = np.asarray(Image.open(image_root / name).convert("RGB"), dtype=np.uint8)
        labels = np.asarray(Image.open(label_root / name), dtype=np.uint8)
        tangent, strength = _photo_tangent_and_strength(image)
        camera = cameras[name]
        native_view = make_native_view(camera)
        for class_id in class_ids:
            mask = labels == class_id
            ring = binary_dilation(mask, iterations=ring_pixels) & ~mask
            packages = []
            for region in (mask, ring):
                package = render(
                    native_view,
                    model,
                    pipe,
                    background,
                    track_gradients=False,
                    inference_only=False,
                    pixel_weights=torch.as_tensor(region.astype(np.float32), device="cuda"),
                )
                weights = package.get("accum_weights")
                if weights is None or weights.shape[0] != model.get_xyz.shape[0]:
                    raise RuntimeError("native rasterizer contribution output changed")
                packages.append(weights.detach().float())
            structure_weights, ring_weights = packages
            chosen = torch.topk(structure_weights, min(top_count, len(structure_weights)), sorted=True)
            chosen_ids = chosen.indices[chosen.values > 0]
            if not len(chosen_ids):
                continue
            ids = chosen_ids.cpu().numpy().astype(np.int64)
            xyz = model.get_xyz[chosen_ids].detach().cpu().numpy()
            scales = model.get_scaling[chosen_ids].detach().cpu().numpy()
            rotations = build_rotation(model.get_rotation[chosen_ids]).detach().cpu().numpy()
            projected = project_gaussians(xyz, scales, rotations, camera)
            structure_density = structure_weights[chosen_ids].cpu().numpy() / max(int(mask.sum()), 1)
            ring_density = ring_weights[chosen_ids].cpu().numpy() / max(int(ring.sum()), 1)
            for local, gaussian_id in enumerate(ids):
                x, y = int(round(projected["u"][local])), int(round(projected["v"][local]))
                if not projected["valid"][local] or not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
                    continue
                if not mask[y, x] or strength[y, x] < minimum_photo_gradient:
                    continue
                ratio = float(structure_density[local] / max(float(ring_density[local]), 1e-20))
                if ratio < minimum_density_ratio:
                    continue
                try:
                    row = tangent_constraint(xyz[local], camera, float(tangent[y, x]))
                except ValueError:
                    continue
                observations[(int(gaussian_id), class_id)].append((name, row, ratio))
        evaluated += 1
        if view_index % 50 == 0:
            print(json.dumps({"bridge_repair_evidence_views": view_index + 1, "total": len(names)}), flush=True)

    point_count = int(model.get_xyz.shape[0])
    tangent_tensor = torch.zeros((point_count, 3), device=model.get_xyz.device, dtype=model.get_xyz.dtype)
    valid_tensor = torch.zeros((point_count,), device=model.get_xyz.device, dtype=torch.bool)
    class_tensor = torch.zeros((point_count,), device=model.get_xyz.device, dtype=torch.uint8)
    accepted: list[dict[str, Any]] = []
    rejected = defaultdict(int)
    by_gaussian: dict[int, list[tuple[int, list[tuple[str, np.ndarray, float]]]]] = defaultdict(list)
    for (gaussian_id, class_id), rows in observations.items():
        by_gaussian[gaussian_id].append((class_id, rows))
    for gaussian_id, class_groups in by_gaussian.items():
        class_groups.sort(key=lambda item: (-len(item[1]), item[0]))
        class_id, rows = class_groups[0]
        second_count = len(class_groups[1][1]) if len(class_groups) > 1 else 0
        if len(rows) < minimum_views:
            rejected["too_few_views"] += 1
            continue
        if second_count and len(rows) < 1.5 * second_count:
            rejected["class_conflict"] += 1
            continue
        unique_views = sorted({row[0] for row in rows})
        if len(unique_views) < minimum_views:
            rejected["too_few_unique_views"] += 1
            continue
        constraints = np.stack([row[1] for row in rows])
        direction, eigenvalues, residual, eigengap = solve_tangent_constraints(constraints)
        if residual > maximum_residual_fraction:
            rejected["high_residual"] += 1
            continue
        if eigengap < minimum_eigengap_ratio:
            rejected["weak_eigengap"] += 1
            continue
        tangent_tensor[gaussian_id] = torch.as_tensor(direction, device=tangent_tensor.device, dtype=tangent_tensor.dtype)
        valid_tensor[gaussian_id] = True
        class_tensor[gaussian_id] = class_id
        accepted.append({
            "gaussian_id": gaussian_id,
            "class_id": class_id,
            "views": unique_views,
            "observation_count": len(rows),
            "tangent": direction.tolist(),
            "eigenvalues": eigenvalues.tolist(),
            "residual_fraction": residual,
            "eigengap_ratio": eigengap,
            "density_ratio": quantiles(row[2] for row in rows),
        })
    evidence = {
        "schema_version": 1,
        "views": evaluated,
        "point_count": point_count,
        "candidate_gaussian_class_groups": len(observations),
        "accepted_gaussians": len(accepted),
        "accepted_by_class": {str(class_id): sum(row["class_id"] == class_id for row in accepted) for class_id in class_ids},
        "rejected": dict(rejected),
        "observation_count": quantiles(row["observation_count"] for row in accepted),
        "residual_fraction": quantiles(row["residual_fraction"] for row in accepted),
        "eigengap_ratio": quantiles(row["eigengap_ratio"] for row in accepted),
        "criteria": {
            "top_count": top_count,
            "ring_pixels": ring_pixels,
            "minimum_views": minimum_views,
            "minimum_density_ratio": minimum_density_ratio,
            "maximum_residual_fraction": maximum_residual_fraction,
            "minimum_eigengap_ratio": minimum_eigengap_ratio,
            "minimum_photo_gradient": minimum_photo_gradient,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"summary": evidence, "accepted": accepted}, ensure_ascii=False, allow_nan=False))
        handle.write("\n")
    return tangent_tensor, valid_tensor, class_tensor, evidence


def install(
    *,
    model_class: type,
    densification_module: Any,
    repair_iteration: int,
    evidence_builder: Callable[[Any], tuple[Any, Any, Any, dict[str, Any]]],
    transverse_multiplier: float,
) -> dict[str, Any]:
    """Install process-local evidence and LAS hooks for exactly one event."""
    import torch
    from utils.general_utils import build_rotation, inverse_sigmoid

    original_scores = densification_module._compute_improvedgs_scores
    original_split = model_class.long_axis_split
    state: dict[str, Any] = {
        "repair_iteration": int(repair_iteration),
        "transverse_multiplier": float(transverse_multiplier),
        "evidence_calls": 0,
        "split_calls": 0,
        "native_selected_parents": 0,
        "repaired_parents": 0,
        "repaired_by_class": {},
        "repaired_parent_ids": [],
    }

    def scores(context: Any, gaussians: Any, render_state: dict[str, Any], iteration: int) -> Any:
        values = original_scores(context, gaussians, render_state, iteration)
        gaussians._bridge_repair_active = False
        if int(iteration) == int(repair_iteration):
            tangents, valid, classes, evidence = evidence_builder(gaussians)
            if tangents.shape != gaussians.get_xyz.shape or valid.shape != (gaussians.get_xyz.shape[0],):
                raise ValueError("bridge tangent evidence shape mismatch")
            gaussians._bridge_repair_tangents = tangents
            gaussians._bridge_repair_valid = valid
            gaussians._bridge_repair_classes = classes
            gaussians._bridge_repair_active = True
            state["evidence_calls"] += 1
            state["evidence"] = evidence
        return values

    def split(model: Any, scores_value: Any, budget: int, filter_mask: Any, split_distance: float, opacity_reduction: float) -> int:
        if not bool(getattr(model, "_bridge_repair_active", False)):
            return original_split(model, scores_value, budget, filter_mask, split_distance, opacity_reduction)
        if budget <= 0 or scores_value.numel() == 0 or not torch.any(filter_mask):
            return 0
        padded = torch.zeros((model.get_xyz.shape[0]), dtype=torch.float32, device="cuda")
        padded[: scores_value.shape[0]] = scores_value.detach().float().clamp_min(0)
        padded[~filter_mask] = 0
        positive_count = int((padded > 0).sum().item())
        if positive_count == 0:
            return 0
        budget = min(int(budget), positive_count)
        selected_indices = torch.multinomial(padded, budget, replacement=False)
        selected_mask = torch.zeros_like(padded, dtype=torch.bool)
        selected_mask[selected_indices] = True
        selected_ids = torch.nonzero(selected_mask, as_tuple=False).flatten()

        parent_scales = model.get_scaling[selected_mask]
        max_values, max_indices = parent_scales.max(dim=1, keepdim=True)
        axis_mask = torch.zeros_like(parent_scales, dtype=torch.bool).scatter(1, max_indices, True)
        axis_offsets = parent_scales * axis_mask * 3.0 * float(split_distance)
        axis_offsets = torch.cat((axis_offsets, -axis_offsets), dim=0)
        rotation_mats = build_rotation(model._rotation[selected_mask]).repeat(2, 1, 1)
        parent_xyz = model.get_xyz[selected_mask].repeat(2, 1)
        new_xyz = torch.bmm(rotation_mats, axis_offsets.unsqueeze(-1)).squeeze(-1) + parent_xyz
        rate_w = max(1.0 - float(split_distance), 1e-6)
        rate_h = math.sqrt(max(1.0 - float(split_distance) ** 2, 1e-6))
        new_scales = parent_scales.scatter(1, max_indices, max_values * rate_w / rate_h).repeat(2, 1) * rate_h
        new_rotation = model._rotation[selected_mask].repeat(2, 1)

        repair_parent_mask = model._bridge_repair_valid[selected_ids]
        repair_local = torch.nonzero(repair_parent_mask, as_tuple=False).flatten()
        if len(repair_local):
            parent_rotation = build_rotation(model._rotation[selected_ids[repair_local]])
            child_xyz, child_scales, child_rotation = directional_children(
                model.get_xyz[selected_ids[repair_local]],
                parent_scales[repair_local],
                parent_rotation,
                model._bridge_repair_tangents[selected_ids[repair_local]],
                split_distance=float(split_distance),
                transverse_multiplier=float(transverse_multiplier),
            )
            child_indices = torch.cat((repair_local, repair_local + len(selected_ids)))
            new_xyz[child_indices] = child_xyz
            new_scales[child_indices] = child_scales
            new_rotation[child_indices] = child_rotation

        new_scaling = model.scaling_inverse_activation(new_scales)
        new_opacity = inverse_sigmoid(model.get_opacity[selected_mask] * float(opacity_reduction)).repeat(2, 1)
        new_features_dc = model._features_dc[selected_mask].repeat(2, 1, 1)
        new_features_rest = model._features_rest[selected_mask].repeat(2, 1, 1)
        model.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat((selected_mask, torch.zeros(2 * len(selected_ids), device="cuda", dtype=torch.bool)))
        model.prune_points(prune_filter)

        repaired_ids = selected_ids[repair_local]
        repaired_classes = model._bridge_repair_classes[selected_ids][repair_local].detach().cpu().tolist()
        counts = {str(class_id): repaired_classes.count(class_id) for class_id in sorted(set(repaired_classes))}
        state["split_calls"] += 1
        state["native_selected_parents"] += len(selected_ids)
        state["repaired_parents"] += len(repaired_ids)
        state["repaired_by_class"] = counts
        state["repaired_parent_ids"] = repaired_ids.detach().cpu().tolist()
        model._bridge_repair_active = False
        return budget

    densification_module._compute_improvedgs_scores = scores
    model_class.long_axis_split = split
    state["original_scores"] = original_scores
    state["original_split"] = original_split
    return state

