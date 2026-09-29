"""Evidence-directed ImprovedGS split, independent of the native LAS gate."""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from src.bridge_gaussian_support import project_gaussians, quantiles, undirected_angle_error
from src.improvedgs_bridge_repair import (
    _photo_tangent_and_strength,
    directional_children,
    solve_tangent_constraints,
    tangent_constraint,
)


def build_mixed_geometry_evidence(
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
    center_tolerance_pixels: int = 2,
    minimum_views: int = 3,
    minimum_mixed_fraction: float = 0.5,
    minimum_background_fraction: float = 0.35,
    maximum_residual_fraction: float = 0.05,
    minimum_eigengap_ratio: float = 4.0,
    minimum_photo_gradient: float = 0.03,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Select cross-view bridge contributors with mixed structure/ring duty.

    Semantic masks localize the structure and its immediate ring.  The solved
    3D tangent still comes only from real-photo gradients and official camera
    Jacobians.  A point is eligible only when it is a leading structure
    contributor and repeatedly also contributes to the ring.
    """
    import torch
    from utils.general_utils import build_rotation

    class_ids = (1, 2, 3)
    names = sorted(path.name for path in label_root.glob("*.png"))
    if len(names) != 300:
        raise ValueError("evidence-directed repair requires all 300 registered masks")
    observations: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for view_index, name in enumerate(names):
        image = np.asarray(Image.open(image_root / name).convert("RGB"), dtype=np.uint8)
        labels = np.asarray(Image.open(label_root / name), dtype=np.uint8)
        tangent, strength = _photo_tangent_and_strength(image)
        camera = cameras[name]
        native_view = make_native_view(camera)
        for class_id in class_ids:
            mask = labels == class_id
            ring = binary_dilation(mask, iterations=ring_pixels) & ~mask
            near_structure = binary_dilation(mask, iterations=center_tolerance_pixels)
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
                if not near_structure[y, x] or strength[y, x] < minimum_photo_gradient:
                    continue
                structure_value = float(structure_density[local])
                ring_value = float(ring_density[local])
                fraction = ring_value / max(structure_value + ring_value, 1e-20)
                try:
                    constraint = tangent_constraint(xyz[local], camera, float(tangent[y, x]))
                except ValueError:
                    continue
                orientation = float(np.degrees(undirected_angle_error(
                    np.asarray([projected["angle_rad"][local]]), np.asarray([tangent[y, x]])
                )[0]))
                observations[(int(gaussian_id), class_id)].append({
                    "view": name,
                    "constraint": constraint,
                    "near_background_fraction": fraction,
                    "orientation_error_deg": orientation,
                    "major_radius_px": float(projected["major_radius"][local]),
                    "minor_radius_px": float(projected["minor_radius"][local]),
                    "structure_density": structure_value,
                    "ring_density": ring_value,
                })
        if view_index % 50 == 0:
            print(json.dumps({"directed_evidence_views": view_index + 1, "groups": len(observations)}), flush=True)

    point_count = int(model.get_xyz.shape[0])
    device, dtype = model.get_xyz.device, model.get_xyz.dtype
    tangent_tensor = torch.zeros((point_count, 3), device=device, dtype=dtype)
    valid_tensor = torch.zeros((point_count,), device=device, dtype=torch.bool)
    class_tensor = torch.zeros((point_count,), device=device, dtype=torch.uint8)
    priority_tensor = torch.zeros((point_count,), device=device, dtype=torch.float32)
    accepted: list[dict[str, Any]] = []
    rejected: dict[str, int] = defaultdict(int)
    by_gaussian: dict[int, list[tuple[int, list[dict[str, Any]]]]] = defaultdict(list)
    for (gaussian_id, class_id), rows in observations.items():
        by_gaussian[gaussian_id].append((class_id, rows))
    for gaussian_id, groups in sorted(by_gaussian.items()):
        groups.sort(key=lambda item: (-len(item[1]), item[0]))
        class_id, rows = groups[0]
        second_count = len(groups[1][1]) if len(groups) > 1 else 0
        unique_views = sorted({row["view"] for row in rows})
        if len(unique_views) < minimum_views:
            rejected["too_few_views"] += 1
            continue
        if second_count and len(rows) < 1.5 * second_count:
            rejected["class_conflict"] += 1
            continue
        mixed_rows = [row for row in rows if row["near_background_fraction"] >= minimum_background_fraction]
        mixed_fraction = len(mixed_rows) / max(len(rows), 1)
        if mixed_fraction < minimum_mixed_fraction:
            rejected["not_repeatedly_mixed"] += 1
            continue
        constraints = np.stack([row["constraint"] for row in rows])
        direction, eigenvalues, residual, eigengap = solve_tangent_constraints(constraints)
        if residual > maximum_residual_fraction:
            rejected["high_residual"] += 1
            continue
        if eigengap < minimum_eigengap_ratio:
            rejected["weak_eigengap"] += 1
            continue
        background_p50 = float(np.median([row["near_background_fraction"] for row in rows]))
        orientation_p50 = float(np.median([row["orientation_error_deg"] for row in rows]))
        priority = len(rows) * background_p50 * (1.0 + orientation_p50 / 90.0)
        tangent_tensor[gaussian_id] = torch.as_tensor(direction, device=device, dtype=dtype)
        valid_tensor[gaussian_id] = True
        class_tensor[gaussian_id] = class_id
        priority_tensor[gaussian_id] = priority
        accepted.append({
            "gaussian_id": gaussian_id,
            "class_id": class_id,
            "defect": "existing_mixed_or_wrong_geometry",
            "views": unique_views,
            "observation_count": len(rows),
            "mixed_observation_fraction": mixed_fraction,
            "near_background_fraction": quantiles(row["near_background_fraction"] for row in rows),
            "orientation_error_deg": quantiles(row["orientation_error_deg"] for row in rows),
            "major_radius_px": quantiles(row["major_radius_px"] for row in rows),
            "minor_radius_px": quantiles(row["minor_radius_px"] for row in rows),
            "tangent": direction.tolist(),
            "eigenvalues": eigenvalues.tolist(),
            "residual_fraction": residual,
            "eigengap_ratio": eigengap,
            "priority": priority,
        })
    evidence = {
        "schema_version": 1,
        "views": len(names),
        "point_count": point_count,
        "candidate_gaussian_class_groups": len(observations),
        "accepted_gaussians": len(accepted),
        "accepted_by_class": {str(class_id): sum(row["class_id"] == class_id for row in accepted)
                              for class_id in class_ids},
        "rejected": dict(sorted(rejected.items())),
        "observation_count": quantiles(row["observation_count"] for row in accepted),
        "mixed_observation_fraction": quantiles(row["mixed_observation_fraction"] for row in accepted),
        "priority": quantiles(row["priority"] for row in accepted),
        "criteria": {
            "top_count": top_count,
            "ring_pixels": ring_pixels,
            "center_tolerance_pixels": center_tolerance_pixels,
            "minimum_views": minimum_views,
            "minimum_mixed_fraction": minimum_mixed_fraction,
            "minimum_background_fraction": minimum_background_fraction,
            "maximum_residual_fraction": maximum_residual_fraction,
            "minimum_eigengap_ratio": minimum_eigengap_ratio,
            "minimum_photo_gradient": minimum_photo_gradient,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"summary": evidence, "accepted": accepted}, ensure_ascii=False, allow_nan=False) + "\n")
    return tangent_tensor, valid_tensor, class_tensor, priority_tensor, evidence


def compose_evidence_directed_selection(
    native_selected: Any,
    evidence_valid: Any,
    evidence_priority: Any,
    budget: int,
) -> tuple[Any, Any, Any]:
    """Replace native slots with evidence targets while preserving point budget."""
    import torch

    forced = torch.nonzero(evidence_valid, as_tuple=False).flatten()
    if len(forced) > budget:
        order = torch.topk(evidence_priority[forced], int(budget), sorted=True).indices
        forced = forced[order]
    forced_mask = torch.zeros_like(evidence_valid, dtype=torch.bool)
    forced_mask[forced] = True
    native_remaining = native_selected[~forced_mask[native_selected]]
    keep_count = max(int(budget) - len(forced), 0)
    kept_native = native_remaining[:keep_count]
    final = torch.cat((forced, kept_native))
    if len(final) != int(budget) or len(torch.unique(final)) != len(final):
        raise RuntimeError("evidence-directed selection failed to preserve unique split budget")
    return final, forced, kept_native


def install(
    *,
    model_class: type,
    densification_module: Any,
    repair_iteration: int,
    evidence_builder: Callable[[Any], tuple[Any, Any, Any, Any, dict[str, Any]]],
    transverse_multiplier: float,
) -> dict[str, Any]:
    """Install one direct evidence-triggered, fixed-budget split event."""
    import torch
    from utils.general_utils import build_rotation, inverse_sigmoid

    original_scores = densification_module._compute_improvedgs_scores
    original_split = model_class.long_axis_split
    state: dict[str, Any] = {
        "repair_iteration": int(repair_iteration),
        "transverse_multiplier": float(transverse_multiplier),
        "selection_mode": "evidence_directed_fixed_budget",
        "evidence_calls": 0,
        "split_calls": 0,
        "native_selected_parents": 0,
        "repaired_parents": 0,
        "displaced_native_parents": 0,
        "native_evidence_overlap": 0,
        "repaired_by_class": {},
        "repaired_parent_ids": [],
    }

    def scores(context: Any, gaussians: Any, render_state: dict[str, Any], iteration: int) -> Any:
        values = original_scores(context, gaussians, render_state, iteration)
        gaussians._bridge_repair_active = False
        if int(iteration) == int(repair_iteration):
            tangents, valid, classes, priority, evidence = evidence_builder(gaussians)
            if tangents.shape != gaussians.get_xyz.shape or valid.shape != (gaussians.get_xyz.shape[0],):
                raise ValueError("bridge evidence shape mismatch")
            gaussians._bridge_repair_tangents = tangents
            gaussians._bridge_repair_valid = valid
            gaussians._bridge_repair_classes = classes
            gaussians._bridge_repair_priority = priority
            gaussians._bridge_repair_active = True
            state["evidence_calls"] += 1
            state["evidence"] = evidence
        return values

    def split(model: Any, scores_value: Any, budget: int, filter_mask: Any,
              split_distance: float, opacity_reduction: float) -> int:
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
        native_selected = torch.multinomial(padded, budget, replacement=False)
        selected_ids, forced_ids, kept_native = compose_evidence_directed_selection(
            native_selected, model._bridge_repair_valid, model._bridge_repair_priority, budget,
        )
        selected_mask = torch.zeros_like(padded, dtype=torch.bool)
        selected_mask[selected_ids] = True
        selected_ids_sorted = torch.nonzero(selected_mask, as_tuple=False).flatten()

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

        repair_mask = model._bridge_repair_valid[selected_ids_sorted]
        repair_local = torch.nonzero(repair_mask, as_tuple=False).flatten()
        if len(repair_local):
            parent_rotation = build_rotation(model._rotation[selected_ids_sorted[repair_local]])
            child_xyz, child_scales, child_rotation = directional_children(
                model.get_xyz[selected_ids_sorted[repair_local]],
                parent_scales[repair_local],
                parent_rotation,
                model._bridge_repair_tangents[selected_ids_sorted[repair_local]],
                split_distance=float(split_distance),
                transverse_multiplier=float(transverse_multiplier),
            )
            child_indices = torch.cat((repair_local, repair_local + len(selected_ids_sorted)))
            new_xyz[child_indices] = child_xyz
            new_scales[child_indices] = child_scales
            new_rotation[child_indices] = child_rotation

        new_scaling = model.scaling_inverse_activation(new_scales)
        new_opacity = inverse_sigmoid(model.get_opacity[selected_mask] * float(opacity_reduction)).repeat(2, 1)
        new_features_dc = model._features_dc[selected_mask].repeat(2, 1, 1)
        new_features_rest = model._features_rest[selected_mask].repeat(2, 1, 1)
        model.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat((selected_mask, torch.zeros(2 * len(selected_ids_sorted), device="cuda", dtype=torch.bool)))
        model.prune_points(prune_filter)

        forced_set = set(forced_ids.detach().cpu().tolist())
        native_set = set(native_selected.detach().cpu().tolist())
        repaired_ids = selected_ids_sorted[repair_local]
        repaired_classes = model._bridge_repair_classes[selected_ids_sorted][repair_local].detach().cpu().tolist()
        state["split_calls"] += 1
        state["native_selected_parents"] = len(native_selected)
        state["repaired_parents"] = len(repaired_ids)
        state["displaced_native_parents"] = len(forced_set - native_set)
        state["native_evidence_overlap"] = len(forced_set & native_set)
        state["retained_native_parents"] = len(kept_native)
        state["repaired_by_class"] = {
            str(class_id): repaired_classes.count(class_id) for class_id in sorted(set(repaired_classes))
        }
        state["repaired_parent_ids"] = repaired_ids.detach().cpu().tolist()
        model._bridge_repair_active = False
        return budget

    densification_module._compute_improvedgs_scores = scores
    model_class.long_axis_split = split
    state["original_scores"] = original_scores
    state["original_split"] = original_split
    return state

