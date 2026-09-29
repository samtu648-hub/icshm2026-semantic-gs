"""Responsibility-preserving asymmetric bridge clone for ImprovedGS."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.improvedgs_bridge_repair import align_rotation_to_tangent, rotation_matrix_to_quaternion
from src.improvedgs_evidence_directed_split import (
    build_mixed_geometry_evidence,
    compose_evidence_directed_selection,
)


def alpha_conserving_partition(opacity: Any, bridge_fraction: Any) -> tuple[Any, Any]:
    """Partition alpha so colocated context and bridge children reproduce it.

    If ``a_c = 1-(1-a)^(1-r)`` and ``a_b = 1-(1-a)^r``, then
    ``1-(1-a_c)(1-a_b) == a``.  The function supports NumPy arrays and
    Torch tensors without importing Torch in the lightweight test runtime.
    """
    if hasattr(opacity, "clamp"):
        base = (1.0 - opacity).clamp(min=1e-8, max=1.0)
        fraction = bridge_fraction.clamp(min=1e-4, max=1.0 - 1e-4)
        bridge = 1.0 - base.pow(fraction)
        context = 1.0 - base.pow(1.0 - fraction)
        return context, bridge
    opacity_array = np.asarray(opacity, dtype=np.float64)
    fraction_array = np.clip(np.asarray(bridge_fraction, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    base = np.clip(1.0 - opacity_array, 1e-8, 1.0)
    bridge = 1.0 - np.power(base, fraction_array)
    context = 1.0 - np.power(base, 1.0 - fraction_array)
    return context, bridge


def build_responsibility_clone_evidence(
    *, output_path: Path, **kwargs: Any
) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
    """Reuse RGB172 evidence and add a measured bridge-responsibility tensor."""
    tangents, valid, classes, priority, evidence = build_mixed_geometry_evidence(
        output_path=output_path, **kwargs
    )
    import torch

    bridge_fraction = torch.zeros_like(priority, dtype=torch.float32)
    with gzip.open(output_path, "rt", encoding="utf-8") as handle:
        payload = json.loads(handle.readline())
    accepted = payload["accepted"]
    values: list[float] = []
    for row in accepted:
        background = float(row["near_background_fraction"]["p50"])
        fraction = min(max(1.0 - background, 1e-4), 1.0 - 1e-4)
        bridge_fraction[int(row["gaussian_id"])] = fraction
        row["bridge_responsibility_fraction"] = fraction
        values.append(fraction)
    evidence["bridge_responsibility_fraction"] = {
        key: float(value)
        for key, value in zip(
            ("min", "p10", "p25", "p50", "p75", "p90", "max"),
            np.quantile(np.asarray(values, dtype=np.float64), (0, .1, .25, .5, .75, .9, 1)),
        )
    } if values else {}
    evidence["opacity_partition"] = "analytic front-to-back alpha conservation at the shared center"
    # Rewrite the evidence record with the responsibility value used by the
    # operator so the executed allocation is independently auditable.
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"summary": evidence, "accepted": accepted}, ensure_ascii=False, allow_nan=False) + "\n")
    return tangents, valid, classes, priority, bridge_fraction, evidence


def responsibility_child_geometry(
    parent_xyz: Any,
    parent_scales: Any,
    parent_rotation_matrix: Any,
    parent_rotation_raw: Any,
    tangent: Any,
    *,
    transverse_multiplier: float,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Return exact context children and centered, tangent-aligned bridge children."""
    import torch

    if not 0.0 < transverse_multiplier <= 1.0:
        raise ValueError("transverse_multiplier must lie in (0,1]")
    max_values, max_indices_2d = parent_scales.max(dim=1, keepdim=True)
    max_indices = max_indices_2d[:, 0]
    aligned = align_rotation_to_tangent(parent_rotation_matrix, max_indices, tangent)
    bridge_scales = parent_scales * float(transverse_multiplier)
    bridge_scales = bridge_scales.scatter(1, max_indices_2d, max_values)
    bridge_rotation = rotation_matrix_to_quaternion(aligned)
    # Both children start at the observed support center. The context child is
    # an exact geometric continuation of the parent; only alpha is partitioned.
    return (
        parent_xyz.clone(),
        parent_scales.clone(),
        parent_rotation_raw.clone(),
        parent_xyz.clone(),
        bridge_scales,
        bridge_rotation,
    )


def _capture_context_optimizer_state(model: Any, parent_ids: Any) -> dict[str, dict[str, Any]]:
    """Capture per-Gaussian Adam state for the context continuation child."""
    captured: dict[str, dict[str, Any]] = {}
    point_count = int(model.get_xyz.shape[0])
    for group in model.optimizer.param_groups:
        parameter = group["params"][0]
        stored = model.optimizer.state.get(parameter)
        if not stored:
            continue
        values: dict[str, Any] = {}
        for key in ("exp_avg", "exp_avg_sq", "last_seen_step"):
            value = stored.get(key)
            if hasattr(value, "shape") and value.ndim >= 1 and value.shape[0] == point_count:
                values[key] = value[parent_ids].clone()
        if values:
            captured[str(group["name"])] = values
    return captured


def _restore_context_optimizer_state(
    model: Any,
    captured: dict[str, dict[str, Any]],
    appended_context_ids: Any,
) -> int:
    restored = 0
    for group in model.optimizer.param_groups:
        values = captured.get(str(group["name"]))
        stored = model.optimizer.state.get(group["params"][0])
        if not values or not stored:
            continue
        for key, value in values.items():
            stored[key][appended_context_ids] = value
            restored += int(value.shape[0])
    return restored


def install(
    *,
    model_class: type,
    densification_module: Any,
    repair_iteration: int,
    evidence_builder: Callable[[Any], tuple[Any, Any, Any, Any, Any, dict[str, Any]]],
    transverse_multiplier: float,
) -> dict[str, Any]:
    """Install one fixed-budget asymmetric responsibility-clone event."""
    import torch
    from utils.general_utils import build_rotation, inverse_sigmoid

    original_scores = densification_module._compute_improvedgs_scores
    original_split = model_class.long_axis_split
    state: dict[str, Any] = {
        "repair_iteration": int(repair_iteration),
        "transverse_multiplier": float(transverse_multiplier),
        "selection_mode": "responsibility_asymmetric_fixed_budget",
        "evidence_calls": 0,
        "split_calls": 0,
        "native_selected_parents": 0,
        "repaired_parents": 0,
        "displaced_native_parents": 0,
        "native_evidence_overlap": 0,
        "context_optimizer_state_values_restored": 0,
        "repaired_by_class": {},
        "repaired_parent_ids": [],
    }

    def scores(context: Any, gaussians: Any, render_state: dict[str, Any], iteration: int) -> Any:
        values = original_scores(context, gaussians, render_state, iteration)
        gaussians._bridge_repair_active = False
        if int(iteration) == int(repair_iteration):
            tangents, valid, classes, priority, fraction, evidence = evidence_builder(gaussians)
            if tangents.shape != gaussians.get_xyz.shape or valid.shape != (gaussians.get_xyz.shape[0],):
                raise ValueError("bridge responsibility evidence shape mismatch")
            gaussians._bridge_repair_tangents = tangents
            gaussians._bridge_repair_valid = valid
            gaussians._bridge_repair_classes = classes
            gaussians._bridge_repair_priority = priority
            gaussians._bridge_repair_fraction = fraction
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
        old_point_count = int(model.get_xyz.shape[0])
        padded = torch.zeros((old_point_count,), dtype=torch.float32, device="cuda")
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
        selected_count = len(selected_ids_sorted)

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
        new_opacity_activated = model.get_opacity[selected_mask].repeat(2, 1) * float(opacity_reduction)

        repair_mask = model._bridge_repair_valid[selected_ids_sorted]
        repair_local = torch.nonzero(repair_mask, as_tuple=False).flatten()
        repaired_ids = selected_ids_sorted[repair_local]
        captured = _capture_context_optimizer_state(model, repaired_ids)
        if len(repair_local):
            geometry = responsibility_child_geometry(
                model.get_xyz[repaired_ids],
                parent_scales[repair_local],
                build_rotation(model._rotation[repaired_ids]),
                model._rotation[repaired_ids],
                model._bridge_repair_tangents[repaired_ids],
                transverse_multiplier=float(transverse_multiplier),
            )
            context_xyz, context_scales, context_rotation, bridge_xyz, bridge_scales, bridge_rotation = geometry
            bridge_slots = repair_local + selected_count
            new_xyz[repair_local], new_xyz[bridge_slots] = context_xyz, bridge_xyz
            new_scales[repair_local], new_scales[bridge_slots] = context_scales, bridge_scales
            new_rotation[repair_local], new_rotation[bridge_slots] = context_rotation, bridge_rotation
            fraction = model._bridge_repair_fraction[repaired_ids].unsqueeze(1)
            context_opacity, bridge_opacity = alpha_conserving_partition(model.get_opacity[repaired_ids], fraction)
            new_opacity_activated[repair_local] = context_opacity
            new_opacity_activated[bridge_slots] = bridge_opacity

        new_scaling = model.scaling_inverse_activation(new_scales)
        new_opacity = inverse_sigmoid(new_opacity_activated.clamp(1e-8, 1.0 - 1e-8))
        new_features_dc = model._features_dc[selected_mask].repeat(2, 1, 1)
        new_features_rest = model._features_rest[selected_mask].repeat(2, 1, 1)
        model.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        appended_context = old_point_count + repair_local
        state["context_optimizer_state_values_restored"] = _restore_context_optimizer_state(
            model, captured, appended_context
        )
        prune_filter = torch.cat((selected_mask, torch.zeros(2 * selected_count, device="cuda", dtype=torch.bool)))
        model.prune_points(prune_filter)

        forced_set = set(forced_ids.detach().cpu().tolist())
        native_set = set(native_selected.detach().cpu().tolist())
        repaired_classes = model._bridge_repair_classes[selected_ids_sorted][repair_local].detach().cpu().tolist()
        responsibility = model._bridge_repair_fraction[repaired_ids].detach().cpu().numpy()
        state["split_calls"] += 1
        state["native_selected_parents"] = len(native_selected)
        state["repaired_parents"] = len(repaired_ids)
        state["displaced_native_parents"] = len(forced_set - native_set)
        state["native_evidence_overlap"] = len(forced_set & native_set)
        state["retained_native_parents"] = len(kept_native)
        state["repaired_by_class"] = {
            str(class_id): repaired_classes.count(class_id) for class_id in sorted(set(repaired_classes))
        }
        state["bridge_responsibility_fraction"] = {
            "min": float(responsibility.min()), "p50": float(np.median(responsibility)),
            "max": float(responsibility.max()),
        } if len(responsibility) else {}
        state["repaired_parent_ids"] = repaired_ids.detach().cpu().tolist()
        model._bridge_repair_active = False
        return budget

    densification_module._compute_improvedgs_scores = scores
    model_class.long_axis_split = split
    state["original_scores"] = original_scores
    state["original_split"] = original_split
    return state

