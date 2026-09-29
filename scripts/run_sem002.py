#!/usr/bin/env python3
"""Run SEM-002 contribution-aware visibility supervision on frozen Scaffold RGB.

The implementation intentionally reuses SEM-001's split, labels, optimizer,
camera convention and RGB isolation checks.  The only methodological switch is
the stop-gradient contributor gate in ``src.scaffold_visibility_supervision``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem001 as s0  # noqa: E402
from src.scaffold_semantic_sidecar import (  # noqa: E402
    AnchorSemanticSidecar,
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    frozen_rgb_named_parameters,
    load_frozen_scaffold,
    pipeline,
    render_frozen_rgb,
    render_semantic_probabilities,
    sha256,
)
from src.scaffold_visibility_supervision import (  # noqa: E402
    GateName,
    load_visibility_extension,
    per_pixel_retention_summary,
    render_visibility_semantic_probabilities,
)


CLASS_NAMES = s0.CLASS_NAMES
CLASS_COUNT = len(CLASS_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sem001-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Git-ignored experiment output root")
    parser.add_argument("--extension-root", type=Path, required=True, help="Git-ignored torch extension cache")
    parser.add_argument("--mode", choices=("equivalence", "diagnostic", "preflight", "full"), required=True)
    parser.add_argument("--iterations", type=int, help="Only accepted when it equals the pre-registered mode budget")
    parser.add_argument("--diagnostic-start", type=int, default=0, help="Zero-based training-view offset for a bounded diagnostic chunk")
    parser.add_argument("--diagnostic-count", type=int, help="Maximum training views in a bounded diagnostic chunk")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("class_names") != list(CLASS_NAMES) or payload.get("semantic_storage") != "per_anchor_logits":
        raise ValueError("Semantic checkpoint is not a five-class per-anchor-logit sidecar")
    return payload


def load_sem001_sidecar(path: Path, anchor_count: int, rgb_hashes: dict[str, str]) -> AnchorSemanticSidecar:
    payload = _checkpoint_payload(path)
    if payload.get("experiment_id") != "SEM-001":
        raise ValueError("No-gate equivalence input must be the frozen SEM-001 checkpoint")
    if payload.get("frozen_rgb_hashes") != rgb_hashes:
        raise ValueError("SEM-001 checkpoint frozen-RGB hashes do not match the loaded RGB artifact")
    sidecar = AnchorSemanticSidecar(anchor_count)
    sidecar.load_state_dict(payload)
    return sidecar


def save_sem002_checkpoint(path: Path, sidecar: AnchorSemanticSidecar, config: dict[str, Any], rgb_hashes: dict[str, str]) -> None:
    import torch

    payload = {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "parent_experiment": "SEM-001",
        "class_names": list(CLASS_NAMES),
        "semantic_storage": "per_anchor_logits",
        "visibility_gate": config["visibility_gate"],
        "frozen_rgb_hashes": rgb_hashes,
        "semantic_anchor_logits": sidecar.logits.detach().cpu(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def reload_sem002_sidecar(path: Path, anchor_count: int, config: dict[str, Any], rgb_hashes: dict[str, str]) -> AnchorSemanticSidecar:
    payload = _checkpoint_payload(path)
    if payload.get("experiment_id") != config["experiment_id"]:
        raise ValueError("SEM-002 checkpoint experiment ID mismatch")
    if payload.get("visibility_gate") != config["visibility_gate"]:
        raise ValueError("SEM-002 checkpoint visibility gate mismatch")
    if payload.get("frozen_rgb_hashes") != rgb_hashes:
        raise ValueError("SEM-002 checkpoint frozen RGB hash mismatch")
    sidecar = AnchorSemanticSidecar(anchor_count)
    sidecar.load_state_dict(payload)
    return sidecar


def gate_from_config(config: dict[str, Any]) -> tuple[GateName, float]:
    gate = config["visibility_gate"]
    name = str(gate["name"])
    if name not in {"cumulative", "relative_max"}:
        raise ValueError("SEM-002 full/preflight config must name a visibility gate")
    threshold = float(gate["threshold"])
    if not 0.0 < threshold <= 1.0:
        raise ValueError("Visibility threshold must be in (0, 1]")
    return name, threshold  # type: ignore[return-value]


def render_semantics(
    view: Any,
    model: Any,
    sidecar: AnchorSemanticSidecar,
    pipe: Any,
    extension: Any,
    gate: GateName | None,
    threshold: float | None,
) -> tuple[Any, dict[str, Any] | None]:
    if gate is None:
        return render_semantic_probabilities(view, model, sidecar, pipe), None
    return render_visibility_semantic_probabilities(
        view, model, sidecar, pipe, gate=gate, threshold=float(threshold), extension=extension
    )


def _rgb_array(rendered: Any) -> np.ndarray:
    return s0.tensor_to_rgb(rendered)


def _semantic_external_output(
    output_dir: Path,
    data_root: Path,
    image_name: str,
    appearance_camera_id: int,
    model: Any,
    sidecar: AnchorSemanticSidecar,
    pipe: Any,
    extension: Any,
    gate: GateName | None,
    threshold: float | None,
) -> dict[str, Any]:
    selected, camera, view = s0.official_view(data_root, image_name, appearance_camera_id)
    rgb = _rgb_array(render_frozen_rgb(view, model, pipe))
    with __import__("torch").no_grad():
        pinhole_probabilities, _stats = render_semantics(view, model, sidecar, pipe, extension, gate, threshold)
    official_rgb = s0.redistort_rgb(rgb.astype(np.float32) / 255.0, camera)
    official_probabilities = s0.redistort_semantic_probabilities(pinhole_probabilities.detach().cpu().numpy().astype(np.float32), camera)
    logits = np.log(np.clip(official_probabilities, np.finfo(np.float32).tiny, 1.0))
    labels = logits.argmax(axis=0).astype(np.uint8)
    if labels.shape != (camera.height, camera.width) or int(labels.max()) >= CLASS_COUNT:
        raise RuntimeError("External semantic output is not a valid five-class image")
    root = output_dir / "external_camera"
    root.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(official_rgb.clip(0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(root / "rgb.png")
    Image.fromarray(labels, mode="L").save(root / "semantic.png")
    np.save(root / "semantic_logits.npy", logits)
    return {
        "selected_pose_name": selected,
        "input_pose_convention": "CAMERA_FROM_WORLD",
        "official_to_model_sim3": {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]},
        "input_camera_model": "SIMPLE_RADIAL",
        "output_grid": "official_SIMPLE_RADIAL",
        "rgb_shape": list(official_rgb.shape),
        "semantic_logits_shape": list(logits.shape),
        "semantic_png_shape": list(labels.shape),
        "semantic_png_dtype": str(labels.dtype),
        "semantic_png_min": int(labels.min()),
        "semantic_png_max": int(labels.max()),
        "uses_test_rgb_or_labels": False,
    }


def _retention_accumulator() -> dict[str, float]:
    return {
        "foreground_mass": 0.0,
        "retained_mass": 0.0,
        "contributors": 0.0,
        "retained_contributors": 0.0,
        "pixels_with_foreground": 0.0,
        "max_contribution_sum": 0.0,
    }


def update_retention(accumulator: dict[str, float], stats: dict[str, Any]) -> None:
    import torch

    foreground = stats["foreground_mass"].detach()
    retained = stats["retained_contribution_mass"].detach()
    contributors = stats["contributor_count"].detach().to(torch.float32)
    retained_contributors = stats["retained_contributor_count"].detach().to(torch.float32)
    valid = foreground > 0.0
    if not bool(valid.any()):
        return
    accumulator["foreground_mass"] += float(foreground[valid].sum().item())
    accumulator["retained_mass"] += float(retained[valid].sum().item())
    accumulator["contributors"] += float(contributors[valid].sum().item())
    accumulator["retained_contributors"] += float(retained_contributors[valid].sum().item())
    accumulator["pixels_with_foreground"] += float(valid.sum().item())
    accumulator["max_contribution_sum"] += float(stats["max_contribution"].detach()[valid].sum().item())


def finalize_retention(accumulator: dict[str, float]) -> dict[str, float]:
    return {
        "pixels_with_foreground": accumulator["pixels_with_foreground"],
        "retained_contribution_mass_fraction": accumulator["retained_mass"] / max(accumulator["foreground_mass"], np.finfo(np.float64).tiny),
        "retained_contributor_fraction": accumulator["retained_contributors"] / max(accumulator["contributors"], 1.0),
        "max_contribution_mean": accumulator["max_contribution_sum"] / max(accumulator["pixels_with_foreground"], 1.0),
    }


def evaluate(
    output_dir: Path,
    names: list[str],
    labels_root: Path,
    data_root: Path,
    appearance_camera_id: int,
    model: Any,
    sidecar: AnchorSemanticSidecar,
    pipe: Any,
    extension: Any,
    gate: GateName,
    threshold: float,
) -> dict[str, Any]:
    import torch

    combined = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    per_view: list[dict[str, Any]] = []
    retention = _retention_accumulator()
    removed_mass_in_bg_to_stay = 0.0
    removed_mass_total = 0.0
    render_root = output_dir / "validation_semantic_png"
    render_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = s0.read_label(labels_root / name)
        _, _camera, view = s0.official_view(data_root, name, appearance_camera_id)
        with torch.no_grad():
            probabilities, stats = render_semantics(view, model, sidecar, pipe, extension, gate, threshold)
        if stats is None:
            raise AssertionError("A visibility evaluation requires visibility statistics")
        prediction = probabilities.argmax(dim=0).cpu().numpy().astype(np.uint8)
        Image.fromarray(prediction, mode="L").save(render_root / name)
        local = np.bincount(
            CLASS_COUNT * target.reshape(-1).astype(np.int64) + prediction.reshape(-1), minlength=CLASS_COUNT**2
        ).reshape(CLASS_COUNT, CLASS_COUNT)
        combined += local
        view_metrics = s0.confusion_metrics(local)
        local_retention = per_pixel_retention_summary(stats)
        per_view.append({"view": name, **view_metrics, "visibility": local_retention})
        update_retention(retention, stats)
        removed = (stats["foreground_mass"] - stats["retained_contribution_mass"]).detach().cpu().numpy()
        removed_mass_total += float(removed.sum())
        removed_mass_in_bg_to_stay += float(removed[(target == 0) & (prediction == 2)].sum())
    metrics = s0.confusion_metrics(combined)
    worst_count = max(1, int(np.ceil(len(per_view) * 0.10)))
    worst = sorted(per_view, key=lambda record: record["mIoU_without_background"])[:worst_count]
    difficult = [record for record in per_view if 140 <= int(Path(record["view"]).stem) <= 146]
    metrics.update(
        {
            "view_count": len(names),
            "confusion_target_rows_prediction_columns": combined.tolist(),
            "per_view": per_view,
            "worst_10_percent": {
                "count": worst_count,
                "views": [item["view"] for item in worst],
                "mean_mIoU_without_background": float(np.mean([item["mIoU_without_background"] for item in worst])),
            },
            "challenge_140_146": {
                "views": [item["view"] for item in difficult],
                "mean_mIoU_without_background": float(np.mean([item["mIoU_without_background"] for item in difficult])) if difficult else None,
            },
            "visibility": {
                **finalize_retention(retention),
                "removed_contribution_mass_in_bg_to_stay_prediction_regions": removed_mass_in_bg_to_stay,
                "removed_contribution_mass_fraction_in_bg_to_stay_prediction_regions": removed_mass_in_bg_to_stay / max(removed_mass_total, np.finfo(np.float64).tiny),
                "definition": "The overlap is contribution mass omitted by the stop-gradient gate, spatially intersected with GT-background/predicted-stay_cable pixels. It is an evaluation diagnostic, not a threshold-selection input.",
            },
            "local_protocol_note": "Fixed local full-image confusion-matrix protocol; official final mIoU rules have not been released.",
        }
    )
    with (output_dir / "validation_per_view.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "view", "mIoU_with_background", "mIoU_without_background", "stay_cable_IoU", "stay_cable_precision",
            "stay_cable_recall", "background_to_stay_cable_false_positive_pixels",
            "background_to_stay_cable_false_positive_fraction_of_gt_background", "retained_contribution_mass_fraction",
            "retained_contributor_fraction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in per_view:
            stay = record["stay_cable"]
            visibility = record["visibility"]
            writer.writerow({
                "view": record["view"],
                "mIoU_with_background": record["mIoU_with_background"],
                "mIoU_without_background": record["mIoU_without_background"],
                "stay_cable_IoU": stay["IoU"],
                "stay_cable_precision": stay["precision"],
                "stay_cable_recall": stay["recall"],
                "background_to_stay_cable_false_positive_pixels": record["background_to_stay_cable_false_positive_pixels"],
                "background_to_stay_cable_false_positive_fraction_of_gt_background": record["background_to_stay_cable_false_positive_fraction_of_gt_background"],
                "retained_contribution_mass_fraction": visibility["retained_contribution_mass_fraction"],
                "retained_contributor_fraction": visibility["retained_contributor_fraction"],
            })
    return metrics


def gradient_retention_diagnostic(
    name: str,
    labels_root: Path,
    data_root: Path,
    appearance_camera_id: int,
    model: Any,
    sidecar: AnchorSemanticSidecar,
    pipe: Any,
    extension: Any,
    gate: GateName,
    threshold: float,
    weights: Any,
) -> dict[str, Any]:
    """Compare gated vs ungated semantic anchor gradients on one training view.

    It does not update parameters and uses only a pre-registered training image.
    """
    import torch

    target = torch.from_numpy(s0.read_label(labels_root / name).astype(np.int64, copy=False)).cuda(non_blocking=True)
    _, _camera, view = s0.official_view(data_root, name, appearance_camera_id)
    gated, gated_stats = render_semantics(view, model, sidecar, pipe, extension, gate, threshold)
    if gated_stats is None:
        raise AssertionError("Gated diagnostic requires contribution tensors")
    direct_gated = torch.autograd.grad(
        gated_stats["semantic_numerator"].sum(), gated_stats["offset_probabilities"], retain_graph=True
    )[0][:, 0].detach()
    gated_loss = torch.nn.functional.nll_loss(torch.log(gated.clamp_min(1e-8)).unsqueeze(0), target.unsqueeze(0), weight=weights)
    gated_grad = torch.autograd.grad(gated_loss, sidecar.logits, retain_graph=False)[0].detach()
    ungated, _ = render_semantics(view, model, sidecar, pipe, extension, None, None)
    ungated_loss = torch.nn.functional.nll_loss(torch.log(ungated.clamp_min(1e-8)).unsqueeze(0), target.unsqueeze(0), weight=weights)
    ungated_grad = torch.autograd.grad(ungated_loss, sidecar.logits, retain_graph=False)[0].detach()
    # Cumulative threshold 1.0 is the adapter's all-contributor reference.  It
    # is not a candidate and is used only to quantify the binary gate mask.
    full, full_stats = render_visibility_semantic_probabilities(
        view, model, sidecar, pipe, gate="cumulative", threshold=1.0, extension=extension
    )
    del full
    direct_full = torch.autograd.grad(
        full_stats["semantic_numerator"].sum(), full_stats["offset_probabilities"], retain_graph=False
    )[0][:, 0].detach()
    if not torch.equal(gated_stats["offset_anchor_indices"], full_stats["offset_anchor_indices"]):
        raise RuntimeError("Visibility diagnostic generated-Gaussian ordering changed between gated and full paths")
    anchor_count = int(sidecar.logits.shape[0])
    direct_gated_per_anchor = torch.zeros(anchor_count, dtype=torch.float32, device="cuda")
    direct_full_per_anchor = torch.zeros(anchor_count, dtype=torch.float32, device="cuda")
    direct_gated_per_anchor.scatter_add_(0, gated_stats["offset_anchor_indices"], direct_gated)
    direct_full_per_anchor.scatter_add_(0, full_stats["offset_anchor_indices"], direct_full)
    denominator = ungated_grad.abs().sum().clamp_min(torch.finfo(ungated_grad.dtype).tiny)
    active = ungated_grad.abs().sum(dim=1) > 0
    retained = gated_grad.abs().sum(dim=1) > 0
    return {
        "training_view": name,
        "gated_loss": float(gated_loss.item()),
        "ungated_loss": float(ungated_loss.item()),
        "loss_gradient_abs_ratio_gated_to_ungated": float((gated_grad.abs().sum() / denominator).item()),
        "anchors_with_ungated_gradient": int(active.sum().item()),
        "anchors_with_gated_gradient": int(retained.sum().item()),
        "anchor_gradient_support_fraction_retained": float((retained & active).sum().item() / max(int(active.sum().item()), 1)),
        "contribution_gate_semantic_probability_gradient_mass_fraction": float((direct_gated.sum() / direct_full.sum().clamp_min(torch.finfo(direct_full.dtype).tiny)).item()),
        "generated_gaussians_with_full_semantic_gradient": int((direct_full > 0).sum().item()),
        "generated_gaussians_with_gated_semantic_gradient": int((direct_gated > 0).sum().item()),
        "anchors_with_full_semantic_probability_gradient": int((direct_full_per_anchor > 0).sum().item()),
        "anchors_with_gated_semantic_probability_gradient": int((direct_gated_per_anchor > 0).sum().item()),
        "anchor_semantic_probability_gradient_mass_fraction": float((direct_gated_per_anchor.sum() / direct_full_per_anchor.sum().clamp_min(torch.finfo(direct_full_per_anchor.dtype).tiny)).item()),
        "absolute_gradient_by_logit_class_gated": {name: float(value) for name, value in zip(CLASS_NAMES, gated_grad.abs().sum(dim=0).cpu().tolist())},
        "absolute_gradient_by_logit_class_ungated": {name: float(value) for name, value in zip(CLASS_NAMES, ungated_grad.abs().sum(dim=0).cpu().tolist())},
        "note": "The direct probability-gradient mass is the exact stop-gradient contribution-mask quantity. Loss-gradient norms can exceed 1 after probability renormalization and are reported separately. RGB tensors are not differentiable inputs.",
    }


def run_equivalence(
    args: argparse.Namespace,
    config: dict[str, Any],
    model: Any,
    rgb_hashes: dict[str, str],
    extension: Any,
) -> dict[str, Any]:
    import torch

    sidecar = load_sem001_sidecar(args.sem001_checkpoint, int(model.get_anchor.shape[0]), rgb_hashes)
    pipe = pipeline()
    view_name = str(config["equivalence"]["image_name"])
    _, _camera, view = s0.official_view(args.data_root, view_name, int(config["frozen_rgb"]["appearance_camera_id"]))
    with torch.no_grad():
        sem001_logits = render_semantic_probabilities(view, model, sidecar, pipe)
        sem002_no_gate_logits, _ = render_semantics(view, model, sidecar, pipe, extension, None, None)
        rgb_before = _rgb_array(render_frozen_rgb(view, model, pipe))
        rgb_after = _rgb_array(render_frozen_rgb(view, model, pipe))
        # This numerically validates the custom rasterizer while the actual
        # disabled-gate path remains the exact SEM-001 renderer above.
        full_visibility, _stats = render_visibility_semantic_probabilities(
            view, model, sidecar, pipe, gate="cumulative", threshold=1.0, extension=extension
        )
    exact_logits = bool(torch.equal(sem001_logits, sem002_no_gate_logits))
    exact_png = bool(torch.equal(sem001_logits.argmax(dim=0), sem002_no_gate_logits.argmax(dim=0)))
    rgb_max = int(np.abs(rgb_before.astype(np.int16) - rgb_after.astype(np.int16)).max())
    custom_difference = float((sem001_logits - full_visibility).abs().max().item())
    custom_png_equal = bool(torch.equal(sem001_logits.argmax(dim=0), full_visibility.argmax(dim=0)))
    if not exact_logits or not exact_png or rgb_max != 0:
        raise RuntimeError("SEM-020-EQ failed: disabled visibility path is not exactly SEM-001")
    tolerance = float(config["equivalence"]["custom_full_visibility_max_abs_tolerance"])
    if custom_difference > tolerance or not custom_png_equal:
        raise RuntimeError("Custom contribution adapter cannot reproduce ungated class IDs within its registered numerical tolerance")
    return {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "sem001_checkpoint_sha256": sha256(args.sem001_checkpoint),
        "no_gate_logits_bit_identical": exact_logits,
        "no_gate_class_png_bit_identical": exact_png,
        "rgb_max_absolute_pixel_difference": rgb_max,
        "custom_full_visibility_cumulative_threshold": 1.0,
        "custom_full_visibility_max_absolute_probability_difference": custom_difference,
        "custom_full_visibility_class_png_bit_identical": custom_png_equal,
        "custom_full_visibility_tolerance": tolerance,
        "frozen_rgb_trainable_parameters": [name for name, parameter in frozen_rgb_named_parameters(model) if parameter.requires_grad],
    }


def run_training_side_diagnostic(
    args: argparse.Namespace,
    config: dict[str, Any],
    split: dict[str, Any],
    model: Any,
    extension: Any,
) -> dict[str, Any]:
    import torch

    pipe = pipeline()
    # Zero logits are sufficient: visibility is fully determined by frozen RGB
    # geometry/opacity, not semantic values or labels.
    sidecar = AnchorSemanticSidecar(int(model.get_anchor.shape[0]))
    all_training_names = list(split["semantic_train_labeled"])
    start = int(args.diagnostic_start)
    if start < 0 or start >= len(all_training_names):
        raise ValueError("diagnostic-start is outside the semantic training camera list")
    count = len(all_training_names) - start if args.diagnostic_count is None else int(args.diagnostic_count)
    if count <= 0:
        raise ValueError("diagnostic-count must be positive")
    training_names = all_training_names[start : start + count]
    entries = config["training_side_threshold_diagnostic"]["gates"]
    summaries: dict[str, Any] = {}
    for entry in entries:
        gate = str(entry["name"])
        if gate not in {"cumulative", "relative_max"}:
            raise ValueError("Invalid diagnostic gate")
        threshold = float(entry["threshold"])
        retention = _retention_accumulator()
        started = time.perf_counter()
        for index, name in enumerate(training_names, start=1):
            _, _camera, view = s0.official_view(args.data_root, name, int(config["frozen_rgb"]["appearance_camera_id"]))
            with torch.no_grad():
                _probabilities, stats = render_visibility_semantic_probabilities(
                    view, model, sidecar, pipe, gate=gate, threshold=threshold, extension=extension  # type: ignore[arg-type]
                )
            update_retention(retention, stats)
            if index == 1 or index % 25 == 0 or index == len(training_names):
                print(json.dumps({"diagnostic_gate": gate, "threshold": threshold, "view": index, "views": len(training_names), **finalize_retention(retention)}, sort_keys=True), flush=True)
        summaries[gate] = {
            "threshold": threshold,
            "selection_rule": entry["selection_rule"],
            "training_views": len(training_names),
            "training_view_start": start,
            "training_view_end_exclusive": start + len(training_names),
            "elapsed_seconds": time.perf_counter() - started,
            "raw_accumulator": retention,
            **finalize_retention(retention),
        }
    return {
        "experiment_id": config["experiment_id"],
        "status": "completed_training_side_only_diagnostic",
        "uses_validation_rgb_or_labels": False,
        "uses_unlabeled_rgb": False,
        "training_view_chunk": {
            "start": start,
            "end_exclusive": start + len(training_names),
            "total_training_views": len(all_training_names),
            "image_names": training_names,
        },
        "threshold_diagnostics": summaries,
    }


def run_preflight_or_full(
    args: argparse.Namespace,
    config: dict[str, Any],
    split: dict[str, Any],
    label_manifest: dict[str, Any],
    model: Any,
    rgb_hashes: dict[str, str],
    extension: Any,
) -> dict[str, Any]:
    import torch

    gate, threshold = gate_from_config(config)
    budget = config[args.mode]
    iterations = int(args.iterations if args.iterations is not None else budget["iterations"])
    if iterations != int(budget["iterations"]):
        raise ValueError("Iteration override differs from registered mode budget")
    if args.mode == "full" and iterations != 10000:
        raise ValueError("SEM-002 main-fold comparison is fixed at 10,000 iterations")
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.reset_peak_memory_stats()
    # This is the SEM-001 *initialization* (all semantic logits zero), not a
    # continuation from the trained SEM-001 checkpoint.
    sidecar = AnchorSemanticSidecar(int(model.get_anchor.shape[0]), initial_background_logit=float(config["training"]["initial_background_logit"]))
    optimizer = torch.optim.AdamW([sidecar.logits], lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    weights = torch.tensor(s0.class_weights(label_manifest), device="cuda")
    pipe = pipeline()
    sampler = np.random.default_rng(seed)
    train_names = list(split["semantic_train_labeled"])
    losses: list[float] = []
    training_retention = _retention_accumulator()
    anchor_gradient_support_views = torch.zeros(int(model.get_anchor.shape[0]), dtype=torch.int32, device="cuda")
    gradient_logit_abs = torch.zeros(CLASS_COUNT, dtype=torch.float64, device="cuda")
    checkpoint_steps = {int(item) for item in budget.get("checkpoint_iterations", [])}
    started = time.perf_counter()
    for step in range(1, iterations + 1):
        name = train_names[int(sampler.integers(len(train_names)))]
        target = torch.from_numpy(s0.read_label(args.labels_root / name).astype(np.int64, copy=False)).cuda(non_blocking=True)
        _, _camera, view = s0.official_view(args.data_root, name, int(config["frozen_rgb"]["appearance_camera_id"]))
        probabilities, stats = render_semantics(view, model, sidecar, pipe, extension, gate, threshold)
        if stats is None:
            raise AssertionError("SEM-002 training requires contributor statistics")
        loss = torch.nn.functional.nll_loss(torch.log(probabilities.clamp_min(1e-8)).unsqueeze(0), target.unsqueeze(0), weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        frozen_with_grad = [name for name, parameter in frozen_rgb_named_parameters(model) if parameter.requires_grad or parameter.grad is not None]
        if frozen_with_grad:
            raise RuntimeError("Frozen RGB isolation failed: RGB parameters received semantic gradients: " + ", ".join(frozen_with_grad))
        if sidecar.logits.grad is None or not bool(torch.isfinite(sidecar.logits.grad).all()):
            raise RuntimeError("Semantic sidecar gradients are missing or non-finite")
        gradient = sidecar.logits.grad.detach()
        anchor_gradient_support_views += (gradient.abs().sum(dim=1) > 0).to(torch.int32)
        gradient_logit_abs += gradient.abs().sum(dim=0).to(torch.float64)
        update_retention(training_retention, stats)
        optimizer.step()
        if not bool(torch.isfinite(sidecar.logits).all()):
            raise RuntimeError("Semantic sidecar logits became non-finite")
        losses.append(float(loss.detach().cpu()))
        if step in checkpoint_steps:
            save_sem002_checkpoint(args.output_dir / "checkpoints" / f"semantic_sidecar_iter_{step}.pth", sidecar, config, rgb_hashes)
        if step == 1 or step % int(budget["log_every"]) == 0 or step == iterations:
            print(json.dumps({
                "step": step,
                "loss": losses[-1],
                "mean_recent_loss": float(np.mean(losses[-min(len(losses), int(budget["log_every"])):])),
                "max_memory_allocated_mib": int(torch.cuda.max_memory_allocated() / 1024**2),
                **finalize_retention(training_retention),
            }, sort_keys=True), flush=True)
    training_seconds = time.perf_counter() - started
    isolation = s0.RGB_isolation_evidence(
        args.output_dir,
        args.data_root,
        list(config["isolation"]["views"]),
        int(config["frozen_rgb"]["appearance_camera_id"]),
        model,
        pipe,
        rgb_hashes,
        args.rgb_run_dir,
        int(config["frozen_rgb"]["iteration"]),
    )
    final_checkpoint = args.output_dir / "checkpoints" / "semantic_sidecar.pth"
    save_sem002_checkpoint(final_checkpoint, sidecar, config, rgb_hashes)
    reloaded = reload_sem002_sidecar(final_checkpoint, int(model.get_anchor.shape[0]), config, rgb_hashes)
    with torch.no_grad():
        if not torch.equal(sidecar.logits, reloaded.logits):
            raise RuntimeError("Saved/reloaded SEM-002 sidecar logits are not bit-identical")
    external = _semantic_external_output(
        args.output_dir,
        args.data_root,
        str(config["external_camera"]["image_name"]),
        int(config["frozen_rgb"]["appearance_camera_id"]),
        model,
        reloaded,
        pipe,
        extension,
        gate,
        threshold,
    )
    validation_names = list(split["semantic_validation_labeled"])
    if args.mode == "preflight":
        validation_names = validation_names[: int(budget["validation_views"])]
    evaluation = evaluate(
        args.output_dir,
        validation_names,
        args.labels_root,
        args.data_root,
        int(config["frozen_rgb"]["appearance_camera_id"]),
        model,
        reloaded,
        pipe,
        extension,
        gate,
        threshold,
    )
    gradient_diagnostic = gradient_retention_diagnostic(
        str(config["gradient_diagnostic_training_view"]),
        args.labels_root,
        args.data_root,
        int(config["frozen_rgb"]["appearance_camera_id"]),
        model,
        reloaded,
        pipe,
        extension,
        gate,
        threshold,
        weights,
    )
    result = {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "status": "completed" if args.mode == "full" else "completed_preflight_not_a_quality_baseline",
        "mode": args.mode,
        "iterations": iterations,
        "seed": seed,
        "visibility_gate": config["visibility_gate"],
        "train_views": len(train_names),
        "validation_views": len(validation_names),
        "class_weights": {name: float(value) for name, value in zip(CLASS_NAMES, weights.cpu().numpy())},
        "loss": {"first": losses[0], "last": losses[-1], "mean": float(np.mean(losses))},
        "training_seconds": training_seconds,
        "gpu_hours": training_seconds / 3600.0,
        "peak_cuda_allocated_mib": int(torch.cuda.max_memory_allocated() / 1024**2),
        "checkpoint": {"path": str(final_checkpoint), "sha256": sha256(final_checkpoint), "bytes": final_checkpoint.stat().st_size, "reloaded": True},
        "rgb_isolation": isolation,
        "external_camera": external,
        "training_visibility": {
            **finalize_retention(training_retention),
            "anchors_with_effective_semantic_gradient": int((anchor_gradient_support_views > 0).sum().item()),
            "effective_support_views_per_anchor": {
                "mean": float(anchor_gradient_support_views.to(torch.float32).mean().item()),
                "median": float(anchor_gradient_support_views.to(torch.float32).median().item()),
                "max": int(anchor_gradient_support_views.max().item()),
            },
            "absolute_gradient_by_logit_class": {name: float(value) for name, value in zip(CLASS_NAMES, gradient_logit_abs.cpu().tolist())},
            "note": "Support and class quantities are semantic-gradient evidence aggregated over sampled training views; RGB geometry/opacity are detached inputs.",
        },
        "gradient_retention_diagnostic": gradient_diagnostic,
        "evaluation": evaluation,
    }
    return result


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    if args.mode == "equivalence" and config["experiment_id"] != "SEM-020-EQ":
        raise ValueError("Equivalence mode requires the SEM-020-EQ config")
    if args.mode == "diagnostic" and config["experiment_id"] != "SEM-020-DIAG":
        raise ValueError("Diagnostic mode requires the SEM-020-DIAG config")
    if args.mode == "preflight" and not str(config["experiment_id"]).startswith(("SEM-021-PF", "SEM-022-PF")):
        raise ValueError("Visibility preflight mode requires SEM-021-PF or SEM-022-PF config")
    if args.mode == "full" and config["experiment_id"] not in {"SEM-021", "SEM-022"}:
        raise ValueError("Visibility full mode requires SEM-021 or SEM-022 config")
    split = load_json(REPO_ROOT / config["data"]["semantic_split"])
    label_manifest = load_json(REPO_ROOT / config["data"]["label_manifest"])
    if split["split_id"] != config["data"]["split_id"] or label_manifest["split_id"] != split["split_id"]:
        raise ValueError("Config, semantic split and label manifest do not match the frozen SEM-001 protocol")
    if label_manifest["class_mapping"] != {name: index for index, name in enumerate(CLASS_NAMES)}:
        raise ValueError("Official semantic mapping mismatch; class 2 must be stay_cable")
    if args.mode in {"preflight", "full"} and (not args.labels_root.is_dir() or not args.sem001_checkpoint.is_file()):
        raise FileNotFoundError("SEM-002 requires existing SEM-001 labels and checkpoint evidence")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "invocation.json", {
        "config": str(args.config), "mode": args.mode, "data_root": str(args.data_root), "source_root": str(args.source_root),
        "uses_validation_for_threshold_selection": False, "uses_test_rgb_or_labels": False,
    })
    settings = ScaffoldSettings(
        train_camera_count=int(config["frozen_rgb"]["train_camera_count"]), n_offsets=int(config["frozen_rgb"]["n_offsets"])
    )
    point_cloud = args.rgb_run_dir / "point_cloud" / f"iteration_{config['frozen_rgb']['iteration']}" / "point_cloud.ply"
    rgb_hashes = frozen_rgb_hashes(args.rgb_run_dir, int(config["frozen_rgb"]["iteration"]))
    configure_scaffold_import(args.source_root)
    model = load_frozen_scaffold(args.source_root, point_cloud, settings)
    extension = load_visibility_extension(args.extension_root)
    if args.mode == "equivalence":
        result = run_equivalence(args, config, model, rgb_hashes, extension)
    elif args.mode == "diagnostic":
        result = run_training_side_diagnostic(args, config, split, model, extension)
    else:
        result = run_preflight_or_full(args, config, split, label_manifest, model, rgb_hashes, extension)
    write_json(args.output_dir / "result.json", result)
    print(json.dumps({"experiment_id": result["experiment_id"], "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
