#!/usr/bin/env python3
"""Run SEM-003 train-side conflict diagnosis and Dirichlet semantic controls.

SEM-003 never re-enables SEM-021/022 contribution gates.  The existing
visibility adapter is used only in diagnostic mode to measure the frozen RGB
contribution assigned to *training* labels.  Training and evaluation use the
ordinary SEM-001 all-contributor rasterization path.
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
from src.scaffold_dirichlet_evidence import (  # noqa: E402
    ConflictPolicy,
    DirichletAnchorSidecar,
    conflict_statistics,
    masked_concentration_regularizer,
    render_dirichlet_semantic_probabilities,
)
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
from src.scaffold_visibility_supervision import load_visibility_extension, render_visibility_semantic_probabilities  # noqa: E402


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
    parser.add_argument("--output-dir", type=Path, required=True, help="Git-ignored external experiment output root")
    parser.add_argument("--extension-root", type=Path, required=True, help="Git-ignored torch extension cache")
    parser.add_argument("--mode", choices=("diagnostic", "equivalence", "preflight", "full"), required=True)
    parser.add_argument("--evidence-artifact", type=Path, help="External SEM-003-DIAG .pth artifact; required outside diagnostic mode")
    parser.add_argument("--iterations", type=int, help="Must exactly equal the registered mode budget")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sem001_payload(path: Path, rgb_hashes: dict[str, str]) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("experiment_id") != "SEM-001":
        raise ValueError("SEM-003 must convert the frozen final SEM-001 checkpoint")
    if payload.get("semantic_storage") != "per_anchor_logits" or payload.get("class_names") != list(CLASS_NAMES):
        raise ValueError("SEM-001 checkpoint is not the expected five-class per-anchor semantic sidecar")
    if payload.get("frozen_rgb_hashes") != rgb_hashes:
        raise ValueError("SEM-001 checkpoint frozen RGB hashes do not match loaded RGB artifact")
    if not isinstance(payload.get("semantic_anchor_logits"), torch.Tensor):
        raise ValueError("SEM-001 checkpoint has no semantic anchor logits")
    return payload


def _policy_from_config(config: dict[str, Any]) -> ConflictPolicy:
    value = config["conflict_diagnostic"]["eligibility_policy"]
    return ConflictPolicy(
        minimum_support_views=int(value["minimum_support_views"]),
        minimum_background_fraction=float(value["minimum_background_fraction"]),
        minimum_stay_cable_fraction=float(value["minimum_stay_cable_fraction"]),
        maximum_protected_fraction=float(value["maximum_protected_fraction"]),
        conflict_quantile=float(value["conflict_quantile"]),
    )


def _policy_dict(policy: ConflictPolicy) -> dict[str, Any]:
    return {
        "minimum_support_views": policy.minimum_support_views,
        "minimum_background_fraction": policy.minimum_background_fraction,
        "minimum_stay_cable_fraction": policy.minimum_stay_cable_fraction,
        "maximum_protected_fraction": policy.maximum_protected_fraction,
        "conflict_quantile": policy.conflict_quantile,
    }


def _load_evidence_artifact(path: Path, config: dict[str, Any], anchor_count: int) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"Missing SEM-003 training-side diagnostic artifact: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("experiment_id") != "SEM-003-DIAG":
        raise ValueError("Evidence artifact does not belong to SEM-003-DIAG")
    if payload.get("split_id") != config["data"]["split_id"]:
        raise ValueError("Evidence artifact split differs from SEM-003 config")
    if payload.get("policy") != _policy_dict(_policy_from_config(config)):
        raise ValueError("Evidence artifact eligibility policy differs from the frozen config")
    mask = payload.get("eligibility")
    if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (anchor_count,):
        raise ValueError("Evidence artifact eligibility mask does not match frozen RGB anchors")
    if int(mask.sum()) == 0:
        raise RuntimeError("SEM-003 diagnostic found no eligible bg/stay conflict anchors; candidate training must not start")
    return payload


def _view_probability_output(view: Any, model: Any, sidecar: DirichletAnchorSidecar, pipe: Any):
    return render_dirichlet_semantic_probabilities(view, model, sidecar, pipe)


def _external_render(
    output_dir: Path,
    data_root: Path,
    image_name: str,
    appearance_camera_id: int,
    model: Any,
    sidecar: DirichletAnchorSidecar,
    pipe: Any,
) -> dict[str, Any]:
    selected, camera, view = s0.official_view(data_root, image_name, appearance_camera_id)
    rgb = s0.tensor_to_rgb(render_frozen_rgb(view, model, pipe))
    with __import__("torch").no_grad():
        probabilities = _view_probability_output(view, model, sidecar, pipe).detach().cpu().numpy().astype(np.float32)
    official_rgb = s0.redistort_rgb(rgb.astype(np.float32) / 255.0, camera)
    official_probabilities = s0.redistort_semantic_probabilities(probabilities, camera)
    logits = np.log(np.clip(official_probabilities, np.finfo(np.float32).tiny, 1.0))
    labels = logits.argmax(axis=0).astype(np.uint8)
    if labels.shape != (camera.height, camera.width) or int(labels.min()) < 0 or int(labels.max()) >= CLASS_COUNT:
        raise RuntimeError("External semantic output is not a valid five-class ID PNG")
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
        "rgb_shape": list(official_rgb.shape),
        "semantic_logits_shape": list(logits.shape),
        "semantic_png_shape": list(labels.shape),
        "semantic_png_dtype": str(labels.dtype),
        "semantic_png_min": int(labels.min()),
        "semantic_png_max": int(labels.max()),
        "uses_test_rgb_or_labels": False,
    }


def _evaluate(
    output_dir: Path,
    names: list[str],
    labels_root: Path,
    data_root: Path,
    appearance_camera_id: int,
    model: Any,
    sidecar: DirichletAnchorSidecar,
    pipe: Any,
) -> dict[str, Any]:
    import torch

    combined = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    per_view: list[dict[str, Any]] = []
    render_root = output_dir / "validation_semantic_png"
    render_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = s0.read_label(labels_root / name)
        _, _camera, view = s0.official_view(data_root, name, appearance_camera_id)
        with torch.no_grad():
            prediction = _view_probability_output(view, model, sidecar, pipe).argmax(dim=0).cpu().numpy().astype(np.uint8)
        Image.fromarray(prediction, mode="L").save(render_root / name)
        local = np.bincount(CLASS_COUNT * target.reshape(-1).astype(np.int64) + prediction.reshape(-1), minlength=CLASS_COUNT**2).reshape(CLASS_COUNT, CLASS_COUNT)
        combined += local
        per_view.append({"view": name, **s0.confusion_metrics(local)})
    metrics = s0.confusion_metrics(combined)
    worst_count = max(1, int(np.ceil(len(per_view) * 0.10)))
    worst = sorted(per_view, key=lambda item: item["mIoU_without_background"])[:worst_count]
    difficult = [item for item in per_view if 140 <= int(Path(item["view"]).stem) <= 146]
    metrics.update({
        "view_count": len(names),
        "confusion_target_rows_prediction_columns": combined.tolist(),
        "per_view": per_view,
        "worst_10_percent": {"count": worst_count, "views": [item["view"] for item in worst], "mean_mIoU_without_background": float(np.mean([item["mIoU_without_background"] for item in worst]))},
        "challenge_140_146": {"views": [item["view"] for item in difficult], "mean_mIoU_without_background": float(np.mean([item["mIoU_without_background"] for item in difficult])) if difficult else None},
        "local_protocol_note": "Fixed local full-image confusion-matrix protocol; official final mIoU rules are not yet released.",
    })
    with (output_dir / "validation_per_view.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["view", "mIoU_with_background", "mIoU_without_background", "stay_cable_IoU", "stay_cable_precision", "stay_cable_recall", "background_to_stay_cable_false_positive_pixels", "background_to_stay_cable_false_positive_fraction_of_gt_background"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in per_view:
            stay = record["stay_cable"]
            writer.writerow({"view": record["view"], "mIoU_with_background": record["mIoU_with_background"], "mIoU_without_background": record["mIoU_without_background"], "stay_cable_IoU": stay["IoU"], "stay_cable_precision": stay["precision"], "stay_cable_recall": stay["recall"], "background_to_stay_cable_false_positive_pixels": record["background_to_stay_cable_false_positive_pixels"], "background_to_stay_cable_false_positive_fraction_of_gt_background": record["background_to_stay_cable_false_positive_fraction_of_gt_background"]})
    return metrics


def _save_checkpoint(path: Path, sidecar: DirichletAnchorSidecar, config: dict[str, Any], rgb_hashes: dict[str, str], evidence_sha256: str | None) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 3,
        "experiment_id": config["experiment_id"],
        "parent_experiment": "SEM-001",
        "class_names": list(CLASS_NAMES),
        "semantic_storage": "per_anchor_dirichlet_evidence",
        "frozen_rgb_hashes": rgb_hashes,
        "source_sem001_checkpoint_sha256": sha256(Path(config["_runtime_sem001_checkpoint"])),
        "evidence_artifact_sha256": evidence_sha256,
        **sidecar.checkpoint_state(),
    }
    torch.save(payload, path)


def _reload_checkpoint(path: Path, sidecar: DirichletAnchorSidecar, config: dict[str, Any], rgb_hashes: dict[str, str], evidence_sha256: str | None) -> DirichletAnchorSidecar:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("experiment_id") != config["experiment_id"] or payload.get("semantic_storage") != "per_anchor_dirichlet_evidence":
        raise ValueError("Dirichlet semantic checkpoint schema/experiment mismatch")
    if payload.get("frozen_rgb_hashes") != rgb_hashes or payload.get("evidence_artifact_sha256") != evidence_sha256:
        raise ValueError("Dirichlet semantic checkpoint provenance mismatch")
    restored = DirichletAnchorSidecar(sidecar.class_logits.detach(), initial_evidence_strength=1.0, device=sidecar.class_logits.device)
    restored.load_checkpoint_state(payload)
    return restored


def _run_diagnostic(args: argparse.Namespace, config: dict[str, Any], split: dict[str, Any], model: Any, extension: Any) -> dict[str, Any]:
    import torch

    policy = _policy_from_config(config)
    sidecar = AnchorSemanticSidecar(int(model.get_anchor.shape[0]))
    pipe = pipeline()
    anchors = int(model.get_anchor.shape[0])
    evidence = torch.zeros((anchors, CLASS_COUNT), dtype=torch.float64, device="cuda")
    support_views = torch.zeros(anchors, dtype=torch.int32, device="cuda")
    class_support_views = torch.zeros((anchors, CLASS_COUNT), dtype=torch.int32, device="cuda")
    view_summaries: list[dict[str, Any]] = []
    names = list(split["semantic_train_labeled"])
    started = time.perf_counter()
    for index, name in enumerate(names, start=1):
        target = torch.from_numpy(s0.read_label(args.labels_root / name).astype(np.int64, copy=False)).to(device="cuda", non_blocking=True)
        one_hot = torch.nn.functional.one_hot(target, num_classes=CLASS_COUNT).permute(2, 0, 1).to(dtype=torch.float32)
        _, _camera, view = s0.official_view(args.data_root, name, int(config["frozen_rgb"]["appearance_camera_id"]))
        _probabilities, stats = render_visibility_semantic_probabilities(view, model, sidecar, pipe, gate="cumulative", threshold=1.0, extension=extension)
        # ``numerator`` is an all-contributor, frozen-RGB weighted sum. Its
        # derivative w.r.t. per-offset class probability is exactly the label
        # evidence r(g,p)y(p,c); no contributor is filtered or reweighted.
        offset_evidence = torch.autograd.grad((stats["semantic_numerator"] * one_hot).sum(), stats["offset_probabilities"], retain_graph=False)[0].detach()
        anchor_indices = stats["offset_anchor_indices"].detach()
        anchor_evidence = torch.zeros((anchors, CLASS_COUNT), dtype=torch.float64, device="cuda")
        anchor_evidence.scatter_add_(0, anchor_indices[:, None].expand(-1, CLASS_COUNT), offset_evidence.to(torch.float64))
        evidence += anchor_evidence
        supported = anchor_evidence.sum(dim=1) > 0
        support_views += supported.to(torch.int32)
        class_support_views += (anchor_evidence > 0).to(torch.int32)
        if index == 1 or index % 25 == 0 or index == len(names):
            view_summaries.append({"view": name, "index": index, "anchors_with_label_evidence": int(supported.sum().item())})
            print(json.dumps({"diagnostic_view": index, "training_views": len(names), "anchors_with_label_evidence": int(supported.sum().item())}, sort_keys=True), flush=True)
    stats = conflict_statistics(evidence, support_views, policy)
    artifact = {
        "schema_version": 1,
        "experiment_id": "SEM-003-DIAG",
        "split_id": config["data"]["split_id"],
        "policy": _policy_dict(policy),
        "uses_validation_rgb_or_labels": False,
        "uses_unlabeled_rgb": False,
        "contribution_definition": "sum of exact frozen RGB all-contributor marginal contributions r(g,p) times train-side one-hot label; cumulative threshold=1 is extraction only, not a training gate",
        "evidence": evidence.detach().cpu(),
        "support_views": support_views.detach().cpu(),
        "class_support_views": class_support_views.detach().cpu(),
        "eligibility": stats["eligibility"].detach().cpu(),
        "base_eligibility": stats["base_eligibility"].detach().cpu(),
        "background_stay_conflict": stats["background_stay_conflict"].detach().cpu(),
        "protected_fraction": stats["protected_fraction"].detach().cpu(),
        "conflict_threshold": float(stats["conflict_threshold"]),
    }
    artifact_path = args.output_dir / "diagnostic_evidence.pth"
    torch.save(artifact, artifact_path)
    summary = {
        "schema_version": 1,
        "experiment_id": "SEM-003-DIAG",
        "status": "completed_training_side_only_diagnostic",
        "training_views": len(names),
        "split_id": config["data"]["split_id"],
        "policy": _policy_dict(policy),
        "conflict_threshold": float(stats["conflict_threshold"]),
        "anchors": anchors,
        "base_eligible_anchors": int(stats["base_eligibility"].sum().item()),
        "eligible_anchors": int(stats["eligibility"].sum().item()),
        "eligible_fraction": float(stats["eligibility"].to(torch.float32).mean().item()),
        "support_views": {"mean": float(support_views.to(torch.float32).mean().item()), "median": float(support_views.to(torch.float32).median().item()), "max": int(support_views.max().item())},
        "artifact": {"path": str(artifact_path), "sha256": sha256(artifact_path), "bytes": artifact_path.stat().st_size},
        "elapsed_seconds": time.perf_counter() - started,
        "view_progress": view_summaries,
        "uses_validation_rgb_or_labels": False,
        "uses_unlabeled_rgb": False,
    }
    write_json(args.output_dir / "result.json", summary)
    return summary


def _run_equivalence(args: argparse.Namespace, config: dict[str, Any], model: Any, rgb_hashes: dict[str, str], sem001_payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    sidecar = DirichletAnchorSidecar(sem001_payload["semantic_anchor_logits"], initial_evidence_strength=float(config["dirichlet"]["initial_evidence_strength"]), device="cuda")
    pipe = pipeline()
    _, _camera, view = s0.official_view(args.data_root, str(config["equivalence"]["image_name"]), int(config["frozen_rgb"]["appearance_camera_id"]))
    sem001 = AnchorSemanticSidecar(int(model.get_anchor.shape[0]))
    sem001.load_state_dict(sem001_payload)
    with torch.no_grad():
        reference = render_semantic_probabilities(view, model, sem001, pipe)
        converted = _view_probability_output(view, model, sidecar, pipe)
        rgb_before = s0.tensor_to_rgb(render_frozen_rgb(view, model, pipe))
        rgb_after = s0.tensor_to_rgb(render_frozen_rgb(view, model, pipe))
    difference = float((reference - converted).abs().max().item())
    labels_equal = bool(torch.equal(reference.argmax(dim=0), converted.argmax(dim=0)))
    rgb_max = int(np.abs(rgb_before.astype(np.int16) - rgb_after.astype(np.int16)).max())
    tolerance = float(config["equivalence"]["max_probability_difference"])
    if difference > tolerance or not labels_equal or rgb_max != 0:
        raise RuntimeError("SEM-003 Dirichlet conversion equivalence failed")
    result = {"experiment_id": config["experiment_id"], "status": "completed", "source_sem001_checkpoint_sha256": sha256(args.sem001_checkpoint), "max_probability_difference": difference, "class_png_bit_identical": labels_equal, "rgb_max_absolute_pixel_difference": rgb_max, "frozen_rgb_trainable_parameters": [name for name, parameter in frozen_rgb_named_parameters(model) if parameter.requires_grad]}
    write_json(args.output_dir / "result.json", result)
    return result


def _run_training(args: argparse.Namespace, config: dict[str, Any], split: dict[str, Any], label_manifest: dict[str, Any], model: Any, rgb_hashes: dict[str, str], sem001_payload: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    import torch

    budget = config[args.mode]
    iterations = int(args.iterations if args.iterations is not None else budget["iterations"])
    if iterations != int(budget["iterations"]):
        raise ValueError("Iteration override differs from pre-registered budget")
    if args.mode == "full" and iterations != 10000:
        raise ValueError("SEM-003 main-fold runs are exactly 10,000 iterations")
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.reset_peak_memory_stats()
    sidecar = DirichletAnchorSidecar(sem001_payload["semantic_anchor_logits"], initial_evidence_strength=float(config["dirichlet"]["initial_evidence_strength"]), device="cuda")
    conflict_enabled = bool(config["conflict_regularizer"]["enabled"])
    eligibility = evidence["eligibility"].to(device="cuda", dtype=torch.bool)
    if conflict_enabled:
        # Ordinary NLL always trains class logits for every anchor.  The
        # concentration scalar is intentionally mutable only on the frozen
        # diagnostic mask, so protected/non-conflict anchors cannot receive the
        # new conflict regularizer through an incidental CE gradient.
        sidecar.log_evidence_strength.register_hook(lambda gradient: gradient * eligibility.to(dtype=gradient.dtype))
        parameters = [sidecar.class_logits, sidecar.log_evidence_strength]
    else:
        sidecar.log_evidence_strength.requires_grad_(False)
        parameters = [sidecar.class_logits]
    optimizer = torch.optim.AdamW(parameters, lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    weights = torch.tensor(s0.class_weights(label_manifest), device="cuda")
    pipe = pipeline()
    sampler = np.random.default_rng(seed)
    names = list(split["semantic_train_labeled"])
    losses: list[float] = []
    conflict_losses: list[float] = []
    checkpoint_steps = {int(value) for value in budget.get("checkpoint_iterations", [])}
    evidence_sha256 = sha256(args.evidence_artifact) if args.evidence_artifact is not None else None
    started = time.perf_counter()
    for step in range(1, iterations + 1):
        name = names[int(sampler.integers(len(names)))]
        target = torch.from_numpy(s0.read_label(args.labels_root / name).astype(np.int64, copy=False)).to(device="cuda", non_blocking=True)
        _, _camera, view = s0.official_view(args.data_root, name, int(config["frozen_rgb"]["appearance_camera_id"]))
        probabilities = _view_probability_output(view, model, sidecar, pipe)
        nll = torch.nn.functional.nll_loss(torch.log(probabilities.clamp_min(1e-8)).unsqueeze(0), target.unsqueeze(0), weight=weights)
        concentration = masked_concentration_regularizer(sidecar.log_evidence_strength, eligibility, target_evidence_strength=float(config["conflict_regularizer"]["target_evidence_strength"])) if conflict_enabled else nll * 0.0
        loss = nll + float(config["conflict_regularizer"]["lambda"]) * concentration
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        frozen_with_grad = [name for name, parameter in frozen_rgb_named_parameters(model) if parameter.requires_grad or parameter.grad is not None]
        if frozen_with_grad:
            raise RuntimeError("Frozen RGB isolation failed: " + ", ".join(frozen_with_grad))
        if sidecar.class_logits.grad is None or not bool(torch.isfinite(sidecar.class_logits.grad).all()):
            raise RuntimeError("Dirichlet class logits have missing/non-finite gradients")
        if conflict_enabled and sidecar.log_evidence_strength.grad is not None and bool((sidecar.log_evidence_strength.grad[~eligibility] != 0).any()):
            raise RuntimeError("Conflict concentration gradient reached a non-eligible anchor")
        optimizer.step()
        if not bool(torch.isfinite(sidecar.class_logits).all()) or not bool(torch.isfinite(sidecar.log_evidence_strength).all()):
            raise RuntimeError("Dirichlet semantic parameters became non-finite")
        losses.append(float(loss.detach().cpu()))
        conflict_losses.append(float(concentration.detach().cpu()))
        if step in checkpoint_steps:
            _save_checkpoint(args.output_dir / "checkpoints" / f"semantic_sidecar_iter_{step}.pth", sidecar, config, rgb_hashes, evidence_sha256)
        if step == 1 or step % int(budget["log_every"]) == 0 or step == iterations:
            recent = losses[-min(len(losses), int(budget["log_every"])) :]
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": losses[-1],
                        "conflict_loss": conflict_losses[-1],
                        "mean_recent_loss": float(np.mean(recent)),
                        "max_memory_allocated_mib": int(torch.cuda.max_memory_allocated() / 1024**2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    training_seconds = time.perf_counter() - started
    isolation = s0.RGB_isolation_evidence(args.output_dir, args.data_root, list(config["isolation"]["views"]), int(config["frozen_rgb"]["appearance_camera_id"]), model, pipe, rgb_hashes, args.rgb_run_dir, int(config["frozen_rgb"]["iteration"]))
    isolation["semantic_optimizer_parameter_allowlist"] = ["dirichlet_class_logits"] + (["dirichlet_log_evidence_strength_on_diagnostic_mask"] if conflict_enabled else [])
    checkpoint = args.output_dir / "checkpoints" / "semantic_sidecar.pth"
    _save_checkpoint(checkpoint, sidecar, config, rgb_hashes, evidence_sha256)
    reloaded = _reload_checkpoint(checkpoint, sidecar, config, rgb_hashes, evidence_sha256)
    with torch.no_grad():
        if not torch.equal(sidecar.class_logits, reloaded.class_logits) or not torch.equal(sidecar.log_evidence_strength, reloaded.log_evidence_strength):
            raise RuntimeError("Dirichlet semantic checkpoint reload is not bit-identical")
    external = _external_render(args.output_dir, args.data_root, str(config["external_camera"]["image_name"]), int(config["frozen_rgb"]["appearance_camera_id"]), model, reloaded, pipe)
    validation = list(split["semantic_validation_labeled"])
    if args.mode == "preflight":
        validation = validation[: int(budget["validation_views"])]
    evaluation = _evaluate(args.output_dir, validation, args.labels_root, args.data_root, int(config["frozen_rgb"]["appearance_camera_id"]), model, reloaded, pipe)
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed" if args.mode == "full" else "completed_preflight_not_a_quality_baseline",
        "mode": args.mode,
        "iterations": iterations,
        "seed": seed,
        "source_sem001_checkpoint_sha256": sha256(args.sem001_checkpoint),
        "evidence_artifact_sha256": evidence_sha256,
        "eligible_conflict_anchors": int(eligibility.sum().item()),
        "train_views": len(names),
        "validation_views": len(validation),
        "class_weights": {name: float(value) for name, value in zip(CLASS_NAMES, weights.cpu().numpy())},
        "loss": {"first": losses[0], "last": losses[-1], "mean": float(np.mean(losses)), "conflict_first": conflict_losses[0], "conflict_last": conflict_losses[-1]},
        "training_seconds": training_seconds,
        "gpu_hours": training_seconds / 3600.0,
        "peak_cuda_allocated_mib": int(torch.cuda.max_memory_allocated() / 1024**2),
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size, "reloaded": True},
        "rgb_isolation": isolation,
        "external_camera": external,
        "evaluation": evaluation,
    }


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    expected = {"diagnostic": "SEM-003-DIAG", "equivalence": "SEM-003-CTRL-PF", "preflight": {"SEM-003-CTRL-PF", "SEM-003-PF"}, "full": {"SEM-003-CTRL", "SEM-003"}}
    allowed = expected[args.mode]
    if isinstance(allowed, str):
        if config["experiment_id"] != allowed:
            raise ValueError(f"{args.mode} mode requires {allowed}")
    elif config["experiment_id"] not in allowed:
        raise ValueError(f"Unexpected SEM-003 config for {args.mode} mode")
    split = load_json(REPO_ROOT / config["data"]["semantic_split"])
    manifest = load_json(REPO_ROOT / config["data"]["label_manifest"])
    if split["split_id"] != config["data"]["split_id"] or manifest["split_id"] != split["split_id"]:
        raise ValueError("Config, semantic split and label manifest mismatch")
    if manifest["class_mapping"] != {name: index for index, name in enumerate(CLASS_NAMES)}:
        raise ValueError("Official semantic mapping mismatch; class ID 2 must be stay_cable")
    if args.mode != "diagnostic" and args.evidence_artifact is None:
        raise ValueError("SEM-003 equivalence/preflight/full requires the frozen training-side evidence artifact")
    if not args.labels_root.is_dir() or not args.sem001_checkpoint.is_file():
        raise FileNotFoundError("SEM-003 requires released training labels and the frozen SEM-001 checkpoint")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "invocation.json", {"config": str(args.config), "mode": args.mode, "uses_validation_for_eligibility": False, "uses_test_rgb_or_labels": False, "global_visibility_gate_enabled": False})
    try:
        settings = ScaffoldSettings(train_camera_count=int(config["frozen_rgb"]["train_camera_count"]), n_offsets=int(config["frozen_rgb"]["n_offsets"]))
        point_cloud = args.rgb_run_dir / "point_cloud" / f"iteration_{config['frozen_rgb']['iteration']}" / "point_cloud.ply"
        rgb_hashes = frozen_rgb_hashes(args.rgb_run_dir, int(config["frozen_rgb"]["iteration"]))
        configure_scaffold_import(args.source_root)
        model = load_frozen_scaffold(args.source_root, point_cloud, settings)
        config["_runtime_sem001_checkpoint"] = str(args.sem001_checkpoint)
        sem001_payload = _sem001_payload(args.sem001_checkpoint, rgb_hashes)
        if args.mode == "diagnostic":
            extension = load_visibility_extension(args.extension_root)
            result = _run_diagnostic(args, config, split, model, extension)
        elif args.mode == "equivalence":
            _load_evidence_artifact(args.evidence_artifact, config, int(model.get_anchor.shape[0]))
            result = _run_equivalence(args, config, model, rgb_hashes, sem001_payload)
        else:
            evidence = _load_evidence_artifact(args.evidence_artifact, config, int(model.get_anchor.shape[0]))
            result = _run_training(args, config, split, manifest, model, rgb_hashes, sem001_payload, evidence)
            write_json(args.output_dir / "result.json", result)
    except Exception as error:
        write_json(args.output_dir / "failure.json", {
            "experiment_id": config["experiment_id"],
            "mode": args.mode,
            "status": "blocked_runtime_gpu_unavailable",
            "exception_type": type(error).__name__,
            "exception": str(error),
            "training_or_checkpoint_started": False,
            "uses_test_rgb_or_labels": False,
        })
        raise
    print(json.dumps({"experiment_id": result["experiment_id"], "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
