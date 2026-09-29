#!/usr/bin/env python3
"""Evaluate strict spatial-OOF anchor consensus for selective cable recovery.

SEM-130 is deliberately a training-only experiment.  It begins from the
already registered SEM-123 ``p20_d02_any`` recovery candidate, but asks a new
question for each proposed pixel: does the same dominant frozen Scaffold
anchor receive cable evidence in *other* cameras from the target's spatial OOF
fold?  The source OOF model excludes every camera in that fold, including the
current target camera.

The mechanism never deletes a hard-0.30 ``stay_cable`` pixel and never changes
SEM-001 deck, tower or foundation pixels.  It reads all source arrays and
camera metadata before any label.  Calibration labels select one fixed
minimum-source-view count; the disjoint check half evaluates it once.  The
74-view development set, reference RGB, unlabelled images, hidden inputs and
external-camera rendering are intentionally absent from this interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem001 as sem001
import run_sem080_structural_diag as common
import run_sem110_protected_fusion_oof as sem110

from src.scaffold_semantic_sidecar import (
    ScaffoldSettings,
    configure_scaffold_import,
    frozen_rgb_hashes,
    frozen_rgb_named_parameters,
    load_frozen_scaffold,
    pipeline,
    render_frozen_rgb,
)
from src.scaffold_visibility_supervision import (
    load_visibility_extension,
    render_frozen_contribution_diagnostics,
)
from src.semantic_band_recovery import recover_selective_band
from src.semantic_crossview_consensus import (
    add_view_support,
    apply_retained_recovery,
    dominant_anchor_map,
    oof_fold_by_name,
    other_view_support_map,
    retain_recovery_by_crossview_support,
    view_anchor_cable_evidence,
)
from src.semantic_protected_fusion import (
    CLASS_COUNT,
    fuse_background_stay_cable,
    local_confusion,
    merge_confusions,
    metrics_from_confusion,
    partition_training_names,
    validate_probabilities,
)
from src.semantic_render_domain import semantic_rgb
from src.semantic_rendered_training import select_rendered_training_views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--oof-split", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sem001-checkpoint", type=Path, required=True)
    parser.add_argument("--sem001-training-root", type=Path, required=True)
    parser.add_argument("--sem101-oof-root", type=Path, required=True)
    parser.add_argument("--sem101-oof-summary", type=Path, required=True)
    parser.add_argument("--sem110-result", type=Path, required=True)
    parser.add_argument("--sem123-result", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--audit-rgb-only",
        action="store_true",
        help="After a recorded calibration stop, run only the frozen RGB isolation audit; never read labels.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_score(metrics: dict[str, Any], weights: dict[str, float]) -> float:
    return float(
        float(weights["mIoU_with_background"]) * float(metrics["mIoU_with_background"])
        + float(weights["stay_cable_iou"]) * float(metrics["per_class"]["stay_cable"]["iou"])
        + float(weights["stay_cable_recall"]) * float(metrics["per_class"]["stay_cable"]["recall"])
    )


def validate_config(config: dict[str, Any]) -> None:
    require(
        config.get("experiment_id") == "SEM-130-ANCHOR-CROSSVIEW-CONSENSUS-OOF",
        "SEM-130 received another experiment config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-130 may run only from its registered config",
    )
    rule = config["crossview_consensus"]
    require(
        rule["recovery_rule"]
        == {
            "lower_threshold": 0.20,
            "upper_threshold": 0.30,
            "probability_minimum": 0.20,
            "maximum_seed_distance": 2.0,
            "prior_mode": "any",
        },
        "SEM-130 recovery rule is not the frozen SEM-123 p20_d02_any candidate",
    )
    support_grid = [int(value) for value in rule["minimum_source_view_grid"]]
    require(support_grid == [1, 2, 3, 4, 6, 8], "SEM-130 support grid differs from registration")
    require(
        int(rule["minimum_dominant_pixels_per_source_view"]) == 8,
        "SEM-130 source-pixel minimum differs from registration",
    )
    require(
        float(rule["minimum_mean_oof_stay_cable_probability"]) == 0.30,
        "SEM-130 source probability minimum differs from registration",
    )
    weights = config["selection"]["score_weights"]
    require(
        weights == {"mIoU_with_background": 0.55, "stay_cable_iou": 0.30, "stay_cable_recall": 0.15},
        "SEM-130 selection score differs from registration",
    )


def _anchor_map(
    *,
    view: Any,
    model: Any,
    pipe: Any,
    extension: Any,
    anchor_count: int,
) -> np.ndarray:
    import torch

    with torch.inference_mode():
        diagnostics = render_frozen_contribution_diagnostics(
            view, model, pipe, extension=extension
        )
    return dominant_anchor_map(
        diagnostics["dominant_offset_index"].detach().cpu().numpy(),
        diagnostics["offset_anchor_indices"].detach().cpu().numpy(),
        anchor_count=anchor_count,
    )


def _teacher(path: Path) -> np.ndarray:
    value = sem110.teacher_probabilities(path)
    return validate_probabilities(value, shape=(989, 1320))


def collect_fold_support_counts(
    *,
    names: list[str],
    fold_by_name: dict[str, int],
    teacher_paths: dict[str, Path],
    data_root: Path,
    appearance_camera_id: int,
    model: Any,
    pipe: Any,
    extension: Any,
    anchor_count: int,
    minimum_dominant_pixels: int,
    minimum_mean_probability: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build fold-local support counts using no target labels or real RGB."""
    fold_count = max(fold_by_name.values()) + 1
    counts = np.zeros((fold_count, anchor_count), dtype=np.uint16)
    per_view: list[dict[str, Any]] = []
    for index, name in enumerate(names, start=1):
        _selected, _camera, view = sem001.official_view(
            data_root, name, appearance_camera_id
        )
        anchors = _anchor_map(
            view=view,
            model=model,
            pipe=pipe,
            extension=extension,
            anchor_count=anchor_count,
        )
        supported = view_anchor_cable_evidence(
            anchors,
            _teacher(teacher_paths[name])[2],
            anchor_count=anchor_count,
            minimum_dominant_pixels=minimum_dominant_pixels,
            minimum_mean_cable_probability=minimum_mean_probability,
        )
        add_view_support(counts, fold=fold_by_name[name], supported_anchor_ids=supported)
        per_view.append(
            {
                "view": name,
                "fold": int(fold_by_name[name]),
                "dominant_anchor_pixels": int(np.count_nonzero(anchors >= 0)),
                "supported_anchor_count": int(supported.size),
            }
        )
        if index == 1 or index % 20 == 0 or index == len(names):
            print(
                json.dumps(
                    {"phase": "unlabelled_anchor_evidence", "completed_views": index, "total_views": len(names)},
                    sort_keys=True,
                ),
                flush=True,
            )
    return counts, {
        "source_labels_opened": False,
        "source_real_rgb_opened": False,
        "source_views": len(names),
        "per_view": per_view,
        "per_fold_supported_anchor_count": {
            str(fold): int(np.count_nonzero(counts[fold]))
            for fold in range(counts.shape[0])
        },
        "per_fold_total_support_observations": {
            str(fold): int(counts[fold].sum(dtype=np.int64))
            for fold in range(counts.shape[0])
        },
    }


def _view_state(
    *,
    name: str,
    base_paths: dict[str, Path],
    teacher_paths: dict[str, Path],
    fold_by_name: dict[str, int],
    fold_support_counts: np.ndarray,
    data_root: Path,
    appearance_camera_id: int,
    model: Any,
    pipe: Any,
    extension: Any,
    anchor_count: int,
    rule: dict[str, Any],
) -> dict[str, Any]:
    """Make one label-free hard/recovery state plus its source-count map."""
    base = sem110.image(base_paths[name], context="SEM-130 frozen SEM-001 map")
    teacher = _teacher(teacher_paths[name])
    recovery_rule = rule["recovery_rule"]
    hard, raw_selected, fixed = recover_selective_band(
        base,
        teacher,
        probability_minimum=float(recovery_rule["probability_minimum"]),
        maximum_seed_distance=float(recovery_rule["maximum_seed_distance"]),
        prior_mode=str(recovery_rule["prior_mode"]),
        lower_threshold=float(recovery_rule["lower_threshold"]),
        upper_threshold=float(recovery_rule["upper_threshold"]),
    )
    # ``recover_selective_band`` returns the raw candidate as ``hard`` plus
    # its band selection.  Keep only actual hard-background-to-cable changes,
    # so consensus can never overwrite an existing hard prediction.
    hard030, fixed_hard = fuse_background_stay_cable(
        base, teacher, stay_cable_threshold=float(recovery_rule["upper_threshold"])
    )
    require(np.array_equal(fixed, fixed_hard), "SEM-130 recovery/fusion protected masks differ")
    raw_additions = raw_selected & (hard030 != 2)
    raw_recovery = apply_retained_recovery(hard030, raw_additions, raw_additions)
    require(np.array_equal(hard, raw_recovery), "SEM-130 raw recovery differs from frozen SEM-123 predicate")
    _selected, _camera, view = sem001.official_view(data_root, name, appearance_camera_id)
    anchors = _anchor_map(
        view=view,
        model=model,
        pipe=pipe,
        extension=extension,
        anchor_count=anchor_count,
    )
    current_supported = view_anchor_cable_evidence(
        anchors,
        teacher[2],
        anchor_count=anchor_count,
        minimum_dominant_pixels=int(rule["minimum_dominant_pixels_per_source_view"]),
        minimum_mean_cable_probability=float(rule["minimum_mean_oof_stay_cable_probability"]),
    )
    source_support = other_view_support_map(
        anchors,
        fold_support_counts[int(fold_by_name[name])],
        current_supported,
    )
    return {
        "base": base,
        "hard030": hard030,
        "raw_recovery": raw_recovery,
        "raw_additions": raw_additions,
        "fixed": fixed,
        "source_support": source_support,
        "anchor_pixels": int(np.count_nonzero(anchors >= 0)),
    }


def _record_metrics(
    *,
    confusion: np.ndarray,
    recovered_pixels: int,
    raw_recovered_pixels: int,
    retained_target_counts: np.ndarray,
    raw_target_counts: np.ndarray,
    protected_predictions_exact: bool,
    minimum_source_views: int | None,
    score_weights: dict[str, float],
) -> dict[str, Any]:
    metrics = metrics_from_confusion(confusion)
    result = {
        "metrics": metrics,
        "selection_score": _metric_score(metrics, score_weights),
        "recovered_pixels": int(recovered_pixels),
        "vetoed_recovery_pixels": int(raw_recovered_pixels - recovered_pixels),
        "recovered_target_counts": retained_target_counts.astype(np.int64).tolist(),
        "vetoed_target_counts": (raw_target_counts - retained_target_counts).astype(np.int64).tolist(),
        "protected_prediction_pixels_exact": bool(protected_predictions_exact),
    }
    if minimum_source_views is not None:
        result["minimum_source_views"] = int(minimum_source_views)
    return result


def evaluate_grid(
    *,
    names: list[str],
    labels_root: Path,
    base_paths: dict[str, Path],
    teacher_paths: dict[str, Path],
    fold_by_name: dict[str, int],
    fold_support_counts: np.ndarray,
    data_root: Path,
    appearance_camera_id: int,
    model: Any,
    pipe: Any,
    extension: Any,
    anchor_count: int,
    rule: dict[str, Any],
    support_grid: list[int],
    score_weights: dict[str, float],
    phase: str,
    write_predictions: Path | None = None,
) -> dict[str, Any]:
    """Evaluate every registered source-count predicate on one label half."""
    hard_confusions: list[np.ndarray] = []
    raw_confusions: list[np.ndarray] = []
    candidate_confusions = {value: [] for value in support_grid}
    candidate_recovered = {value: 0 for value in support_grid}
    candidate_targets = {value: np.zeros(CLASS_COUNT, dtype=np.int64) for value in support_grid}
    hard_protected = True
    raw_protected = True
    candidate_protected = {value: True for value in support_grid}
    raw_recovered = 0
    raw_targets = np.zeros(CLASS_COUNT, dtype=np.int64)
    anchor_pixels = 0
    prediction_roots: dict[str, Path] | None = None
    written_protected = True
    written_recovered = 0
    if write_predictions is not None:
        require(
            len(support_grid) == 1,
            "SEM-130 may write predictions only for one already-selected source count",
        )
        prediction_roots = {
            "hard030": write_predictions / "hard030",
            "raw_sem123_p20_d02_any": write_predictions / "raw_sem123_p20_d02_any",
            "crossview": write_predictions / "crossview",
        }
        for root in prediction_roots.values():
            root.mkdir(parents=True, exist_ok=True)

    for index, name in enumerate(names, start=1):
        target = sem110.image(labels_root / name, context=f"SEM-130 {phase} label")
        state = _view_state(
            name=name,
            base_paths=base_paths,
            teacher_paths=teacher_paths,
            fold_by_name=fold_by_name,
            fold_support_counts=fold_support_counts,
            data_root=data_root,
            appearance_camera_id=appearance_camera_id,
            model=model,
            pipe=pipe,
            extension=extension,
            anchor_count=anchor_count,
            rule=rule,
        )
        hard_confusions.append(local_confusion(target, state["hard030"]))
        raw_confusions.append(local_confusion(target, state["raw_recovery"]))
        hard_protected = hard_protected and bool(
            np.array_equal(state["base"][state["fixed"]], state["hard030"][state["fixed"]])
        )
        raw_protected = raw_protected and bool(
            np.array_equal(state["base"][state["fixed"]], state["raw_recovery"][state["fixed"]])
        )
        raw_recovered += int(np.count_nonzero(state["raw_additions"]))
        raw_targets += np.bincount(target[state["raw_additions"]], minlength=CLASS_COUNT)
        anchor_pixels += int(state["anchor_pixels"])
        for minimum_source_views in support_grid:
            retained = retain_recovery_by_crossview_support(
                state["raw_additions"],
                state["source_support"],
                minimum_source_views=minimum_source_views,
            )
            candidate = apply_retained_recovery(
                state["hard030"], state["raw_additions"], retained
            )
            candidate_confusions[minimum_source_views].append(local_confusion(target, candidate))
            candidate_recovered[minimum_source_views] += int(np.count_nonzero(retained))
            candidate_targets[minimum_source_views] += np.bincount(
                target[retained], minlength=CLASS_COUNT
            )
            candidate_protected[minimum_source_views] = candidate_protected[minimum_source_views] and bool(
                np.array_equal(state["base"][state["fixed"]], candidate[state["fixed"]])
            )
            if prediction_roots is not None:
                Image.fromarray(state["hard030"], mode="L").save(prediction_roots["hard030"] / name)
                Image.fromarray(state["raw_recovery"], mode="L").save(
                    prediction_roots["raw_sem123_p20_d02_any"] / name
                )
                Image.fromarray(candidate, mode="L").save(prediction_roots["crossview"] / name)
                written_protected = written_protected and bool(
                    np.array_equal(state["base"][state["fixed"]], candidate[state["fixed"]])
                )
                written_recovered += int(np.count_nonzero(retained))
        if index == 1 or index % 20 == 0 or index == len(names):
            print(
                json.dumps(
                    {"phase": phase, "completed_views": index, "total_views": len(names)},
                    sort_keys=True,
                ),
                flush=True,
            )

    hard_metrics = metrics_from_confusion(merge_confusions(hard_confusions))
    raw_metrics = metrics_from_confusion(merge_confusions(raw_confusions))
    raw_record = _record_metrics(
        confusion=merge_confusions(raw_confusions),
        recovered_pixels=raw_recovered,
        raw_recovered_pixels=raw_recovered,
        retained_target_counts=raw_targets,
        raw_target_counts=raw_targets,
        protected_predictions_exact=raw_protected,
        minimum_source_views=None,
        score_weights=score_weights,
    )
    candidates = [
        _record_metrics(
            confusion=merge_confusions(candidate_confusions[value]),
            recovered_pixels=candidate_recovered[value],
            raw_recovered_pixels=raw_recovered,
            retained_target_counts=candidate_targets[value],
            raw_target_counts=raw_targets,
            protected_predictions_exact=candidate_protected[value],
            minimum_source_views=value,
            score_weights=score_weights,
        )
        for value in support_grid
    ]
    result = {
        "view_count": len(names),
        "anchor_pixels": anchor_pixels,
        "hard030": {
            "metrics": hard_metrics,
            "selection_score": _metric_score(hard_metrics, score_weights),
            "protected_prediction_pixels_exact": hard_protected,
        },
        "raw_sem123_p20_d02_any": raw_record,
        "candidates": candidates,
    }
    if prediction_roots is not None:
        result["prediction_outputs"] = {
            "roots": {key: str(value.resolve()) for key, value in prediction_roots.items()},
            "selected_recovery_pixels": written_recovered,
            "protected_prediction_pixels_exact": written_protected,
        }
    return result


def candidate_gate(
    candidate: dict[str, Any],
    hard: dict[str, Any],
    raw: dict[str, Any],
    *,
    gates: dict[str, Any],
    require_recall_gain: bool,
) -> dict[str, Any]:
    candidate_metrics = candidate["metrics"]
    hard_metrics = hard["metrics"]
    raw_metrics = raw["metrics"]
    checks = {
        "protected_prediction_pixels_exact": candidate["protected_prediction_pixels_exact"] is True,
        "nonempty_recovery": int(candidate["recovered_pixels"]) > 0,
        "mIoU_with_background_drop_within_limit": (
            float(hard_metrics["mIoU_with_background"]) - float(candidate_metrics["mIoU_with_background"])
            <= float(gates["mIoU_with_background_drop_max"])
        ),
        "stay_cable_iou_drop_within_limit": (
            float(hard_metrics["per_class"]["stay_cable"]["iou"])
            - float(candidate_metrics["per_class"]["stay_cable"]["iou"])
            <= float(gates["stay_cable_iou_drop_max"])
        ),
        "background_to_stay_cable_not_above_raw_recovery": (
            int(candidate_metrics["background_to_stay_cable_pixels"])
            <= int(raw_metrics["background_to_stay_cable_pixels"])
        ),
    }
    if require_recall_gain:
        checks["stay_cable_recall_gain_minimum"] = (
            float(candidate_metrics["per_class"]["stay_cable"]["recall"])
            - float(hard_metrics["per_class"]["stay_cable"]["recall"])
            >= float(gates["stay_cable_recall_gain_minimum"])
        )
    else:
        checks["stay_cable_recall_not_below_hard030"] = (
            float(candidate_metrics["per_class"]["stay_cable"]["recall"])
            >= float(hard_metrics["per_class"]["stay_cable"]["recall"])
        )
    return {"checks": checks, "passed": bool(all(checks.values()))}


def select_calibration_candidate(
    records: list[dict[str, Any]],
    hard: dict[str, Any],
    raw: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any] | None:
    eligible: list[dict[str, Any]] = []
    for record in records:
        record["calibration_gate"] = candidate_gate(
            record, hard, raw, gates=gates, require_recall_gain=True
        )
        if record["calibration_gate"]["passed"]:
            eligible.append(record)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda record: (
            float(record["selection_score"]),
            float(record["metrics"]["per_class"]["stay_cable"]["iou"]),
            -int(record["metrics"]["background_to_stay_cable_pixels"]),
            int(record["minimum_source_views"]),
        ),
    )


def _write_selected_predictions(
    *,
    names: list[str],
    output: Path,
    minimum_source_views: int,
    base_paths: dict[str, Path],
    teacher_paths: dict[str, Path],
    fold_by_name: dict[str, int],
    fold_support_counts: np.ndarray,
    data_root: Path,
    appearance_camera_id: int,
    model: Any,
    pipe: Any,
    extension: Any,
    anchor_count: int,
    rule: dict[str, Any],
) -> dict[str, Any]:
    roots = {name: output / name for name in ("hard030", "raw_sem123_p20_d02_any", "crossview")}
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    protected_exact = True
    added = 0
    for name in names:
        state = _view_state(
            name=name,
            base_paths=base_paths,
            teacher_paths=teacher_paths,
            fold_by_name=fold_by_name,
            fold_support_counts=fold_support_counts,
            data_root=data_root,
            appearance_camera_id=appearance_camera_id,
            model=model,
            pipe=pipe,
            extension=extension,
            anchor_count=anchor_count,
            rule=rule,
        )
        retained = retain_recovery_by_crossview_support(
            state["raw_additions"], state["source_support"], minimum_source_views=minimum_source_views
        )
        candidate = apply_retained_recovery(state["hard030"], state["raw_additions"], retained)
        Image.fromarray(state["hard030"], mode="L").save(roots["hard030"] / name)
        Image.fromarray(state["raw_recovery"], mode="L").save(roots["raw_sem123_p20_d02_any"] / name)
        Image.fromarray(candidate, mode="L").save(roots["crossview"] / name)
        protected_exact = protected_exact and bool(
            np.array_equal(state["base"][state["fixed"]], candidate[state["fixed"]])
        )
        added += int(np.count_nonzero(retained))
    return {
        "roots": {key: str(value.resolve()) for key, value in roots.items()},
        "selected_recovery_pixels": added,
        "protected_prediction_pixels_exact": protected_exact,
    }


def _panel(target: np.ndarray, hard: np.ndarray, raw: np.ndarray, consensus: np.ndarray) -> np.ndarray:
    delta = np.zeros((*target.shape, 3), dtype=np.uint8)
    restored = (consensus == 2) & (hard != 2)
    delta[restored & (target == 2)] = np.asarray([56, 220, 80], dtype=np.uint8)
    delta[restored & (target == 0)] = np.asarray([240, 56, 56], dtype=np.uint8)
    delta[restored & (target != 0) & (target != 2)] = np.asarray([255, 172, 35], dtype=np.uint8)
    return np.concatenate((semantic_rgb(target), semantic_rgb(hard), semantic_rgb(raw), semantic_rgb(consensus), delta), axis=1)


def _write_training_panels(
    *,
    names: list[str],
    labels_root: Path,
    predictions: dict[str, str],
    output: Path,
) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    roots = {key: Path(value) for key, value in predictions.items()}
    paths: list[str] = []
    for name in names:
        target = sem110.image(labels_root / name, context="SEM-130 training panel label")
        image = _panel(
            target,
            sem110.image(roots["hard030"] / name, context="SEM-130 hard panel map"),
            sem110.image(roots["raw_sem123_p20_d02_any"] / name, context="SEM-130 raw panel map"),
            sem110.image(roots["crossview"] / name, context="SEM-130 consensus panel map"),
        )
        path = output / name
        Image.fromarray(image).save(path)
        paths.append(str(path.resolve()))
    return paths


def _rgb_isolation(
    *,
    names: list[str],
    data_root: Path,
    appearance_camera_id: int,
    model: Any,
    pipe: Any,
    extension: Any,
    anchor_count: int,
    rgb_run_dir: Path,
    iteration: int,
) -> dict[str, Any]:
    """Show that diagnostic reads leave frozen RGB bytes unchanged."""
    import torch

    before: dict[str, str] = {}
    after: dict[str, str] = {}
    maximum_difference = 0
    for name in names:
        _selected, _camera, view = sem001.official_view(data_root, name, appearance_camera_id)
        with torch.inference_mode():
            rgb_before = sem001.tensor_to_rgb(render_frozen_rgb(view, model, pipe))
        _ = _anchor_map(
            view=view,
            model=model,
            pipe=pipe,
            extension=extension,
            anchor_count=anchor_count,
        )
        with torch.inference_mode():
            rgb_after = sem001.tensor_to_rgb(render_frozen_rgb(view, model, pipe))
        before[name] = hashlib.sha256(rgb_before.tobytes()).hexdigest()
        after[name] = hashlib.sha256(rgb_after.tobytes()).hexdigest()
        maximum_difference = max(
            maximum_difference,
            int(np.abs(rgb_before.astype(np.int16) - rgb_after.astype(np.int16)).max()),
        )
    artifact_hashes = frozen_rgb_hashes(rgb_run_dir, iteration)
    return {
        "views": names,
        "rgb_artifact_hashes_before": artifact_hashes,
        "rgb_artifact_hashes_after": frozen_rgb_hashes(rgb_run_dir, iteration),
        "rgb_artifact_hashes_equal": artifact_hashes == frozen_rgb_hashes(rgb_run_dir, iteration),
        "rgb_tensor_hashes_before": before,
        "rgb_tensor_hashes_after": after,
        "rgb_tensor_hashes_equal": before == after,
        "rgb_max_absolute_pixel_difference": maximum_difference,
    }


def main() -> int:
    args = parse_args()
    config = common.load_json(args.config)
    split = common.load_json(args.split)
    oof_split = common.load_json(args.oof_split)
    validate_config(config)
    require(args.labels_root.is_dir(), "SEM-130 authorised training labels are missing")
    access_manifest_path = args.output_dir / "input_access_manifest.json"
    resumable_files = {
        access_manifest_path.name,
        "fold_anchor_support_counts.npz",
        "anchor_support_summary.json",
    }
    if args.audit_rgb_only:
        resumable_files.update({"result.json", "rgb_isolation_after_calibration_failure.json"})
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        require(
            {path.name for path in args.output_dir.iterdir()}.issubset(resumable_files),
            "SEM-130 may resume only its immutable pre-label anchor-support cache",
        )
    require(sha256(args.sem001_checkpoint) == config["inputs"]["sem001_checkpoint_sha256"], "SEM-130 SEM-001 checkpoint hash differs")
    require(sha256(args.sem101_oof_summary) == config["inputs"]["sem101_oof_summary_sha256"], "SEM-130 SEM-101 OOF summary hash differs")
    require(sha256(args.sem110_result) == config["inputs"]["sem110_result_sha256"], "SEM-130 SEM-110 result hash differs")
    require(sha256(args.sem123_result) == config["inputs"]["sem123_result_sha256"], "SEM-130 SEM-123 result hash differs")
    require(common.load_json(args.sem110_result).get("status") == "completed_oof_gate_passed", "SEM-130 requires qualified SEM-110 training OOF")
    require(common.load_json(args.sem123_result).get("status") == "completed_development_sweep", "SEM-130 requires completed SEM-123 recovery registration")

    names = select_rendered_training_views(split, expected_count=int(config["data"]["training_images"]))
    partition = partition_training_names(names, expected_count=len(names))
    fold_by_name, oof_evidence = oof_fold_by_name(oof_split, names)
    base_paths = sem110.validate_base_manifest(
        args.sem001_training_root, names, config["inputs"]["sem001_checkpoint_sha256"]
    )
    teacher_paths = sem110.validate_oof_teacher(args.sem101_oof_root, oof_split, names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_config_sha256_before_execution": sha256(args.config),
        "partition": partition,
        "training_views": len(names),
        "crossview_source_contract": oof_evidence,
        "inputs_validated_before_labels": {
            "sem001_training_predictions": True,
            "sem101_spatial_oof_probabilities": True,
            "sem101_oof_summary": True,
            "sem110_training_result": True,
            "sem123_recovery_registration": True,
            "frozen_geometry_camera_metadata": True,
            "fixed_validation_views_opened": False,
            "real_rgb_input_opened": False,
            "unlabeled_images_opened": False,
            "hidden_test_input_opened": False,
            "external_camera_mode_present": False,
        },
    }
    if access_manifest_path.exists():
        require(
            common.load_json(access_manifest_path) == input_manifest,
            "SEM-130 resumable pre-label input manifest differs from current registration",
        )
    else:
        common.write_json(access_manifest_path, input_manifest)

    import torch

    frozen = config["frozen_rgb"]
    settings = ScaffoldSettings(
        train_camera_count=int(frozen["train_camera_count"]),
        n_offsets=int(frozen["n_offsets"]),
    )
    point_cloud = args.rgb_run_dir / "point_cloud" / f"iteration_{frozen['iteration']}" / "point_cloud.ply"
    configure_scaffold_import(args.source_root)
    model = load_frozen_scaffold(args.source_root, point_cloud, settings)
    require(
        not any(parameter.requires_grad for _name, parameter in frozen_rgb_named_parameters(model)),
        "SEM-130 frozen RGB model unexpectedly has trainable parameters",
    )
    pipe = pipeline()
    extension = load_visibility_extension(args.extension_root)
    anchor_count = int(model.get_anchor.shape[0])
    started = time.perf_counter()

    if args.audit_rgb_only:
        previous_result_path = args.output_dir / "result.json"
        require(previous_result_path.is_file(), "SEM-130 RGB audit requires a recorded calibration-stop result")
        previous = common.load_json(previous_result_path)
        require(
            previous.get("status") == "completed_calibration_gate_failed",
            "SEM-130 RGB audit is permitted only after calibration stopped before check",
        )
        audit = _rgb_isolation(
            names=[str(name) for name in config["outputs"]["training_visual_samples"]],
            data_root=args.data_root,
            appearance_camera_id=int(frozen["appearance_camera_id"]),
            model=model,
            pipe=pipe,
            extension=extension,
            anchor_count=anchor_count,
            rgb_run_dir=args.rgb_run_dir,
            iteration=int(frozen["iteration"]),
        )
        audit.update(
            {
                "schema_version": 1,
                "experiment_id": config["experiment_id"],
                "purpose": "Post-calibration-stop frozen RGB isolation audit; no labels, validation views, unlabelled imagery or external camera input.",
                "labels_opened": False,
                "fixed_validation_views_opened": False,
                "result_sha256_before_audit": sha256(previous_result_path),
            }
        )
        common.write_json(args.output_dir / "rgb_isolation_after_calibration_failure.json", audit)
        print(
            json.dumps(
                {
                    "status": "completed_rgb_isolation_after_calibration_failure",
                    "rgb_max_absolute_pixel_difference": audit["rgb_max_absolute_pixel_difference"],
                    "rgb_hashes_equal": audit["rgb_tensor_hashes_equal"],
                },
                sort_keys=True,
            )
        )
        return 0

    rule = config["crossview_consensus"]
    support_path = args.output_dir / "fold_anchor_support_counts.npz"
    support_summary_path = args.output_dir / "anchor_support_summary.json"
    if support_path.exists() or support_summary_path.exists():
        require(
            support_path.is_file() and support_summary_path.is_file(),
            "SEM-130 cached anchor support requires both its NPZ and summary",
        )
        with np.load(support_path) as archive:
            require(archive.files == ["support_view_counts"], "SEM-130 cached support NPZ schema differs")
            support_counts = archive["support_view_counts"]
        require(
            support_counts.dtype == np.uint16
            and support_counts.shape == (max(fold_by_name.values()) + 1, anchor_count),
            "SEM-130 cached support count shape differs from frozen geometry/folds",
        )
        cached_summary = common.load_json(support_summary_path)
        require(
            cached_summary.get("experiment_id") == config["experiment_id"]
            and int(cached_summary.get("anchor_count", -1)) == anchor_count
            and cached_summary.get("support_counts_sha256") == sha256(support_path),
            "SEM-130 cached support summary does not match its frozen count table",
        )
        print(json.dumps({"phase": "unlabelled_anchor_evidence", "resumed_cached_support": True}, sort_keys=True), flush=True)
    else:
        support_counts, support_summary = collect_fold_support_counts(
            names=names,
            fold_by_name=fold_by_name,
            teacher_paths=teacher_paths,
            data_root=args.data_root,
            appearance_camera_id=int(frozen["appearance_camera_id"]),
            model=model,
            pipe=pipe,
            extension=extension,
            anchor_count=anchor_count,
            minimum_dominant_pixels=int(rule["minimum_dominant_pixels_per_source_view"]),
            minimum_mean_probability=float(rule["minimum_mean_oof_stay_cable_probability"]),
        )
        np.savez_compressed(support_path, support_view_counts=support_counts)
        common.write_json(
            support_summary_path,
            {
                "schema_version": 1,
                "experiment_id": config["experiment_id"],
                "anchor_count": anchor_count,
                "minimum_dominant_pixels_per_source_view": int(rule["minimum_dominant_pixels_per_source_view"]),
                "minimum_mean_oof_stay_cable_probability": float(rule["minimum_mean_oof_stay_cable_probability"]),
                "support_counts_path": support_path.name,
                "support_counts_sha256": sha256(support_path),
                **support_summary,
            },
        )

    support_grid = [int(value) for value in rule["minimum_source_view_grid"]]
    score_weights = config["selection"]["score_weights"]
    calibration = evaluate_grid(
        names=partition["calibration"],
        labels_root=args.labels_root,
        base_paths=base_paths,
        teacher_paths=teacher_paths,
        fold_by_name=fold_by_name,
        fold_support_counts=support_counts,
        data_root=args.data_root,
        appearance_camera_id=int(frozen["appearance_camera_id"]),
        model=model,
        pipe=pipe,
        extension=extension,
        anchor_count=anchor_count,
        rule=rule,
        support_grid=support_grid,
        score_weights=score_weights,
        phase="calibration",
    )
    selected = select_calibration_candidate(
        calibration["candidates"],
        calibration["hard030"],
        calibration["raw_sem123_p20_d02_any"],
        config["selection"]["calibration_gates"],
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "parent_experiment": config["parent_experiment"],
        "seed": config["seed"],
        "input_access_manifest_sha256": sha256(access_manifest_path),
        "anchor_support_summary_sha256": sha256(support_summary_path),
        "anchor_count": anchor_count,
        "training_partition": partition,
        "calibration": calibration,
        "fixed_validation_views_opened": False,
        "real_rgb_input_opened": False,
        "unlabeled_images_opened": False,
        "hidden_test_input_opened": False,
        "external_camera_mode_present": False,
        "rgb_modified": False,
        "fallback": "RGB-030-FULL Scaffold-GS + SEM-001",
    }
    if selected is None:
        result.update(
            {
                "status": "completed_calibration_gate_failed",
                "selected_minimum_source_views": None,
                "check_labels_opened": False,
                "next_step": "Drop SEM-130: no registered cross-view predicate qualified on calibration. Do not read the check or 74-view development labels.",
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        common.write_json(args.output_dir / "result.json", result)
        print(json.dumps({"status": result["status"], "selected_minimum_source_views": None}, sort_keys=True))
        return 0

    selected_support = int(selected["minimum_source_views"])
    check = evaluate_grid(
        names=partition["check"],
        labels_root=args.labels_root,
        base_paths=base_paths,
        teacher_paths=teacher_paths,
        fold_by_name=fold_by_name,
        fold_support_counts=support_counts,
        data_root=args.data_root,
        appearance_camera_id=int(frozen["appearance_camera_id"]),
        model=model,
        pipe=pipe,
        extension=extension,
        anchor_count=anchor_count,
        rule=rule,
        support_grid=[selected_support],
        score_weights=score_weights,
        phase="check",
        write_predictions=args.output_dir / "check_predictions",
    )
    check_candidate = check["candidates"][0]
    check_gate = candidate_gate(
        check_candidate,
        check["hard030"],
        check["raw_sem123_p20_d02_any"],
        gates=config["selection"]["check_gates"],
        require_recall_gain=False,
    )
    predictions = check["prediction_outputs"]
    panel_names = [str(name) for name in config["outputs"]["training_visual_samples"]]
    panel_predictions = _write_selected_predictions(
        names=panel_names,
        output=args.output_dir / "training_panel_predictions",
        minimum_source_views=selected_support,
        base_paths=base_paths,
        teacher_paths=teacher_paths,
        fold_by_name=fold_by_name,
        fold_support_counts=support_counts,
        data_root=args.data_root,
        appearance_camera_id=int(frozen["appearance_camera_id"]),
        model=model,
        pipe=pipe,
        extension=extension,
        anchor_count=anchor_count,
        rule=rule,
    )
    panels = _write_training_panels(
        names=panel_names,
        labels_root=args.labels_root,
        predictions=panel_predictions["roots"],
        output=args.output_dir / "training_panels",
    )
    rgb_isolation = _rgb_isolation(
        names=panel_names,
        data_root=args.data_root,
        appearance_camera_id=int(frozen["appearance_camera_id"]),
        model=model,
        pipe=pipe,
        extension=extension,
        anchor_count=anchor_count,
        rgb_run_dir=args.rgb_run_dir,
        iteration=int(frozen["iteration"]),
    )
    passed = bool(check_gate["passed"])
    result.update(
        {
            "status": "completed_oof_check_passed" if passed else "completed_oof_check_failed",
            "selected_minimum_source_views": selected_support,
            "selection_rule": config["selection"]["rule"],
            "selected_calibration_record": selected,
            "check_labels_opened": True,
            "check": {"evaluation": check, "selected_candidate": check_candidate, "gate": check_gate},
            "check_predictions": predictions,
            "training_panels": panels,
            "rgb_isolation": rgb_isolation,
            "runtime_seconds": time.perf_counter() - started,
            "next_step": (
                "A separate SEM-131 developer audit may apply this immutable selected source-count rule to the already-authorised 74 views; do not change SEM-130 parameters."
                if passed
                else "Drop SEM-130: independent training check did not qualify the selected rule. Do not read 74-view development labels or retune the registered source-count grid."
            ),
        }
    )
    common.write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_minimum_source_views": selected_support,
                "check_gate_passed": passed,
                "check_mIoU_with_background": check_candidate["metrics"]["mIoU_with_background"],
                "check_stay_cable_iou": check_candidate["metrics"]["per_class"]["stay_cable"]["iou"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
