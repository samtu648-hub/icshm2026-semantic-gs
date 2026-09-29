#!/usr/bin/env python3
"""Run the sealed, read-only SEM-402 rasterized contribution diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sem398_boundary_split as base
from src.semantic_intraview_conflict import mask_signal, select_conflicted_parents

CONFIG = ROOT / "experiments/runs/SEM-402-RGB152-INTRAVIEW-MASK-CONFLICT-DIAG/config.yaml"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {str(level): 0.0 for level in [0.0, 0.25, 0.5, 0.75, 1.0]}
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    result = np.quantile(values, levels)
    return {str(level): float(value) for level, value in zip(levels, result)}


def rgb_replay_is_exact(state, name: str) -> bool:
    torch = state.torch
    with torch.no_grad():
        rendered = state.render(state.views[name], state.model, state.pipe, state.bg)["render"]
        actual = state.np.rint(rendered.permute(1, 2, 0).cpu().numpy() * 255).astype(state.np.uint8)
    expected = state.np.asarray(state.Image.open(ROOT / state.config["parent_export"] / name))
    return bool(state.np.array_equal(actual, expected))


def run(config: dict, output: Path, result: dict) -> None:
    state = base.setup(config)
    torch = state.torch
    if set(state.semantic_names) & set(state.holdout_names):
        raise ValueError("Training and confirmation label roles overlap")
    if not rgb_replay_is_exact(state, "001.png"):
        raise AssertionError("Frozen RGB152 native replay mismatch before diagnostic")

    common_path = ROOT / config["semantic_common"]
    common = torch.load(common_path, map_location="cpu", weights_only=False)
    evidence = common["evidence"]
    eligible = evidence["eligible"].numpy().astype(bool)
    previous = evidence["candidate"].numpy().astype(np.int64)
    point_count = len(state.model.get_xyz)
    if len(eligible) != point_count or len(previous) != config["split_budget"]:
        raise ValueError("Sealed SEM400 evidence does not match the RGB152 parent")
    if int(eligible.sum()) < config["split_budget"]:
        raise ValueError("Insufficient sealed eligible parents")

    for name in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
        getattr(state.model, name).requires_grad_(False)
    probe = torch.ones(point_count, device="cuda", dtype=torch.float32, requires_grad=True)
    zeros = torch.zeros_like(probe)
    override = torch.stack([probe, zeros, zeros], dim=1).contiguous()
    geometry = base.GeometryView(state.model)

    signal_sum = torch.zeros(point_count, device="cuda", dtype=torch.float64)
    foreground_sum = torch.zeros_like(signal_sum)
    background_sum = torch.zeros_like(signal_sum)
    visible_views = torch.zeros(point_count, device="cuda", dtype=torch.int32)
    ambiguous_views = torch.zeros_like(visible_views)
    visibility_epsilon = float(config["contribution_visibility_epsilon"])
    signal_epsilon = float(config["mask_signal_epsilon"])
    threshold = float(config["mask_signal_threshold"])
    minimum_gradient = 0.0
    started = time.monotonic()

    for index, name in enumerate(state.semantic_names, 1):
        if name in state.holdout_names:
            raise RuntimeError(f"Forbidden confirmation label requested: {name}")
        target = torch.from_numpy(state.labels[name].astype("int64")).cuda()
        cable = target == int(config["class_id_of_interest"])
        rendered = state.render(
            state.views[name], geometry, state.pipe, state.bg, override_color=override
        )["render"][0]
        if rendered.shape != target.shape:
            raise ValueError(f"Rendered/label shape mismatch for {name}")
        foreground = torch.autograd.grad(rendered[cable].sum(), probe, retain_graph=True)[0]
        background = torch.autograd.grad(rendered[~cable].sum(), probe, retain_graph=False)[0]
        minimum_gradient = min(minimum_gradient, float(foreground.min()), float(background.min()))
        if minimum_gradient < -1e-6:
            raise FloatingPointError(f"Rasterized color contribution became negative: {minimum_gradient}")
        foreground = foreground.detach().double().clamp_min_(0)
        background = background.detach().double().clamp_min_(0)
        total = foreground + background
        visible = total > visibility_epsilon
        signal = (foreground - background).abs() / (total + signal_epsilon)
        foreground_sum.add_(foreground)
        background_sum.add_(background)
        signal_sum[visible] += signal[visible]
        visible_views.add_(visible.to(visible_views.dtype))
        ambiguous_views.add_((visible & (signal < threshold)).to(ambiguous_views.dtype))
        if index % 25 == 0 or index == len(state.semantic_names):
            print(
                json.dumps(
                    {
                        "processed_training_labels": index,
                        "total": len(state.semantic_names),
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )

    signal_sum_np = signal_sum.cpu().numpy()
    foreground_np = foreground_sum.cpu().numpy()
    background_np = background_sum.cpu().numpy()
    visible_np = visible_views.cpu().numpy()
    ambiguous_np = ambiguous_views.cpu().numpy()
    selection = select_conflicted_parents(
        signal_sum_np,
        visible_np,
        ambiguous_np,
        eligible,
        signal_threshold=threshold,
        minimum_visible_views=int(config["minimum_visible_views"]),
        minimum_ambiguous_views=int(config["minimum_ambiguous_views"]),
    )
    candidate = selection.ranked_eligible[: config["split_budget"]]
    overlap = int(np.intersect1d(candidate, previous, assume_unique=True).size)
    different = int(config["split_budget"] - overlap)
    qualified_ids = np.flatnonzero(selection.qualified)
    qualified_visible_median = float(np.median(visible_np[qualified_ids])) if len(qualified_ids) else 0.0
    covered = eligible & (visible_np >= int(config["minimum_visible_views"]))
    gates_config = config["gates"]
    gates = {
        "qualified_parents": len(qualified_ids) >= gates_config["qualified_parents_at_least"],
        "median_visible_views": qualified_visible_median
        >= gates_config["median_visible_views_at_least"],
        "different_full_budget_parents": different
        >= gates_config["different_full_budget_parents_at_least"],
    }
    training_authorized = all(gates.values())

    evidence_path = output / "evidence.npz"
    np.savez_compressed(
        evidence_path,
        eligible=eligible,
        previous_selection=previous,
        candidate_selection=candidate,
        foreground_mass=foreground_np.astype(np.float32),
        background_mass=background_np.astype(np.float32),
        visible_views=visible_np.astype(np.uint16),
        ambiguous_views=ambiguous_np.astype(np.uint16),
        mean_mask_signal=selection.mean_signal.astype(np.float32),
        qualified=selection.qualified,
    )
    summary = {
        "point_count": point_count,
        "eligible_parents": int(eligible.sum()),
        "training_labels_read": len(state.semantic_names),
        "confirmation_labels_read": 0,
        "covered_eligible_parents": int(covered.sum()),
        "qualified_parents": int(len(qualified_ids)),
        "qualified_visible_views_median": qualified_visible_median,
        "full_budget_overlap_with_sem398": overlap,
        "different_full_budget_parents": different,
        "covered_mean_mask_signal_quantiles": quantiles(selection.mean_signal[covered]),
        "covered_visible_view_quantiles": quantiles(visible_np[covered]),
        "covered_ambiguous_view_quantiles": quantiles(ambiguous_np[covered]),
        "eligible_foreground_mass_positive": int((eligible & (foreground_np > visibility_epsilon)).sum()),
        "eligible_background_mass_positive": int((eligible & (background_np > visibility_epsilon)).sum()),
        "minimum_raw_contribution_gradient": minimum_gradient,
        "rgb_png_exact_before": True,
        "gates": gates,
        "training_authorized": training_authorized,
        "interpretation_scope": "within_view_rasterized_alpha_transmittance_contribution_conflict",
    }
    if not rgb_replay_is_exact(state, "001.png"):
        raise AssertionError("Frozen RGB152 native replay mismatch after diagnostic")
    summary["rgb_png_exact_after"] = True
    write(output / "summary.json", summary)
    result.update(
        summary,
        evidence_sha256=sha256(evidence_path),
        summary_sha256=sha256(output / "summary.json"),
        semantic_common_sha256=sha256(common_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="r1")
    args = parser.parse_args()
    config = read(CONFIG)
    output = ROOT / config["output"] / f"diagnostic-{args.attempt}"
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "experiment_id": config["experiment_id"],
        "status": "starting",
        "seed": config["seed"],
        "config_sha256": sha256(CONFIG),
        "script_sha256": sha256(Path(__file__)),
        "split_sha256": sha256(ROOT / config["split_path"]),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    write(output / "started.json", result)
    started = time.monotonic()
    try:
        run(config, output, result)
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed_preserved", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write(output / "result.json", result)


if __name__ == "__main__":
    main()
