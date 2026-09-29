#!/usr/bin/env python3
"""Run the sealed, label-free SEM-406 weighted depth-mode preflight."""
from __future__ import annotations

import argparse
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
from scripts.run_sem403_ray_responsibility_preflight import diagnostic_colors, native_forward, rgb_replay_is_exact
from scripts.run_sem405_depth_layer_preflight import js_divergence_bits, quantiles, read, sha256, write
from src.gaussian_ray_responsibility import decode_raster_buffers, replay_pixel, split_depth_modes

CONFIG = ROOT / "experiments/runs/SEM-406-RGB152-DEPTH-MODE-PREFLIGHT/config.yaml"


def run(config: dict, output: Path, result: dict) -> None:
    state = base.setup(config)
    torch = state.torch
    views = list(config["preflight_views"])
    protected = set(state.holdout_names) | set(state.split["primary_fold"]["buffer_excluded_all_cameras"])
    if not set(views).issubset(state.semantic_names) or set(views) & protected:
        raise ValueError("SEM406 views must be unprotected semantic-training camera poses")
    if set(views) & {"001.png", "120.png", "300.png"}:
        raise ValueError("SEM406 must not reuse SEM405 camera samples")
    if not rgb_replay_is_exact(state, "001.png"):
        raise AssertionError("Frozen RGB152 native replay mismatch before preflight")

    common_path = ROOT / config["semantic_common"]
    common = torch.load(common_path, map_location="cpu", weights_only=False)
    logits = common["logits"]
    if len(logits) != len(state.model.get_xyz):
        raise ValueError("SEM400 logits do not match RGB152 geometry")
    gaussian_probability = logits.softmax(1).numpy().astype(np.float64)
    colors = diagnostic_colors(torch, len(state.model.get_xyz))
    colors_cpu = colors.cpu().numpy().astype(np.float64)

    rows: list[dict] = []
    replay_errors: list[float] = []
    explained: list[float] = []
    clear_js: list[float] = []
    nonempty = eligible = clear = clear_disagree = depth_order_violations = 0
    started = time.monotonic()

    for name in views:
        native, background = native_forward(state, name, colors)
        instances, image, _, geom, binning, image_buffer, _ = native
        height, width = image.shape[1:]
        decoded = decode_raster_buffers(
            torch, geom, binning, image_buffer, len(state.model.get_xyz), int(instances), int(height * width)
        )
        cpu = {
            "depths": decoded.depths.cpu().numpy(),
            "means2d": decoded.means2d.cpu().numpy(),
            "conic_opacity": decoded.conic_opacity.cpu().numpy(),
            "n_contrib": decoded.n_contrib.cpu().numpy(),
            "ranges": decoded.ranges.cpu().numpy(),
            "point_list": decoded.point_list.cpu().numpy(),
        }
        image_cpu = image.cpu().numpy().astype(np.float64)
        background_cpu = background.cpu().numpy().astype(np.float64)
        xs = np.rint(np.linspace(0, width - 1, int(config["sample_grid_x"]))).astype(np.int64)
        ys = np.rint(np.linspace(0, height - 1, int(config["sample_grid_y"]))).astype(np.int64)
        before = len(rows)
        for y in ys:
            for x in xs:
                trace = replay_pixel(
                    int(x), int(y), int(width), int(height), cpu["means2d"], cpu["conic_opacity"],
                    cpu["depths"], cpu["point_list"], cpu["ranges"], cpu["n_contrib"],
                    block_x=int(config["block_x"]), block_y=int(config["block_y"]),
                )
                nonempty += int(bool(len(trace.weights)))
                violations = int(np.sum(np.diff(trace.depths) < -1e-6))
                depth_order_violations += violations
                replay_color = (
                    (colors_cpu[trace.gaussian_ids] * trace.weights[:, None]).sum(axis=0)
                    if len(trace.weights) else np.zeros(3, dtype=np.float64)
                ) + trace.final_transmittance * background_cpu
                replay_error = float(np.max(np.abs(replay_color - image_cpu[:, int(y), int(x)])))
                replay_errors.append(replay_error)
                mode = split_depth_modes(
                    trace,
                    gaussian_probability,
                    minimum_side_mass_fraction=float(config["minimum_side_mass_fraction"]),
                    minimum_explained_depth_variance=float(config["minimum_explained_depth_variance"]),
                )
                eligible += int(mode.eligible)
                js = None
                front_label = None
                back_label = None
                if mode.eligible:
                    explained.append(mode.explained_depth_variance)
                if mode.clear:
                    clear += 1
                    front_label = int(np.argmax(mode.front_probability))
                    back_label = int(np.argmax(mode.back_probability))
                    clear_disagree += int(front_label != back_label)
                    js = js_divergence_bits(mode.front_probability, mode.back_probability)
                    clear_js.append(js)
                rows.append({
                    "name": name,
                    "x": int(x),
                    "y": int(y),
                    "accepted_contributors": int(len(trace.weights)),
                    "color_max_abs_error": replay_error,
                    "depth_order_violations": violations,
                    "mass_eligible": mode.eligible,
                    "clear_two_mode": mode.clear,
                    "split_index": mode.split_index,
                    "front_mass_fraction": mode.front_mass_fraction,
                    "back_mass_fraction": mode.back_mass_fraction,
                    "front_depth_mean": mode.front_depth_mean,
                    "back_depth_mean": mode.back_depth_mean,
                    "explained_depth_variance": mode.explained_depth_variance,
                    "front_label": front_label,
                    "back_label": back_label,
                    "mode_js_divergence_bits": js,
                })
        print(json.dumps({"completed_view": name, "sampled_pixels": len(rows) - before}), flush=True)

    sample_count = len(rows)
    nonempty_fraction = nonempty / sample_count
    eligible_fraction = eligible / sample_count
    clear_fraction = clear / sample_count
    disagreement = clear_disagree / clear if clear else 0.0
    js_quantiles = quantiles(clear_js)
    replay_max = max(replay_errors)
    gate_config = config["gates"]
    gates = {
        "color_replay": replay_max <= gate_config["color_replay_max_abs_error_at_most"],
        "nonempty_trace_fraction": nonempty_fraction >= gate_config["nonempty_trace_fraction_at_least"],
        "mass_eligible_split_fraction": eligible_fraction >= gate_config["mass_eligible_split_fraction_at_least"],
        "clear_two_mode_fraction": clear_fraction >= gate_config["clear_two_mode_fraction_at_least"],
        "clear_mode_argmax_disagreement": disagreement >= gate_config["clear_mode_argmax_disagreement_at_least"],
        "clear_mode_js_divergence_p75": js_quantiles["0.75"] >= gate_config["clear_mode_js_divergence_p75_at_least"],
        "depth_order": depth_order_violations <= gate_config["depth_order_violation_count_at_most"],
    }
    evidence_path = output / "sample_modes.json"
    write(evidence_path, rows)
    summary = {
        "labels_read": 0,
        "protected_cameras_read": 0,
        "sem405_cameras_read": 0,
        "sampled_pixels": sample_count,
        "nonempty_traces": nonempty,
        "nonempty_trace_fraction": nonempty_fraction,
        "mass_eligible_splits": eligible,
        "mass_eligible_split_fraction": eligible_fraction,
        "clear_two_mode_traces": clear,
        "clear_two_mode_fraction": clear_fraction,
        "clear_mode_argmax_disagreements": clear_disagree,
        "clear_mode_argmax_disagreement_rate": disagreement,
        "eligible_explained_depth_variance_quantiles": quantiles(explained),
        "clear_mode_js_divergence_bits_quantiles": js_quantiles,
        "color_replay_max_abs_error": replay_max,
        "depth_order_violation_count": depth_order_violations,
        "rgb_png_exact_before": True,
        "gates": gates,
        "label_diagnostic_authorized": all(gates.values()),
    }
    if not rgb_replay_is_exact(state, "001.png"):
        raise AssertionError("Frozen RGB152 native replay mismatch after preflight")
    summary["rgb_png_exact_after"] = True
    write(output / "summary.json", summary)
    result.update(
        summary,
        evidence_sha256=sha256(evidence_path),
        summary_sha256=sha256(output / "summary.json"),
        diagnostic_seconds=time.monotonic() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="r1")
    args = parser.parse_args()
    config = read(CONFIG)
    output = ROOT / config["output"] / f"preflight-{args.attempt}"
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "experiment_id": config["experiment_id"],
        "status": "starting",
        "seed": config["seed"],
        "config_sha256": sha256(CONFIG),
        "script_sha256": sha256(Path(__file__)),
        "split_sha256": sha256(ROOT / config["split_path"]),
        "semantic_common_sha256": sha256(ROOT / config["semantic_common"]),
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
