#!/usr/bin/env python3
"""Run the sealed, label-free SEM-403 native ray replay preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sem398_boundary_split as base
from src.gaussian_ray_responsibility import decode_raster_buffers, replay_pixel, trace_statistics

CONFIG = ROOT / "experiments/runs/SEM-403-RGB152-RAY-RESPONSIBILITY-PREFLIGHT/config.yaml"


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


def quantiles(values: list[float]) -> dict[str, float]:
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    if not values:
        return {str(level): 0.0 for level in levels}
    result = np.quantile(np.asarray(values, dtype=np.float64), levels)
    return {str(level): float(value) for level, value in zip(levels, result)}


def rgb_replay_is_exact(state, name: str) -> bool:
    torch = state.torch
    with torch.no_grad():
        rendered = state.render(state.views[name], state.model, state.pipe, state.bg)["render"]
        actual = state.np.rint(rendered.permute(1, 2, 0).cpu().numpy() * 255).astype(state.np.uint8)
    expected = state.np.asarray(state.Image.open(ROOT / state.config["parent_export"] / name))
    return bool(state.np.array_equal(actual, expected))


def diagnostic_colors(torch, point_count: int):
    ids = torch.arange(point_count, device="cuda", dtype=torch.int64)
    return torch.stack(
        [
            (ids % 251).float() / 250.0,
            ((ids // 251) % 241).float() / 240.0,
            ((ids // (251 * 241)) % 239).float() / 238.0,
        ],
        dim=1,
    ).contiguous()


def native_forward(state, name: str, colors):
    import diff_gaussian_rasterization

    torch = state.torch
    model = state.model
    view = state.views[name]
    height, width = int(view.image_height), int(view.image_width)
    tanfovx = math.tan(view.FoVx * 0.5)
    tanfovy = math.tan(view.FoVy * 0.5)
    empty = torch.empty(0, device="cuda", dtype=torch.float32)
    background = torch.tensor([0.05, 0.10, 0.15], device="cuda", dtype=torch.float32)
    args = (
        background,
        model.get_xyz.detach(),
        colors,
        model.get_opacity.detach(),
        model.get_scaling.detach(),
        model.get_rotation.detach(),
        1.0,
        empty,
        view.world_view_transform,
        view.full_proj_transform,
        tanfovx,
        tanfovy,
        height,
        width,
        empty,
        int(model.active_sh_degree),
        view.camera_center,
        False,
        bool(state.pipe.antialiasing),
        bool(state.pipe.debug),
    )
    return diff_gaussian_rasterization._C.rasterize_gaussians(*args), background


def run(config: dict, output: Path, result: dict) -> None:
    state = base.setup(config)
    torch = state.torch
    views = list(config["preflight_views"])
    if not set(views).issubset(state.semantic_names):
        raise ValueError("SEM403 preflight views must be semantic-training cameras")
    if set(views) & (set(state.holdout_names) | set(state.split["primary_fold"]["buffer_excluded_all_cameras"])):
        raise ValueError("SEM403 preflight attempted a protected camera")
    if not rgb_replay_is_exact(state, "001.png"):
        raise AssertionError("Frozen RGB152 native replay mismatch before preflight")

    colors = diagnostic_colors(torch, len(state.model.get_xyz))
    colors_cpu = colors.cpu().numpy().astype(np.float64)
    rows: list[dict] = []
    view_summaries: list[dict] = []
    all_color_errors: list[float] = []
    all_t_errors: list[float] = []
    all_depth_errors: list[float] = []
    all_stats: list[dict] = []
    nonempty = 0
    depth_order_violations = 0
    started = time.monotonic()

    for name in views:
        native, background = native_forward(state, name, colors)
        instances, image, _, geom, binning, image_buffer, invdepth = native
        height, width = image.shape[1:]
        decoded = decode_raster_buffers(
            torch,
            geom,
            binning,
            image_buffer,
            len(state.model.get_xyz),
            int(instances),
            int(height * width),
        )
        decoded_cpu = {
            "depths": decoded.depths.cpu().numpy(),
            "means2d": decoded.means2d.cpu().numpy(),
            "conic_opacity": decoded.conic_opacity.cpu().numpy(),
            "final_transmittance": decoded.final_transmittance.cpu().numpy(),
            "n_contrib": decoded.n_contrib.cpu().numpy(),
            "ranges": decoded.ranges.cpu().numpy(),
            "point_list": decoded.point_list.cpu().numpy(),
        }
        image_cpu = image.cpu().numpy().astype(np.float64)
        invdepth_cpu = invdepth[0].cpu().numpy().astype(np.float64)
        background_cpu = background.cpu().numpy().astype(np.float64)
        xs = np.rint(np.linspace(0, width - 1, int(config["sample_grid_x"]))).astype(np.int64)
        ys = np.rint(np.linspace(0, height - 1, int(config["sample_grid_y"]))).astype(np.int64)
        before_rows = len(rows)
        for y in ys:
            for x in xs:
                trace = replay_pixel(
                    int(x),
                    int(y),
                    int(width),
                    int(height),
                    decoded_cpu["means2d"],
                    decoded_cpu["conic_opacity"],
                    decoded_cpu["depths"],
                    decoded_cpu["point_list"],
                    decoded_cpu["ranges"],
                    decoded_cpu["n_contrib"],
                    block_x=int(config["block_x"]),
                    block_y=int(config["block_y"]),
                )
                stats = trace_statistics(trace)
                all_stats.append(stats)
                if len(trace.weights):
                    nonempty += 1
                violations = int(np.sum(np.diff(trace.depths) < -1e-6))
                depth_order_violations += violations
                replay_color = (
                    (colors_cpu[trace.gaussian_ids] * trace.weights[:, None]).sum(axis=0)
                    if len(trace.weights)
                    else np.zeros(3, dtype=np.float64)
                ) + trace.final_transmittance * background_cpu
                replay_invdepth = float(np.sum(trace.weights / trace.depths)) if len(trace.weights) else 0.0
                native_t = float(decoded_cpu["final_transmittance"][int(y) * width + int(x)])
                color_error = float(np.max(np.abs(replay_color - image_cpu[:, int(y), int(x)])))
                t_error = abs(trace.final_transmittance - native_t)
                depth_error = abs(replay_invdepth - float(invdepth_cpu[int(y), int(x)]))
                all_color_errors.append(color_error)
                all_t_errors.append(t_error)
                all_depth_errors.append(depth_error)
                rows.append(
                    {
                        "name": name,
                        "x": int(x),
                        "y": int(y),
                        "color_max_abs_error": color_error,
                        "transmittance_abs_error": t_error,
                        "inverse_depth_abs_error": depth_error,
                        "depth_order_violations": violations,
                        **stats,
                    }
                )
        view_summaries.append(
            {
                "name": name,
                "width": int(width),
                "height": int(height),
                "instances": int(instances),
                "geom_buffer_bytes": int(geom.numel()),
                "binning_buffer_bytes": int(binning.numel()),
                "image_buffer_bytes": int(image_buffer.numel()),
                "sampled_pixels": len(rows) - before_rows,
            }
        )
        print(json.dumps({"completed_view": name, "sampled_pixels": len(rows) - before_rows}), flush=True)

    sample_count = len(rows)
    nonempty_fraction = nonempty / sample_count
    gates_config = config["gates"]
    maxima = {
        "color": max(all_color_errors),
        "transmittance": max(all_t_errors),
        "inverse_depth": max(all_depth_errors),
    }
    gates = {
        "color_replay": maxima["color"] <= gates_config["color_replay_max_abs_error_at_most"],
        "transmittance_replay": maxima["transmittance"]
        <= gates_config["transmittance_replay_max_abs_error_at_most"],
        "inverse_depth_replay": maxima["inverse_depth"]
        <= gates_config["inverse_depth_replay_max_abs_error_at_most"],
        "nonempty_trace_fraction": nonempty_fraction >= gates_config["nonempty_trace_fraction_at_least"],
        "depth_order": depth_order_violations <= gates_config["depth_order_violation_count_at_most"],
    }
    evidence_path = output / "sample_traces.json"
    write(evidence_path, rows)
    summary = {
        "labels_read": 0,
        "protected_cameras_read": 0,
        "views": view_summaries,
        "sampled_pixels": sample_count,
        "nonempty_traces": nonempty,
        "nonempty_trace_fraction": nonempty_fraction,
        "depth_order_violation_count": depth_order_violations,
        "color_replay_max_abs_error": maxima["color"],
        "transmittance_replay_max_abs_error": maxima["transmittance"],
        "inverse_depth_replay_max_abs_error": maxima["inverse_depth"],
        "accepted_contributor_quantiles": quantiles([float(row["accepted_contributors"]) for row in all_stats]),
        "top_share_quantiles": quantiles([float(row["top_share"]) for row in all_stats if row["accepted_contributors"]]),
        "effective_contributor_quantiles": quantiles(
            [float(row["effective_contributors"]) for row in all_stats if row["accepted_contributors"]]
        ),
        "depth_std_quantiles": quantiles([float(row["depth_std"]) for row in all_stats if row["accepted_contributors"]]),
        "prefix80_contributor_quantiles": quantiles(
            [float(row["prefix80_contributors"]) for row in all_stats if row["accepted_contributors"]]
        ),
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
