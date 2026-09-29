#!/usr/bin/env python3
"""Run the sealed SEM-404 training-label surface-prefix diagnostic."""
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
from src.gaussian_ray_responsibility import (
    decode_raster_buffers,
    replay_pixel,
    responsibility_prediction,
    trace_statistics,
)

CONFIG = ROOT / "experiments/runs/SEM-404-RGB152-SURFACE-PREFIX-DIAG/config.yaml"
STRATA = ("cable_boundary", "cable_interior", "noncable_near", "noncable_far")


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


def deterministic_sample(mask: np.ndarray, limit: int) -> np.ndarray:
    ids = np.flatnonzero(mask.reshape(-1))
    if len(ids) <= limit:
        return ids
    positions = np.linspace(0, len(ids) - 1, limit, dtype=np.int64)
    return ids[positions]


def label_strata(labels: np.ndarray, cable_id: int, boundary: float, far: float) -> dict[str, np.ndarray]:
    from scipy.ndimage import distance_transform_edt

    cable = labels == cable_id
    inside_distance = distance_transform_edt(cable)
    outside_distance = distance_transform_edt(~cable)
    return {
        "cable_boundary": cable & (inside_distance <= boundary),
        "cable_interior": cable & (inside_distance > boundary),
        "noncable_near": (~cable) & (outside_distance <= boundary),
        "noncable_far": (~cable) & (outside_distance > far),
    }


def empty_counts() -> dict[str, int | float]:
    return {
        "count": 0,
        "standard_correct": 0,
        "surface_correct": 0,
        "changed": 0,
        "corrected": 0,
        "harmed": 0,
        "tail_present": 0,
        "front_tail_disagree": 0,
        "true_probability_delta_sum": 0.0,
    }


def summarize_counts(counts: dict[str, int | float]) -> dict[str, int | float]:
    count = int(counts["count"])
    changed = int(counts["changed"])
    corrected = int(counts["corrected"])
    harmed = int(counts["harmed"])
    tail_present = int(counts["tail_present"])
    return {
        **counts,
        "standard_accuracy": float(counts["standard_correct"] / count) if count else 0.0,
        "surface_accuracy": float(counts["surface_correct"] / count) if count else 0.0,
        "accuracy_delta": float((counts["surface_correct"] - counts["standard_correct"]) / count)
        if count
        else 0.0,
        "change_fraction": changed / count if count else 0.0,
        "net_corrections": corrected - harmed,
        "correction_precision": corrected / (corrected + harmed) if corrected + harmed else 0.0,
        "front_tail_disagreement_rate": float(counts["front_tail_disagree"] / tail_present)
        if tail_present
        else 0.0,
        "mean_true_probability_delta": float(counts["true_probability_delta_sum"] / count)
        if count
        else 0.0,
    }


def native_semantic_forward(state, name: str, colors, background):
    """Render one three-channel semantic block and retain its own replay buffers."""
    import diff_gaussian_rasterization

    torch = state.torch
    model = state.model
    view = state.views[name]
    empty = torch.empty(0, device="cuda", dtype=torch.float32)
    return diff_gaussian_rasterization._C.rasterize_gaussians(
        background,
        model.get_xyz.detach(),
        colors.contiguous(),
        model.get_opacity.detach(),
        model.get_scaling.detach(),
        model.get_rotation.detach(),
        1.0,
        empty,
        view.world_view_transform,
        view.full_proj_transform,
        math.tan(view.FoVx * 0.5),
        math.tan(view.FoVy * 0.5),
        int(view.image_height),
        int(view.image_width),
        empty,
        int(model.active_sh_degree),
        view.camera_center,
        False,
        bool(state.pipe.antialiasing),
        bool(state.pipe.debug),
    )


def run(config: dict, output: Path, result: dict) -> None:
    state = base.setup(config)
    torch = state.torch
    protected = set(state.holdout_names) | set(state.split["primary_fold"]["buffer_excluded_all_cameras"])
    if set(state.semantic_names) & protected:
        raise ValueError("SEM404 training labels overlap protected roles")
    common_path = ROOT / config["semantic_common"]
    common = torch.load(common_path, map_location="cpu", weights_only=False)
    logits = common["logits"]
    if len(logits) != len(state.model.get_xyz):
        raise ValueError("SEM400 logits do not match RGB152 geometry")
    gaussian_probability = logits.softmax(1).numpy().astype(np.float64)
    logits_gpu = logits.cuda()
    gaussian_probability_gpu = logits_gpu.softmax(1)
    counts = {name: empty_counts() for name in STRATA}
    available = {name: 0 for name in STRATA}
    rows: list[dict] = []
    semantic_replay_errors: list[float] = []
    prefix_counts: dict[str, list[float]] = {name: [] for name in STRATA}
    effective_counts: dict[str, list[float]] = {name: [] for name in STRATA}
    started = time.monotonic()

    for view_index, name in enumerate(state.semantic_names, 1):
        if name in protected:
            raise RuntimeError(f"Protected camera requested: {name}")
        labels = state.labels[name]
        strata = label_strata(
            labels,
            int(config["class_id_of_interest"]),
            float(config["boundary_distance_px"]),
            float(config["far_noncable_distance_px"]),
        )
        selected: list[tuple[str, int]] = []
        for stratum in STRATA:
            available[stratum] += int(strata[stratum].sum())
            ids = deterministic_sample(strata[stratum], int(config["samples_per_view_per_stratum"]))
            selected.extend((stratum, int(pixel_id)) for pixel_id in ids)

        first_native = native_semantic_forward(
            state,
            name,
            gaussian_probability_gpu[:, :3],
            torch.tensor([1.0, 0.0, 0.0], device="cuda"),
        )
        instances, first_image, _, geom, binning, image_buffer, _ = first_native
        height, width = labels.shape
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
            "n_contrib": decoded.n_contrib.cpu().numpy(),
            "ranges": decoded.ranges.cpu().numpy(),
            "point_list": decoded.point_list.cpu().numpy(),
        }
        first_image_cpu = first_image.cpu().numpy().astype(np.float64)
        del first_native, first_image, geom, binning, image_buffer, decoded

        second_colors = torch.cat(
            [gaussian_probability_gpu[:, 3:], torch.zeros_like(gaussian_probability_gpu[:, :1])],
            dim=1,
        )
        second_native = native_semantic_forward(
            state, name, second_colors, torch.zeros(3, device="cuda")
        )
        second_instances, second_image, _, second_geom, second_binning, second_image_buffer, _ = second_native
        second_decoded = decode_raster_buffers(
            torch,
            second_geom,
            second_binning,
            second_image_buffer,
            len(state.model.get_xyz),
            int(second_instances),
            int(height * width),
        )
        second_cpu = {
            "depths": second_decoded.depths.cpu().numpy(),
            "means2d": second_decoded.means2d.cpu().numpy(),
            "conic_opacity": second_decoded.conic_opacity.cpu().numpy(),
            "n_contrib": second_decoded.n_contrib.cpu().numpy(),
            "ranges": second_decoded.ranges.cpu().numpy(),
            "point_list": second_decoded.point_list.cpu().numpy(),
        }
        second_image_cpu = second_image[:2].cpu().numpy().astype(np.float64)
        native_semantic = np.concatenate([first_image_cpu, second_image_cpu], axis=0)
        del second_native, second_image, second_geom, second_binning, second_image_buffer, second_decoded

        for stratum, pixel_id in selected:
            y, x = divmod(pixel_id, width)
            trace = replay_pixel(
                x,
                y,
                width,
                height,
                decoded_cpu["means2d"],
                decoded_cpu["conic_opacity"],
                decoded_cpu["depths"],
                decoded_cpu["point_list"],
                decoded_cpu["ranges"],
                decoded_cpu["n_contrib"],
                block_x=int(config["block_x"]),
                block_y=int(config["block_y"]),
            )
            prediction = responsibility_prediction(
                trace,
                gaussian_probability,
                surface_mass_fraction=float(config["surface_mass_fraction"]),
            )
            second_trace = replay_pixel(
                x,
                y,
                width,
                height,
                second_cpu["means2d"],
                second_cpu["conic_opacity"],
                second_cpu["depths"],
                second_cpu["point_list"],
                second_cpu["ranges"],
                second_cpu["n_contrib"],
                block_x=int(config["block_x"]),
                block_y=int(config["block_y"]),
            )
            second_prediction = responsibility_prediction(
                second_trace,
                gaussian_probability,
                surface_mass_fraction=float(config["surface_mass_fraction"]),
            )
            replay_error = max(
                float(np.max(np.abs(prediction.standard_probability[:3] - native_semantic[:3, y, x]))),
                float(np.max(np.abs(second_prediction.standard_probability[3:] - native_semantic[3:, y, x]))),
            )
            semantic_replay_errors.append(replay_error)
            target = int(labels[y, x])
            standard_label = int(np.argmax(native_semantic[:, y, x]))
            surface_label = int(np.argmax(prediction.surface_probability))
            tail_label = (
                int(np.argmax(prediction.tail_probability))
                if prediction.tail_probability is not None
                else None
            )
            standard_correct = standard_label == target
            surface_correct = surface_label == target
            counter = counts[stratum]
            counter["count"] += 1
            counter["standard_correct"] += int(standard_correct)
            counter["surface_correct"] += int(surface_correct)
            counter["changed"] += int(standard_label != surface_label)
            counter["corrected"] += int((not standard_correct) and surface_correct)
            counter["harmed"] += int(standard_correct and (not surface_correct))
            counter["tail_present"] += int(tail_label is not None)
            counter["front_tail_disagree"] += int(
                tail_label is not None and surface_label != tail_label
            )
            counter["true_probability_delta_sum"] += float(
                prediction.surface_probability[target] - native_semantic[target, y, x]
            )
            stats = trace_statistics(trace)
            prefix_counts[stratum].append(float(prediction.surface_contributors))
            effective_counts[stratum].append(float(stats["effective_contributors"]))
            rows.append(
                {
                    "name": name,
                    "stratum": stratum,
                    "x": x,
                    "y": y,
                    "target": target,
                    "standard_label": standard_label,
                    "surface_label": surface_label,
                    "tail_label": tail_label,
                    "semantic_replay_max_abs_error": replay_error,
                    "surface_contributors": prediction.surface_contributors,
                    "surface_mass": prediction.surface_mass,
                    "true_probability_delta": float(
                        prediction.surface_probability[target] - native_semantic[target, y, x]
                    ),
                    **stats,
                }
            )
        del native_semantic, decoded_cpu, second_cpu
        if view_index % 10 == 0 or view_index == len(state.semantic_names):
            print(
                json.dumps(
                    {
                        "processed_training_labels": view_index,
                        "total": len(state.semantic_names),
                        "sampled_pixels": len(rows),
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )

    summary_by_stratum = {name: summarize_counts(counts[name]) for name in STRATA}
    boundary = summary_by_stratum["cable_boundary"]
    interior_tail = int(counts["cable_interior"]["tail_present"]) + int(
        counts["noncable_far"]["tail_present"]
    )
    interior_disagreement = int(counts["cable_interior"]["front_tail_disagree"]) + int(
        counts["noncable_far"]["front_tail_disagree"]
    )
    interior_disagreement_rate = interior_disagreement / interior_tail if interior_tail else 0.0
    enrichment = float(boundary["front_tail_disagreement_rate"] - interior_disagreement_rate)
    semantic_error = max(semantic_replay_errors) if semantic_replay_errors else float("inf")
    gates_config = config["gates"]
    cable_interior_drop = float(
        summary_by_stratum["cable_interior"]["standard_accuracy"]
        - summary_by_stratum["cable_interior"]["surface_accuracy"]
    )
    noncable_near_drop = float(
        summary_by_stratum["noncable_near"]["standard_accuracy"]
        - summary_by_stratum["noncable_near"]["surface_accuracy"]
    )
    gates = {
        "semantic_replay": semantic_error <= float(config["semantic_replay_max_abs_error"]),
        "sampled_cable_boundary_pixels": int(boundary["count"])
        >= gates_config["sampled_cable_boundary_pixels_at_least"],
        "changed_cable_boundary_pixels": int(boundary["changed"])
        >= gates_config["changed_cable_boundary_pixels_at_least"],
        "cable_boundary_net_corrections": int(boundary["net_corrections"])
        >= gates_config["cable_boundary_net_corrections_at_least"],
        "cable_boundary_correction_precision": float(boundary["correction_precision"])
        >= gates_config["cable_boundary_correction_precision_at_least"],
        "cable_interior_accuracy_drop": cable_interior_drop
        <= gates_config["cable_interior_accuracy_drop_at_most"],
        "noncable_near_accuracy_drop": noncable_near_drop
        <= gates_config["noncable_near_accuracy_drop_at_most"],
        "boundary_front_tail_disagreement_enrichment": enrichment
        >= gates_config["boundary_front_tail_disagreement_enrichment_at_least"],
    }
    evidence_path = output / "sample_rows.json"
    write(evidence_path, rows)
    summary = {
        "training_labels_read": len(state.semantic_names),
        "confirmation_labels_read": 0,
        "buffer_labels_read": 0,
        "sampled_pixels": len(rows),
        "available_pixels_by_stratum": available,
        "semantic_replay_max_abs_error": semantic_error,
        "strata": summary_by_stratum,
        "interior_front_tail_disagreement_rate": interior_disagreement_rate,
        "boundary_front_tail_disagreement_enrichment": enrichment,
        "cable_interior_accuracy_drop": cable_interior_drop,
        "noncable_near_accuracy_drop": noncable_near_drop,
        "surface_contributor_quantiles": {
            name: {str(level): float(value) for level, value in zip(
                [0.0, 0.25, 0.5, 0.75, 1.0], np.quantile(prefix_counts[name], [0, 0.25, 0.5, 0.75, 1])
            )}
            for name in STRATA
        },
        "effective_contributor_quantiles": {
            name: {str(level): float(value) for level, value in zip(
                [0.0, 0.25, 0.5, 0.75, 1.0], np.quantile(effective_counts[name], [0, 0.25, 0.5, 0.75, 1])
            )}
            for name in STRATA
        },
        "gates": gates,
        "training_authorized": all(gates.values()),
    }
    write(output / "summary.json", summary)
    result.update(
        summary,
        evidence_sha256=sha256(evidence_path),
        summary_sha256=sha256(output / "summary.json"),
        semantic_common_sha256=sha256(common_path),
        diagnostic_seconds=time.monotonic() - started,
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
