#!/usr/bin/env python3
"""Run the sealed SEM-395 frozen cross-view reliability diagnostic."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import CLASSES, sha256, verify_hashes, write_new_json
from src.quality_semantic_audit import independent_confusion, metrics
from src.semantic_crossview_reliability import (
    PAIR_STAGE_NAMES,
    build_confirmation_candidate,
    combine_source_evidence,
    depth_proxy,
    evaluate_pair,
    replacement_counts,
    select_diagnostic_plan,
)

EID = "SEM-395-RGB150-CROSSVIEW-RELIABILITY-DIAG"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/quality/SEM-395/r1"
OUTPUT = DEFAULT_OUTPUT
CODE_PATHS = (
    "scripts/run_sem395_crossview_reliability.py",
    "scripts/prepare_rgb_colmap_scene.py",
    "src/camera_adapter.py",
    "src/quality_backbone.py",
    "src/semantic_crossview_reliability.py",
    "src/quality_cameras.py",
    "src/quality_native_gaussian.py",
    "src/quality_protocol.py",
    "src/quality_semantic_audit.py",
    "tests/test_sem395_crossview_reliability.py",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_path(relative: str) -> Path:
    path = ROOT / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(
        ROOT.resolve()
    ):
        raise ValueError(f"Repository-relative path required: {relative}")
    return path


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1 or config.get("experiment_id") != EID:
        raise ValueError("SEM-395 config identity changed")
    if config.get("training") is not False or config.get("seed") != 395:
        raise ValueError("SEM-395 is a fixed seed-395 no-training diagnostic")
    access = config["access_contract"]
    if (
        access["target_count"] != 24
        or access["sources_per_target"] != 2
        or access["maximum_pairs"] != 48
        or access["holdout_rgb_access"] is not False
        or access["holdout_label_access"] is not False
        or access["unlabeled_supervision"] is not False
    ):
        raise ValueError("SEM-395 access contract changed")
    if config["reliability_filter"]["schemes_tested"] != 1:
        raise ValueError("Exactly one reliability filter is permitted")
    if config["budget"]["maximum_pairs"] != access["maximum_pairs"]:
        raise ValueError("SEM-395 pair budgets disagree")
    if CLASSES[2] != "stay_cable":
        raise ValueError("Class ID 2 name changed")
    expected_outcomes = {
        "reliability_filter_has_positive_net_value_and_warrants_a_training_or_fusion_candidate",
        "reliability_filter_is_correct_but_too_sparse",
        "current_rgb150_depth_or_geometry_is_insufficient",
        "semantic_or_annotation_inconsistency_dominates",
    }
    observed = {
        config["outcome_rules"][key]
        for key in (
            "reliable_positive_net_and_coverage",
            "positive_but_too_sparse",
            "geometry_dominant_failure",
            "semantic_or_annotation_dominant_failure",
        )
    }
    if observed != expected_outcomes:
        raise ValueError("Outcome vocabulary changed")
    for key in (
        "dataset_manifest",
        "split_path",
        "camera_graph",
        "trajectory_segments",
        "protocol",
        "label_root",
        "label_manifest",
        "output_path",
    ):
        safe_path(config[key])
    for parent in config["parents"].values():
        for key, value in parent.items():
            if key in {
                "run",
                "scene",
                "source",
                "training_result",
                "evaluation_result",
                "point_cloud",
                "saved_renders",
                "checkpoint",
                "probabilities",
                "reproduction_result",
            }:
                safe_path(value)
    if safe_path(config["output_path"]) != DEFAULT_OUTPUT:
        raise ValueError("SEM-395 output root changed")


def code_hashes() -> dict[str, str]:
    return {path: sha256(ROOT / path) for path in CODE_PATHS}


def output_bytes() -> int:
    return sum(path.stat().st_size for path in OUTPUT.rglob("*") if path.is_file())


def enforce_budget(
    config: dict[str, Any], started: float, *, gpu_stage: bool = False
) -> None:
    elapsed_minutes = (time.monotonic() - started) / 60
    maximum_minutes = config["budget"][
        "maximum_gpu_minutes" if gpu_stage else "maximum_cpu_minutes"
    ]
    if elapsed_minutes > maximum_minutes:
        raise RuntimeError(
            f"SEM-395 {'GPU' if gpu_stage else 'CPU'} time budget exceeded"
        )
    if output_bytes() > config["budget"]["maximum_output_gib"] * 2**30:
        raise RuntimeError("SEM-395 output budget exceeded")


def git_context() -> dict[str, Any]:
    diff = subprocess.check_output(
        ["git", "diff", "--no-ext-diff", "--binary", "HEAD"], cwd=ROOT
    )
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "tracked_worktree_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "note": "The full dirty-worktree implementation is bound by per-file SHA256; the base commit alone is not the implementation identity.",
    }


def verify_parent_inputs(config: dict[str, Any]) -> dict[str, str]:
    rgb, sem = config["parents"]["rgb150"], config["parents"]["sem392"]
    required = {
        rgb["training_result"]: rgb["training_result_sha256"],
        rgb["evaluation_result"]: rgb["evaluation_result_sha256"],
        rgb["point_cloud"]: rgb["point_cloud_sha256"],
        sem["checkpoint"]: sem["checkpoint_sha256"],
        sem["reproduction_result"]: sem["reproduction_result_sha256"],
    }
    actual = {}
    for relative, expected in required.items():
        value = sha256(safe_path(relative))
        if value != expected:
            raise ValueError(f"Frozen parent input changed: {relative}")
        actual[relative] = value
    rgb_result = read_json(safe_path(rgb["training_result"]))
    rgb_eval = read_json(safe_path(rgb["evaluation_result"]))
    reproduction = read_json(safe_path(sem["reproduction_result"]))
    if (
        rgb_result["status"] != "completed_dev30_training"
        or rgb_eval["status"] != "completed"
    ):
        raise ValueError("RGB150 parent is incomplete")
    if (
        reproduction["status"] != "completed"
        or reproduction["pixel_exact_predictions"] != 400
        or reproduction["byte_identical_predictions"] != 400
    ):
        raise ValueError("SEM392 reproduction is incomplete")
    if reproduction["checkpoint_sha256"] != sem["checkpoint_sha256"]:
        raise ValueError("SEM392 reproduction checkpoint binding changed")
    if (
        reproduction["code_sha256"]["src/quality_mask2former.py"]
        != sem["inference_source_sha256"]
    ):
        raise ValueError("SEM392 inference source binding changed")
    source_root = safe_path(rgb["source"])
    source_hashes = verify_hashes(source_root, rgb_result["source"])
    actual.update(
        {
            f"{rgb['source']}/{relative}": value
            for relative, value in source_hashes.items()
        }
    )
    return actual


def role_paths(value: Any, name: str, prefix: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(role_paths(item, name, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list) and name in value:
        found.append(prefix)
    return found


def history_audit(
    candidate: dict[str, Any], dev_split: dict[str, Any]
) -> dict[str, Any]:
    """Audit registered partition reuse; this is not a claim of unlogged file IO."""
    registry = read_json(ROOT / "experiments/registry.yaml")["experiments"]
    split_objects = {
        "dev30_local_composition_v1": dev_split,
        "rgb_r1_spatial_v1": read_json(
            ROOT / "experiments/splits/rgb_r1_spatial_v1.json"
        ),
        "sem_001_scaffold_static_v1": read_json(
            ROOT / "experiments/splits/sem_001_scaffold_static_v1.json"
        ),
        "sem_005_oof_spatial_v1": read_json(
            ROOT / "experiments/splits/sem_005_oof_spatial_v1.json"
        ),
    }
    broad = []
    for row in registry:
        split_id = str(row.get("split_id", ""))
        experiment_id = row["experiment_id"]
        if split_id in {"all_300_labeled", "released_all400_rgb_all300_sem_v1"}:
            if experiment_id.startswith("RGB-"):
                role = (
                    "registered_all400_rgb_training"
                    if row.get("iterations", 0)
                    else "registered_all400_rgb_reference"
                )
            elif experiment_id.startswith("SEM-"):
                role = (
                    "registered_all300_semantic_training"
                    if row.get("iterations", 0)
                    else "registered_all300_semantic_diagnostic_or_evaluation"
                )
            else:
                role = "registered_all_data_reference"
            broad.append(
                {
                    "experiment_id": experiment_id,
                    "role": role,
                    "status": row.get("status"),
                }
            )
    relevant = []
    for row in registry:
        split_text = str(row.get("split_id", ""))
        named = [split_id for split_id in split_objects if split_id in split_text]
        if named:
            relevant.append(
                {
                    "experiment_id": row["experiment_id"],
                    "split_ids": named,
                    "status": row.get("status"),
                    "iterations": row.get("iterations", 0),
                }
            )
    per_view = {}
    for name in candidate["confirmation_labeled"]:
        references = list(broad)
        for row in relevant:
            paths = {
                split_id: role_paths(split_objects[split_id], name)
                for split_id in row["split_ids"]
            }
            references.append(
                {
                    "experiment_id": row["experiment_id"],
                    "role": "registered_partition_reference",
                    "partition_paths": paths,
                    "status": row["status"],
                }
            )
        per_view[name] = {
            "historically_unseen": False,
            "registered_references": references,
        }
    rgb_result = read_json(ROOT / "outputs/quality/RGB-150/full-r1/result.json")
    semantic_result = read_json(
        ROOT / "outputs/quality/SEM-392/r1/mask2former/result.json"
    )
    inference_result = read_json(
        ROOT / "outputs/quality/SEM-392/r1/inference/result.json"
    )
    one_seed_seconds = float(
        rgb_result["seconds"] + semantic_result["seconds"] + inference_result["seconds"]
    )
    return {
        "schema_version": 1,
        "scope": "Best-effort audit of registry-declared split roles. It proves prior reuse where an all-data or named partition record exists; it does not infer unregistered file access.",
        "candidate_views_are_newly_unseen": False,
        "per_view": per_view,
        "future_cost_estimate": {
            "basis": {
                "rgb150_training_seconds": rgb_result["seconds"],
                "sem392_training_seconds": semantic_result["seconds"],
                "sem392_400_view_inference_seconds": inference_result["seconds"],
            },
            "one_seed_matched_chain_gpu_minutes_approx": one_seed_seconds / 60,
            "three_seed_matched_chain_gpu_minutes_approx": 3 * one_seed_seconds / 60,
            "additional_required_work": "Fresh split-specific scene preparation and track re-triangulation, per-seed preflight, once-only confirmation inference, independent scoring, and preserved artifacts.",
            "warning": "Historical runtimes are planning estimates, not reservations or guarantees; confirmation and buffer views must remain inaccessible until final evaluation.",
        },
    }


def prepare(config: dict[str, Any]) -> None:
    output = OUTPUT / "prepare"
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EID,
        "attempt": OUTPUT.name,
        "stage": "prepare",
        "status": "starting",
        "labels_read": 0,
        "holdout_rgb_read": 0,
        "holdout_labels_read": 0,
    }
    write_new_json(output / "started.json", result)
    try:
        parent_hashes = verify_parent_inputs(config)
        split_path = safe_path(config["split_path"])
        graph_path = safe_path(config["camera_graph"])
        segment_path = safe_path(config["trajectory_segments"])
        split, graph, segments = map(read_json, (split_path, graph_path, segment_path))
        fold = split["primary_fold"]
        train_names = list(fold["train_rgb_300"])
        holdout = set(fold["validation_labeled"])
        if len(train_names) != 270 or len(holdout) != 30 or set(train_names) & holdout:
            raise ValueError("Dev30 semantic roles changed")
        selection = config["selection"]
        plan = select_diagnostic_plan(
            graph,
            segments,
            train_names,
            seed=config["seed"],
            target_count=config["access_contract"]["target_count"],
            sources_per_target=config["access_contract"]["sources_per_target"],
            source_minimum_shared_points=selection["source_minimum_shared_points"],
            source_maximum_axis_angle_deg=selection["source_maximum_axis_angle_deg"],
            source_minimum_center_distance=selection["source_minimum_center_distance"],
            source_maximum_center_distance=selection["source_maximum_center_distance"],
            second_source_minimum_separation=selection[
                "second_source_minimum_separation"
            ],
        )
        touched = {row["target"] for row in plan["targets"]} | {
            source["name"] for row in plan["targets"] for source in row["sources"]
        }
        if not touched <= set(train_names) or touched & holdout:
            raise ValueError(
                "Target/source plan escapes the 270-view training universe"
            )
        visual_count = selection["visualization_target_count"]
        indices = [
            int(np.floor(index * (len(plan["targets"]) - 1) / (visual_count - 1) + 0.5))
            for index in range(visual_count)
        ]
        plan["visualization_targets"] = [
            plan["targets"][index]["target"] for index in indices
        ]
        plan["selection_inputs"] = [
            config["camera_graph"],
            config["trajectory_segments"],
            config["split_path"],
        ]
        plan["selection_uses_labels_rgb_pixels_model_outputs_or_errors"] = False

        reproduction = read_json(
            safe_path(config["parents"]["sem392"]["reproduction_result"])
        )
        probability_hashes = {}
        probability_root = safe_path(config["parents"]["sem392"]["probabilities"])
        for name in sorted(touched):
            filename = Path(name).with_suffix(".npz").name
            expected = reproduction["probability_hashes"][filename]
            actual = sha256(probability_root / filename)
            if actual != expected:
                raise ValueError(f"Sealed SEM392 probability changed: {filename}")
            probability_hashes[filename] = actual

        labeled_names = sorted(set(train_names) | holdout)
        confirmation = build_confirmation_candidate(
            graph,
            segments,
            labeled_names,
            seed=config["confirmation_split_candidate"]["seed"],
            region_count=config["confirmation_split_candidate"]["regions"],
            views_per_region=config["confirmation_split_candidate"]["views_per_region"],
            guard_radius=config["confirmation_split_candidate"]["guard_radius"],
            minimum_region_separation=config["confirmation_split_candidate"][
                "minimum_region_separation"
            ],
            minimum_rgb_train=config["confirmation_split_candidate"][
                "minimum_rgb_train"
            ],
            minimum_semantic_train=config["confirmation_split_candidate"][
                "minimum_semantic_train"
            ],
        )
        audit = history_audit(confirmation, split)
        access = {
            "schema_version": 1,
            "experiment_id": EID,
            "allowed_target_names": sorted(row["target"] for row in plan["targets"]),
            "allowed_source_names": sorted(
                {source["name"] for row in plan["targets"] for source in row["sources"]}
            ),
            "allowed_label_names_after_evidence_seal": sorted(touched),
            "forbidden_holdout_names": sorted(holdout),
            "forbidden_unlabeled_names": sorted(
                set(graph["adjacency"]) - set(labeled_names)
            ),
            "labels_before_score": False,
        }
        write_new_json(output / "target-source-plan.json", plan)
        write_new_json(output / "access.json", access)
        write_new_json(output / "confirmation-split-candidate.json", confirmation)
        write_new_json(output / "historical-use-audit.json", audit)
        protocol_path = safe_path(config["protocol"])
        result.update(
            status="completed",
            config_sha256=sha256(CONFIG),
            protocol_sha256=sha256(protocol_path),
            split_sha256=sha256(split_path),
            dataset_manifest_sha256=sha256(safe_path(config["dataset_manifest"])),
            camera_graph_sha256=sha256(graph_path),
            trajectory_segments_sha256=sha256(segment_path),
            label_manifest_sha256_without_label_access=sha256(
                safe_path(config["label_manifest"])
            ),
            code_sha256=code_hashes(),
            git=git_context(),
            parent_hashes=parent_hashes,
            selected_probability_hashes=probability_hashes,
            artifacts={
                "target-source-plan.json": sha256(output / "target-source-plan.json"),
                "access.json": sha256(output / "access.json"),
                "confirmation-split-candidate.json": sha256(
                    output / "confirmation-split-candidate.json"
                ),
                "historical-use-audit.json": sha256(
                    output / "historical-use-audit.json"
                ),
            },
            counts={
                "targets": plan["target_count"],
                "pairs": plan["pair_count"],
                "unique_touched_training_views": len(touched),
                "covered_segments": len(plan["covered_segments"]),
                "visualization_targets": len(plan["visualization_targets"]),
            },
        )
        enforce_budget(config, started)
    except BaseException as exc:
        result.update(
            status="failed_preserved",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


def require_prepare(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    output = OUTPUT / "prepare"
    result = read_json(output / "result.json")
    if result["status"] != "completed" or result["labels_read"] != 0:
        raise ValueError("SEM-395 prepare did not complete without labels")
    if result["config_sha256"] != sha256(CONFIG) or result["protocol_sha256"] != sha256(
        safe_path(config["protocol"])
    ):
        raise ValueError("SEM-395 preregistration changed")
    if result["code_sha256"] != code_hashes():
        raise ValueError("SEM-395 sealed implementation changed")
    for key, relative in (
        ("split_sha256", config["split_path"]),
        ("dataset_manifest_sha256", config["dataset_manifest"]),
        ("camera_graph_sha256", config["camera_graph"]),
        ("trajectory_segments_sha256", config["trajectory_segments"]),
        ("label_manifest_sha256_without_label_access", config["label_manifest"]),
    ):
        if result[key] != sha256(safe_path(relative)):
            raise ValueError(f"SEM-395 sealed prerequisite changed: {relative}")
    verify_parent_inputs(config)
    verify_hashes(output, result["artifacts"])
    probability_root = safe_path(config["parents"]["sem392"]["probabilities"])
    for filename, digest in result["selected_probability_hashes"].items():
        if sha256(probability_root / filename) != digest:
            raise ValueError(f"SEM-395 sealed probability changed: {filename}")
    plan, access = (
        read_json(output / "target-source-plan.json"),
        read_json(output / "access.json"),
    )
    if plan["target_count"] != 24 or plan["pair_count"] != 48:
        raise ValueError("SEM-395 sealed plan budget changed")
    return result, plan, access


def runtime_environment(
    artifact_root: Path, source: Path
) -> tuple[dict[str, str], Path]:
    cuda = artifact_root / "r1/toolchain/cuda-12.8"
    python = artifact_root / "r1/envs/r1-cuda128/bin/python"
    if not python.is_file() or not cuda.is_dir():
        raise ValueError("Registered CUDA128 runtime is unavailable")
    environment = os.environ.copy()
    environment.update(
        CUDA_HOME=str(cuda),
        PATH=str(cuda / "bin") + os.pathsep + environment.get("PATH", ""),
        LD_LIBRARY_PATH=str(cuda / "lib64")
        + os.pathsep
        + environment.get("LD_LIBRARY_PATH", ""),
        PYTHONPATH=os.pathsep.join(
            map(
                str,
                [
                    source,
                    source / "submodules/diff-gaussian-rasterization",
                    source / "submodules/simple-knn",
                    ROOT,
                ],
            )
        ),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return environment, python


def wait_for_gpu_lock() -> tuple[Any, float, str]:
    lock = (ROOT / "outputs/quality/gpu.lock").open("a")
    queued = time.monotonic()
    fcntl.flock(lock, fcntl.LOCK_EX)
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    )
    others = [
        line
        for line in processes.splitlines()
        if line.strip() and "todesk" not in line.lower()
    ]
    if others:
        lock.close()
        raise RuntimeError(f"GPU process active outside the shared lock: {others}")
    return lock, time.monotonic() - queued, processes


def render_depth(config: dict[str, Any], artifact_root: Path) -> None:
    _, plan, _ = require_prepare(config)
    output = OUTPUT / "render"
    output.mkdir(parents=True, exist_ok=False)
    depth_dir = output / "depth"
    depth_dir.mkdir()
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EID,
        "attempt": OUTPUT.name,
        "stage": "render",
        "status": "waiting_gpu_lock",
        "labels_read": 0,
        "holdout_rgb_read": 0,
        "holdout_labels_read": 0,
        "prepare_result_sha256": sha256(OUTPUT / "prepare/result.json"),
    }
    write_new_json(output / "started.json", result)
    lock = None
    try:
        source = safe_path(config["parents"]["rgb150"]["source"])
        environment, expected_python = runtime_environment(artifact_root, source)
        if Path(sys.prefix).resolve() != expected_python.parent.parent.resolve():
            raise ValueError(
                "Render must use the registered CUDA128 Python environment"
            )
        os.environ.update(environment)
        sys.path[:0] = [
            str(source),
            str(source / "submodules/diff-gaussian-rasterization"),
            str(source / "submodules/simple-knn"),
        ]
        lock, result["queue_seconds"], result["gpu_processes_before"] = (
            wait_for_gpu_lock()
        )
        result["status"] = "running"

        import torch  # type: ignore[import-not-found]
        from arguments import PipelineParams  # type: ignore[import-not-found]
        from gaussian_renderer import (  # type: ignore[import-not-found]
            GaussianModel,
            render,
        )
        from PIL import Image

        from src.quality_cameras import read_camera_requests
        from src.quality_native_gaussian import make_native_view

        torch.set_num_threads(4)
        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        model = GaussianModel(3)
        model.load_ply(str(safe_path(config["parents"]["rgb150"]["point_cloud"])))
        parser = argparse.ArgumentParser(add_help=False)
        pipeline = PipelineParams(parser).extract(parser.parse_args([]))
        background = torch.zeros(3, device="cuda")
        unit = torch.ones(
            (model.get_xyz.shape[0], 3), device="cuda", dtype=model.get_xyz.dtype
        )
        cameras = read_camera_requests(
            safe_path(config["parents"]["rgb150"]["scene"]) / "sparse/0"
        )
        names = sorted(
            {row["target"] for row in plan["targets"]}
            | {
                source_row["name"]
                for row in plan["targets"]
                for source_row in row["sources"]
            }
        )
        parent_eval = read_json(
            safe_path(config["parents"]["rgb150"]["evaluation_result"])
        )
        saved_root = safe_path(config["parents"]["rgb150"]["saved_renders"])
        hashes, replay = {}, []
        with torch.inference_mode():
            for index, name in enumerate(names):
                camera = cameras[name]
                if camera.model != "PINHOLE" or (camera.width, camera.height) != (
                    1320,
                    989,
                ):
                    raise ValueError(
                        "SEM-395 rendering requires RGB150 native PINHOLE cameras"
                    )
                view = make_native_view(camera)
                native = render(view, model, pipeline, background)
                white = render(view, model, pipeline, background, override_color=unit)
                if not torch.equal(native["depth"], white["depth"]):
                    raise ValueError(
                        "Native inverse-depth sum changed under white override color"
                    )
                rgb = np.rint(
                    native["render"][:3].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    * 255
                ).astype(np.uint8)
                saved_path = saved_root / name
                if sha256(saved_path) != parent_eval["image_hashes"][name]:
                    raise ValueError(f"Saved RGB150 render hash changed: {name}")
                saved = np.asarray(
                    Image.open(saved_path).convert("RGB"), dtype=np.uint8
                )
                if not np.array_equal(rgb, saved):
                    raise ValueError(
                        f"Frozen RGB150 render does not replay pixel-exactly: {name}"
                    )
                inverse = white["depth"][0].float().cpu().numpy()
                alpha = white["render"][0].float().cpu().numpy()
                if inverse.shape != (989, 1320) or alpha.shape != inverse.shape:
                    raise ValueError("Native depth/alpha shape changed")
                if (
                    not np.isfinite(inverse).all()
                    or not np.isfinite(alpha).all()
                    or alpha.min() < -1e-6
                    or alpha.max() > 1.0001
                ):
                    raise ValueError("Native depth/alpha contains invalid values")
                path = depth_dir / f"{Path(name).stem}.npz"
                np.savez_compressed(
                    path,
                    inverse_depth_sum=inverse.astype(np.float32),
                    alpha=alpha.astype(np.float32),
                )
                hashes[str(path.relative_to(output))] = sha256(path)
                enforce_budget(config, started, gpu_stage=True)
                replay.append(
                    {
                        "name": name,
                        "pixel_exact_rgb": True,
                        "saved_rgb_sha256": sha256(saved_path),
                    }
                )
                if index % 10 == 0:
                    print(
                        json.dumps({"rendered": index + 1, "total": len(names)}),
                        flush=True,
                    )
        torch.cuda.synchronize()
        result.update(
            status="completed",
            config_sha256=sha256(CONFIG),
            code_sha256=code_hashes(),
            unique_views=len(names),
            depth_artifacts=hashes,
            rgb_replay=replay,
            all_selected_rgb_pixel_exact=True,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            runtime={
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
            },
        )
        if result["peak_reserved_mib"] > config["budget"]["maximum_peak_gpu_mib"]:
            raise RuntimeError("SEM-395 GPU memory budget exceeded")
        enforce_budget(config, started, gpu_stage=True)
        verify_parent_inputs(config)
        if sha256(OUTPUT / "prepare/result.json") != result["prepare_result_sha256"]:
            raise ValueError("Prepare seal changed during render")
    except BaseException as exc:
        result.update(
            status="failed_preserved",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        if lock is not None:
            lock.close()
        result["seconds"] = time.monotonic() - started
        result["output_bytes"] = sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()
        )
        write_new_json(output / "result.json", result)


def require_render(config: dict[str, Any]) -> dict[str, Any]:
    prepare_result, _, _ = require_prepare(config)
    result = read_json(OUTPUT / "render/result.json")
    if (
        result["status"] != "completed"
        or result["labels_read"] != 0
        or not result["all_selected_rgb_pixel_exact"]
    ):
        raise ValueError(
            "SEM-395 render did not complete under the access/freeze contract"
        )
    if result["prepare_result_sha256"] != sha256(OUTPUT / "prepare/result.json"):
        raise ValueError("SEM-395 render prerequisite changed")
    if (
        result["config_sha256"] != prepare_result["config_sha256"]
        or result["code_sha256"] != code_hashes()
    ):
        raise ValueError("SEM-395 render implementation changed")
    verify_hashes(OUTPUT / "render", result["depth_artifacts"])
    return result


def load_probability(
    config: dict[str, Any], name: str, expected_hashes: dict[str, str]
) -> np.ndarray:
    filename = Path(name).with_suffix(".npz").name
    path = safe_path(config["parents"]["sem392"]["probabilities"]) / filename
    if sha256(path) != expected_hashes[filename]:
        raise ValueError(f"SEM392 probability cache changed: {filename}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {config["parents"]["sem392"]["probability_key"]}:
            raise ValueError(f"Unexpected probability archive schema: {filename}")
        probability = np.asarray(
            archive[config["parents"]["sem392"]["probability_key"]]
        )
    if (
        probability.dtype != np.float32
        or list(probability.shape) != config["parents"]["sem392"]["probability_shape"]
    ):
        raise ValueError(f"Probability dtype/shape changed: {filename}")
    if (
        not np.isfinite(probability).all()
        or np.any(probability < 0)
        or not np.allclose(probability.sum(0), 1, atol=2e-5, rtol=0)
    ):
        raise ValueError(f"Invalid probability simplex: {filename}")
    return probability


def load_depth(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = OUTPUT / "render/depth" / f"{Path(name).stem}.npz"
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"inverse_depth_sum", "alpha"}:
            raise ValueError("Unexpected depth archive schema")
        inverse = np.asarray(archive["inverse_depth_sum"], dtype=np.float32)
        alpha = np.asarray(archive["alpha"], dtype=np.float32)
    return depth_proxy(inverse, alpha), alpha, inverse


def evidence(config: dict[str, Any]) -> None:
    prepare_result, plan, _ = require_prepare(config)
    require_render(config)
    output = OUTPUT / "evidence"
    output.mkdir(parents=True, exist_ok=False)
    pair_dir, target_dir = output / "pairs", output / "targets"
    pair_dir.mkdir()
    target_dir.mkdir()
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EID,
        "attempt": OUTPUT.name,
        "stage": "evidence",
        "status": "running",
        "labels_read": 0,
        "holdout_rgb_read": 0,
        "holdout_labels_read": 0,
        "prepare_result_sha256": sha256(OUTPUT / "prepare/result.json"),
        "render_result_sha256": sha256(OUTPUT / "render/result.json"),
    }
    write_new_json(output / "started.json", result)
    try:
        from src.quality_cameras import read_camera_requests

        cameras = read_camera_requests(
            safe_path(config["parents"]["rgb150"]["scene"]) / "sparse/0"
        )
        probability_hashes = prepare_result["selected_probability_hashes"]
        thresholds = config["reliability_filter"]
        rows, artifacts = [], {}
        depth_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        probability_cache: dict[str, np.ndarray] = {}

        def depths(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if name not in depth_cache:
                depth_cache[name] = load_depth(name)
            return depth_cache[name]

        def probabilities(name: str) -> np.ndarray:
            if name not in probability_cache:
                probability_cache[name] = load_probability(
                    config, name, probability_hashes
                )
            return probability_cache[name]

        for target_number, target_row in enumerate(plan["targets"]):
            target = target_row["target"]
            target_depth, target_alpha, _ = depths(target)
            pair_arrays = []
            for source_number, source_row in enumerate(target_row["sources"]):
                source = source_row["name"]
                source_depth, source_alpha, _ = depths(source)
                evaluated = evaluate_pair(
                    target_depth,
                    target_alpha,
                    source_depth,
                    source_alpha,
                    probabilities(source),
                    cameras[target],
                    cameras[source],
                    thresholds,
                )
                arrays = evaluated.pop("arrays")
                path = pair_dir / f"{Path(target).stem}__{Path(source).stem}.npz"
                np.savez_compressed(path, **arrays)
                relative = str(path.relative_to(output))
                artifacts[relative] = sha256(path)
                pair_arrays.append(arrays)
                rows.append(
                    {
                        "target": target,
                        "source": source,
                        "source_number": source_number + 1,
                        "segment_id": target_row["segment_id"],
                        "pair_metadata": source_row,
                        "pair_artifact": relative,
                        **evaluated,
                    }
                )
            combined = combine_source_evidence(
                pair_arrays[0], pair_arrays[1], target_depth.size
            )
            counts = combined.pop("counts")
            target_path = target_dir / f"{Path(target).stem}.npz"
            np.savez_compressed(target_path, **combined)
            relative = str(target_path.relative_to(output))
            artifacts[relative] = sha256(target_path)
            target_row["evidence_artifact"] = relative
            target_row["evidence_counts"] = counts
            if target_number % 4 == 0:
                print(
                    json.dumps(
                        {"targets": target_number + 1, "total": len(plan["targets"])}
                    ),
                    flush=True,
                )
            enforce_budget(config, started)
        write_new_json(output / "pair-summary.json", rows)
        write_new_json(
            output / "target-summary.json",
            [
                {
                    "target": row["target"],
                    "segment_id": row["segment_id"],
                    "sources": row["sources"],
                    "evidence_artifact": row["evidence_artifact"],
                    "evidence_counts": row["evidence_counts"],
                }
                for row in plan["targets"]
            ],
        )
        artifacts["pair-summary.json"] = sha256(output / "pair-summary.json")
        artifacts["target-summary.json"] = sha256(output / "target-summary.json")
        result.update(
            status="completed",
            config_sha256=sha256(CONFIG),
            code_sha256=code_hashes(),
            pair_count=len(rows),
            target_count=len(plan["targets"]),
            artifacts=artifacts,
            aggregate_stage_counts={
                stage: int(sum(row["stage_counts"][stage] for row in rows))
                for stage in PAIR_STAGE_NAMES
            },
            aggregate_occlusion_counts={
                key: int(sum(row["occlusion_counts"][key] for row in rows))
                for key in (
                    "projected_behind_source_surface",
                    "projected_in_front_of_source_surface",
                )
            },
        )
        enforce_budget(config, started)
        verify_parent_inputs(config)
        if sha256(OUTPUT / "render/result.json") != result["render_result_sha256"]:
            raise ValueError("Render seal changed during evidence construction")
    except BaseException as exc:
        result.update(
            status="failed_preserved",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        result["output_bytes"] = sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()
        )
        write_new_json(output / "result.json", result)


def require_evidence(config: dict[str, Any]) -> dict[str, Any]:
    require_prepare(config)
    require_render(config)
    result = read_json(OUTPUT / "evidence/result.json")
    if (
        result["status"] != "completed"
        or result["labels_read"] != 0
        or result["pair_count"] != 48
        or result["target_count"] != 24
    ):
        raise ValueError("SEM-395 evidence did not complete without labels")
    if result["prepare_result_sha256"] != sha256(
        OUTPUT / "prepare/result.json"
    ) or result["render_result_sha256"] != sha256(OUTPUT / "render/result.json"):
        raise ValueError("SEM-395 evidence prerequisite changed")
    if (
        result["config_sha256"] != sha256(CONFIG)
        or result["code_sha256"] != code_hashes()
    ):
        raise ValueError("SEM-395 evidence implementation changed")
    verify_hashes(OUTPUT / "evidence", result["artifacts"])
    return result


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def group_name(value: float, boundaries: tuple[float, float]) -> str:
    return (
        "low"
        if value < boundaries[0]
        else "medium"
        if value < boundaries[1]
        else "high"
    )


def update_group(
    store: dict[str, dict[str, Any]],
    dimension: str,
    key: str,
    truth: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
) -> None:
    bucket = store.setdefault(dimension, {}).setdefault(
        key,
        {
            "baseline_confusion": np.zeros((5, 5), np.int64),
            "counterfactual_confusion": np.zeros((5, 5), np.int64),
            "corrected": 0,
            "damaged": 0,
        },
    )
    bucket["baseline_confusion"] += np.bincount(
        truth.astype(np.int64) * 5 + old, minlength=25
    ).reshape(5, 5)
    bucket["counterfactual_confusion"] += np.bincount(
        truth.astype(np.int64) * 5 + new, minlength=25
    ).reshape(5, 5)
    bucket["corrected"] += int(np.count_nonzero((old != truth) & (new == truth)))
    bucket["damaged"] += int(np.count_nonzero((old == truth) & (new != truth)))


def finish_groups(store: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for dimension, buckets in store.items():
        output[dimension] = {}
        for key, row in sorted(buckets.items()):
            baseline = metrics(row["baseline_confusion"])
            candidate = metrics(row["counterfactual_confusion"])
            delta = (
                100
                * (candidate["mIoU_with_background"] - baseline["mIoU_with_background"])
                if candidate["mIoU_with_background"] is not None
                and baseline["mIoU_with_background"] is not None
                else None
            )
            output[dimension][key] = {
                "pixels": int(row["baseline_confusion"].sum()),
                "corrected": row["corrected"],
                "damaged": row["damaged"],
                "net_correct": row["corrected"] - row["damaged"],
                "baseline": baseline,
                "counterfactual": candidate,
                "five_class_miou_delta_pp": delta,
            }
    return output


def semantic_rejection_count(
    stage_counts: dict[str, int], target_summaries: list[dict[str, Any]]
) -> int:
    return int(
        stage_counts["source_low_confidence"]
        + sum(
            row["evidence_counts"]["source_semantic_disagreement"]
            for row in target_summaries
        )
    )


def palette_image(labels: np.ndarray) -> Any:
    from PIL import Image

    palette = np.asarray(
        [[28, 31, 38], [44, 129, 201], [245, 197, 66], [206, 72, 89], [89, 175, 112]],
        dtype=np.uint8,
    )
    return Image.fromarray(palette[np.asarray(labels, dtype=np.uint8)])


def make_panel(
    config: dict[str, Any],
    target_row: dict[str, Any],
    evidence_arrays: dict[str, np.ndarray],
    target_label: np.ndarray,
    baseline: np.ndarray,
    label_cache: dict[str, np.ndarray],
    output: Path,
) -> None:
    from PIL import Image, ImageDraw

    target = target_row["target"]
    source1, source2 = (row["name"] for row in target_row["sources"])
    scene_images = safe_path(config["parents"]["rgb150"]["scene"]) / "images"
    render_images = safe_path(config["parents"]["rgb150"]["saved_renders"])
    indices, consensus = (
        evidence_arrays["target_index"],
        evidence_arrays["consensus_class"],
    )
    projected = np.full(target_label.size, 0, dtype=np.uint8)
    projected[indices] = consensus
    projected = projected.reshape(target_label.shape)
    reliable = np.zeros(target_label.size, dtype=bool)
    reliable[indices] = True
    reliable = reliable.reshape(target_label.shape)
    old = baseline.ravel()[indices]
    truth = target_label.ravel()[indices]
    corrected = np.zeros(target_label.size, dtype=bool)
    damaged = np.zeros(target_label.size, dtype=bool)
    corrected[indices] = (old != truth) & (consensus == truth)
    damaged[indices] = (old == truth) & (consensus != truth)
    photo = Image.open(scene_images / target).convert("RGB")
    overlay = np.asarray(photo).copy()
    overlay[reliable] = (
        0.45 * overlay[reliable] + 0.55 * np.array([69, 181, 120])
    ).astype(np.uint8)
    overlay.reshape(-1, 3)[corrected] = [30, 220, 90]
    overlay.reshape(-1, 3)[damaged] = [235, 45, 55]
    tiles = [
        ("target input photo", photo),
        ("frozen RGB150 render", Image.open(render_images / target).convert("RGB")),
        ("target GT", palette_image(target_label)),
        ("SEM392 target", palette_image(baseline)),
        (f"source 1: {source1}", Image.open(scene_images / source1).convert("RGB")),
        (f"source 2: {source2}", Image.open(scene_images / source2).convert("RGB")),
        ("projected agreed evidence", palette_image(projected)),
        ("reliable green / correction bright / damage red", Image.fromarray(overlay)),
    ]
    tile_size, title_height = (396, 297), 28
    canvas = Image.new(
        "RGB",
        (tile_size[0] * 4, (tile_size[1] + title_height) * 2),
        255,
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(tiles):
        x = (index % 4) * tile_size[0]
        y = (index // 4) * (tile_size[1] + title_height)
        canvas.paste(
            image.resize(tile_size, Image.Resampling.NEAREST), (x, y + title_height)
        )
        draw.text((x + 5, y + 7), title, fill="black")
    canvas.save(output / f"{Path(target).stem}.png")


def score(config: dict[str, Any]) -> None:
    prepare_result, plan, access = require_prepare(config)
    evidence_result = require_evidence(config)
    target_summaries = read_json(OUTPUT / "evidence/target-summary.json")
    if [row["target"] for row in target_summaries] != [
        row["target"] for row in plan["targets"]
    ]:
        raise ValueError("SEM-395 evidence target summary differs from the sealed plan")
    output = OUTPUT / "score"
    output.mkdir(parents=True, exist_ok=False)
    panel_dir = output / "panels"
    panel_dir.mkdir()
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EID,
        "attempt": OUTPUT.name,
        "stage": "score",
        "status": "running",
        "prepare_result_sha256": sha256(OUTPUT / "prepare/result.json"),
        "render_result_sha256": sha256(OUTPUT / "render/result.json"),
        "evidence_result_sha256": sha256(OUTPUT / "evidence/result.json"),
        "holdout_rgb_read": 0,
        "holdout_labels_read": 0,
    }
    write_new_json(output / "started.json", result)
    label_access, label_cache = {}, {}
    try:
        from PIL import Image

        allowed = set(access["allowed_label_names_after_evidence_seal"])
        forbidden = set(access["forbidden_holdout_names"]) | set(
            access["forbidden_unlabeled_names"]
        )
        label_root = safe_path(config["label_root"])
        label_manifest = read_json(safe_path(config["label_manifest"]))
        expected_mapping = {name: index for index, name in enumerate(CLASSES)}
        if (
            label_manifest["class_mapping"] != expected_mapping
            or label_manifest["image_count"] != 300
        ):
            raise ValueError("SEM-395 label manifest schema changed")
        label_hashes = {
            row["image"]: row["label_png_sha256"] for row in label_manifest["records"]
        }
        if (
            len(label_manifest["records"]) != 300
            or len(label_hashes) != 300
            or not allowed <= label_hashes.keys()
            or allowed & forbidden
        ):
            raise ValueError("SEM-395 allowed labels differ from the sealed manifest")

        def label(name: str) -> np.ndarray:
            if name not in allowed or name in forbidden:
                raise ValueError(f"Forbidden SEM-395 label access: {name}")
            if name not in label_cache:
                path = label_root / name
                digest = sha256(path)
                if digest != label_hashes[name]:
                    raise ValueError(f"Sealed training label changed: {name}")
                array = np.asarray(Image.open(path), dtype=np.uint8)
                if array.shape != (989, 1320) or np.any(array > 4):
                    raise ValueError(f"Invalid five-class training label: {name}")
                label_cache[name] = array
                label_access[name] = digest
            return label_cache[name]

        probability_hashes = prepare_result["selected_probability_hashes"]
        total_baseline = np.zeros((5, 5), np.int64)
        total_counterfactual = np.zeros((5, 5), np.int64)
        reliable_confusion = np.zeros((5, 5), np.int64)
        source_target_confusion = np.zeros((5, 5), np.int64)
        total_counts = defaultdict(int)
        source_error_categories = defaultdict(int)
        groups: dict[str, dict[str, Any]] = {}
        rows = []
        visual = set(plan["visualization_targets"])
        foreground_pixels, reliable_foreground = 0, 0
        for target_row in plan["targets"]:
            target = target_row["target"]
            truth = label(target)
            probability = load_probability(config, target, probability_hashes)
            baseline = np.argmax(probability, axis=0).astype(np.uint8)
            target_confidence = probability.max(0)
            arrays = load_npz_dict(
                OUTPUT / "evidence/targets" / f"{Path(target).stem}.npz"
            )
            indices, consensus = arrays["target_index"], arrays["consensus_class"]
            candidate = baseline.copy()
            candidate.ravel()[indices] = consensus
            baseline_cm = independent_confusion(truth, baseline)
            candidate_cm = independent_confusion(truth, candidate)
            total_baseline += baseline_cm
            total_counterfactual += candidate_cm
            reliable_confusion += np.bincount(
                truth.ravel()[indices].astype(np.int64) * 5 + consensus,
                minlength=25,
            ).reshape(5, 5)
            source_target_confusion += np.bincount(
                consensus.astype(np.int64) * 5 + truth.ravel()[indices], minlength=25
            ).reshape(5, 5)
            counts = replacement_counts(truth, baseline, indices, consensus)
            for key, value in counts.items():
                total_counts[key] += value
            foreground_pixels += int(np.count_nonzero(truth > 0))
            reliable_foreground += int(np.count_nonzero(truth.ravel()[indices] > 0))

            source_labels = [label(source["name"]) for source in target_row["sources"]]
            source_gt = [
                source_labels[index].ravel()[
                    arrays[f"{'first' if index == 0 else 'second'}_source_index"]
                ]
                for index in range(2)
            ]
            target_gt = truth.ravel()[indices]
            source_correct = [(consensus == sampled) for sampled in source_gt]
            target_disagree = consensus != target_gt
            source_error_categories["target_disagreement"] += int(
                np.count_nonzero(target_disagree)
            )
            source_error_categories[
                "target_disagreement_both_source_predictions_correct_in_own_views"
            ] += int(
                np.count_nonzero(
                    target_disagree & source_correct[0] & source_correct[1]
                )
            )
            source_error_categories[
                "target_disagreement_any_source_prediction_wrong_in_own_view"
            ] += int(
                np.count_nonzero(
                    target_disagree & ~(source_correct[0] & source_correct[1])
                )
            )
            source_error_categories["nearest_source_gt_disagreement"] += int(
                np.count_nonzero(source_gt[0] != source_gt[1])
            )

            old, new = baseline.ravel()[indices], consensus
            selected_truth = target_gt
            confidence_values = target_confidence.ravel()[indices]
            opacity_values = arrays["target_alpha"].astype(np.float32)
            for key in ("low", "medium", "high"):
                if key == "low":
                    mask = confidence_values < 0.80
                elif key == "medium":
                    mask = (confidence_values >= 0.80) & (confidence_values < 0.95)
                else:
                    mask = confidence_values >= 0.95
                update_group(
                    groups,
                    "target_confidence",
                    key,
                    selected_truth[mask],
                    old[mask],
                    new[mask],
                )
                if key == "low":
                    mask = opacity_values < 0.93
                elif key == "medium":
                    mask = (opacity_values >= 0.93) & (opacity_values < 0.98)
                else:
                    mask = opacity_values >= 0.98
                update_group(
                    groups,
                    "target_opacity",
                    key,
                    selected_truth[mask],
                    old[mask],
                    new[mask],
                )
            minimum_shared = min(
                source["shared_point3d"] for source in target_row["sources"]
            )
            covisibility = group_name(float(minimum_shared), (500.0, 1500.0))
            update_group(groups, "covisibility", covisibility, selected_truth, old, new)
            update_group(
                groups,
                "trajectory_segment",
                target_row["segment_id"],
                selected_truth,
                old,
                new,
            )
            for class_id, class_name in enumerate(CLASSES):
                mask = selected_truth == class_id
                update_group(
                    groups,
                    "target_gt_class",
                    class_name,
                    selected_truth[mask],
                    old[mask],
                    new[mask],
                )

            row = {
                "target": target,
                "segment_id": target_row["segment_id"],
                "sources": target_row["sources"],
                "minimum_shared_point3d": minimum_shared,
                "covisibility_group": covisibility,
                "reliable_fraction": float(indices.size / truth.size),
                "counts": counts,
                "baseline": metrics(baseline_cm),
                "counterfactual": metrics(candidate_cm),
                "five_class_miou_delta_pp": 100
                * (
                    metrics(candidate_cm)["mIoU_with_background"]
                    - metrics(baseline_cm)["mIoU_with_background"]
                ),
            }
            rows.append(row)
            if target in visual:
                make_panel(
                    config, target_row, arrays, truth, baseline, label_cache, panel_dir
                )
            enforce_budget(config, started)

        baseline_metrics, counterfactual_metrics = (
            metrics(total_baseline),
            metrics(total_counterfactual),
        )
        target_pixels = len(plan["targets"]) * 989 * 1320
        reliable_fraction = total_counts["eligible_pixels"] / target_pixels
        foreground_coverage = (
            reliable_foreground / foreground_pixels if foreground_pixels else 0.0
        )
        stage_counts = evidence_result["aggregate_stage_counts"]
        geometry_stages = PAIR_STAGE_NAMES[:-2]
        geometry_rejects = sum(stage_counts[name] for name in geometry_stages)
        semantic_rejects = semantic_rejection_count(stage_counts, target_summaries)
        if total_counts["net_correct_pixels"] > 0:
            if reliable_fraction >= 0.01 and foreground_coverage >= 0.005:
                outcome = config["outcome_rules"]["reliable_positive_net_and_coverage"]
            else:
                outcome = config["outcome_rules"]["positive_but_too_sparse"]
        elif geometry_rejects > semantic_rejects:
            outcome = config["outcome_rules"]["geometry_dominant_failure"]
        else:
            outcome = config["outcome_rules"]["semantic_or_annotation_dominant_failure"]
        pair_pixels = sum(stage_counts.values())
        rejection = {
            "pair_pixel_denominator": pair_pixels,
            "first_rejection_counts": stage_counts,
            "first_rejection_ratios": {
                key: value / pair_pixels for key, value in stage_counts.items()
            },
            "geometry_rejects": geometry_rejects,
            "semantic_confidence_plus_two_source_disagreement_rejects": semantic_rejects,
            "occlusion_direction_counts": evidence_result["aggregate_occlusion_counts"],
        }
        summary = {
            "schema_version": 1,
            "experiment_id": EID,
            "scope": config["scope"],
            "outcome": outcome,
            "outcome_rule": config["outcome_rules"]["classification_rule"],
            "counts": dict(total_counts),
            "coverage": {
                "target_pixels": target_pixels,
                "reliable_consensus_pixels": total_counts["eligible_pixels"],
                "reliable_consensus_fraction": reliable_fraction,
                "target_foreground_gt_pixels": foreground_pixels,
                "reliable_target_foreground_gt_pixels": reliable_foreground,
                "reliable_target_foreground_fraction": foreground_coverage,
            },
            "baseline": baseline_metrics,
            "counterfactual": counterfactual_metrics,
            "counterfactual_five_class_miou_delta_pp": 100
            * (
                counterfactual_metrics["mIoU_with_background"]
                - baseline_metrics["mIoU_with_background"]
            ),
            "counterfactual_foreground_miou_delta_pp": 100
            * (
                counterfactual_metrics["mIoU_without_background"]
                - baseline_metrics["mIoU_without_background"]
            ),
            "reliable_source_class_vs_target_gt_confusion": source_target_confusion.tolist(),
            "target_gt_vs_reliable_source_class_confusion": reliable_confusion.tolist(),
            "source_error_categories": dict(source_error_categories),
            "groups": finish_groups(groups),
            "rejection": rejection,
            "per_view": rows,
            "interpretation_limits": [
                "All scores use SEM392 training-side labels and are fit diagnostics, not holdout or blind evidence.",
                "The RGB150 alpha-normalized inverse-depth reciprocal is a surface proxy, not geometric ground truth or expected Z.",
                "The counterfactual replacement is diagnostic and must not be reported as training gain or deployed output.",
                "Source nearest-pixel GT comparisons can mix geometry, visibility, and annotation-boundary differences; they do not identify a unique cause.",
            ],
        }
        write_new_json(output / "summary.json", summary)
        write_new_json(
            output / "label-access.json",
            {
                "schema_version": 1,
                "opened_only_after_evidence_seal": True,
                "allowed_partition": "primary_fold.train_rgb_300",
                "labels": label_access,
                "label_count": len(label_access),
                "holdout_labels_read": 0,
                "unlabeled_labels_read": 0,
            },
        )
        html = [
            "<!doctype html><meta charset='utf-8'><title>SEM-395 fixed panels</title>",
            "<h1>SEM-395 fixed pre-registered panels</h1>",
        ]
        for name in plan["visualization_targets"]:
            html.append(
                f"<h2>{name}</h2><img src='panels/{Path(name).stem}.png' style='max-width:100%'>"
            )
        (output / "index.html").write_text("\n".join(html) + "\n", encoding="utf-8")
        artifacts = {
            "summary.json": sha256(output / "summary.json"),
            "label-access.json": sha256(output / "label-access.json"),
            "index.html": sha256(output / "index.html"),
        }
        for path in sorted(panel_dir.glob("*.png")):
            artifacts[str(path.relative_to(output))] = sha256(path)
        result.update(
            status="completed",
            outcome=outcome,
            labels_read=len(label_access),
            allowed_labels_read=sorted(label_access),
            config_sha256=sha256(CONFIG),
            code_sha256=code_hashes(),
            summary=summary,
            artifacts=artifacts,
        )
        enforce_budget(config, started)
        if set(label_access) != allowed:
            raise ValueError(
                "Score stage did not audit exactly the sealed target/source label set"
            )
        if sha256(OUTPUT / "evidence/result.json") != result["evidence_result_sha256"]:
            raise ValueError("Evidence seal changed during scoring")
        verify_parent_inputs(config)
    except BaseException as exc:
        result.update(
            status="failed_preserved",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        result["labels_read"] = len(label_access)
        result["allowed_labels_read"] = sorted(label_access)
        label_audit_path = output / "label-access.json"
        if not label_audit_path.exists():
            write_new_json(
                label_audit_path,
                {
                    "schema_version": 1,
                    "opened_only_after_evidence_seal": True,
                    "status": "partial_failed",
                    "allowed_partition": "primary_fold.train_rgb_300",
                    "labels": label_access,
                    "label_count": len(label_access),
                    "holdout_labels_read": 0,
                    "unlabeled_labels_read": 0,
                },
            )
        result["seconds"] = time.monotonic() - started
        result["output_bytes"] = sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()
        )
        write_new_json(output / "result.json", result)


def main() -> None:
    global OUTPUT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--attempt", default="r1")
    parser.add_argument(
        "--stage", choices=("prepare", "render", "evidence", "score"), required=True
    )
    args = parser.parse_args()
    config = read_json(CONFIG)
    validate_config(config)
    if (
        not args.attempt.startswith("r")
        or not args.attempt[1:].isdigit()
        or int(args.attempt[1:]) < 1
    ):
        raise ValueError("Attempt must be r followed by a positive integer")
    OUTPUT = safe_path(config["output_path"]).parent / args.attempt
    artifact_root = args.artifact_root.resolve()
    if args.stage == "prepare":
        prepare(config)
    elif args.stage == "render":
        render_depth(config, artifact_root)
    elif args.stage == "evidence":
        evidence(config)
    else:
        score(config)


if __name__ == "__main__":
    main()
