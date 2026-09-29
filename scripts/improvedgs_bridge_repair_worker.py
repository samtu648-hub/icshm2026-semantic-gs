#!/usr/bin/env python3
"""Run RGB170 with one process-local multiview bridge LAS repair event."""

from __future__ import annotations

import argparse
import inspect
import json
import random
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, write_new_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--access", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--repair-record", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repair-iteration", type=int, required=True)
    parser.add_argument("--top-count", type=int, default=16)
    parser.add_argument("--transverse-multiplier", type=float, default=0.5)
    parser.add_argument(
        "--selection-mode",
        choices=(
            "native-intersection",
            "evidence-directed-fixed-budget",
            "responsibility-asymmetric-fixed-budget",
        ),
        default="native-intersection",
    )
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.evidence.exists() or args.repair_record.exists():
        raise FileExistsError("worker evidence output already exists")

    started = time.monotonic()
    record: dict[str, object] = {
        "schema_version": 1,
        "seed": args.seed,
        "repair_iteration": args.repair_iteration,
        "selection_mode": args.selection_mode,
        "status": "initializing",
        "sample_trace": [],
    }
    repair_state: dict[str, object] | None = None
    try:
        sys.path.insert(0, str(args.source))
        import numpy as np
        import torch
        import scene
        from scene import dataset_readers, training_runtime
        from scene.gaussian_model import GaussianModel
        from scene.methods import densification_stage
        from utils import general_utils
        from gaussian_renderer import render
        from src.improvedgs_bridge_repair import build_multiview_tangent_evidence, install
        from src.improvedgs_evidence_directed_split import (
            build_mixed_geometry_evidence,
            install as install_evidence_directed,
        )
        from src.improvedgs_responsibility_clone import (
            build_responsibility_clone_evidence,
            install as install_responsibility_clone,
        )
        from src.quality_cameras import read_camera_requests
        from src.quality_native_gaussian import make_native_view

        access = json.loads(args.access.read_text())
        train_names = set(access["train"])
        evaluation_names = set(access["evaluation"])
        if len(train_names) != 400 or evaluation_names or access.get("partition") != "released_all400_v1":
            raise ValueError("Expected the registered released_all400_v1 access set")
        label_names = sorted(path.name for path in args.labels.glob("*.png"))
        if len(label_names) != 300:
            raise ValueError("Expected all 300 registered semantic masks")

        original_reader = scene.sceneLoadTypeCallbacks["Colmap"]

        def explicit_reader(*reader_args, **reader_kwargs):
            bound = inspect.signature(original_reader).bind(*reader_args, **reader_kwargs)
            bound.apply_defaults()
            bound.arguments["eval"] = False
            scene_path = Path(bound.arguments["path"])
            info = original_reader(*bound.args, **bound.kwargs)
            cameras = list(info.train_cameras) + list(info.test_cameras)

            def canonical_name(camera) -> str:
                name = Path(camera.image_name).name
                return name if name.endswith(".png") else name + ".png"

            actual = [canonical_name(camera) for camera in cameras]
            if len(actual) != len(set(actual)) or set(actual) != train_names:
                raise ValueError("ImprovedGS reader changed the registered all-400 image universe")
            train = [camera for camera in cameras if canonical_name(camera) in train_names]
            canonical_ply = scene_path / "sparse/0/points3D.ply"
            common_cloud = dataset_readers.fetchPly(str(canonical_ply))
            if len(common_cloud.points) != 72_292 or not np.isfinite(common_cloud.points).all():
                raise ValueError("Unexpected common initialization point cloud")
            record["actual_train"] = sorted(canonical_name(camera) for camera in train)
            record["actual_evaluation"] = []
            record["initialization"] = {
                "canonical_ply_sha256": sha256(canonical_ply),
                "common_point_count": len(common_cloud.points),
                "policy": "RGB170 original 72,292-point initialization; all400 photometric training",
            }
            return info._replace(
                train_cameras=train,
                test_cameras=[],
                point_cloud=common_cloud,
                ply_path=str(canonical_ply),
                nerf_normalization=dataset_readers.getNerfppNorm(train),
            )

        scene.sceneLoadTypeCallbacks["Colmap"] = explicit_reader
        original_safe_state = general_utils.safe_state

        def seeded_state(*state_args, **state_kwargs):
            original_safe_state(*state_args, **state_kwargs)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

        general_utils.safe_state = seeded_state
        camera_rng = torch.Generator(device="cpu")
        camera_rng.manual_seed(args.seed + 17_031)

        def audited_sample(scene_object, viewpoint_stack):
            if not viewpoint_stack:
                viewpoint_stack.extend(scene_object.getTrainCameras().copy())
            sample_index = int(torch.randint(0, len(viewpoint_stack), (1,), generator=camera_rng).item())
            camera = viewpoint_stack.pop(sample_index)
            trace = record["sample_trace"]
            if len(trace) < 1_000:
                name = Path(camera.image_name).name
                trace.append(name if name.endswith(".png") else name + ".png")
            return camera

        training_runtime.sample_training_viewpoint = audited_sample
        official_cameras = read_camera_requests(Path(access["root"]) / "scene/sparse/0") if "root" in access else read_camera_requests(args.access.parent / "scene/sparse/0")
        image_root = args.access.parent / "scene/images"

        def evidence_builder(model):
            common = dict(
                model=model,
                render=render,
                make_native_view=make_native_view,
                pipe=record["active_pipe"],
                background=record["active_background"],
                cameras=official_cameras,
                image_root=image_root,
                label_root=args.labels,
                output_path=args.repair_record,
                top_count=args.top_count,
                ring_pixels=6,
                minimum_views=3,
                maximum_residual_fraction=0.05,
                minimum_eigengap_ratio=4.0,
                minimum_photo_gradient=0.03,
            )
            if args.selection_mode == "responsibility-asymmetric-fixed-budget":
                return build_responsibility_clone_evidence(
                    **common,
                    center_tolerance_pixels=2,
                    minimum_mixed_fraction=0.5,
                    minimum_background_fraction=0.35,
                )
            if args.selection_mode == "evidence-directed-fixed-budget":
                return build_mixed_geometry_evidence(
                    **common,
                    center_tolerance_pixels=2,
                    minimum_mixed_fraction=0.5,
                    minimum_background_fraction=0.35,
                )
            return build_multiview_tangent_evidence(**common, minimum_density_ratio=1.5)

        # The training context owns pipe/background. Capture them immediately
        # before the one repair event through the score wrapper's context.
        original_compute = densification_stage._compute_improvedgs_scores

        def capture_context(context, gaussians, render_state, iteration):
            record["active_pipe"] = context.pipe
            record["active_background"] = render_state["bg"]
            return original_compute(context, gaussians, render_state, iteration)

        densification_stage._compute_improvedgs_scores = capture_context
        install_function = {
            "native-intersection": install,
            "evidence-directed-fixed-budget": install_evidence_directed,
            "responsibility-asymmetric-fixed-budget": install_responsibility_clone,
        }[args.selection_mode]
        repair_state = install_function(
            model_class=GaussianModel,
            densification_module=densification_stage,
            repair_iteration=args.repair_iteration,
            evidence_builder=evidence_builder,
            transverse_multiplier=args.transverse_multiplier,
        )

        torch.cuda.reset_peak_memory_stats()
        record["torch"] = torch.__version__
        record["cuda"] = torch.version.cuda
        record["device"] = torch.cuda.get_device_name()
        upstream_args = args.upstream_args[1:] if args.upstream_args[:1] == ["--"] else args.upstream_args
        sys.argv = [str(args.source / "train.py"), *upstream_args]
        runpy.run_path(str(args.source / "train.py"), run_name="__main__")
        torch.cuda.synchronize()
        if repair_state["evidence_calls"] != 1 or repair_state["split_calls"] != 1:
            raise RuntimeError(f"Repair event did not execute exactly once: {repair_state}")
        if int(repair_state["repaired_parents"]) <= 0:
            raise RuntimeError("Repair event found no native-selected confirmed bridge parents")
        record["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        record["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        record["status"] = "completed_native_execution"
    except BaseException as exc:
        record.update(status="failed_native_execution", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if repair_state is not None:
            record["repair"] = {key: value for key, value in repair_state.items() if not key.startswith("original_")}
        record.pop("active_pipe", None)
        record.pop("active_background", None)
        try:
            import torch

            if torch.cuda.is_initialized():
                record["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
                record["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        except Exception:
            pass
        record["seconds"] = time.monotonic() - started
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        write_new_json(args.evidence, record)


if __name__ == "__main__":
    main()
