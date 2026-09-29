#!/usr/bin/env python3
"""Run pinned ImprovedGS with bridge-protected reliability-weighted EAS maps."""

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
    parser.add_argument("--reliability-root", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.evidence.exists():
        raise FileExistsError(args.evidence)

    started = time.monotonic()
    record: dict[str, object] = {
        "schema_version": 1,
        "seed": args.seed,
        "status": "initializing",
        "sample_trace": [],
        "mask_manifest_sha256": sha256(args.mask_manifest),
    }
    eas_state: dict[str, object] | None = None
    try:
        sys.path.insert(0, str(args.source))
        import numpy as np
        import torch
        import scene
        from scene import dataset_readers, training_runtime, training_context
        from utils import general_utils
        from src.improvedgs_bridge_protected_eas import install_bridge_protected_eas

        access = json.loads(args.access.read_text())
        train_names = set(access["train"])
        evaluation_names = set(access["evaluation"])
        if len(train_names) != 400 or evaluation_names or access.get("partition") != "released_all400_v1":
            raise ValueError("Expected the registered released_all400_v1 access set")
        manifest = json.loads(args.mask_manifest.read_text())
        manifest_names = {row["name"] for row in manifest["files"]}
        if manifest.get("count") != 400 or manifest_names != train_names:
            raise ValueError("Reliability manifest does not match all400")
        for row in manifest["files"]:
            path = args.reliability_root / row["name"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise ValueError(f"Reliability map hash mismatch: {path}")

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
                raise ValueError("ImprovedGS reader changed the registered all400 image universe")
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
                "policy": "RGB170 original 72,292-point initialization; all400 real-photo RGB training",
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
        eas_state = install_bridge_protected_eas(
            training_context_module=training_context,
            reliability_root=args.reliability_root,
            expected_names=train_names,
        )
        torch.cuda.reset_peak_memory_stats()
        record["torch"] = torch.__version__
        record["cuda"] = torch.version.cuda
        record["device"] = torch.cuda.get_device_name()
        upstream_args = args.upstream_args[1:] if args.upstream_args[:1] == ["--"] else args.upstream_args
        sys.argv = [str(args.source / "train.py"), *upstream_args]
        runpy.run_path(str(args.source / "train.py"), run_name="__main__")
        torch.cuda.synchronize()
        if eas_state["prepare_calls"] != 1 or eas_state["prepared_views"] != 400:
            raise RuntimeError(f"Bridge-protected EAS did not prepare all400 exactly once: {eas_state}")
        record["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        record["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        record["status"] = "completed_native_execution"
    except BaseException as exc:
        record.update(status="failed_native_execution", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if eas_state is not None:
            record["bridge_protected_eas"] = {
                key: value for key, value in eas_state.items() if key != "original_prepare"
            }
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
