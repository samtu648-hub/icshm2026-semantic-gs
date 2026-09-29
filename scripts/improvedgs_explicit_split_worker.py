#!/usr/bin/env python3
"""Run pinned ImprovedGS with an audited explicit train/evaluation partition."""
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
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--expected-train", type=int, required=True)
    parser.add_argument("--expected-evaluation", type=int, required=True)
    parser.add_argument("--expected-universe", type=int, required=True)
    parser.add_argument("--expected-points", type=int, required=True)
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.evidence.exists():
        raise FileExistsError(args.evidence)

    started = time.monotonic()
    record: dict[str, object] = {
        "schema_version": 1,
        "seed": args.seed,
        "status": "initializing",
        "access_sha256": sha256(args.access),
        "sample_trace": [],
    }
    try:
        sys.path.insert(0, str(args.source))
        import numpy as np
        import torch
        import scene
        from scene import dataset_readers, training_runtime
        from utils import general_utils

        access = json.loads(args.access.read_text(encoding="utf-8"))
        train_names = set(access["train"])
        evaluation_names = set(access["evaluation"])
        if (
            access.get("partition") != args.partition
            or len(train_names) != args.expected_train
            or len(evaluation_names) != args.expected_evaluation
            or train_names & evaluation_names
            or len(train_names | evaluation_names) != args.expected_universe
        ):
            raise ValueError("Access record differs from the registered explicit partition")

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
                return name if Path(name).suffix else name + ".png"

            actual = [canonical_name(camera) for camera in cameras]
            if len(actual) != len(set(actual)) or set(actual) != train_names | evaluation_names:
                raise ValueError("ImprovedGS reader changed the registered image universe")
            explicit_test = {
                line.strip()
                for line in (scene_path / "sparse/0/test.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            if explicit_test != evaluation_names:
                raise ValueError("Scene test.txt differs from the registered evaluation role")
            train = [camera for camera in cameras if canonical_name(camera) in train_names]
            evaluation = [camera for camera in cameras if canonical_name(camera) in evaluation_names]
            canonical_ply = scene_path / "sparse/0/points3D.ply"
            common_cloud = dataset_readers.fetchPly(str(canonical_ply))
            if len(common_cloud.points) != args.expected_points or not np.isfinite(common_cloud.points).all():
                raise ValueError("Unexpected training-only initialization point cloud")
            record["actual_train"] = sorted(canonical_name(camera) for camera in train)
            record["actual_evaluation"] = sorted(canonical_name(camera) for camera in evaluation)
            record["initialization"] = {
                "canonical_ply_sha256": sha256(canonical_ply),
                "common_point_count": len(common_cloud.points),
                "native_reader_point_count": len(info.point_cloud.points),
                "policy": f"DLT points and colors from {len(train_names)} training RGB observations only",
            }
            return info._replace(
                train_cameras=train,
                test_cameras=evaluation,
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
                trace.append(canonical_name_for_trace(camera))
            return camera

        def canonical_name_for_trace(camera) -> str:
            name = Path(camera.image_name).name
            return name if Path(name).suffix else name + ".png"

        training_runtime.sample_training_viewpoint = audited_sample
        torch.cuda.reset_peak_memory_stats()
        record.update(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name())
        upstream_args = args.upstream_args[1:] if args.upstream_args[:1] == ["--"] else args.upstream_args
        sys.argv = [str(args.source / "train.py"), *upstream_args]
        runpy.run_path(str(args.source / "train.py"), run_name="__main__")
        torch.cuda.synchronize()
        record.update(
            status="completed_native_execution",
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        )
    except BaseException as exc:
        record.update(status="failed_native_execution", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            import torch

            if torch.cuda.is_initialized():
                record.update(
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                )
        except Exception:
            pass
        record["seconds"] = time.monotonic() - started
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        write_new_json(args.evidence, record)


if __name__ == "__main__":
    main()
