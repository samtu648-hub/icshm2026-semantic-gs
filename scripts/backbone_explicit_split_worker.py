#!/usr/bin/env python3
"""Run Standard 3DGS with an audited explicit train/evaluation partition."""

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
    parser.add_argument(
        "--script", choices=("train.py", "render.py"), default="train.py"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--partition", default="dev30_local_composition_v1")
    parser.add_argument("--expected-train", type=int, default=370)
    parser.add_argument("--expected-evaluation", type=int, default=30)
    parser.add_argument("--expected-universe", type=int, default=400)
    parser.add_argument(
        "--split-shape", choices=("native", "long-axis"), default="native"
    )
    parser.add_argument("--split-distance", type=float, default=0.45)
    parser.add_argument("--split-opacity-multiplier", type=float, default=0.6)
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.evidence.exists():
        raise FileExistsError(f"Worker evidence already exists: {args.evidence}")

    started = time.monotonic()
    record: dict[str, object] = {
        "seed": args.seed,
        "script": args.script,
        "status": "initializing",
        "access_sha256": sha256(args.access),
    }
    try:
        sys.path.insert(0, str(args.source))
        import numpy as np
        import torch  # type: ignore[import-not-found]
        import scene  # type: ignore[import-not-found]
        from gaussian_renderer import network_gui  # type: ignore[import-not-found]
        from scene import dataset_readers  # type: ignore[import-not-found]
        from utils import general_utils  # type: ignore[import-not-found]

        if args.split_shape == "long-axis":
            from src.standard_long_axis_split import install

            split_state = install(
                scene.GaussianModel,
                enabled=True,
                split_distance=args.split_distance,
                opacity_multiplier=args.split_opacity_multiplier,
                preserve_native_rng_draw=True,
            )
        else:
            split_state = {
                "enabled": False,
                "method": "native",
                "split_distance": None,
                "opacity_multiplier": None,
                "preserve_native_rng_draw": False,
            }
        record["split_shape"] = {
            key: value for key, value in split_state.items() if key != "original_split"
        }
        record["camera_rng"] = {
            "implementation": "Python random module in upstream train.py",
            "candidate_hook_uses_python_random": False,
        }

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
            raise ValueError(
                "Access record differs from the registered explicit partition"
            )

        original = scene.sceneLoadTypeCallbacks["Colmap"]

        def explicit_reader(*reader_args, **kwargs):
            bound = inspect.signature(original).bind(*reader_args, **kwargs)
            bound.apply_defaults()
            bound.arguments["eval"] = False
            scene_path = Path(bound.arguments["path"])
            info = original(*bound.args, **bound.kwargs)
            cameras = list(info.train_cameras) + list(info.test_cameras)

            def camera_name(camera) -> str:
                name = Path(camera.image_name).name
                return name if Path(name).suffix else f"{name}.png"

            actual_names = [camera_name(camera) for camera in cameras]
            if (
                len(actual_names) != len(set(actual_names))
                or set(actual_names) != train_names | evaluation_names
            ):
                raise ValueError("Upstream reader changed the explicit image universe")
            train = [camera for camera in cameras if camera_name(camera) in train_names]
            evaluation = [
                camera for camera in cameras if camera_name(camera) in evaluation_names
            ]
            explicit_test = {
                line.strip()
                for line in (scene_path / "sparse/0/test.txt")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            }
            if explicit_test != evaluation_names:
                raise ValueError(
                    "Scene test.txt differs from the registered evaluation role"
                )
            canonical = scene_path / "sparse/0/points3D.ply"
            common_point_cloud = dataset_readers.fetchPly(str(canonical))
            if (
                not len(common_point_cloud.points)
                or not np.isfinite(common_point_cloud.points).all()
            ):
                raise ValueError("Invalid training-only initialization point cloud")
            record["initialization"] = {
                "canonical_ply_sha256": sha256(canonical),
                "native_reader_point_count": len(info.point_cloud.points),
                "common_point_count": len(common_point_cloud.points),
                "native_reader_matched_common": np.array_equal(
                    info.point_cloud.points, common_point_cloud.points
                ),
                "policy": f"DLT re-triangulation and colors from the {len(train_names)} training RGB observations only",
            }
            record["actual_train"] = sorted(camera_name(camera) for camera in train)
            record["actual_evaluation"] = sorted(
                camera_name(camera) for camera in evaluation
            )
            return info._replace(
                train_cameras=train,
                test_cameras=evaluation,
                point_cloud=common_point_cloud,
                ply_path=str(canonical),
                nerf_normalization=dataset_readers.getNerfppNorm(train),
            )

        scene.sceneLoadTypeCallbacks["Colmap"] = explicit_reader
        original_seed = general_utils.safe_state

        def seeded(*seed_args, **seed_kwargs):
            original_seed(*seed_args, **seed_kwargs)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

        general_utils.safe_state = seeded
        network_gui.init = lambda *unused_args, **unused_kwargs: None
        network_gui.try_connect = lambda *unused_args, **unused_kwargs: None
        torch.cuda.reset_peak_memory_stats()
        record.update(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
        )
        upstream_args = (
            args.upstream_args[1:]
            if args.upstream_args[:1] == ["--"]
            else args.upstream_args
        )
        sys.argv = [str(args.source / args.script), *upstream_args]
        runpy.run_path(str(args.source / args.script), run_name="__main__")
        torch.cuda.synchronize()
        record.update(
            status="completed_native_execution",
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        )
        record["split_shape"] = {
            key: value for key, value in split_state.items() if key != "original_split"
        }
    except BaseException as exc:
        record.update(
            status="failed_native_execution", error=f"{type(exc).__name__}: {exc}"
        )
        raise
    finally:
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_initialized():
                record.update(
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                )
        except Exception:
            pass
        record["seconds"] = time.monotonic() - started
        write_new_json(args.evidence, record)


if __name__ == "__main__":
    main()
