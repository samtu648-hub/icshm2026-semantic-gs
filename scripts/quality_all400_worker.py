#!/usr/bin/env python3
"""Run a pinned upstream in a fresh process with explicit camera access auditing.

Only dataset partitioning/common initialization, random seed and GUI disabling
are adapted here. PGSR's implicit >2px TXT filter must not change the common PLY.
The native training/rasterization recipe is not replaced by a surrogate.
"""
from __future__ import annotations

import argparse
import json
import random
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, write_new_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--access", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--script", choices=("train.py", "render.py"), default="train.py")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--scaffold-lifecycle-evidence", type=Path)
    p.add_argument("--trusted-checkpoint-load", action="store_true")
    p.add_argument("upstream_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.evidence.exists():
        raise FileExistsError("Worker evidence exists")
    started = time.monotonic()
    record = {"seed": args.seed, "script": args.script, "status": "initializing"}
    try:
        sys.path.insert(0, str(args.source))
        import numpy as np
        import torch
        import scene
        import scene.gaussian_model as scaffold_gaussian_model
        from scene import dataset_readers
        from utils import general_utils
        from gaussian_renderer import network_gui
        access = json.loads(args.access.read_text())
        train_names, test_names = set(access["train"]), set(access["evaluation"])
        if len(train_names) != 400 or test_names or access.get("partition") != "released_all400_v1":
            raise ValueError("Expected registered all400 training and empty evaluation")
        original = scene.sceneLoadTypeCallbacks["Colmap"]

        def explicit_reader(path, images, evaluate, *rest, **kwargs):
            info = original(path, images, False, *rest, **kwargs)
            cameras = list(info.train_cameras) + list(info.test_cameras)
            def name(cam):
                n = Path(cam.image_name).name
                return n if n.endswith(".png") else n + ".png"
            actual = [name(c) for c in cameras]
            if len(actual) != len(set(actual)) or set(actual) != train_names | test_names:
                raise ValueError("Upstream reader changed explicit image universe")
            train = [c for c in cameras if name(c) in train_names]
            test = [c for c in cameras if name(c) in test_names]
            if info.point_cloud is None:
                raise ValueError("Missing training-only initialization")
            # PGSR always regenerates flat-sparse PLY from TXT and silently drops
            # points above its own 2px error cutoff. The project already fixed a
            # common training-only DLT PLY. Load that identical input explicitly.
            canonical = Path(path) / "sparse/0/points3D.ply"
            common_pcd = dataset_readers.fetchPly(str(canonical))
            if not len(common_pcd.points) or not np.isfinite(common_pcd.points).all():
                raise ValueError("Invalid common initialization point cloud")
            record["initialization"] = {"canonical_ply_sha256": sha256(canonical),
                "native_reader_point_count": len(info.point_cloud.points), "common_point_count": len(common_pcd.points),
                "native_reader_matched_common": np.array_equal(info.point_cloud.points, common_pcd.points),
                "policy": "same RGB110 303-derived DLT initialization; all400 photometric training"}
            record["actual_train"] = sorted(name(c) for c in train)
            record["actual_evaluation"] = sorted(name(c) for c in test)
            return info._replace(train_cameras=train, test_cameras=test,
                                 point_cloud=common_pcd, ply_path=str(canonical),
                                 nerf_normalization=dataset_readers.getNerfppNorm(train))

        scene.sceneLoadTypeCallbacks["Colmap"] = explicit_reader
        original_seed = general_utils.safe_state
        def seeded(*a, **kw):
            original_seed(*a, **kw)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        general_utils.safe_state = seeded
        network_gui.init = lambda *a, **kw: None
        network_gui.try_connect = lambda *a, **kw: None
        if args.trusted_checkpoint_load:
            original_torch_load = torch.load

            def explicit_trusted_load(*load_args, **load_kwargs):
                load_kwargs.setdefault("weights_only", False)
                return original_torch_load(*load_args, **load_kwargs)

            torch.load = explicit_trusted_load
            record["trusted_checkpoint_load"] = "explicit_weights_only_false_for_self_generated_checkpoint"
        if args.scaffold_lifecycle_evidence is not None:
            if args.script != "train.py":
                raise ValueError("Scaffold lifecycle instrumentation is training-only")
            lifecycle = {"adjust_events": [], "save_events": []}
            active_adjust: dict[str, int] | None = None
            original_growing = scaffold_gaussian_model.GaussianModel.anchor_growing
            original_adjust = scaffold_gaussian_model.GaussianModel.adjust_anchor
            original_save = scene.Scene.save

            def observed_growing(self, *method_args, **method_kwargs):
                nonlocal active_adjust
                before = int(self.get_anchor.shape[0])
                result = original_growing(self, *method_args, **method_kwargs)
                if active_adjust is not None:
                    active_adjust["grown"] += int(self.get_anchor.shape[0]) - before
                return result

            def observed_adjust(self, *method_args, **method_kwargs):
                nonlocal active_adjust
                before = int(self.get_anchor.shape[0])
                active_adjust = {"call_index": len(lifecycle["adjust_events"]), "anchors_before": before, "grown": 0}
                try:
                    result = original_adjust(self, *method_args, **method_kwargs)
                    after = int(self.get_anchor.shape[0])
                    active_adjust["anchors_after"] = after
                    active_adjust["pruned"] = before + active_adjust["grown"] - after
                    lifecycle["adjust_events"].append(active_adjust)
                    return result
                finally:
                    active_adjust = None

            def observed_save(self, iteration):
                result = original_save(self, iteration)
                lifecycle["save_events"].append({
                    "iteration": int(iteration),
                    "elapsed_seconds": time.monotonic() - started,
                    "anchors": int(self.gaussians.get_anchor.shape[0]),
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                })
                return result

            scaffold_gaussian_model.GaussianModel.anchor_growing = observed_growing
            scaffold_gaussian_model.GaussianModel.adjust_anchor = observed_adjust
            scene.Scene.save = observed_save
            record["scaffold_lifecycle"] = lifecycle
        torch.cuda.reset_peak_memory_stats()
        record["torch"] = torch.__version__
        record["cuda"] = torch.version.cuda
        record["device"] = torch.cuda.get_device_name()
        upstream_args = args.upstream_args[1:] if args.upstream_args[:1] == ["--"] else args.upstream_args
        sys.argv = [str(args.source / args.script), *upstream_args]
        runpy.run_path(str(args.source / args.script), run_name="__main__")
        torch.cuda.synchronize()
        record["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        record["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        record["status"] = "completed_native_execution"
    except BaseException as exc:
        record.update(status="failed_native_execution", error=f"{type(exc).__name__}: {exc}")
        description = str(exc).lower()
        record["failure_category"] = ("gpu_memory" if "out of memory" in description else
                                      "numeric" if isinstance(exc, FloatingPointError) else
                                      "dependency_or_compile" if isinstance(exc, (ImportError, ModuleNotFoundError)) else "runtime")
        raise
    finally:
        # Failed/OOM attempts also retain the high-water mark; failure is not a
        # quality rejection. Do not mask the original exception if CUDA failed.
        try:
            import torch
            if torch.cuda.is_initialized():
                record["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
                record["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        except Exception:
            pass
        record["seconds"] = time.monotonic() - started
        if args.scaffold_lifecycle_evidence is not None:
            write_new_json(args.scaffold_lifecycle_evidence, record.get("scaffold_lifecycle", {}))
        write_new_json(args.evidence, record)


if __name__ == "__main__":
    main()
