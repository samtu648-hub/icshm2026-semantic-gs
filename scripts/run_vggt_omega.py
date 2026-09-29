#!/usr/bin/env python3
"""Dry-plan, CPU-validate, or execute the fixed GEO-025-PF VGGT-Omega smoke.

The dry-run path is dependency-free and has no model, torch, network, or GPU
import. Real dependencies are imported only after immutable artifact gates.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry_prior.vggt_omega import (  # noqa: E402
    SUPPORTED_VIEW_COUNTS,
    build_vggt_omega_manifest,
    training_names_from_split,
    verify_fixed_artifacts,
)


DEFAULT_GRAPH = ROOT / "experiments" / "geometry_camera_graph.json"
DEFAULT_SPLIT = ROOT / "experiments" / "splits" / "rgb_r1_spatial_v1.json"
GPU_LOCK = ROOT / "outputs" / "quality" / "gpu.lock"
EXPERIMENT_CONFIGS = {
    "smoke": ROOT / "experiments" / "runs" / "GEO-025-PF" / "config.yaml",
    "full-training": ROOT / "experiments" / "runs" / "GEO-025-P0" / "config.yaml",
}
EXECUTION_BINDING_FILES = (
    Path("scripts/run_vggt_omega.py"),
    Path("src/geometry_prior/vggt_omega.py"),
    Path("src/geometry_prior/vggt_omega_adapter.py"),
    Path("src/geometry_prior/camera_io.py"),
    Path("src/geometry_prior/output_schema.py"),
    Path("configs/geometry/evaluation_v2.json"),
    Path("experiments/splits/rgb_r1_spatial_v1.json"),
    Path("experiments/geometry_camera_graph.json"),
)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_output_dir(path: Path) -> Path:
    output_dir = path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"attempt output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_split(path: Path) -> tuple[dict, list[str]]:
    split = json.loads(path.read_text(encoding="utf-8"))
    return split, training_names_from_split(split)


def build_manifest(
    args: argparse.Namespace, split: dict, training_names: list[str]
) -> dict:
    image_dir = args.image_dir
    if image_dir is None:
        if args.scene_root is None:
            raise ValueError("--image-dir is required unless --scene-root is supplied")
        image_dir = args.scene_root / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {image_dir}")
    available = {path.name for path in image_dir.glob("*.png") if path.is_file()}
    training_available = sorted(set(training_names) & available)
    graph = json.loads(args.camera_graph.read_text(encoding="utf-8"))
    anchors = training_available if args.scope == "full-training" else None
    keyword_args = {
        "view_count": args.view_count,
        "split_id": split.get("split_id"),
        "scope": args.scope.replace("-", "_"),
    }
    if anchors is not None:
        keyword_args["anchors"] = anchors
    return build_vggt_omega_manifest(
        graph,
        training_available,
        **keyword_args,
    )


def require_real_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.source_root is None or args.checkpoint is None:
        raise ValueError("--source-root and --checkpoint are required for real modes")
    license_path = args.checkpoint_license or (args.checkpoint.parent / "LICENSE.txt")
    return args.source_root, args.checkpoint, license_path


def query_compute_apps() -> tuple[str, list[str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    raw = completed.stdout
    disallowed = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and "todesk" not in line.lower()
    ]
    return raw, disallowed


def optional_cuda_failure_metrics() -> dict[str, Any] | None:
    if "torch" not in sys.modules:
        return None
    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
        device = torch.device("cuda")
        return {
            "device_name": torch.cuda.get_device_name(device),
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    except Exception as error:
        return {"collection_error": f"{type(error).__name__}: {error}"}


def run_provenance(
    argv: list[str], experiment_id: str, experiment_config: Path
) -> dict[str, Any]:
    project_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    worktree_dirty = bool(
        subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    )
    binding_files = (*EXECUTION_BINDING_FILES, experiment_config.relative_to(ROOT))
    return {
        "experiment_id": experiment_id,
        "project_git_commit": project_commit,
        "project_worktree_dirty": worktree_dirty,
        "execution_bindings_sha256": {
            str(path): sha256_file(ROOT / path) for path in binding_files
        },
        "experiment_config": str(experiment_config.relative_to(ROOT)),
        "seed": 0,
        "command": [sys.executable, str(Path(__file__).resolve()), *argv],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--validate-load", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--camera-graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-license", type=Path)
    parser.add_argument(
        "--view-count", type=int, choices=SUPPORTED_VIEW_COUNTS, default=25
    )
    parser.add_argument("--repeat-count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--scope", choices=("smoke", "full-training"), default="smoke")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(effective_argv)
    experiment_id = "GEO-025-PF" if args.scope == "smoke" else "GEO-025-P0"
    experiment_config = EXPERIMENT_CONFIGS[args.scope]
    required_repeats = 2 if args.scope == "smoke" else 1
    if args.repeat_count != required_repeats:
        raise ValueError(
            f"--scope {args.scope} requires --repeat-count {required_repeats}"
        )
    output_dir = prepare_output_dir(args.output_dir)
    started = time.monotonic()
    provenance = run_provenance(effective_argv, experiment_id, experiment_config)
    stage = "arguments"
    try:
        if args.dry_run:
            stage = "split_and_window_plan"
            split, training_names = load_split(args.split)
            provenance["split_id"] = split.get("split_id")
            manifest = build_manifest(args, split, training_names)
            manifest["amp_dtype"] = args.amp_dtype
            provenance["elapsed_seconds"] = time.monotonic() - started
            provenance["exit_status"] = 0
            manifest["run_provenance"] = provenance
            output = output_dir / "window_manifest.json"
            atomic_write_json(output, manifest)
            print(
                json.dumps(
                    {
                        "status": "dry_run",
                        "model_execution": "not_run",
                        "view_count": args.view_count,
                        "windows": manifest["window_count"],
                        "manifest": str(output),
                    },
                    sort_keys=True,
                )
            )
            return 0

        stage = "fixed_artifact_gates"
        source_root, checkpoint, checkpoint_license = require_real_paths(args)
        gates = verify_fixed_artifacts(source_root, checkpoint, checkpoint_license)
        adapter = importlib.import_module("geometry_prior.vggt_omega_adapter")

        if args.validate_load:
            stage = "strict_cpu_checkpoint_load"
            result = adapter.validate_checkpoint_load(
                source_root.resolve(), checkpoint.resolve()
            )
            result["artifact_gates"] = gates
            provenance["elapsed_seconds"] = time.monotonic() - started
            provenance["exit_status"] = 0
            result["run_provenance"] = provenance
            atomic_write_json(output_dir / "validate_load.json", result)
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "result": str(output_dir / "validate_load.json"),
                    }
                )
            )
            return 0

        if args.scene_root is None:
            raise ValueError("--scene-root is required for --execute")
        stage = "split_and_window_plan"
        split, training_names = load_split(args.split)
        provenance["split_id"] = split.get("split_id")
        manifest = build_manifest(args, split, training_names)
        manifest["amp_dtype"] = args.amp_dtype
        manifest["run_provenance"] = provenance.copy()
        atomic_write_json(output_dir / "window_manifest.json", manifest)

        stage = "gpu_lock"
        GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
        with GPU_LOCK.open("a", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            stage = "gpu_compute_process_check"
            compute_apps, disallowed = query_compute_apps()
            if disallowed:
                raise RuntimeError(
                    f"GPU has non-ToDesk compute processes after lock: {disallowed}"
                )
            stage = "real_execution_and_npz_audit"
            result = adapter.run_vggt_omega_windows(
                source_root=source_root.resolve(),
                checkpoint=checkpoint.resolve(),
                scene_root=args.scene_root.resolve(),
                output_dir=output_dir,
                manifest=manifest,
                split=split,
                artifact_gates=gates,
                repeat_count=args.repeat_count,
                amp_dtype=args.amp_dtype,
                experiment_id=experiment_id,
                run_scope=args.scope.replace("-", "_"),
            )
            result["gpu_compute_apps_after_lock"] = compute_apps
            provenance["elapsed_seconds"] = time.monotonic() - started
            provenance["exit_status"] = 0
            result["run_provenance"] = provenance
            atomic_write_json(output_dir / "run_summary.json", result)
            completed_manifest = {
                **manifest,
                "model_execution": "completed",
                "status": result["status"],
                "records": result["records"],
                "audit": result["audit"],
                "runtime_summary": {
                    "model_load_seconds": result["model_load_seconds"],
                    "inference_seconds": result["inference_seconds"],
                    "wall_seconds": result["wall_seconds"],
                    "torch_peak_allocated_bytes": result["torch_peak_allocated_bytes"],
                    "torch_peak_reserved_bytes": result["torch_peak_reserved_bytes"],
                },
            }
            atomic_write_json(output_dir / "window_manifest.json", completed_manifest)
            atomic_write_json(
                output_dir / "progress.json",
                {
                    "status": "completed",
                    "experiment_id": experiment_id,
                    "run_scope": args.scope.replace("-", "_"),
                    "completed_windows": manifest["window_count"],
                    "total_windows": manifest["window_count"],
                    "record_count": len(result["records"]),
                    "run_summary": "run_summary.json",
                    "window_manifest": "window_manifest.json",
                },
            )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "summary": str(output_dir / "run_summary.json"),
                }
            )
        )
        return 0
    except Exception as error:
        atomic_write_json(
            output_dir / "failure.json",
            {
                "status": "failed_preserved",
                "stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "cuda_failure_metrics": optional_cuda_failure_metrics(),
                "run_provenance": {
                    **provenance,
                    "elapsed_seconds": time.monotonic() - started,
                    "exit_status": 1,
                },
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
