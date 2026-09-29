#!/usr/bin/env python3
"""Run a registered MapAnything P0 geometry job without hidden data access.

Use ``--dry-run`` to inspect only the explicit training-side co-visibility
windows.  Real execution requires a local official source checkout and an
exact local Apache checkpoint snapshot; it never falls back to Hub download.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry_prior.mapanything_adapter import (  # noqa: E402
    build_window_manifest,
    run_mapanything_windows,
    sha256_file,
    source_commit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True, help="Prepared undistorted PINHOLE scene")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--camera-graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=4)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--minimum-correspondences", type=int, default=64)
    parser.add_argument("--minimum-depth", type=float, default=1e-6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run == args.execute:
        parser.error("provide exactly one of --dry-run or --execute")
    if not (args.scene_root / "images").is_dir() or not (args.scene_root / "sparse" / "0").is_dir():
        parser.error("--scene-root must contain images/ and sparse/0/")
    for path in (args.split, args.camera_graph, args.source_root, args.checkpoint_root, args.checkpoint_file):
        if not path.exists():
            parser.error(f"required path does not exist: {path}")
    actual_source_commit = source_commit(args.source_root)
    if actual_source_commit != args.source_commit:
        parser.error(f"source commit mismatch: expected {args.source_commit}, got {actual_source_commit}")
    actual_checkpoint_sha256 = sha256_file(args.checkpoint_file)
    if actual_checkpoint_sha256 != args.checkpoint_sha256:
        parser.error("checkpoint SHA-256 does not match the registered value")

    manifest = build_window_manifest(
        graph_path=args.camera_graph,
        split_path=args.split,
        window_size=args.window_size,
        overlap=args.overlap,
        max_windows=args.max_windows,
        experiment_id=args.experiment_id,
        source_commit_value=actual_source_commit,
        checkpoint_sha256=actual_checkpoint_sha256,
        checkpoint_revision=args.checkpoint_revision,
    )
    manifest["split_path"] = str(args.split)
    manifest["checkpoint_file_name"] = args.checkpoint_file.name
    manifest["checkpoint_root_name"] = args.checkpoint_root.name
    manifest["amp_dtype"] = args.amp_dtype
    manifest_path = args.output_dir / "window_manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "manifest": str(manifest_path), "window_count": manifest["window_count"]}))
        return 0

    # Explicit opt-in means a new process can be used for each OOM retry.
    result = run_mapanything_windows(
        scene_root=args.scene_root,
        output_dir=args.output_dir,
        window_manifest=manifest,
        checkpoint_root=args.checkpoint_root,
        checkpoint_file=args.checkpoint_file,
        amp_dtype=args.amp_dtype,
        minimum_correspondences=args.minimum_correspondences,
        minimum_depth=args.minimum_depth,
    )
    manifest.update(result)
    manifest["pid"] = os.getpid()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "manifest": str(manifest_path), "records": len(result["records"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
