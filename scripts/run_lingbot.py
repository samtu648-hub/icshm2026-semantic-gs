#!/usr/bin/env python3
"""Prepare or run segmented LingBot-Map geometry-prior jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry_prior.forward_manifest import make_window_manifest, require_optional_package  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=4)
    parser.add_argument("--camera-num-iterations", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--attention", choices=("sdpa", "flash"), default="sdpa")
    parser.add_argument("--cpu-offload", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    names = sorted(path.name for path in args.image_dir.glob("*.png"))
    segments = json.loads(args.segments.read_text())
    manifest = make_window_manifest(names, args.output_dir / "window_manifest.json", model="LingBot-Map-long", window_size=args.window_size, overlap=args.overlap, segments=segments)
    manifest["runtime"] = {"image_size": args.image_size, "camera_num_iterations": args.camera_num_iterations, "attention": args.attention, "cpu_offload": args.cpu_offload}
    (args.output_dir / "window_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print({"status": "dry_run", "segments": segments["segment_count"], "windows": manifest["window_count"], "manifest": str(args.output_dir / "window_manifest.json")})
        return 0
    require_optional_package("lingbot", args.source_root, args.checkpoint)
    raise RuntimeError("LingBot adapter is not enabled until the approved source API is pinned")


if __name__ == "__main__":
    raise SystemExit(main())
