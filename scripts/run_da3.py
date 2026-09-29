#!/usr/bin/env python3
"""Prepare or run a windowed Depth Anything 3 geometry-prior job."""

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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segments", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=4)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    names = sorted(path.name for path in args.image_dir.glob("*.png"))
    segments = json.loads(args.segments.read_text()) if args.segments else None
    manifest = make_window_manifest(names, args.output_dir / "window_manifest.json", model="DepthAnything3-da3-large", window_size=args.window_size, overlap=args.overlap, segments=segments, max_windows=args.max_windows)
    if args.dry_run:
        print({"status": "dry_run", "windows": manifest["window_count"], "manifest": str(args.output_dir / "window_manifest.json")})
        return 0
    require_optional_package("depth_anything_3", args.source_root, args.checkpoint)
    raise RuntimeError("DA3 adapter is not enabled until the approved source API is pinned")


if __name__ == "__main__":
    raise SystemExit(main())
