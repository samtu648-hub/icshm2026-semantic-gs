#!/usr/bin/env python3
"""Run the frozen Standard/AbsGS worker with one screen-radius lifecycle mode."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("native", "preserve_existing"), required=True)
    parser.add_argument("--lifecycle-output", type=Path, required=True)
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    source = Path(rest[rest.index("--source") + 1])
    sys.path[:0] = [
        str(source),
        str(source / "submodules/diff-gaussian-rasterization"),
        str(source / "submodules/simple-knn"),
    ]
    import backbone_explicit_split_worker as worker
    from src.absgs_screen_radius_lifecycle import ScreenRadiusLifecycle

    lifecycle = ScreenRadiusLifecycle(args.mode, args.lifecycle_output)
    lifecycle.install()
    sys.argv = [str(ROOT / "scripts/backbone_explicit_split_worker.py"), *rest]
    try:
        worker.main()
    finally:
        # Preserve partial lifecycle evidence even when the native worker fails.
        lifecycle.finalize()
        lifecycle.restore()


if __name__ == "__main__":
    main()
