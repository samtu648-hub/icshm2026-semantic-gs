#!/usr/bin/env python3
"""Install read-only trajectory observers around the unchanged native worker."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-config", type=Path, required=True)
    p.add_argument("--trace-output", type=Path, required=True)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    a = p.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    source = Path(rest[rest.index("--source") + 1])
    scene = Path(rest[rest.index("--source_path") + 1])
    sys.path[:0] = [
        str(source),
        str(source / "submodules/diff-gaussian-rasterization"),
        str(source / "submodules/simple-knn"),
    ]
    from src.absgs_trajectory import TrajectoryObserver
    import backbone_explicit_split_worker as worker

    observer = TrajectoryObserver(
        json.loads(a.trace_config.read_text()), a.trace_output, scene
    )
    observer.install()
    sys.argv = [str(ROOT / "scripts/backbone_explicit_split_worker.py"), *rest]
    try:
        worker.main()
        observer.finalize()
    finally:
        observer.restore()


if __name__ == "__main__":
    main()
