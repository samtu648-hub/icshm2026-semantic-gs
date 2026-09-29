#!/usr/bin/env python3
"""Install an independent transient dropout hook, then run the native worker."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maximum", type=float, required=True)
    p.add_argument("--dropout-seed", type=int, required=True)
    p.add_argument("--dropout-record", type=Path, required=True)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    a = p.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    source = Path(rest[rest.index("--source") + 1])
    sys.path[:0] = [
        str(source),
        str(source / "submodules/diff-gaussian-rasterization"),
        str(source / "submodules/simple-knn"),
    ]
    from src.gaussian_training_dropout import GaussianDropout
    import backbone_explicit_split_worker as worker

    hook = GaussianDropout(a.maximum, 30000, a.dropout_seed)
    hook.install()
    sys.argv = [str(ROOT / "scripts/backbone_explicit_split_worker.py"), *rest]
    try:
        worker.main()
    finally:
        from src.quality_protocol import write_new_json

        write_new_json(
            a.dropout_record,
            {
                "maximum": a.maximum,
                "seed": a.dropout_seed,
                "stats": hook.stats,
                "camera_trace": hook.trace,
            },
        )
        hook.restore()


if __name__ == "__main__":
    main()
