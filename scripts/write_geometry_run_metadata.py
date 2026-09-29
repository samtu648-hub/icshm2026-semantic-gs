#!/usr/bin/env python3
"""Write lightweight timing evidence for one continuous GeometryPrior run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--start-time", required=True)
    parser.add_argument("--end-time", required=True)
    parser.add_argument("--start-seconds", type=int, required=True)
    parser.add_argument("--end-seconds", type=int, required=True)
    parser.add_argument("--record-iterations", nargs="+", type=int, default=[1000, 3000, 5000, 10000])
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": args.experiment_id,
                "iterations": args.record_iterations,
                "continuous_training": f"single_process_0_to_{max(args.record_iterations)}",
                "start_time": args.start_time,
                "end_time": args.end_time,
                "wall_seconds": args.end_seconds - args.start_seconds,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
