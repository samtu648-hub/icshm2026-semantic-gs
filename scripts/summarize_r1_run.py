#!/usr/bin/env python3
"""Extract auditable, non-ranking resource evidence from one completed R1 run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_elapsed_seconds(value: str) -> float:
    fields = [float(field) for field in value.strip().split(":")]
    if len(fields) == 2:
        return fields[0] * 60.0 + fields[1]
    if len(fields) == 3:
        return fields[0] * 3600.0 + fields[1] * 60.0 + fields[2]
    raise ValueError(f"Unrecognized elapsed time: {value!r}")


def parse_time_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    elapsed_match = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(.+)", text)
    rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    exit_match = re.search(r"Exit status:\s*(\d+)", text)
    if elapsed_match is None or rss_match is None or exit_match is None:
        raise ValueError(f"Missing GNU time summary in {path}; only summarize after the wrapper exits")
    return {
        "wall_seconds": parse_elapsed_seconds(elapsed_match.group(1)),
        "host_max_rss_kib": int(rss_match.group(1)),
        "wrapper_exit_status": int(exit_match.group(1)),
    }


def peak_gpu_memory(path: Path) -> int | None:
    if not path.is_file():
        return None
    peak: int | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                value = int(float(row["memory_used_mib"]))
            except (KeyError, TypeError, ValueError):
                continue
            peak = value if peak is None else max(peak, value)
    return peak


def file_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--gpu-samples", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checkpoint = run_dir / f"chkpnt{args.iteration}.pth"
    point_cloud = run_dir / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    metric_summary = run_dir / "r1_metrics_summary.json"
    mapping = run_dir / "r1_render_mapping.json"
    result: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "run_dir": str(run_dir),
        "iteration": args.iteration,
        **parse_time_log(args.log),
        "sampled_peak_gpu_memory_mib": peak_gpu_memory(args.gpu_samples),
        "run_directory_bytes": directory_size(run_dir),
        "checkpoint_bytes": file_size(checkpoint),
        "point_cloud_bytes": file_size(point_cloud),
        "checkpoint_exists": checkpoint.is_file(),
        "point_cloud_exists": point_cloud.is_file(),
        "metrics_summary_exists": metric_summary.is_file(),
        "render_mapping_exists": mapping.is_file(),
    }
    if metric_summary.is_file():
        result["metrics_summary"] = json.loads(metric_summary.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["wrapper_exit_status"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
