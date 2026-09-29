#!/usr/bin/env python3
"""Compare two fixed-protocol GeometryPrior per-view RGB metric CSV files.

The candidate-minus-control paired bootstrap avoids inferring a winner from
separately aggregated means.  It is intentionally independent from organizer
metrics, which have not yet been released.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


LOWER_IS_BETTER = {"LPIPS_Alex"}


def read_rows(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "view" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a view column")
        rows: dict[str, dict[str, float]] = {}
        for row in reader:
            view = row["view"]
            if not view or view in rows:
                raise ValueError(f"{path} has missing or duplicate view {view!r}")
            rows[view] = {key: float(value) for key, value in row.items() if key != "view" and value is not None}
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def aggregate(values: np.ndarray, lower_is_better: bool) -> dict[str, Any]:
    count = max(1, math.ceil(len(values) * 0.10))
    ordered = np.sort(values)
    worst = ordered[-count:] if lower_is_better else ordered[:count]
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "worst_10_percent_mean": float(np.mean(worst)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-views", nargs="*", default=[f"{index}.png" for index in range(140, 147)])
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")

    control = read_rows(args.control)
    candidate = read_rows(args.candidate)
    if set(control) != set(candidate):
        raise ValueError("Control and candidate must contain exactly the same views")
    views = sorted(control)
    metrics = sorted(set.intersection(*(set(row) for row in control.values()), *(set(row) for row in candidate.values())))
    if not metrics:
        raise ValueError("No common metric columns")
    hard_views = [view for view in args.hard_views if view in control]
    rng = np.random.default_rng(args.seed)
    result: dict[str, Any] = {
        "schema_version": 1,
        "comparison": "candidate_minus_control",
        "control_csv": str(args.control),
        "candidate_csv": str(args.candidate),
        "view_count": len(views),
        "hard_views_requested": args.hard_views,
        "hard_views_present": hard_views,
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed, "interval": "95_percentile"},
        "metrics": {},
    }
    sample_indices = rng.integers(0, len(views), size=(args.bootstrap_samples, len(views)))
    for metric in metrics:
        lower_is_better = metric in LOWER_IS_BETTER
        control_values = np.asarray([control[view][metric] for view in views], dtype=np.float64)
        candidate_values = np.asarray([candidate[view][metric] for view in views], dtype=np.float64)
        paired_delta = candidate_values - control_values
        bootstrap = np.mean(paired_delta[sample_indices], axis=1)
        metric_record: dict[str, Any] = {
            "direction": "lower_is_better" if lower_is_better else "higher_is_better",
            "control": aggregate(control_values, lower_is_better),
            "candidate": aggregate(candidate_values, lower_is_better),
            "candidate_minus_control": {
                "mean": float(np.mean(paired_delta)),
                "median": float(np.median(paired_delta)),
                "relative_to_control_mean_percent": float(100.0 * np.mean(paired_delta) / np.mean(control_values)),
                "paired_bootstrap_mean_95_percentile_ci": [float(value) for value in np.percentile(bootstrap, [2.5, 97.5])],
            },
        }
        if hard_views:
            hard_control = np.asarray([control[view][metric] for view in hard_views], dtype=np.float64)
            hard_candidate = np.asarray([candidate[view][metric] for view in hard_views], dtype=np.float64)
            metric_record["hard_views"] = {
                "control_mean": float(np.mean(hard_control)),
                "candidate_mean": float(np.mean(hard_candidate)),
                "candidate_minus_control_mean": float(np.mean(hard_candidate - hard_control)),
            }
        result["metrics"][metric] = metric_record
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
