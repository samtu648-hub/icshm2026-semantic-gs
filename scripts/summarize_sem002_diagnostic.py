#!/usr/bin/env python3
"""Deterministically merge bounded SEM-020-DIAG training-only chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def finalize(raw: dict[str, float]) -> dict[str, float]:
    tiny = float.fromhex("0x1.0p-1022")
    return {
        "pixels_with_foreground": raw["pixels_with_foreground"],
        "retained_contribution_mass_fraction": raw["retained_mass"] / max(raw["foreground_mass"], tiny),
        "retained_contributor_fraction": raw["retained_contributors"] / max(raw["contributors"], 1.0),
        "max_contribution_mean": raw["max_contribution_sum"] / max(raw["pixels_with_foreground"], 1.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", type=Path, action="append", required=True, help="SEM-020-DIAG chunk result.json; repeat in any order")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chunks = [json.loads(path.read_text(encoding="utf-8")) for path in args.chunk]
    if any(item.get("status") != "completed_training_side_only_diagnostic" for item in chunks):
        raise ValueError("Every input must be a completed SEM-020-DIAG chunk")
    total = chunks[0]["training_view_chunk"]["total_training_views"]
    ranges = sorted((int(item["training_view_chunk"]["start"]), int(item["training_view_chunk"]["end_exclusive"])) for item in chunks)
    cursor = 0
    for start, end in ranges:
        if start != cursor or end <= start:
            raise ValueError("Diagnostic chunks must be contiguous, non-overlapping and start at zero")
        cursor = end
    if cursor != total:
        raise ValueError(f"Diagnostic chunks cover 0:{cursor}, not all 0:{total} training cameras")
    merged: dict[str, Any] = {
        "schema_version": 2,
        "experiment_id": "SEM-020-DIAG",
        "status": "completed_training_side_only_diagnostic",
        "uses_validation_rgb_or_labels": False,
        "uses_unlabeled_rgb": False,
        "training_view_chunks": [{"start": start, "end_exclusive": end} for start, end in ranges],
        "threshold_diagnostics": {},
    }
    gates = sorted(chunks[0]["threshold_diagnostics"])
    for gate in gates:
        summaries = [item["threshold_diagnostics"][gate] for item in chunks]
        if any(summary["threshold"] != summaries[0]["threshold"] or summary["selection_rule"] != summaries[0]["selection_rule"] for summary in summaries):
            raise ValueError(f"{gate} chunks disagree on the frozen threshold or rule")
        raw = {key: 0.0 for key in summaries[0]["raw_accumulator"]}
        for summary in summaries:
            for key, value in summary["raw_accumulator"].items():
                raw[key] += float(value)
        merged["threshold_diagnostics"][gate] = {
            "threshold": summaries[0]["threshold"],
            "selection_rule": summaries[0]["selection_rule"],
            "training_views": total,
            "raw_accumulator": raw,
            **finalize(raw),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": merged["experiment_id"], "status": merged["status"], "training_views": total}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
