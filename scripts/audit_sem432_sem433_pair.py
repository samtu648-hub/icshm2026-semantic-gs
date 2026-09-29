#!/usr/bin/env python3
"""Audit the full-step stochastic pairing contract of SEM432 and SEM433."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/quality/QUAL-019/pair-audit.json"


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    a_path = ROOT / "outputs/quality/SEM-432/r1/mask2former/training.jsonl"
    b_path = ROOT / "outputs/quality/SEM-433/r1/mask2former/training.jsonl"
    a, b = read_lines(a_path), read_lines(b_path)
    if len(a) != 6000 or len(b) != 6000:
        raise ValueError("Both full training journals are required")
    fields = (
        "sample",
        "attempt_start_cpu_rng_sha256",
        "attempt_start_cuda_rng_sha256",
        "lr_factor",
    )
    checks = {field: all(x[field] == y[field] for x, y in zip(a, b)) for field in fields}
    result = {
        "schema_version": 1,
        "status": "completed_exact" if all(checks.values()) else "failed_pair_mismatch",
        "experiment_id": "QUAL-019-PAIRED-PHOTO-DOMAIN",
        "rows_per_arm": 6000,
        "checks": checks,
        "control_skip_steps": [row["step"] for row in a if row["skips"]],
        "candidate_skip_steps": [row["step"] for row in b if row["skips"]],
        "control_total_amp_skips": sum(len(row["skips"]) for row in a),
        "candidate_total_amp_skips": sum(len(row["skips"]) for row in b),
        "interpretation": "Different overflow-retry locations do not change the next successful update RNG because each retry restores the step-start CPU/CUDA states.",
        "control_journal_sha256": digest(a_path),
        "candidate_journal_sha256": digest(b_path),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
