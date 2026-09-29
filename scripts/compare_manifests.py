#!/usr/bin/env python3
"""Compare two dataset manifests without exposing their source paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


FIELDS = (
    "counts",
    "image_resolutions",
    "shape_label_counts",
    "official_class_mapping",
    "camera",
    "aggregate_sha256",
    "fingerprinted_file_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    differences = [field for field in FIELDS if reference.get(field) != candidate.get(field)]
    if differences:
        print("Dataset manifests differ:", file=sys.stderr)
        for field in differences:
            print(f"  {field}", file=sys.stderr)
        return 1
    print("Dataset manifests match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

