#!/usr/bin/env python3
"""Materialize the canonical binary initialization PLY from points3D.txt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_all400_initialization import write_ply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args()
    sparse = args.scene / "sparse/0"
    target = sparse / "points3D.ply"
    if target.exists():
        raise FileExistsError(target)
    points = []
    for line in (sparse / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8:
            raise ValueError("Malformed points3D.txt row")
        points.append(
            {
                "xyz": tuple(float(value) for value in fields[1:4]),
                "rgb": tuple(int(value) for value in fields[4:7]),
            }
        )
    if not points:
        raise ValueError("No points found")
    write_ply(target, points)
    print(f"Wrote {len(points)} points to {target}")


if __name__ == "__main__":
    main()
