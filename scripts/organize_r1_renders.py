#!/usr/bin/env python3
"""Rename a native 3DGS test render sequence to frozen R1 validation filenames."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True, help="Native zero-padded PNG render directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Empty output directory for validation-view names.")
    parser.add_argument("--mapping-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    views = list(split["primary_fold"]["validation_labeled"])
    source_paths = sorted(path for path in args.source_dir.glob("*.png") if path.is_file())
    if len(source_paths) != len(views):
        raise ValueError(f"Expected {len(views)} test renders, found {len(source_paths)} in {args.source_dir}")
    if [path.name for path in source_paths] != [f"{index:05d}.png" for index in range(len(views))]:
        raise ValueError("Native render sequence must use contiguous zero-padded filenames")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    mapping: list[dict[str, str]] = []
    for view, source_path in zip(views, source_paths):
        destination = args.output_dir / view
        try:
            os.link(source_path, destination)
            method = "hardlink"
        except OSError:
            shutil.copy2(source_path, destination)
            method = "copy"
        mapping.append({"view": view, "source": source_path.name, "destination": destination.name, "method": method})
    args.mapping_json.parent.mkdir(parents=True, exist_ok=True)
    args.mapping_json.write_text(json.dumps({"views": mapping}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"view_count": len(mapping), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
