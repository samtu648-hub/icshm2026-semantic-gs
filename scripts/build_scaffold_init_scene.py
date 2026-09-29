#!/usr/bin/env python3
"""Create an external Scaffold scene that uses a fused P0 initialization PLY."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusion.scaffold_init import build_scaffold_init_scene  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", type=Path, required=True)
    parser.add_argument("--init-ply", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_scaffold_init_scene(args.source_scene, args.init_ply, args.destination), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
