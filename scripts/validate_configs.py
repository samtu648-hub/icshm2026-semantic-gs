#!/usr/bin/env python3
"""Validate JSON-compatible YAML files using only the Python standard library."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    files = [
        root / "configs" / "local.example.yaml",
        root / "configs" / "server.example.yaml",
        root / "experiments" / "registry.yaml",
        root / "experiments" / "templates" / "config.yaml",
    ]
    files.extend(sorted((root / "experiments" / "runs").glob("*/config.yaml")))
    files.extend(sorted((root / "experiments" / "runs").glob("*/audit_config.yaml")))
    failures: list[str] = []
    for path in files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{path.relative_to(root)}: {exc}")
    if failures:
        print("Configuration validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"Validated {len(files)} JSON-compatible YAML files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
