"""Portable artifact-root defaults for the public research snapshot."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def artifact_root_default() -> Path:
    value = Path(os.environ.get("ICSHM_OUTPUT_ROOT", "outputs")).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_artifact_root(description: str | None = None) -> Path:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root_default())
    value = parser.parse_args().artifact_root.expanduser()
    return value if value.is_absolute() else ROOT / value
