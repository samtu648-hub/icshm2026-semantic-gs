#!/usr/bin/env python3
"""Run the RGB173 asymmetric responsibility-clone bridge repair."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_rgb171_bridge_repair import main


if __name__ == "__main__":
    defaults = [
        "--experiment-id", "RGB-173-IMPROVEDGS-RESPONSIBILITY-CLONE",
        "--output-root", str(ROOT / "outputs/quality/RGB-173"),
        "--selection-mode", "responsibility-asymmetric-fixed-budget",
    ]
    sys.argv[1:1] = defaults
    main()
