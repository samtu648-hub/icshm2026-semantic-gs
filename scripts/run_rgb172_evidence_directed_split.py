#!/usr/bin/env python3
"""Run the RGB172 evidence-directed fixed-budget bridge repair."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_rgb171_bridge_repair import main


if __name__ == "__main__":
    defaults = [
        "--experiment-id", "RGB-172-IMPROVEDGS-EVIDENCE-DIRECTED-SPLIT",
        "--output-root", str(ROOT / "outputs/quality/RGB-172"),
        "--selection-mode", "evidence-directed-fixed-budget",
    ]
    sys.argv[1:1] = defaults
    main()
