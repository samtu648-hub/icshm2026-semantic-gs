#!/usr/bin/env python3
"""Run the orthogonal ImprovedGS growth-window and budget comparisons serially."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for script in ("run_rgb169_full_queue.py", "run_rgb170_full_queue.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
