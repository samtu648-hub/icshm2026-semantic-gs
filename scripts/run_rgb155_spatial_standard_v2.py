#!/usr/bin/env python3
"""Run the original-Standard arm on spatial_confirmation_v2."""

from pathlib import Path
import run_rgb153_spatial_standard as base

base.EID = "RGB-155-STANDARD-SIGNED-SPATIAL30-V2"
base.CONFIG = Path(__file__).resolve().parents[1] / "experiments/runs/RGB-155-STANDARD-SIGNED-SPATIAL30-V2/config.yaml"
base.CODE_PATHS = ("scripts/run_rgb155_spatial_standard_v2.py",) + base.CODE_PATHS

if __name__ == "__main__":
    base.main()
