#!/usr/bin/env python3
"""Run the AbsGS arm on spatial_confirmation_v2."""

from pathlib import Path
import run_rgb152_spatial as base

base.EID = "RGB-154-ABSGRAD-SPATIAL30-V2"
base.CONFIG = Path(__file__).resolve().parents[1] / "experiments/runs/RGB-154-ABSGRAD-SPATIAL30-V2/config.yaml"
base.CODE_PATHS = ("scripts/run_rgb154_spatial_absgs_v2.py",) + base.CODE_PATHS

if __name__ == "__main__":
    base.main()
