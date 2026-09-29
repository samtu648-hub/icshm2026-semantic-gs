#!/usr/bin/env python3
"""Run the Mask2Former Standard arm on spatial_confirmation_v2."""

from pathlib import Path
import run_sem396_spatial as base

base.EID = "SEM-413-RGB155-SPATIAL255"
base.CONFIG = Path(__file__).resolve().parents[1] / "experiments/runs/SEM-413-RGB155-SPATIAL255/config.yaml"
base.CODE_PATHS = ("scripts/run_sem413_rgb155_spatial255.py",) + base.CODE_PATHS

if __name__ == "__main__":
    base.main()
