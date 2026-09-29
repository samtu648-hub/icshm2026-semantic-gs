#!/usr/bin/env python3
"""Run the seed-aware second semantic seed on RGB153 Standard renders."""

from __future__ import annotations

from pathlib import Path

import run_sem396_spatial as base


base.EID = "SEM-411-RGB153-SPATIAL256-SEED20260919"
base.CONFIG = Path(__file__).resolve().parents[1] / "experiments/runs/SEM-411-RGB153-SPATIAL256-SEED20260919/config.yaml"
base.CODE_PATHS = (
    "scripts/run_sem411_rgb153_spatial_seed20260919.py",
    "scripts/run_sem396_spatial.py",
    "scripts/run_quality_all300.py",
    "src/quality_mask2former.py",
    "src/quality_mask2former_training.py",
    "src/quality_static_training.py",
    "src/quality_semantic_audit.py",
    "src/quality_seed_initialization.py",
)


if __name__ == "__main__":
    base.main()
