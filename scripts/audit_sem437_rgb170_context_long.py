#!/usr/bin/env python3
"""Independently audit SEM437 using the sealed PNG predictions."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import audit_sem436_rgb170_long_all300 as base

base.EID = "SEM-437-RGB170-MASK2FORMER-CONTEXT50-LONG12K-ALL300"

if __name__ == "__main__":
    base.main()
