#!/usr/bin/env python3
"""Run the seed-thread-corrected second-seed confirmation pair."""

from __future__ import annotations

import sys

import run_sem390_sem391_confirmation as runner


runner.ARMS = {
    "control": "SEM-392-RGB150-DEV270-SEED20260919-MEMORY",
    "candidate": "SEM-393-RGB150-DEV270-CONTEXT50-SEED20260919-MEMORY",
}
runner.SCRIPT = "scripts/run_sem392_sem393_confirmation.py"


if __name__ == "__main__":
    sys.exit(runner.main())
