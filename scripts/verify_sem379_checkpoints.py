#!/usr/bin/env python3
"""Verify that both saved SEM379 checkpoints contain only finite tensors."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.quality_protocol import sha256, write_new_json


def main() -> None:
    started = time.monotonic()
    torch.set_num_threads(4)
    batch = ROOT / "outputs/quality/SEM-379/r1"
    rows = []
    for name in ("model-3000.pt", "model-6000.pt"):
        path = batch / "mask2former" / name
        state = torch.load(path, map_location="cpu", weights_only=True)["model"]
        finite = all(torch.isfinite(tensor).all().item() for tensor in state.values())
        rows.append({"name": name, "tensors": len(state), "finite": finite, "sha256": sha256(path)})
        del state
    assert all(row["finite"] for row in rows)
    write_new_json(batch / "checkpoint_audit.json", {
        "schema_version": 1, "status": "completed",
        "experiment_id": "SEM-379-CABLE-REGION-SAMPLING",
        "checkpoints": rows, "seconds": time.monotonic() - started,
        "script_sha256": sha256(Path(__file__)),
    })


if __name__ == "__main__":
    main()
