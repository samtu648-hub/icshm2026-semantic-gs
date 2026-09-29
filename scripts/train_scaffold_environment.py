#!/usr/bin/env python3
"""Isolated process adapter for original Scaffold training with an environment."""
import argparse
import json
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--candidate", action="store_true")
    p.add_argument("--run-dir", type=Path, required=True)
    args, native = p.parse_known_args()
    from src.scaffold_semantic_sidecar import configure_scaffold_import
    configure_scaffold_import(args.source_root)
    import torch
    from src.scaffold_environment import Environment, install
    environment = Environment(enabled=args.candidate)
    checkpoint = args.run_dir / "environment.pth"
    install(environment, checkpoint_path=checkpoint)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    sys.argv = [str(args.source_root / "train.py"), *native]
    runpy.run_path(str(args.source_root / "train.py"), run_name="__main__")
    torch.cuda.synchronize()
    loaded = Environment(enabled=args.candidate)
    loaded.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True)["state_dict"])
    exact = torch.equal(loaded.rgb, environment.rgb)
    assert exact and torch.isfinite(environment.rgb).all()
    report = {"seconds": time.perf_counter()-start, "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20, "environment_reload_exact": exact, "environment_mean": float(environment.rgb.mean()), "environment_max": float(environment.rgb.max()), "candidate": args.candidate}
    (args.run_dir / "environment_execution.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
