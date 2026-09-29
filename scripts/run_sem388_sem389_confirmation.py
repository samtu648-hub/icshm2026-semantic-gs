#!/usr/bin/env python3
"""Run the preregistered second-seed native/context confirmation pair."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_sem385_context_mix as base
import run_sem386_context_memory as memory


ARMS = {
    "control": "SEM-388-RGB150-DEV270-SEED20260919-MEMORY",
    "candidate": "SEM-389-RGB150-DEV270-CONTEXT50-SEED20260919-MEMORY",
}
SCRIPT = "scripts/run_sem388_sem389_confirmation.py"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def bound_arm(config_path: Path, config: dict):
    previous = base.EID, base.CONFIG, base.PARENT, base.CODE_PATHS
    base.EID = config["experiment_id"]
    base.CONFIG = config_path
    base.PARENT = ROOT / config["parent_semantic_run"]
    base.CODE_PATHS = tuple(
        dict.fromkeys((*previous[3], "scripts/run_sem386_context_memory.py", SCRIPT))
    )
    try:
        yield
    finally:
        base.EID, base.CONFIG, base.PARENT, base.CODE_PATHS = previous


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument(
        "--stage",
        choices=("prepare", "preflight", "train", "inference"),
        required=True,
    )
    args = parser.parse_args()
    eid = ARMS[args.arm]
    config_path = ROOT / "experiments/runs" / eid / "config.yaml"
    config = read(config_path)
    if (
        config["experiment_id"] != eid
        or config["seed"] != 20260919
        or not config["memory_policy"]["swin_activation_recomputation"]
    ):
        raise ValueError("Confirmation arm identity, seed, or memory policy changed")
    expected_fraction = 0.0 if args.arm == "control" else 0.5
    if float(config["context_fraction"]) != expected_fraction:
        raise ValueError("Confirmation arm context fraction changed")

    artifact_root = args.artifact_root.resolve()
    with bound_arm(config_path, config):
        if args.stage == "prepare":
            if args.arm == "control":
                with base.shared_identity():
                    base.shared.prepare(config)
            else:
                base.prepare(config)
        elif args.stage == "inference":
            base.inference(config, artifact_root)
        else:
            with memory.memory_policy(config, artifact_root, args.stage):
                getattr(base, args.stage)(config, artifact_root)


if __name__ == "__main__":
    main()
