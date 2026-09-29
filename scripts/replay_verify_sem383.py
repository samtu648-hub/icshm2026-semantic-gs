#!/usr/bin/env python3
"""Replay the bound SEM383 verifier with deterministic model construction."""

from __future__ import annotations

import importlib
import random
import runpy


def reset_model_rng(seed, numpy_module, torch_module) -> None:
    random.seed(seed)
    numpy_module.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def main() -> None:
    from scripts import run_sem383

    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    original_make_model = run_sem383.make_model

    def seeded_make_model(config, model_dir):
        reset_model_rng(config["seed"], np, torch)
        return original_make_model(config, model_dir)

    run_sem383.make_model = seeded_make_model
    runpy.run_module("scripts.verify_sem383", run_name="__main__")


if __name__ == "__main__":
    main()
