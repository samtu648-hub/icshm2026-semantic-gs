#!/usr/bin/env python3
"""SEM386: preserve SEM385 inputs with native Swin activation recomputation.

The serialized base model recipe remains SEM384's. The separate, explicit
memory-policy record describes the runtime-only checkpoint flags; model state,
optimizer groups, losses and inference are unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_sem385_context_mix as base

EID = "SEM-386-RGB150-DEV270-CONTEXT50-MEMORY"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"


def enable_recomputation(model):
    """Use the pinned upstream BasicLayer implementation, without new weights."""
    layers = list(model.backbone.layers)
    if len(layers) != 4 or any(
        not hasattr(layer, "use_checkpoint") for layer in layers
    ):
        raise ValueError("Expected the four native Swin stages")
    for layer in layers:
        layer.use_checkpoint = True
    return {
        "implementation": "native_swin_basiclayer_checkpoint",
        "stages": 4,
        "blocks": [len(layer.blocks) for layer in layers],
        "runtime_override": "backbone.layers[*].use_checkpoint=True",
        "base_resolved_yaml_unchanged": True,
        "inference_uses_native_noncheckpoint_model": True,
    }


@contextmanager
def memory_policy(config, artifact_root, stage):
    base.shared.setup_runtime(artifact_root)
    import src.quality_mask2former as api
    import src.quality_mask2former_training as training
    import torch

    original_loader, original_update = api.load_official_model, training.amp_update
    policy = {
        "experiment_id": config["experiment_id"],
        "stage": stage,
        "model_loads": 0,
        "validated_updates": 0,
        "stage_max_gradient_norm": [0.0] * 4,
    }

    def loader(*args, **kwargs):
        model, loaded = original_loader(*args, **kwargs)
        policy.update(enable_recomputation(model))
        policy["model_loads"] += 1
        return model, loaded

    def update(model, *args, **kwargs):
        result = original_update(model, *args, **kwargs)
        # Exercise every backbone stage in the fresh 4+4 preflight. Later
        # training continues using the parent's finite-gradient checks.
        if stage == "preflight":
            norms = []
            for index, layer in enumerate(model.backbone.layers):
                grads = [p.grad for p in layer.parameters() if p.requires_grad]
                if not grads or any(
                    g is None or not torch.isfinite(g).all() for g in grads
                ):
                    raise RuntimeError("Missing or nonfinite Swin stage gradients")
                # DropPath can legitimately zero a particular block/update.
                norm = float(
                    torch.stack([g.float().square().sum() for g in grads]).sum().sqrt()
                )
                norms.append(norm)
                policy["stage_max_gradient_norm"][index] = max(
                    norm, policy["stage_max_gradient_norm"][index]
                )
            result["swin_stage_gradient_norms"] = norms
            policy["validated_updates"] += 1
            if policy["validated_updates"] == 8 and not all(
                policy["stage_max_gradient_norm"]
            ):
                raise RuntimeError(
                    "A Swin stage had zero gradients throughout the eight updates"
                )
        return result

    api.load_official_model, training.amp_update = loader, update
    try:
        yield
    finally:
        api.load_official_model, training.amp_update = original_loader, original_update
        target = (
            ROOT
            / config["output_path"]
            / ("preflight" if stage == "preflight" else "mask2former")
        )
        if target.exists():
            base.write_new_json(target / "memory-policy.json", policy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("prepare", "preflight", "train", "inference"), required=True
    )
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    if (
        config["experiment_id"] != EID
        or not config["memory_policy"]["swin_activation_recomputation"]
    ):
        raise ValueError("SEM386 identity or memory policy mismatch")
    base.EID, base.CONFIG = EID, CONFIG
    base.CODE_PATHS = (*base.CODE_PATHS, "scripts/run_sem386_context_memory.py")
    if args.stage == "prepare":
        base.prepare(config)
    elif args.stage == "inference":
        base.inference(config, args.artifact_root.resolve())
    else:
        with memory_policy(config, args.artifact_root.resolve(), args.stage):
            getattr(base, args.stage)(config, args.artifact_root.resolve())


if __name__ == "__main__":
    main()
