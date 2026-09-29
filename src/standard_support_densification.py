"""RGB142/143: opacity-marginal evidence changes split ranking only.

No depth, labels, parameter gradients, clone or pruning rules are modified.
Imported only in the isolated experiment worker, never by a frozen model.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path


def support_weight(signed_evidence, exposure, prior=4.0, strength=0.5):
    return 1.0 + strength * signed_evidence / (prior + exposure)


def install(model_class, config, output):
    import torch

    output = Path(output)
    original_stats = model_class.add_densification_stats
    original_split = model_class.densify_and_split
    original_densify = model_class.densify_and_prune
    state = {"iteration": 0, "camera": None, "seconds": 0.0}

    def observe(iteration, camera, loss, model):
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite photometric loss at {iteration}")
        state.update(iteration=iteration, camera=camera.image_name)
        if iteration % 100 == 0:
            with (output / "numeric.jsonl").open("a") as f:
                f.write(json.dumps({"iteration": iteration, "camera": camera.image_name,
                    "loss": float(loss), "points": len(model.get_xyz)}) + "\n")

    def reset(model):
        model._mv_e = torch.zeros(len(model.get_xyz), device="cuda")
        model._mv_r = torch.zeros_like(model._mv_e)
        model._mv_views = set()

    def add_stats(model, screen, visible):
        tick = time.monotonic()
        original_stats(model, screen, visible)
        if not hasattr(model, "_mv_e"):
            reset(model)
        # The first native accumulator spans 600 iterations. Evidence instead
        # uses 100-view blocks, aligned with the 400-camera shuffled epochs.
        if state["iteration"] % 100 == 1:
            reset(model)
        name = state["camera"]
        if name in model._mv_views:
            raise RuntimeError("Repeated camera within a support evidence block")
        model._mv_views.add(name)
        g = model._opacity.grad.detach().flatten()
        if not torch.isfinite(g).all() or not torch.isfinite(screen.grad).all():
            raise FloatingPointError("Nonfinite opacity/screen gradient")
        mag = g.abs()
        active = mag[mag > 0]
        tau = active.median() if len(active) else mag.new_tensor(1.0)
        tau = tau.clamp_min(torch.finfo(mag.dtype).tiny)
        denom = mag + tau
        model._mv_e.add_(-g / denom)
        model._mv_r.add_(mag / denom)
        state["seconds"] += time.monotonic() - tick

    def densify(model, *args, **kwargs):
        model._mv_weight = support_weight(model._mv_e, model._mv_r,
            config["prior"], config["strength"])
        original_densify(model, *args, **kwargs)
        reset(model)
        del model._mv_weight

    def split(model, grads, threshold, extent, N=2):
        tick = time.monotonic()
        if N != 2:
            raise ValueError("Budget assumes native binary splitting")
        n = len(model.get_xyz)
        padded = torch.zeros(n, device=grads.device)
        padded[:len(grads)] = grads.flatten()
        eligible = (padded >= threshold) & (model.get_scaling.max(1).values > model.percent_dense * extent)
        ids = eligible.nonzero().flatten()
        # Both arms use this identical, predeclared capacity envelope and cap.
        # Cloning and pruning remain native and can cause final-count differences.
        target = math.ceil(config["initial_points"] +
            (config["population_envelope"] - config["initial_points"]) *
            min(1., max(0., (state["iteration"] - 500) / 14400)))
        k = min(len(ids), config["split_cap"], max(0, target - n))
        weights = torch.ones(n, device=grads.device)
        weights[:len(model._mv_weight)] = model._mv_weight
        raw = padded[ids]
        scores = raw * weights[ids] if config["arm"] == "support" else raw
        # Stable index tie-break; no RNG consumption by scoring or diagnostics.
        rank = torch.argsort(scores, descending=True, stable=True)
        chosen = ids[rank[:k]]
        plain = ids[torch.argsort(raw, descending=True, stable=True)[:k]]
        masked = torch.zeros_like(padded)
        masked[chosen] = padded[chosen]
        def quantiles(values):
            return torch.quantile(values.float(), values.new_tensor([0., .25, .5, .75, 1.])).tolist() if len(values) else []
        row = {"iteration": state["iteration"], "before_split": n,
            "eligible": len(ids), "selected": k, "population_target": target,
            "different_from_plain": int(k - torch.isin(chosen, plain).sum()),
            "weight_quantiles": quantiles(weights[ids]),
            "selected_weight_quantiles": quantiles(weights[chosen]),
            "evidence_views": len(model._mv_views), "arm": config["arm"],
            "support_host_seconds_cumulative": state["seconds"]}
        with (output / "densification.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        original_split(model, masked, threshold, extent, N)
        state["seconds"] += time.monotonic() - tick

    model_class.add_densification_stats = add_stats
    model_class.densify_and_split = split
    model_class.densify_and_prune = densify
    return observe
