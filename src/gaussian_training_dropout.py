"""Independent training-time opacity dropout adaptation, with native indexing."""

from __future__ import annotations

import hashlib
import json


class GaussianDropout:
    def __init__(self, maximum=0.1, total_steps=30000, seed=157):
        if not 0 <= maximum < 1 or total_steps <= 0:
            raise ValueError("Invalid dropout schedule")
        self.maximum, self.total_steps, self.seed = maximum, total_steps, seed
        self.step = 0
        self.generator = None
        self.trace = []
        self.stats = {
            "training_renders": 0,
            "dropped_total": 0,
            "gaussians_seen": 0,
            "max_rate": 0.0,
            "densification_events": 0,
        }

    def install(self):
        import torch
        import gaussian_renderer as renderer

        model_class = renderer.GaussianModel
        original_render = renderer.render
        original_opacity = model_class.get_opacity
        original_lr = model_class.update_learning_rate
        original_densify = model_class.densify_and_prune
        owner = self

        def opacity(model):
            value = original_opacity.fget(model)
            factor = getattr(model, "_temporary_dropout_factor", None)
            return value if factor is None else value * factor

        def lr(model, iteration):
            owner.step = iteration
            return original_lr(model, iteration)

        def densify(model, *args, **kwargs):
            if hasattr(model, "_temporary_dropout_factor"):
                raise RuntimeError("Dropout leaked into pruning")
            owner.stats["densification_events"] += 1
            return original_densify(model, *args, **kwargs)

        def render(camera, model, *args, **kwargs):
            training = torch.is_grad_enabled() and owner.step > 0
            if not training:
                return original_render(camera, model, *args, **kwargs)
            rate = owner.maximum * min(owner.step / owner.total_steps, 1.0)
            owner.trace.append({"step": owner.step, "camera": camera.image_name})
            owner.stats["training_renders"] += 1
            if rate == 0:
                return original_render(camera, model, *args, **kwargs)
            value = original_opacity.fget(model)
            if owner.generator is None:
                owner.generator = torch.Generator(device=value.device).manual_seed(
                    owner.seed
                )
            keep = (
                torch.rand(value.shape, device=value.device, generator=owner.generator)
                >= rate
            )
            model._temporary_dropout_factor = keep.to(value.dtype) / (1 - rate)
            try:
                output = original_render(camera, model, *args, **kwargs)
            finally:
                del model._temporary_dropout_factor
            # Native preprocessing can return radii for zero-opacity points.
            # Exclude such points from max-radii and gradient-count statistics.
            radii = output["radii"] * keep.flatten().to(output["radii"].dtype)
            output["radii"] = radii
            output["visibility_filter"] = (radii > 0).nonzero()
            owner.stats["dropped_total"] += int((~keep).sum())
            owner.stats["gaussians_seen"] += keep.numel()
            owner.stats["max_rate"] = max(owner.stats["max_rate"], rate)
            return output

        model_class.get_opacity = property(opacity)
        model_class.update_learning_rate = lr
        model_class.densify_and_prune = densify
        renderer.render = render
        self.original_render = original_render
        self.render = render
        self.restore = lambda: self._restore(
            renderer,
            model_class,
            original_render,
            original_opacity,
            original_lr,
            original_densify,
        )

    @staticmethod
    def _restore(renderer, cls, render, opacity, lr, densify):
        renderer.render, cls.get_opacity = render, opacity
        cls.update_learning_rate, cls.densify_and_prune = lr, densify
