"""Isolated max_radii2D lifecycle intervention for Standard/AbsGS experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def extend_radius_history(old, total_count: int):
    """Keep existing radii and initialize only newly appended Gaussians to zero."""
    import torch

    old_count = int(old.shape[0])
    if total_count < old_count:
        raise ValueError("Gaussian append unexpectedly reduced the point count")
    if total_count == old_count:
        return old
    return torch.cat(
        (
            old,
            torch.zeros(
                total_count - old_count,
                dtype=old.dtype,
                device=old.device,
            ),
        )
    )


class ScreenRadiusLifecycle:
    """Observe native behavior or preserve old max_radii2D across append calls."""

    def __init__(self, mode: str, output: Path):
        if mode not in {"native", "preserve_existing"}:
            raise ValueError(f"Unsupported screen-radius mode: {mode}")
        self.mode = mode
        self.output = Path(output)
        self.events: list[dict[str, Any]] = []
        self._original_postfix = None
        self._original_prune = None
        self._original_densify_prune = None
        self._active: dict[str, Any] | None = None
        self._prune_index = 0

    def install(self) -> None:
        from scene.gaussian_model import GaussianModel

        if self._original_postfix is not None:
            raise RuntimeError("Screen-radius observer already installed")
        self._original_postfix = GaussianModel.densification_postfix
        self._original_prune = GaussianModel.prune_points
        self._original_densify_prune = GaussianModel.densify_and_prune
        observer = self

        def postfix(model, *args, **kwargs):
            old = model.max_radii2D
            old_count = int(old.shape[0])
            oversized_before = int((old > 20).sum().item())
            observer._original_postfix(model, *args, **kwargs)
            total_count = int(model.max_radii2D.shape[0])
            if observer.mode == "preserve_existing":
                model.max_radii2D = extend_radius_history(old, total_count)
            if observer._active is not None:
                observer._active["append_calls"].append(
                    {
                        "old_count": old_count,
                        "new_count": total_count - old_count,
                        "oversized_before": oversized_before,
                        "oversized_after": int((model.max_radii2D > 20).sum().item()),
                    }
                )

        def prune_points(model, mask):
            if observer._active is not None:
                observer._prune_index += 1
                observer._active["prune_calls"].append(
                    {
                        "index": observer._prune_index,
                        "point_count": int(mask.shape[0]),
                        "mask_count": int(mask.sum().item()),
                        "oversized_available": int(
                            (model.max_radii2D > 20).sum().item()
                        ),
                    }
                )
            return observer._original_prune(model, mask)

        def densify_and_prune(model, *args, **kwargs):
            max_screen_size = kwargs.get("max_screen_size")
            if max_screen_size is None and len(args) >= 5:
                max_screen_size = args[4]
            event = {
                "event_index": len(observer.events) + 1,
                "mode": observer.mode,
                "point_count_before": int(model.get_xyz.shape[0]),
                "max_screen_size": (
                    None if max_screen_size is None else float(max_screen_size)
                ),
                "oversized_before": int((model.max_radii2D > 20).sum().item()),
                "append_calls": [],
                "prune_calls": [],
            }
            observer._active = event
            observer._prune_index = 0
            try:
                result = observer._original_densify_prune(model, *args, **kwargs)
            finally:
                event["point_count_after"] = int(model.get_xyz.shape[0])
                event["oversized_after"] = int(
                    (model.max_radii2D > 20).sum().item()
                )
                observer.events.append(event)
                observer._active = None
            return result

        GaussianModel.densification_postfix = postfix
        GaussianModel.prune_points = prune_points
        GaussianModel.densify_and_prune = densify_and_prune

    def restore(self) -> None:
        if self._original_postfix is None:
            return
        from scene.gaussian_model import GaussianModel

        GaussianModel.densification_postfix = self._original_postfix
        GaussianModel.prune_points = self._original_prune
        GaussianModel.densify_and_prune = self._original_densify_prune
        self._original_postfix = None

    def finalize(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        active = [e for e in self.events if e["max_screen_size"]]
        summary = {
            "schema_version": 1,
            "mode": self.mode,
            "event_count": len(self.events),
            "active_screen_threshold_events": len(active),
            "events_with_oversized_before": sum(
                int(e["oversized_before"] > 0) for e in active
            ),
            "append_calls": sum(len(e["append_calls"]) for e in self.events),
            "append_calls_preserving_oversized": sum(
                int(a["oversized_after"] > 0)
                for e in active
                for a in e["append_calls"]
            ),
            "events": self.events,
        }
        self.output.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

