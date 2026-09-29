"""Conservative photo-teacher gates for paired render-domain supervision.

The teacher and student see the same registered camera, but different RGB domains:
the teacher sees the real photo while the student sees the reconstruction render.
This module deliberately does not permit generic photo-to-render replacement.  The
first diagnostic only recovers large, flip-stable deck/tower regions that the render
model called background.  Predicted ``stay_cable`` and ``foundation`` masks are
therefore invariant by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhotoTeacherGate:
    teacher_classes: tuple[int, ...] = (1, 3)
    render_source_class: int = 0
    minimum_component_pixels: tuple[tuple[int, int], ...] = ((1, 128), (3, 64))

    def component_threshold(self, class_id: int) -> int:
        thresholds = dict(self.minimum_component_pixels)
        if class_id not in thresholds:
            raise ValueError(f"No component threshold for class {class_id}")
        return thresholds[class_id]


def _validate_probability(probability: np.ndarray) -> np.ndarray:
    value = np.asarray(probability)
    if value.ndim != 3 or value.shape[0] != 5 or not value.size:
        raise ValueError("Expected a non-empty [5,H,W] probability array")
    if not np.isfinite(value).all() or (value < 0).any():
        raise ValueError("Probability must be finite and non-negative")
    if not np.allclose(value.sum(axis=0), 1.0, atol=2e-5, rtol=0):
        raise ValueError("Class probabilities must sum to one")
    return value


def _large_components(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    from scipy.ndimage import label

    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("A two-dimensional boolean mask is required")
    components, count = label(
        mask, structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    )
    if not count:
        return np.zeros_like(mask)
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    keep = sizes >= minimum_pixels
    keep[0] = False
    return keep[components]


def apply_photo_teacher_gate(
    render_probability: np.ndarray,
    photo_probability: np.ndarray,
    flipped_photo_probability: np.ndarray,
    gate: PhotoTeacherGate = PhotoTeacherGate(),
) -> dict[str, np.ndarray | dict[str, int]]:
    """Return a conservative hard candidate and its accepted pseudo-label mask.

    ``flipped_photo_probability`` must already be flipped back onto the native grid.
    Only background -> deck/tower changes are possible.  Component filtering uses
    the full flip-stable teacher component, not just its overlap with render
    background, so a reconstructed hole inside a real structure remains eligible.
    """

    render = _validate_probability(render_probability)
    photo = _validate_probability(photo_probability)
    flipped = _validate_probability(flipped_photo_probability)
    if render.shape != photo.shape or render.shape != flipped.shape:
        raise ValueError("All domains must share the same registered pixel grid")

    render_prediction = render.argmax(axis=0).astype(np.uint8)
    photo_prediction = photo.argmax(axis=0).astype(np.uint8)
    flipped_prediction = flipped.argmax(axis=0).astype(np.uint8)
    stable = photo_prediction == flipped_prediction
    accepted = np.zeros(render_prediction.shape, dtype=np.bool_)
    counts: dict[str, int] = {}
    for class_id in gate.teacher_classes:
        teacher_component = stable & (photo_prediction == class_id)
        large = _large_components(
            teacher_component, gate.component_threshold(class_id)
        )
        class_accept = (
            large
            & (render_prediction == gate.render_source_class)
            & (photo_prediction == class_id)
        )
        accepted |= class_accept
        counts[str(class_id)] = int(class_accept.sum())

    candidate = render_prediction.copy()
    candidate[accepted] = photo_prediction[accepted]
    if not np.array_equal(candidate == 2, render_prediction == 2):
        raise AssertionError("stay_cable prediction changed despite protection")
    if not np.array_equal(candidate == 4, render_prediction == 4):
        raise AssertionError("foundation prediction changed despite protection")
    counts["total"] = int(accepted.sum())
    counts["flip_stable"] = int(stable.sum())
    return {
        "render_prediction": render_prediction,
        "photo_prediction": photo_prediction,
        "flipped_photo_prediction": flipped_prediction,
        "accepted_mask": accepted,
        "candidate_prediction": candidate,
        "counts": counts,
    }
