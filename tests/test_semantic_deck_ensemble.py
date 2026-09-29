import numpy as np
import pytest

from src.semantic_deck_ensemble import (
    blend_deck_probability,
    select_ensemble_candidate,
)


def test_probability_blend_is_global_convex_combination():
    first = np.asarray([[0.0, 1.0]], dtype=np.float32)
    second = np.asarray([[1.0, 0.0]], dtype=np.float32)
    observed = blend_deck_probability(first, second, sem160_weight=0.75)
    np.testing.assert_allclose(observed, [[0.25, 0.75]], rtol=0, atol=1e-7)


def test_probability_blend_rejects_nonconvex_weight_and_shape_mismatch():
    value = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="inside"):
        blend_deck_probability(value, value, sem160_weight=1.0)
    with pytest.raises(ValueError, match="shape"):
        blend_deck_probability(
            value, np.zeros((1, 2), dtype=np.float32), sem160_weight=0.5
        )


def test_selection_uses_metric_then_registered_weight_order():
    metrics = {
        "mIoU_with_background": 0.8,
        "per_class": {"deck": {"iou": 0.9, "recall": 0.91, "precision": 0.92}},
    }
    low = {
        "sem160_weight": 0.25,
        "threshold": 0.99,
        "metrics": metrics,
        "gate": {"passed": True},
    }
    high = {**low, "sem160_weight": 0.75}
    assert select_ensemble_candidate([low, high]) is high
    assert select_ensemble_candidate([{**low, "gate": {"passed": False}}]) is None
