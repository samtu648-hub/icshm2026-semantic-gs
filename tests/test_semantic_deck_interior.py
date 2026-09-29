import pytest

try:
    import torch
    import torch.nn.functional as functional
except ImportError:
    torch = None
    functional = None


requires_torch = pytest.mark.skipif(
    torch is None, reason="torch is intentionally absent from the lightweight interpreter"
)

from src.semantic_deck_interior import (
    deck_interior_mask,
    interior_weighted_cross_entropy,
)


@requires_torch
def test_deck_interior_mask_excludes_boundary_band_and_non_deck():
    target = torch.zeros((1, 9, 9), dtype=torch.long)
    target[:, 1:8, 1:8] = 1
    interior = deck_interior_mask(target, radius=2)
    assert interior.shape == target.shape
    assert interior[0, 4, 4] == 1
    assert interior[0, 2, 4] == 0
    assert interior[0, 0, 0] == 0


@requires_torch
def test_multiplier_one_exactly_matches_weighted_cross_entropy():
    logits = torch.tensor(
        [[[[2.0, -1.0], [0.5, -0.5]], [[-2.0, 1.0], [-0.5, 0.5]]]],
        dtype=torch.float32,
    )
    target = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)
    weights = torch.tensor([0.25, 1.75], dtype=torch.float32)
    observed, _ = interior_weighted_cross_entropy(
        logits, target, class_weights=weights, radius=1, multiplier=1.0
    )
    expected = functional.cross_entropy(logits, target, weight=weights)
    torch.testing.assert_close(observed, expected, rtol=0, atol=1e-7)


@requires_torch
def test_interior_error_receives_more_emphasis_than_boundary_error():
    target = torch.zeros((1, 9, 9), dtype=torch.long)
    target[:, 1:8, 1:8] = 1
    weights = torch.ones(2)
    base = torch.full((1, 2, 9, 9), -4.0)
    base[:, 0] = 4.0
    base[:, 1, target[0] == 1] = 4.0
    base[:, 0, target[0] == 1] = -4.0
    center_wrong = base.clone()
    center_wrong[:, 0, 4, 4] = 4.0
    center_wrong[:, 1, 4, 4] = -4.0
    edge_wrong = base.clone()
    edge_wrong[:, 0, 1, 4] = 4.0
    edge_wrong[:, 1, 1, 4] = -4.0
    center_loss, _ = interior_weighted_cross_entropy(
        center_wrong, target, class_weights=weights, radius=2, multiplier=2.0
    )
    edge_loss, _ = interior_weighted_cross_entropy(
        edge_wrong, target, class_weights=weights, radius=2, multiplier=2.0
    )
    assert center_loss > edge_loss


@requires_torch
def test_interior_loss_rejects_invalid_contracts():
    target = torch.zeros((1, 5, 5), dtype=torch.long)
    with pytest.raises(ValueError, match="positive integer"):
        deck_interior_mask(target, radius=0)
    with pytest.raises(ValueError, match="at least one"):
        interior_weighted_cross_entropy(
            torch.zeros((1, 2, 5, 5)),
            target,
            class_weights=torch.ones(2),
            radius=1,
            multiplier=0.5,
        )
