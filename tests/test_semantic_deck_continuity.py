import numpy as np
import pytest

from src.semantic_deck_continuity import (
    closed_deck_mask,
    continuity_gate,
    recover_deck_continuity,
    select_continuity_candidate,
)


def test_closing_fills_small_internal_gap_without_expanding_outer_boundary():
    prediction = np.zeros((9, 9), dtype=np.uint8)
    prediction[2:7, 2:7] = 1
    prediction[4, 4] = 0
    closed = closed_deck_mask(prediction, radius=1)
    assert closed[4, 4]
    assert not closed[1, 4]


def test_recovery_requires_probability_and_preserves_all_existing_classes():
    baseline = np.zeros((11, 11), dtype=np.uint8)
    baseline[2:9, 2:9] = 1
    baseline[5, 4:7] = 0
    baseline[0, 0] = 2
    baseline[0, 1] = 3
    baseline[0, 2] = 4
    pre = baseline.copy()
    pre[5, 6] = 4
    probability = np.zeros((11, 11), dtype=np.float32)
    probability[5, 4:7] = [0.99, 0.95, 0.99]

    candidate, enclosed, additions = recover_deck_continuity(
        baseline, pre, probability, radius=2, threshold=0.98
    )

    assert enclosed[5, 4:7].all()
    assert additions[5, 4]
    assert not additions[5, 5]
    assert not additions[5, 6]
    assert candidate[5, 4] == 1
    assert candidate[5, 5] == 0
    assert candidate[5, 6] == 0
    assert np.array_equal(candidate[baseline == 1], baseline[baseline == 1])
    for class_id in (2, 3, 4):
        assert np.array_equal(candidate == class_id, baseline == class_id)


def test_recovery_rejects_invalid_inputs():
    prediction = np.zeros((5, 5), dtype=np.uint8)
    probability = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="positive integer"):
        closed_deck_mask(prediction, radius=0)
    with pytest.raises(ValueError, match="inside"):
        recover_deck_continuity(
            prediction, prediction, probability, radius=1, threshold=1.0
        )


def test_gate_and_selection_follow_registered_order():
    baseline = {
        "mIoU_with_background": 0.80,
        "confusion_matrix_rows_gt_columns_prediction": [[900, 100], [10, 990]],
        "per_class": {"deck": {"iou": 0.90, "precision": 0.95, "recall": 0.96}},
    }
    candidate = {
        "mIoU_with_background": 0.801,
        "confusion_matrix_rows_gt_columns_prediction": [[895, 105], [8, 992]],
        "per_class": {"deck": {"iou": 0.902, "precision": 0.951, "recall": 0.962}},
    }
    gate = continuity_gate(
        candidate=candidate,
        baseline=baseline,
        protected_predictions_exact=True,
        existing_deck_predictions_exact=True,
        minimum_iou_gain=0.001,
        minimum_recall_gain=0.002,
        minimum_precision=0.95,
        minimum_miou_gain=0.0,
        maximum_background_fp_ratio=1.1,
    )
    assert gate["passed"]
    first = {"radius": 5, "threshold": 0.9, "metrics": candidate, "gate": gate}
    second = {"radius": 3, "threshold": 0.8, "metrics": candidate, "gate": gate}
    assert select_continuity_candidate([first, second]) is second
    failed = {**first, "gate": {"passed": False}}
    assert select_continuity_candidate([failed]) is None
