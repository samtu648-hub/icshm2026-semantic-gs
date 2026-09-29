import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.photo_teacher_gate import PhotoTeacherGate, apply_photo_teacher_gate


def probabilities(labels):
    result = np.full((5, *labels.shape), 0.01, dtype=np.float32)
    for class_id in range(5):
        result[class_id][labels == class_id] = 0.96
    return result


def test_gate_only_recovers_large_flip_stable_deck_and_tower_from_background():
    render = np.zeros((20, 24), dtype=np.uint8)
    render[0:3, 0:3] = 2
    render[3:6, 0:3] = 4
    photo = render.copy()
    photo[8:16, 2:18] = 1  # 128 pixels: accepted deck component.
    photo[0:8, 16:24] = 3  # 64 pixels: accepted tower component.
    photo[18:20, 20:24] = 1  # Too small.
    flipped = photo.copy()
    flipped[8, 2] = 0  # The remaining 127-pixel stable deck component is too small.

    result = apply_photo_teacher_gate(
        probabilities(render), probabilities(photo), probabilities(flipped)
    )

    candidate = result["candidate_prediction"]
    assert np.sum(candidate == 3) == 64
    assert np.sum(candidate == 1) == 0
    assert np.array_equal(candidate == 2, render == 2)
    assert np.array_equal(candidate == 4, render == 4)


def test_gate_rejects_nonregistered_or_invalid_probabilities():
    labels = np.zeros((4, 5), dtype=np.uint8)
    valid = probabilities(labels)
    invalid = valid.copy()
    invalid[0, 0, 0] = 0.5
    try:
        apply_photo_teacher_gate(valid, invalid, valid)
    except ValueError as error:
        assert "sum to one" in str(error)
    else:
        raise AssertionError("Invalid probability mass was accepted")

    try:
        apply_photo_teacher_gate(valid, valid[:, :, :-1], valid)
    except ValueError as error:
        assert "pixel grid" in str(error)
    else:
        raise AssertionError("Mismatched camera grids were accepted")


def test_gate_configuration_rejects_missing_class_threshold():
    labels = np.zeros((8, 8), dtype=np.uint8)
    photo = labels.copy()
    photo[:] = 1
    gate = PhotoTeacherGate(teacher_classes=(1, 3), minimum_component_pixels=((1, 1),))
    try:
        apply_photo_teacher_gate(
            probabilities(labels), probabilities(photo), probabilities(photo), gate
        )
    except ValueError as error:
        assert "class 3" in str(error)
    else:
        raise AssertionError("Missing per-class component threshold was accepted")
