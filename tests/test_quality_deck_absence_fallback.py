import numpy as np

from src.quality_deck_absence_fallback import apply_deck_absence_fallback


def test_existing_native_deck_disables_fallback():
    native = np.array([[1, 0], [2, 3]], dtype=np.uint8)
    fallback = np.ones((2, 2), dtype=np.uint8)
    output, record = apply_deck_absence_fallback(native, fallback, minimum_fallback_deck_pixels=1)
    assert not record["triggered"]
    assert np.array_equal(output, native)


def test_minimum_fallback_support_is_required():
    native = np.zeros((2, 2), dtype=np.uint8)
    fallback = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    output, record = apply_deck_absence_fallback(native, fallback, minimum_fallback_deck_pixels=2)
    assert not record["triggered"]
    assert np.array_equal(output, native)


def test_trigger_adds_deck_only_over_allowed_classes():
    native = np.array([[0, 2], [3, 4]], dtype=np.uint8)
    fallback = np.ones((2, 2), dtype=np.uint8)
    output, record = apply_deck_absence_fallback(native, fallback, minimum_fallback_deck_pixels=2)
    assert record["triggered"]
    assert record["added_deck_pixels"] == 2
    assert np.array_equal(output, np.array([[1, 1], [3, 4]], dtype=np.uint8))
