import numpy as np
import pytest

from src.paired_domain_training import paired_crop, validate_domain_sources


def test_paired_crop_replays_crop_and_flip():
    image = np.arange(989 * 1320 * 3, dtype=np.uint32).reshape(989, 1320, 3).astype(np.uint8)
    trace = {"top": 11, "left": 17, "flip": True}
    actual = paired_crop(image, trace, 32)
    expected = np.ascontiguousarray(image[11:43, 17:49][:, ::-1])
    assert np.array_equal(actual, expected)


def test_paired_crop_rejects_out_of_bounds():
    image = np.zeros((989, 1320, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not fit"):
        paired_crop(image, {"top": 900, "left": 0, "flip": False}, 128)


def test_validate_domain_sources_requires_distinct_aligned_pairs():
    render = np.zeros((989, 1320, 3), dtype=np.uint8)
    photo = render.copy()
    photo[0, 0, 0] = 1
    result = validate_domain_sources({"001.png": render}, {"001.png": photo}, ["001.png"])
    assert result["paired_names"] == 1
    with pytest.raises(ValueError, match="must differ"):
        validate_domain_sources({"001.png": render}, {"001.png": render.copy()}, ["001.png"])
