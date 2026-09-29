"""Inference-only structural cleanup for frozen SEM-001 semantic maps.

This module deliberately works *after* semantic rasterization.  Unlike the
SEM-070 anchor mask, a rejected pixel does not alter an anchor's contribution
to another view or another pixel.  A raw ``stay_cable`` pixel is retained only
when a deterministic Hough line is both sufficiently vertical/continuous and
has one endpoint near predicted ``tower`` and the other near predicted
``deck``.  Everything else transfers its cable probability mass to
``background``.

The predicate accepts semantic probabilities only.  It has no filesystem
access and cannot read RGB images, labels, validation splits, or camera data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.semantic_postprocess import BACKGROUND, CLASS_COUNT, STAY_CABLE

DECK = 1
TOWER = 3

PALETTE = np.asarray(
    [[0, 0, 0], [230, 86, 37], [255, 210, 0], [38, 139, 210], [133, 153, 0]],
    dtype=np.uint8,
)

_RULE_FIELDS = {
    "hough_threshold",
    "minimum_line_length_px",
    "maximum_line_gap_px",
    "minimum_verticality",
    "structure_contact_radius_px",
    "retained_line_half_width_px",
}


def validate_structural_rule(rule: Any) -> dict[str, int | float]:
    """Validate the complete fixed rule used by a structural-cleanup run."""

    if not isinstance(rule, dict) or set(rule) != _RULE_FIELDS:
        raise ValueError(f"structural rule must contain exactly {sorted(_RULE_FIELDS)}")
    integer_fields = {
        "hough_threshold",
        "minimum_line_length_px",
        "maximum_line_gap_px",
        "structure_contact_radius_px",
        "retained_line_half_width_px",
    }
    normalized: dict[str, int | float] = {}
    for field in integer_fields:
        value = rule[field]
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        normalized[field] = int(value)
    if (
        int(normalized["hough_threshold"]) == 0
        or int(normalized["minimum_line_length_px"]) == 0
    ):
        raise ValueError("hough_threshold and minimum_line_length_px must be positive")
    verticality = float(rule["minimum_verticality"])
    if not np.isfinite(verticality) or not 0.0 < verticality <= 1.0:
        raise ValueError("minimum_verticality must be in (0, 1]")
    normalized["minimum_verticality"] = verticality
    return normalized


def _validate_probabilities(probabilities: Any) -> np.ndarray:
    array = np.asarray(probabilities)
    if array.ndim != 3 or array.shape[0] != CLASS_COUNT:
        raise ValueError("semantic probabilities must have shape [5, H, W]")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("semantic probabilities must be finite and non-negative")
    if not np.allclose(array.sum(axis=0), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("semantic probabilities must lie on the five-class simplex")
    return np.ascontiguousarray(array, dtype=np.float32)


def structural_keep_mask(
    probabilities: Any, rule: Any
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Return raw cable pixels supported by a tower-to-deck continuous line."""

    import cv2

    array = _validate_probabilities(probabilities)
    settings = validate_structural_rule(rule)
    labels = array.argmax(axis=0).astype(np.uint8)
    raw_cable = labels == STAY_CABLE
    tower = labels == TOWER
    deck = labels == DECK
    radius = int(settings["structure_contact_radius_px"])
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    tower_contact = cv2.dilate(tower.astype(np.uint8), kernel, iterations=1).astype(
        bool
    )
    deck_contact = cv2.dilate(deck.astype(np.uint8), kernel, iterations=1).astype(bool)
    lines = cv2.HoughLinesP(
        (raw_cable.astype(np.uint8) * 255),
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(settings["hough_threshold"]),
        minLineLength=int(settings["minimum_line_length_px"]),
        maxLineGap=int(settings["maximum_line_gap_px"]),
    )
    retained_raster = np.zeros(raw_cable.shape, dtype=np.uint8)
    line_count = 0 if lines is None else len(lines)
    vertical_line_count = 0
    supported_line_count = 0
    if lines is not None:
        for encoded in lines.reshape(-1, 4):
            x1, y1, x2, y2 = (int(value) for value in encoded)
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length <= 0.0 or abs(y2 - y1) / length < float(
                settings["minimum_verticality"]
            ):
                continue
            vertical_line_count += 1
            first_tower = bool(tower_contact[y1, x1])
            first_deck = bool(deck_contact[y1, x1])
            second_tower = bool(tower_contact[y2, x2])
            second_deck = bool(deck_contact[y2, x2])
            if not ((first_tower and second_deck) or (first_deck and second_tower)):
                continue
            supported_line_count += 1
            cv2.line(
                retained_raster,
                (x1, y1),
                (x2, y2),
                color=1,
                thickness=2 * int(settings["retained_line_half_width_px"]) + 1,
                lineType=cv2.LINE_8,
            )
    retained = raw_cable & retained_raster.astype(bool)
    raw_pixels = int(raw_cable.sum())
    kept_pixels = int(retained.sum())
    return retained, {
        "raw_stay_cable_pixels": raw_pixels,
        "hough_line_count": line_count,
        "vertical_hough_line_count": vertical_line_count,
        "tower_deck_supported_line_count": supported_line_count,
        "retained_stay_cable_pixels": kept_pixels,
        "transferred_to_background_pixels": raw_pixels - kept_pixels,
        "retained_fraction_of_raw_stay_cable": float(kept_pixels / raw_pixels)
        if raw_pixels
        else 1.0,
    }


def apply_structural_probability_transfer(
    probabilities: Any, rule: Any
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Transfer rejected raw cable probability to background without changing RGB."""

    array = _validate_probabilities(probabilities)
    retained, diagnostics = structural_keep_mask(array, rule)
    raw_cable = array.argmax(axis=0) == STAY_CABLE
    rejected = raw_cable & ~retained
    cleaned = array.copy()
    cleaned[BACKGROUND, rejected] += cleaned[STAY_CABLE, rejected]
    cleaned[STAY_CABLE, rejected] = 0.0
    if not np.allclose(cleaned.sum(axis=0), 1.0, rtol=1e-5, atol=1e-5):
        raise RuntimeError(
            "structural probability transfer violated the five-class simplex"
        )
    return cleaned, retained, diagnostics


def structural_triptych(
    raw: np.ndarray, cleaned: np.ndarray, retained: np.ndarray
) -> np.ndarray:
    """Visualize raw semantics, cleaned semantics, and retained cable evidence."""

    raw_labels = np.asarray(raw, dtype=np.uint8)
    clean_labels = np.asarray(cleaned, dtype=np.uint8)
    if (
        raw_labels.ndim != 2
        or clean_labels.shape != raw_labels.shape
        or retained.shape != raw_labels.shape
    ):
        raise ValueError("raw, cleaned and retained maps must share one HxW grid")
    if int(raw_labels.max()) >= CLASS_COUNT or int(clean_labels.max()) >= CLASS_COUNT:
        raise ValueError("semantic map contains a class outside 0..4")
    evidence = np.zeros((*raw_labels.shape, 3), dtype=np.uint8)
    removed = (raw_labels == STAY_CABLE) & ~retained
    evidence[removed] = (235, 65, 54)
    evidence[retained] = PALETTE[STAY_CABLE]
    return np.concatenate(
        (PALETTE[raw_labels], PALETTE[clean_labels], evidence), axis=1
    )
