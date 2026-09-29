"""Training-only cable-network cleanup for frozen SEM-001 probability maps.

SEM-080 incorrectly treated every bridge cable as a direct tower-to-deck
connection.  This predicate instead retains three deterministic line families
from the raw ``stay_cable`` map: a main cable (tower-top to tower-top or image
boundary), a hanger (deck to retained main cable), and a stay cable (tower to
deck).  The module accepts probabilities only; it cannot access RGB, labels,
cameras, validation names, or files.
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
    "structure_contact_radius_px",
    "retained_line_half_width_px",
    "main_max_verticality",
    "stay_min_verticality",
    "hanger_min_verticality",
    "main_tower_top_fraction",
    "main_boundary_band_px",
    "hanger_main_contact_radius_px",
}


def validate_cable_network_rule(rule: Any) -> dict[str, int | float]:
    """Validate the complete fixed cable-network grammar rule."""

    if not isinstance(rule, dict) or set(rule) != _RULE_FIELDS:
        raise ValueError(
            f"cable-network rule must contain exactly {sorted(_RULE_FIELDS)}"
        )
    integer_fields = {
        "hough_threshold",
        "minimum_line_length_px",
        "maximum_line_gap_px",
        "structure_contact_radius_px",
        "retained_line_half_width_px",
        "main_boundary_band_px",
        "hanger_main_contact_radius_px",
    }
    normalized: dict[str, int | float] = {}
    for field in integer_fields:
        value = rule[field]
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        normalized[field] = int(value)
    if not int(normalized["hough_threshold"]) or not int(
        normalized["minimum_line_length_px"]
    ):
        raise ValueError("hough_threshold and minimum_line_length_px must be positive")
    unit_fields = {
        "main_max_verticality",
        "stay_min_verticality",
        "hanger_min_verticality",
        "main_tower_top_fraction",
    }
    for field in unit_fields:
        value = float(rule[field])
        if not np.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{field} must be in (0, 1]")
        normalized[field] = value
    if float(normalized["stay_min_verticality"]) > float(
        normalized["hanger_min_verticality"]
    ):
        raise ValueError("hanger_min_verticality must be at least stay_min_verticality")
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


def _tower_top_mask(tower: np.ndarray, top_fraction: float) -> np.ndarray:
    import cv2

    count, components, statistics, _centroids = cv2.connectedComponentsWithStats(
        tower.astype(np.uint8), connectivity=8
    )
    top = np.zeros(tower.shape, dtype=bool)
    for identifier in range(1, count):
        _x, y, _width, height, _area = statistics[identifier]
        upper_limit = int(y + max(1, int(np.ceil(height * top_fraction))))
        component = components == identifier
        rows = np.arange(tower.shape[0], dtype=np.int32)[:, None]
        top |= component & (rows < upper_limit)
    return top


def _boundary_mask(shape: tuple[int, int], band: int) -> np.ndarray:
    boundary = np.zeros(shape, dtype=bool)
    if band <= 0:
        return boundary
    height, width = shape
    boundary[:band] = True
    boundary[max(0, height - band) :] = True
    boundary[:, :band] = True
    boundary[:, max(0, width - band) :] = True
    return boundary


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    import cv2

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _line_verticality(line: np.ndarray) -> float:
    x1, y1, x2, y2 = (int(value) for value in line)
    length = float(np.hypot(x2 - x1, y2 - y1))
    return abs(y2 - y1) / length if length else 0.0


def _endpoints_touch(first: np.ndarray, second: np.ndarray, line: np.ndarray) -> bool:
    x1, y1, x2, y2 = (int(value) for value in line)
    return bool(
        (first[y1, x1] and second[y2, x2]) or (second[y1, x1] and first[y2, x2])
    )


def _rasterize(
    lines: list[np.ndarray], shape: tuple[int, int], half_width: int
) -> np.ndarray:
    import cv2

    result = np.zeros(shape, dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = (int(value) for value in line)
        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            color=1,
            thickness=2 * half_width + 1,
            lineType=cv2.LINE_8,
        )
    return result.astype(bool)


def cable_network_keep_masks(
    probabilities: Any, rule: Any
) -> tuple[dict[str, np.ndarray], dict[str, int | float]]:
    """Return raw cable pixels retained by main/hanger/stay line templates."""

    import cv2

    array = _validate_probabilities(probabilities)
    settings = validate_cable_network_rule(rule)
    labels = array.argmax(axis=0).astype(np.uint8)
    raw_cable = labels == STAY_CABLE
    tower = labels == TOWER
    deck = labels == DECK
    structure_radius = int(settings["structure_contact_radius_px"])
    tower_contact = _dilate(tower, structure_radius)
    tower_top_contact = _dilate(
        _tower_top_mask(tower, float(settings["main_tower_top_fraction"])),
        structure_radius,
    )
    deck_contact = _dilate(deck, structure_radius)
    boundary = _boundary_mask(raw_cable.shape, int(settings["main_boundary_band_px"]))
    found = cv2.HoughLinesP(
        raw_cable.astype(np.uint8) * 255,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(settings["hough_threshold"]),
        minLineLength=int(settings["minimum_line_length_px"]),
        maxLineGap=int(settings["maximum_line_gap_px"]),
    )
    lines = [] if found is None else [line for line in found.reshape(-1, 4)]
    main_lines: list[np.ndarray] = []
    stay_lines: list[np.ndarray] = []
    for line in lines:
        verticality = _line_verticality(line)
        if verticality <= float(settings["main_max_verticality"]) and (
            _endpoints_touch(tower_top_contact, tower_top_contact, line)
            or _endpoints_touch(tower_top_contact, boundary, line)
        ):
            main_lines.append(line)
        if verticality >= float(settings["stay_min_verticality"]) and _endpoints_touch(
            tower_contact, deck_contact, line
        ):
            stay_lines.append(line)
    half_width = int(settings["retained_line_half_width_px"])
    main_raster = _rasterize(main_lines, raw_cable.shape, half_width) & raw_cable
    main_contact = _dilate(main_raster, int(settings["hanger_main_contact_radius_px"]))
    hanger_lines: list[np.ndarray] = []
    for line in lines:
        if _line_verticality(line) >= float(
            settings["hanger_min_verticality"]
        ) and _endpoints_touch(deck_contact, main_contact, line):
            hanger_lines.append(line)
    hanger_raster = _rasterize(hanger_lines, raw_cable.shape, half_width) & raw_cable
    stay_raster = _rasterize(stay_lines, raw_cable.shape, half_width) & raw_cable
    families = {
        "main_cable": main_raster,
        "hanger": hanger_raster,
        "stay_cable": stay_raster,
    }
    retained = main_raster | hanger_raster | stay_raster
    raw_pixels = int(raw_cable.sum())
    diagnostics: dict[str, int | float] = {
        "raw_stay_cable_pixels": raw_pixels,
        "hough_line_count": len(lines),
        "main_cable_line_count": len(main_lines),
        "hanger_line_count": len(hanger_lines),
        "stay_cable_line_count": len(stay_lines),
        "main_cable_retained_pixels": int(main_raster.sum()),
        "hanger_retained_pixels": int(hanger_raster.sum()),
        "stay_cable_retained_pixels": int(stay_raster.sum()),
        "retained_stay_cable_pixels": int(retained.sum()),
        "transferred_to_background_pixels": int(raw_cable.sum() - retained.sum()),
        "retained_fraction_of_raw_stay_cable": float(retained.sum() / raw_pixels)
        if raw_pixels
        else 1.0,
    }
    return families, diagnostics


def apply_cable_network_probability_transfer(
    probabilities: Any, rule: Any
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int | float]]:
    """Move rejected raw cable mass to background while preserving the simplex."""

    array = _validate_probabilities(probabilities)
    families, diagnostics = cable_network_keep_masks(array, rule)
    retained = np.logical_or.reduce(tuple(families.values()))
    rejected = (array.argmax(axis=0) == STAY_CABLE) & ~retained
    cleaned = array.copy()
    cleaned[BACKGROUND, rejected] += cleaned[STAY_CABLE, rejected]
    cleaned[STAY_CABLE, rejected] = 0.0
    if not np.allclose(cleaned.sum(axis=0), 1.0, rtol=1e-5, atol=1e-5):
        raise RuntimeError(
            "cable-network probability transfer violated the five-class simplex"
        )
    return cleaned, families, diagnostics


def cable_network_triptych(
    raw: np.ndarray, cleaned: np.ndarray, families: dict[str, np.ndarray]
) -> np.ndarray:
    """Return raw / cleaned / family evidence with removed cable in red."""

    raw_labels = np.asarray(raw, dtype=np.uint8)
    clean_labels = np.asarray(cleaned, dtype=np.uint8)
    expected = {"main_cable", "hanger", "stay_cable"}
    if (
        set(families) != expected
        or raw_labels.ndim != 2
        or clean_labels.shape != raw_labels.shape
    ):
        raise ValueError("invalid raw, cleaned or cable family maps")
    if any(np.asarray(mask).shape != raw_labels.shape for mask in families.values()):
        raise ValueError("every cable family map must match the semantic grid")
    evidence = np.zeros((*raw_labels.shape, 3), dtype=np.uint8)
    retained = np.logical_or.reduce(
        tuple(np.asarray(mask, dtype=bool) for mask in families.values())
    )
    evidence[(raw_labels == STAY_CABLE) & ~retained] = (235, 65, 54)
    evidence[np.asarray(families["stay_cable"], dtype=bool)] = (186, 85, 211)
    evidence[np.asarray(families["hanger"], dtype=bool)] = (255, 210, 0)
    evidence[np.asarray(families["main_cable"], dtype=bool)] = (38, 198, 218)
    return np.concatenate(
        (PALETTE[raw_labels], PALETTE[clean_labels], evidence), axis=1
    )
