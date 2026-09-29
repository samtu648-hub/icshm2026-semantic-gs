from src.bridge_defect_manifest import (
    DEFECT_CLEAN,
    DEFECT_MIXED,
    DEFECT_UNSUPPORTED,
    aggregate_gaussian_records,
    classify_view_structure,
)


def test_view_structure_classes_are_exclusive():
    assert classify_view_structure({"top_ellipse_coverage_fraction": 0.49,
                                    "true_like_ellipse_coverage_fraction": 0.48}) == DEFECT_UNSUPPORTED
    assert classify_view_structure({"top_ellipse_coverage_fraction": 0.8,
                                    "true_like_ellipse_coverage_fraction": 0.5}) == DEFECT_MIXED
    assert classify_view_structure({"top_ellipse_coverage_fraction": 0.8,
                                    "true_like_ellipse_coverage_fraction": 0.7}) == DEFECT_CLEAN


def test_mixed_gaussian_requires_cross_view_and_mixed_responsibility():
    rows = []
    for index, kind in enumerate(("artifact_like", "ambiguous", "true_like"), start=1):
        rows.append({
            "top_structure": True, "class": "stay_cable", "gaussian_id": 7,
            "view": f"{index:03d}.png", "kind": kind,
            "near_background_fraction": 0.5, "orientation_error_deg": 45.0,
            "major_radius_px": 100.0, "minor_radius_px": 20.0, "opacity": 0.3,
        })
    output = aggregate_gaussian_records(rows)
    assert len(output) == 1
    assert output[0]["gaussian_id"] == 7
    assert output[0]["defect"] == DEFECT_MIXED

