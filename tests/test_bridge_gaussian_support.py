import math
from types import SimpleNamespace

import numpy as np

from src.bridge_gaussian_support import project_gaussians, summarize_records, undirected_angle_error


def test_project_isotropic_gaussian_with_official_pinhole():
    camera = SimpleNamespace(
        pose=SimpleNamespace(rotation=np.eye(3), translation=np.zeros(3)),
        matrix=np.array([[100.0, 0.0, 50.0], [0.0, 120.0, 40.0], [0.0, 0.0, 1.0]]),
    )
    result = project_gaussians(
        np.array([[0.0, 0.0, 10.0]]),
        np.array([[1.0, 1.0, 1.0]]),
        np.eye(3)[None],
        camera,
    )
    assert np.allclose([result["u"][0], result["v"][0]], [50.0, 40.0])
    assert np.allclose([result["minor_radius"][0], result["major_radius"][0]], [30.0, 36.0])
    assert result["valid"][0]


def test_project_anisotropic_gaussian_preserves_major_axis():
    camera = SimpleNamespace(
        pose=SimpleNamespace(rotation=np.eye(3), translation=np.zeros(3)),
        matrix=np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]),
    )
    result = project_gaussians(
        np.array([[0.0, 0.0, 10.0]]),
        np.array([[2.0, 1.0, 0.5]]),
        np.eye(3)[None],
        camera,
    )
    assert np.allclose([result["minor_radius"][0], result["major_radius"][0]], [30.0, 60.0])


def test_undirected_angle_error_wraps_half_turn():
    values = undirected_angle_error(np.array([0.0, math.pi - 0.1]), np.array([math.pi, 0.0]))
    assert np.allclose(values, [0.0, 0.1])


def test_summary_keeps_empty_groups_json_safe():
    summary = summarize_records([
        {
            "kind": "true_like", "opacity": 0.5, "major_radius_px": 4.0,
            "minor_radius_px": 1.0, "area_proxy_px2": 4.0, "anisotropy": 4.0,
            "near_background_fraction": 0.1, "cross_view_top_count": 3,
            "orientation_error_deg": 8.0,
        }
    ])
    assert summary["true_like"]["count"] == 1
    assert summary["artifact_like"]["opacity"]["p50"] is None
