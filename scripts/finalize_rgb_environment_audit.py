#!/usr/bin/env python3
"""Record the fixed visual review and close a completed RGB-101 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.semantic2d import sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.audit_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "numeric_audit_complete_visual_review_pending":
        raise ValueError("Expected a single pending RGB-101 visual audit")
    if result.get("numeric_gate_passes") is not False:
        raise ValueError("This closer only records the registered numeric-failure branch")
    review = {
        "review_protocol": "Fixed six full-frame four-column panels plus fixed 054/159/160 native deck crops; no image filtering, crop selection, model change, or metric re-evaluation.",
        "reviewer": "Codex visual inspection on 2026-09-07",
        "views": {
            "054.png": "Candidate is somewhat cleaner than the matched control in diffuse water/sky contamination, but the deck remains blurred and double-edged.",
            "140.png": "Candidate is visually close to the control; diffuse water haze remains and no unambiguous structural recovery is visible.",
            "151.png": "Some broad background ghosting softens, while the bridge/tower remains strongly blurred; this is mixed rather than a major repair.",
            "159.png": "Near-shore and deck-side stretching/ghosting persists. Candidate does not provide a clear repair of this hard view.",
            "160.png": "The same difficult shore/deck distortion persists; no reliable improvement over the control is visible.",
            "183.png": "Candidate is close to the control; water smoothness changes but does not restore reference detail.",
        },
        "major_artifacts_improved_count": 0,
        "obvious_new_erosion_detected": False,
        "visual_gate_passes": False,
        "reason": "The fixed panels show mixed background smoothing, not two unmistakable major-artifact repairs; difficult 159/160 remain visibly defective. The fixed historical challenge-retention numeric gate already failed, so this visual judgment cannot promote the model.",
    }
    review_path = args.audit_dir / "visual_review.json"
    review_path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    result.update({
        "status": "completed_candidate_not_promoted",
        "visual_gate_passes": False,
        "promotion": "drop_keep_RGB-030-FULL_plus_SEM-161",
        "visual_review_sha256": sha256(review_path),
        "SEM180_allowed": False,
        "semantic_evaluation_performed": False,
        "decision": "Fixed historical challenge retention failed by -0.188019 dB (limit -0.100 dB); visual review also fails. Do not tune map capacity, learning rate, or loss within this experiment.",
    })
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "visual_review_sha256": result["visual_review_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
