#!/usr/bin/env python3
"""Segment image order into geometry-prior航段 using pose and co-visibility jumps.

Image numbering is only used to establish an initial order.  Breakpoints are
detected from official camera centers, optical-axis changes and shared-point
support; the output explicitly records the weak ordering assumption.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry_prior.camera_io import read_images_txt  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-dir", type=Path, required=True)
    parser.add_argument("--camera-graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-segment-size", type=int, default=4)
    parser.add_argument("--distance-multiplier", type=float, default=4.0)
    parser.add_argument("--angle-degrees", type=float, default=35.0)
    parser.add_argument("--overlap-ratio", type=float, default=0.08)
    return parser.parse_args()


def _sort_images(images):
    def key(image):
        try:
            return (0, int(Path(image.name).stem), image.image_id)
        except ValueError:
            return (1, image.name, image.image_id)

    return sorted(images, key=key)


def segment_images(camera_dir: Path, graph_path: Path, args: argparse.Namespace) -> dict:
    images = _sort_images(list(read_images_txt(camera_dir / "images.txt").values()))
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    adjacency = graph["adjacency"]
    distances = np.asarray([np.linalg.norm(images[i].center - images[i - 1].center) for i in range(1, len(images))])
    positive = distances[distances > 1e-9]
    baseline = float(np.median(positive)) if positive.size else 1.0
    breaks = []
    transitions = []
    for index in range(1, len(images)):
        previous, current = images[index - 1], images[index]
        distance = float(np.linalg.norm(current.center - previous.center))
        axis_cosine = float(np.clip(np.dot(current.optical_axis, previous.optical_axis), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(axis_cosine)))
        previous_neighbors = {edge["target"]: edge for edge in adjacency.get(previous.name, [])}
        overlap = previous_neighbors.get(current.name, {}).get("shared_point3d", 0)
        denom = max(1, min(len(previous.point3d_ids), len(current.point3d_ids)))
        overlap_ratio = float(overlap / denom)
        reasons = []
        if distance > args.distance_multiplier * baseline:
            reasons.append("center_distance_jump")
        if angle > args.angle_degrees:
            reasons.append("optical_axis_jump")
        if overlap_ratio < args.overlap_ratio:
            reasons.append("low_adjacent_coview")
        transition = {
            "before": previous.name,
            "after": current.name,
            "center_distance": distance,
            "optical_axis_angle_deg": angle,
            "shared_point3d": int(overlap),
            "overlap_ratio": overlap_ratio,
            "break": bool(reasons),
            "reasons": reasons,
        }
        transitions.append(transition)
        if reasons:
            breaks.append(index)

    # Prevent one-pair spikes from creating one-image segments.  A breakpoint
    # is retained only when both resulting sides can be represented as a segment.
    filtered = [index for index in breaks if index >= args.min_segment_size and len(images) - index >= args.min_segment_size]
    boundaries = [0, *filtered, len(images)]
    raw_ranges = [[start, end] for start, end in zip(boundaries[:-1], boundaries[1:])]
    # A single jump can be caused by a bad feature track.  Do not hand a
    # one-image pseudo-video to LingBot: merge short ranges into the larger
    # neighboring range while retaining the raw candidates for audit.
    while len(raw_ranges) > 1 and any(end - start < args.min_segment_size for start, end in raw_ranges):
        short_index = next(i for i, (start, end) in enumerate(raw_ranges) if end - start < args.min_segment_size)
        if short_index == 0:
            raw_ranges[1][0] = raw_ranges[0][0]
            del raw_ranges[0]
        elif short_index == len(raw_ranges) - 1:
            raw_ranges[-2][1] = raw_ranges[-1][1]
            del raw_ranges[-1]
        else:
            left_size = raw_ranges[short_index - 1][1] - raw_ranges[short_index - 1][0]
            right_size = raw_ranges[short_index + 1][1] - raw_ranges[short_index + 1][0]
            if left_size >= right_size:
                raw_ranges[short_index - 1][1] = raw_ranges[short_index][1]
                del raw_ranges[short_index]
            else:
                raw_ranges[short_index + 1][0] = raw_ranges[short_index][0]
                del raw_ranges[short_index]
    segments = []
    for segment_id, (start, end) in enumerate(raw_ranges, start=1):
        members = [image.name for image in images[start:end]]
        segments.append({"segment_id": f"seg-{segment_id:03d}", "start_index": start, "end_index_exclusive": end, "images": members})
    accepted_breakpoints = [start for start, _ in raw_ranges[1:]]
    payload = {
        "schema_version": 1,
        "ordering": "numeric image stem when available; only a weak prior, not a video claim",
        "pose_convention": "CAMERA_FROM_WORLD; centers C=-R^T t",
        "thresholds": {
            "distance_multiplier": args.distance_multiplier,
            "median_step_distance": baseline,
            "optical_axis_angle_degrees": args.angle_degrees,
            "minimum_adjacent_overlap_ratio": args.overlap_ratio,
            "min_segment_size": args.min_segment_size,
        },
        "image_count": len(images),
        "segment_count": len(segments),
        "candidate_breakpoints": filtered,
        "breakpoints": accepted_breakpoints,
        "segments": segments,
        "transitions": transitions,
    }
    return payload


def main() -> int:
    args = _args()
    payload = segment_images(args.camera_dir, args.camera_graph, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "segments": payload["segment_count"], "breakpoints": payload["breakpoints"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
