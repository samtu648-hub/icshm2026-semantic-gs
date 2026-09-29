#!/usr/bin/env python3
"""Audit hashes, image contracts, and encoded-video metadata for VIS-002."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source-point-cloud", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=144)
    parser.add_argument("--expected-width", type=int, default=1320)
    parser.add_argument("--expected-height", type=int, default=989)
    parser.add_argument("--expected-fps", type=int, default=24)
    args = parser.parse_args()

    manifest = json.loads((args.renders / "render_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("visualization_id") != "VIS-002-STANDARD-FLYTHROUGH":
        raise ValueError("Unexpected visualization ID")
    if manifest.get("uses_reference_rgb_or_labels") is not False:
        raise ValueError("Render manifest does not prove the no-reference-data contract")
    if manifest.get("source_point_cloud_sha256") != sha256(args.source_point_cloud):
        raise ValueError("Frozen Standard point-cloud hash differs from render manifest")
    outputs = manifest.get("outputs", [])
    if len(outputs) != args.expected_frames:
        raise ValueError(f"Expected {args.expected_frames} frames, found {len(outputs)}")
    for index, entry in enumerate(outputs):
        expected_name = f"frame{index:04d}.png"
        if entry.get("pose_name") != expected_name:
            raise ValueError(f"Unexpected pose order at {index}: {entry.get('pose_name')}")
        path = args.renders / entry["rgb"]["path"]
        if sha256(path) != entry["rgb"]["sha256"]:
            raise ValueError(f"Frame hash mismatch: {path}")
        image = Image.open(path)
        if image.mode != "RGB" or image.size != (args.expected_width, args.expected_height):
            raise ValueError(f"Invalid frame contract: {path}: {image.mode} {image.size}")

    probe = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration,size:stream=codec_name,width,height,avg_frame_rate,nb_frames", "-of", "json", str(args.video)],
            text=True,
        )
    )
    streams = [stream for stream in probe.get("streams", []) if stream.get("codec_name")]
    if len(streams) != 1:
        raise ValueError("Expected exactly one video stream")
    stream = streams[0]
    expected_video_size = (args.expected_width, args.expected_height + args.expected_height % 2)
    if stream.get("codec_name") != "h264" or (int(stream.get("width", 0)), int(stream.get("height", 0))) != expected_video_size:
        raise ValueError(f"Unexpected video codec/dimensions: {stream}")
    if stream.get("avg_frame_rate") != f"{args.expected_fps}/1" or int(stream.get("nb_frames", 0)) != args.expected_frames:
        raise ValueError(f"Unexpected video timeline: {stream}")

    result = {
        "schema_version": 1,
        "status": "passed",
        "rendered_frames": len(outputs),
        "frame_contract": {"mode": "RGB", "width": args.expected_width, "height": args.expected_height},
        "video": {
            "path": args.video.name,
            "sha256": sha256(args.video),
            "bytes": args.video.stat().st_size,
            "codec": stream["codec_name"],
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": stream["avg_frame_rate"],
            "frames": int(stream["nb_frames"]),
            "duration_seconds": float(probe["format"]["duration"]),
        },
        "render_manifest_sha256": sha256(args.renders / "render_manifest.json"),
        "source_point_cloud_sha256": sha256(args.source_point_cloud),
        "uses_reference_rgb_or_labels": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
