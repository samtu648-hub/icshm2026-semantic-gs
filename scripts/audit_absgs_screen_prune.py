#!/usr/bin/env python3
"""Link frozen top contributors to discarded native screen-radius signals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/quality/QUAL-018/r1"


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def arm_audit(arm, contributions):
    trace = ROOT / f"outputs/quality/{arm}/full-r1/trace"
    score_events = [
        event
        for event in rows(trace / "events.jsonl")
        if event["type"] == "densify_begin" and event["screen_threshold"]
    ]
    cache = {}
    output = []
    for record in contributions:
        milestone = record["step"]
        points = record["top_contributors"][:32]
        evidence = {point["id"]: [] for point in points}
        for event in score_events:
            if event["step"] > min(milestone, 14900):
                break
            if event["step"] not in cache:
                with np.load(trace / event["file"]) as z:
                    cache[event["step"]] = (z["ids"].copy(), z["max_radii"].copy())
            ids, radii = cache[event["step"]]
            positions = {
                int(ident): index
                for index, ident in enumerate(ids)
                if int(ident) in evidence
            }
            for ident, index in positions.items():
                if float(radii[index]) > event["screen_threshold"]:
                    evidence[ident].append(
                        {"step": event["step"], "max_radius": float(radii[index])}
                    )
        oversized = [ident for ident, seen in evidence.items() if seen]
        output.append(
            {
                "step": milestone,
                "view": record["view"],
                "remove_top32_delta_psnr_db": record["removals"][2]["delta_psnr_db"],
                "top32_ever_oversized_by_milestone": len(oversized),
                "oversized_ids": oversized,
                "first_oversized_event": {
                    str(ident): seen[0] for ident, seen in evidence.items() if seen
                },
                "max_recorded_radius": {
                    str(ident): max(row["max_radius"] for row in seen)
                    for ident, seen in evidence.items()
                    if seen
                },
            }
        )
    return output


def main():
    analysis = read(OUTPUT / "analysis-result.json")
    result = {
        "status": "passed",
        "definition": "A top-32 frozen contributor is linked only if its exact stable lineage ID had max_radii2D > 20 in a native pre-append score snapshot before the milestone.",
        "arms": {},
    }
    for experiment_id, records in analysis["contributions"].items():
        arm = experiment_id.split("-ABSGS")[0]
        result["arms"][experiment_id] = arm_audit(arm, records)
    destination = OUTPUT / "screen-prune-lineage-audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
