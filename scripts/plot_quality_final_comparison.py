#!/usr/bin/env python3
"""Render fixed SEM350 scientific figures; never select or modify predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PALETTE = np.array([[30, 30, 30], [80, 160, 245], [245, 210, 65],
                    [220, 80, 80], [170, 100, 190]], dtype=np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--comparison-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.comparison_root
    result_path = base / "SEM-350-replay-r1/result.json"
    r = json.loads(result_path.read_text())
    assert r["status"] == "passed_independent_fixed74_reconstruction_and_scoring"
    e = r["evaluation"]
    arms = list(e["metrics"])
    labels = ["Old SEM161", "New B2", "New raw", "Calibrated fusion", "Raw + 0.05"]
    colors = ["#727b86", "#abb7c5", "#167aaf", "#bf6924", "#268761"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    values = [e["metrics"][a]["mIoU_with_background"] * 100 for a in arms]
    ax = axes[0, 0]
    bars = ax.barh(labels, values, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set(title="Pooled five-class mIoU (%)", xlabel="Higher is better")
    classes = list(e["metrics"][arms[0]]["per_class"])
    ax = axes[0, 1]
    for a, label, color in zip(arms, labels, colors):
        ax.plot(classes, [e["metrics"][a]["per_class"][c]["IoU"] * 100 for c in classes],
                marker="o", label=label, color=color)
    ax.set(title="Per-class IoU (%)", ylim=(50, 100))
    ax.legend(fontsize=8, ncol=2)
    ax = axes[1, 0]
    groups = list(e["groups"])
    x = np.arange(len(groups))
    for i, (a, color) in enumerate(zip(arms, colors)):
        ax.bar(x + (i - 2) * .16, [e["groups"][g][a]["pooled"]["mIoU_with_background"] * 100
                                  for g in groups], .16, color=color)
    ax.set_xticks(x, ["Challenge 140-146", "Oblique 158-161", "Near 079-082"])
    ax.set(title="Predeclared groups: pooled mIoU (%)", ylim=(0, 100))
    ax = axes[1, 1]
    x = np.arange(len(arms))
    ax.bar(x - .18, [e["metrics"][a]["confusion"][0][2] / 1000 for a in arms], .36,
           label="GT background -> stay_cable", color="#d3a72b")
    ax.bar(x + .18, [e["metrics"][a]["per_class"]["stay_cable"]["fn"] / 1000 for a in arms],
           .36, label="GT stay_cable missed (all classes)", color="#82689d")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set(title="Cable errors: thousands of pixels", ylabel="Lower is better")
    ax.legend(fontsize=8)
    fig.suptitle("SEM350 | Fixed historical 74 development views | seed 20260805\n"
                 "Shared RGB research setting; no blind-test or automatic promotion claim", fontsize=14)
    fig.savefig(args.output / "overview.png", dpi=160)
    plt.close(fig)

    seal = json.loads((base / "SEM-350-predictions-r1/seal.json").read_text())
    # Diagnostic examples chosen after scoring: the worst raw regression and a
    # regressing challenge view. These images never feed selection or calibration.
    names = ["057.png", "146.png"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 7), layout="constrained")
    roots = {"repo": ROOT, "artifact": args.artifact_root}
    for row, name in enumerate(names):
        rgb = ROOT / "outputs/quality/expert_inputs/Scaffold-development-dualline-20260909-r1/images" / name
        gt = args.artifact_root / "r2/sem_001/labels" / name
        images = [np.asarray(Image.open(rgb).convert("RGB")), PALETTE[np.asarray(Image.open(gt))]]
        for a in ["old_sem161", "new_raw", "new_fused"]:
            item = seal["predictions"][name][a]
            path = roots[item["root"]] / item["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
            images.append(PALETTE[np.asarray(Image.open(path))])
        for ax, im, title in zip(axes[row], images, ["New RGB", "Ground truth", "Old SEM161", "New raw", "Calibrated fusion"]):
            ax.imshow(im, interpolation="nearest")
            ax.set_title(f"{name} | {title}")
            ax.axis("off")
    fig.suptitle("SEM350 | Diagnostic regressions (selected after scoring; no tuning)\n"
                 "Yellow: stay_cable | Blue: deck | Red: tower | Purple: foundation", fontsize=13)
    fig.savefig(args.output / "regressions.png", dpi=160)
    plt.close(fig)
    record = {"experiment_id": r["experiment_id"], "config_sha256": r["config_sha256"],
              "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
              "diagnostic_names": names, "posthoc_display_only": True,
              "figures": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(args.output.glob("*.png"))}}
    (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
