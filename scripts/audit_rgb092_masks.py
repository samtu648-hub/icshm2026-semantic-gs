#!/usr/bin/env python3
"""Audit frozen final uncertainty masks on training images, without updating anything."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from transformers import AutoModel

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.scaffold_transient import TransientController, protection_mask, observation_weights
from src.semantic2d import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    root = args.output_root
    output = root / "r2/rgb_092/mask_audit/full-20260906-r1"
    output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((REPO / "experiments/runs/RGB-092-TRANSIENT/config.yaml").read_text())
    checkpoint = root / "r1/runs/RGB-092-TRANSIENT/transient_30000.pth"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    controller = TransientController.__new__(TransientController)
    controller.spec = cfg["transient"]
    controller.dino = AutoModel.from_pretrained(root / "r2/rgb_092/dino", local_files_only=True).cuda().eval()
    controller.dino.requires_grad_(False)
    head = torch.nn.Conv2d(384, 1, 1).cuda().eval()
    head.load_state_dict(saved["head"], strict=True)
    selected = {"002.png", "028.png", "162.png", "274.png", "295.png", "298.png"}
    names = json.loads((REPO / cfg["data"]["semantic_split"]).read_text())["semantic_train_labeled"]
    rows = []
    for name in names:
        array = np.array(Image.open(root / "rgb_r1_spatial_v1/scene_400_eval/images" / name).convert("RGB"))
        image = torch.from_numpy(array.copy()).permute(2, 0, 1).cuda().float() / 255
        label = torch.from_numpy(np.array(Image.open(root / "r2/sem_001/labels" / name), dtype=np.uint8)).cuda()
        protected = protection_mask(label, cfg["transient"]["protection_radius"])
        with torch.inference_mode():
            feat = controller.features(image)
            sigma = F.interpolate(F.softplus(head(feat)).clamp_min(.1), size=label.shape, mode="bilinear", align_corners=False)[0, 0]
            weights = observation_weights(sigma, protected, 30000)
            masked = weights < 1
        if not torch.all(weights[protected] == 1):
            raise RuntimeError("Frozen mask violated structure protection")
        rows.append({"name": name, "masked_pixels": int(masked.sum()), "eligible_pixels": int((~protected).sum()),
                     "masked_eligible_fraction": float(masked.sum() / (~protected).sum().clamp_min(1)),
                     "mean_weight": float(weights.mean()), "protected_violations": 0})
        if name in selected:
            m = masked.cpu().numpy()
            rgb_mask = np.repeat(np.where(m, 51, 255).astype(np.uint8)[..., None], 3, axis=2)
            boundary = protected.cpu().numpy()
            overlay = array.copy()
            overlay[m] = (.5 * overlay[m] + .5 * np.array([255, 80, 0])).astype(np.uint8)
            panel = Image.new("RGB", (3960, 1025), "white")
            for i, (im, title) in enumerate([(array, "Reference training RGB"), (rgb_mask, "Weight: dark=0.2 white=1"), (overlay, "Orange=downweighted background")]):
                panel.paste(Image.fromarray(im), (i * 1320, 36))
                ImageDraw.Draw(panel).text((i * 1320 + 10, 10), name + " | " + title, fill="black")
            panel.save(output / name)
    result = {"status": "completed_training_only_mask_audit", "checkpoint_sha256": sha256(checkpoint),
              "training_views": len(rows), "validation_views": 0, "per_view": rows,
              "masked_eligible_fraction": sum(r["masked_pixels"] for r in rows) / sum(r["eligible_pixels"] for r in rows),
              "views_over_95pct_eligible_masked": sum(r["masked_eligible_fraction"] > .95 for r in rows),
              "claim_boundary": "No ground-truth transient masks: coverage does not establish transient detection accuracy."}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_view"}))


if __name__ == "__main__":
    main()
