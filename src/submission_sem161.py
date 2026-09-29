"""Frozen SEM-161 inference components; no dataset or training entry point."""
from pathlib import Path

import numpy as np

from src.semantic_cable_binary import fuse_cable_probability, normalize_rgb
from src.semantic_deck_binary import recover_deck
from src.semantic_foundation_verifier import verify_foundation
from src.semantic_protected_fusion import fuse_background_stay_cable
from src.semantic2d import tile_starts


def load_segmenter(config_dir: Path, checkpoint: Path, classes: int, experiment: str):
    import torch
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    config = SegformerConfig.from_pretrained(config_dir, local_files_only=True)
    config.num_labels = classes
    model = SegformerForSemanticSegmentation(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["metadata"]["experiment_id"] != experiment:
        raise ValueError("Wrong frozen semantic experiment")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.cuda().eval().requires_grad_(False)


def sliding(model, image: np.ndarray, *, crop: int, stride: int, classes: int):
    """Original sliding/AMP order, with edge padding only below one crop."""
    import torch
    import torch.nn.functional as F

    height, width = image.shape[:2]
    image = np.pad(image, ((0, max(0, crop-height)), (0, max(0, crop-width)), (0, 0)), mode="edge")
    total = np.zeros((classes, *image.shape[:2]), np.float32)
    count = np.zeros(image.shape[:2], np.float32)
    with torch.inference_mode():
        for y in tile_starts(image.shape[0], crop, stride):
            for x in tile_starts(image.shape[1], crop, stride):
                tile = normalize_rgb(image[y:y+crop, x:x+crop]).unsqueeze(0).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = F.interpolate(model(pixel_values=tile).logits, (crop, crop), mode="bilinear", align_corners=False)
                    p = logits.float().softmax(1)[0].cpu().numpy()
                total[:, y:y+crop, x:x+crop] += p
                count[y:y+crop, x:x+crop] += 1
    if np.any(count == 0):
        raise RuntimeError("Uncovered inference pixels")
    probability = total / count[None]
    # SEM-102 normalizes the five-class map; binary final scripts do not.
    if classes == 5:
        probability /= np.maximum(probability.sum(0, keepdims=True), np.finfo(np.float32).tiny)
    return probability[:, :height, :width]


def fuse(base, five_class, cable_probability, deck_probability):
    cable, _ = fuse_cable_probability(base, cable_probability, threshold=0.70)
    foundation, _, _ = verify_foundation(cable, five_class[4], threshold=0.10)
    deck, _, _ = recover_deck(foundation, cable, deck_probability, threshold=0.99)
    fallback, _ = fuse_background_stay_cable(base, five_class, stay_cable_threshold=0.80)
    return {"SEM-001": base, "SEM-111": fallback, "SEM-141": cable, "SEM-151": foundation, "SEM-161": deck}
