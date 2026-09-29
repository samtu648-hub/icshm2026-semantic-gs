#!/usr/bin/env python3
"""Run the matched render/render control and render/photo candidate on v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_sem396_spatial as spatial
from src.paired_domain_training import paired_mask2former
from src.quality_protocol import sha256, write_new_json


EXPERIMENTS = {
    "control": (
        "SEM-432-PAIRED-RENDER-CONTROL",
        ROOT / "experiments/runs/SEM-432-PAIRED-RENDER-CONTROL/config.yaml",
        "render_render",
    ),
    "candidate": (
        "SEM-433-PAIRED-PHOTO-DOMAIN",
        ROOT / "experiments/runs/SEM-433-PAIRED-PHOTO-DOMAIN/config.yaml",
        "render_photo",
    ),
}
CODE_PATHS = (
    "scripts/run_sem432_sem433_dual_domain.py",
    "scripts/run_sem396_spatial.py",
    "scripts/run_quality_all300.py",
    "src/paired_domain_training.py",
    "src/quality_mask2former.py",
    "src/quality_mask2former_training.py",
    "src/quality_static_training.py",
    "src/quality_semantic_audit.py",
    "src/quality_seed_initialization.py",
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def configure_spatial(eid: str, config_path: Path) -> None:
    spatial.EID = eid
    spatial.CONFIG = config_path
    spatial.CODE_PATHS = CODE_PATHS


def prepare(config: dict, eid: str, config_path: Path, arm: str) -> None:
    configure_spatial(eid, config_path)
    spatial.prepare(config)
    batch = ROOT / config["output_path"]
    split = read(ROOT / config["split_path"])
    names = sorted(split["primary_fold"]["semantic_train_labeled"])
    photo_root = ROOT / config["photo_root"]
    render_root = batch / "export/images"
    photo_hashes = {name: sha256(photo_root / name) for name in names}
    render_hashes = {name: sha256(render_root / name) for name in names}
    write_new_json(
        batch / "paired-source.json",
        {
            "schema_version": 1,
            "experiment_id": eid,
            "arm": arm,
            "first_branch": "render",
            "second_branch": "render" if arm == "render_render" else "photo",
            "pixel_space": "PINHOLE",
            "resolution": [1320, 989],
            "shared_view_crop_flip_and_label": True,
            "training_names": names,
            "render_hashes": render_hashes,
            "photo_hashes": photo_hashes,
            "photo_root": config["photo_root"],
            "confirmation_sources_accessed": False,
            "confirmation_labels_accessed": False,
        },
    )


def train(config: dict, eid: str, config_path: Path, arm: str, artifact_root: Path) -> None:
    configure_spatial(eid, config_path)
    import scripts.run_quality_all300 as all300

    original_mask2former = all300.mask2former
    original_recipe = spatial.training_recipe

    def paired_recipe(current: dict, names: list[str]):
        recipe = original_recipe(current, names)
        recipe.update(
            dual_domain_arm=arm,
            paired_photo_root=str(ROOT / current["photo_root"]),
            paired_steps=current["iterations"],
            paired_crop=current["crop"],
            checkpoint_steps=[3000, 6000],
        )
        return recipe

    all300.mask2former = paired_mask2former
    spatial.training_recipe = paired_recipe
    try:
        spatial.train(config, artifact_root)
    finally:
        all300.mask2former = original_mask2former
        spatial.training_recipe = original_recipe


def inference(config: dict, eid: str, config_path: Path, artifact_root: Path) -> None:
    configure_spatial(eid, config_path)
    spatial.inference(config, artifact_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "inference"), required=True)
    args = parser.parse_args()
    eid, config_path, arm = EXPERIMENTS[args.arm]
    config = read(config_path)
    if config["experiment_id"] != eid or config["dual_domain_arm"] != arm:
        raise ValueError("Paired-domain config identity changed")
    if args.stage == "prepare":
        prepare(config, eid, config_path, arm)
    elif args.stage == "train":
        train(config, eid, config_path, arm, args.artifact_root.resolve())
    else:
        inference(config, eid, config_path, args.artifact_root.resolve())


if __name__ == "__main__":
    main()

