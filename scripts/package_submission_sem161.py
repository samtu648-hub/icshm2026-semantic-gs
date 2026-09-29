#!/usr/bin/env python3
"""Create a new relocatable SEM-161 bundle without changing frozen inputs."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import package_submission_baseline as old

ROOT = Path(__file__).resolve().parents[1]
EXTRA_CODE = ("scripts/render_submission_sem161.py", "src/submission_sem161.py", "src/semantic2d.py", "src/semantic_cable_binary.py", "src/semantic_deck_binary.py", "src/semantic_foundation_verifier.py", "src/semantic_protected_fusion.py", "requirements/submission_sem161_cuda128.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "source-root", "segformer-config", "segformer-license", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if "NVIDIA Source Code License for SegFormer" not in args.segformer_license.read_text():
        raise ValueError("SegFormer original license text required")
    frozen = json.loads((ROOT / "experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json").read_text())
    entries = {row["role"]: row for row in frozen["artifacts"]}
    for row in entries.values():
        if old.sha256(args.artifact_root / row["path"]) != row["sha256"]:
            raise ValueError("Frozen artifact mismatch: "+row["role"])
    assert len(entries) == 18
    subprocess.run([sys.executable, str(ROOT / "scripts/package_submission_baseline.py"), "--rgb-run-dir", str(args.artifact_root / "r1/runs/RGB-030-FULL"), "--semantic-checkpoint", str(args.artifact_root / entries["semantic_3d_sidecar"]["path"]), "--source-root", str(args.source_root), "--output-dir", str(args.output_dir)], check=True)
    lock = json.loads((ROOT / "configs/submission/sem001_baseline_v1.json").read_text())
    lock["submission_id"] = "SUBMIT-030-SEM161-BUNDLE"
    lock["semantic_experiment"] = "SEM-161"
    for role, destination in (("semantic_five_class_2d_checkpoint", "five.pth"), ("stay_cable_binary_checkpoint", "cable.pth"), ("deck_binary_checkpoint", "deck.pth")):
        row = entries[role]
        relative = "model/segformer/"+destination
        target = args.output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.artifact_root / row["path"], target)
        lock["artifacts"].append({"role": role, "bundle_path": relative, "sha256": row["sha256"]})
    shutil.copy2(args.segformer_config, args.output_dir / "model/segformer/config.json")
    shutil.copy2(args.segformer_license, args.output_dir / "model/segformer/LICENSE")
    lock["artifacts"].append({"role": "segformer_config", "bundle_path": "model/segformer/config.json", "sha256": old.sha256(args.segformer_config)})
    for relative in EXTRA_CODE:
        target = args.output_dir / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    old.write_json(args.output_dir / "metadata/sem161_lock.json", lock)
    frozen_manifest = ROOT / "experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json"
    shutil.copy2(frozen_manifest, args.output_dir / "metadata/frozen_manifest.json")
    provenance_path = args.output_dir / "metadata/provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance.update(submission_id="SUBMIT-030-SEM161-BUNDLE", semantic_experiment="SEM-161", artifacts=lock["artifacts"], official_protocol_status="2026-09-07 official announcement: email submission by 2026-09-30; no hidden evaluation has been performed", segformer_model="nvidia/mit-b2", segformer_revision="3bb39e8", segformer_license_sha256=old.sha256(args.segformer_license))
    old.write_json(provenance_path, provenance)
    old.write_json(args.output_dir / "metadata/freeze_verification.json", {"verified": 18, "total": 18, "source_manifest_sha256": old.sha256(frozen_manifest), "bundle_manifest": "metadata/frozen_manifest.json", "artifacts": [{"role": e["role"], "sha256": e["sha256"]} for e in entries.values()]})
    readme = """# Frozen RGB-030-FULL + SEM-161 candidate

No competition images, labels or camera files are included. This is an offline
submission adapter, not an official hidden-evaluation result.

From an arbitrary working directory (absolute --bundle is supported):

    python code/scripts/render_submission_sem161.py --bundle . --cameras CAMERAS.txt --images IMAGES.txt --output-dir NEW_OUTPUT --write-fallbacks

Only SIMPLE_RADIAL COLMAP camera metadata is read. RGB and semantic class-ID PNGs
are written at the input camera dimensions. Class 2 is stay_cable. Output folders
must be empty. Fallbacks: SEM-151, SEM-141, SEM-111, SEM-001. Fixed fusion thresholds:
0.70 cable, 0.10 foundation, 0.99 deck. Low-resolution inputs are edge-padded only
inside the neural branches and cropped back; native inputs use the original tiling.

Runtime: use the separately recorded CUDA/PyTorch/transformers environment and
build the two bundled Scaffold CUDA extensions. Scaffold retains its research-only
license in third_party/scaffold-gs/LICENSE.md; SegFormer pretrained backbone is
NVIDIA mit-b2 (Hugging Face revision 3bb39e8, NVIDIA Source Code License
for SegFormer, non-commercial research/evaluation restriction; see
https://github.com/NVlabs/SegFormer/blob/master/LICENSE).
Redistribution requires reviewing these terms and selecting a project-code license.

The content-addressed metadata/artifact_manifest.json includes inference code and
model weights and `code/requirements/submission_sem161_cuda128.txt`. Verify it
before use. Never use semantic stage PNGs as calibrated
soft probabilities: the final predictor contains hard protected fusion rules.
"""
    (args.output_dir / "README.md").write_text(readme)
    old.write_json(args.output_dir / "metadata/artifact_manifest.json", old._content_manifest(args.output_dir))
    print(json.dumps({"bundle": str(args.output_dir), "frozen_verified": "18/18", "status": "created"}))


if __name__ == "__main__":
    raise SystemExit(main())
