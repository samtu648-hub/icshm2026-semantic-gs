#!/usr/bin/env python3
"""Execute the registered matched RGB control/candidate in isolated sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--mode", choices=["prepare", "smoke", "full"], required=True)
    p.add_argument("--dino-dir", type=Path)
    p.add_argument("--run-suffix", default="")
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    root = args.output_root.resolve()
    if shutil.disk_usage(root).free < 100 * 1024**3:
        raise RuntimeError("Registered disk floor: less than 100 GiB free")
    for rel, expected in [(cfg["dataset_manifest"], cfg["data"]["manifest_sha256"]),
                          (cfg["data"]["split"], cfg["data"]["split_sha256"])]:
        if sha(REPO / rel) != expected:
            raise ValueError("Frozen manifest/split hash mismatch: " + rel)
    original = root / "r1/third_party/scaffold-gs"
    source = root / "r2/rgb_092/third_party/scaffold-gs"
    patch = REPO / "experiments/runs/RGB-092-TRANSIENT/scaffold_transient.patch"
    provenance = source.parent / "source_provenance.json"
    if not provenance.exists():
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=original, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--binary"], cwd=original)
        if commit != cfg["training"]["upstream_commit"] or hashlib.sha256(diff).hexdigest() != cfg["training"]["base_local_diff_sha256"]:
            raise ValueError("Frozen Scaffold source provenance mismatch")
        if not source.exists():
            shutil.copytree(original, source, ignore=shutil.ignore_patterns(".git", "__pycache__"), symlinks=True)
        if sha(source / "train.py") != sha(original / "train.py"):
            raise ValueError("Incomplete source preparation contains unexpected changes")
        subprocess.run(["git", "apply", "--unidiff-zero", "--check", str(patch)], cwd=source, check=True)
        subprocess.run(["git", "apply", "--unidiff-zero", str(patch)], cwd=source, check=True)
        provenance.write_text(json.dumps({"commit": commit, "base_diff_sha256": hashlib.sha256(diff).hexdigest(),
                  "patch_sha256": sha(patch), "train_sha256": sha(source / "train.py")}, indent=2))
    src = json.loads(provenance.read_text())
    if src["train_sha256"] != sha(source / "train.py") or src["patch_sha256"] != sha(patch):
        raise ValueError("Isolated source changed after preparation")
    if args.mode == "prepare":
        print(json.dumps(src))
        return
    if args.run_suffix and args.mode != "smoke":
        raise ValueError("Additional full-run attempts require a new registered experiment")
    ident = cfg["experiment_id"] + ("-SMOKE" if args.mode == "smoke" else "") + args.run_suffix
    run = root / "r1/runs" / ident
    if run.exists():
        raise FileExistsError("Refusing existing run " + str(run))
    run.mkdir(parents=True)
    envroot = root / "r1/envs/r1-cuda128"
    cuda = root / "r1/toolchain/cuda-12.8"
    env = os.environ.copy()
    env.update({"CUDA_HOME": str(cuda), "PATH": f"{envroot}/bin:{cuda}/bin:" + env["PATH"],
       "LD_LIBRARY_PATH": str(cuda / "lib64") + ":" + env.get("LD_LIBRARY_PATH", ""),
       "PYTHONPATH": ":".join(map(str, [REPO, source / "submodules/diff-gaussian-rasterization", source / "submodules/simple-knn", source])),
       "TORCH_CUDA_ARCH_LIST": "12.0", "TORCH_HOME": str(root / "r1/weights/torch"),
       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": "4",
       "ICSHM_TRANSIENT_CONFIG": str(args.config.resolve()),
       "ICSHM_TRANSIENT_LABELS": str(root / "r2/sem_001/labels"),
       "ICSHM_DINO_MODEL": str(args.dino_dir.resolve()) if args.dino_dir else ""})
    if cfg["transient"]["enabled"] and not args.dino_dir:
        raise ValueError("Candidate requires pinned local DINO checkpoint")
    steps = 2600 if args.mode == "smoke" else 30000
    scene = root / "rgb_r1_spatial_v1/scene_400_eval"
    command = [str(envroot / "bin/python"), str(source / "train.py"), "--source_path", str(scene),
       "--model_path", str(run), "--eval", "--resolution", "1", "--iterations", str(steps),
       "--save_iterations", str(steps), "--checkpoint_iterations", str(steps),
       "--test_iterations", str(steps) if args.mode == "full" else "999999", "--skip_metric_eval", "--quiet"]
    record = {"experiment_id": cfg["experiment_id"], "mode": args.mode, "config_sha256": sha(args.config),
       "source": src, "module_sha256": sha(REPO / "src/scaffold_transient.py"),
       "runner_sha256": sha(__file__), "repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
       "command": command, "started_unix": time.time(), "status": "running"}
    parent_dir = root / "r1/runs/RGB-030-FULL"
    frozen_paths = [parent_dir / "chkpnt30000.pth", *sorted((parent_dir / "point_cloud/iteration_30000").glob("*"))]
    record["frozen_parent_hashes"] = {str(path.relative_to(root)): sha(path) for path in frozen_paths if path.is_file()}
    if cfg["transient"]["enabled"]:
        record["dino_hashes"] = {path.name: sha(path) for path in sorted(args.dino_dir.glob("*")) if path.is_file()}
        names = json.loads((REPO / cfg["data"]["semantic_split"]).read_text())["semantic_train_labeled"]
        record["training_label_hashes"] = {name: sha(root / "r2/sem_001/labels" / name) for name in names}
    (run / "execution.json").write_text(json.dumps(record, indent=2))
    try:
        with (run / "execution.log").open("w") as log:
            subprocess.run(command, env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
        checkpoint = run / f"chkpnt{steps}.pth"
        check = subprocess.check_output([str(envroot / "bin/python"), str(REPO / "scripts/validate_candidate_checkpoint.py"),
                "--candidate-root", str(source), "--candidate", "scaffold-gs", "--checkpoint", str(checkpoint),
                "--train-camera-count", "303"], env=env, text=True)
        (run / "reload.json").write_text(check)
        record["checkpoint_sha256"] = sha(checkpoint)
        record["ply_sha256"] = sha(run / f"point_cloud/iteration_{steps}/point_cloud.ply")
        if cfg["transient"]["enabled"]:
            record["uncertainty_checkpoint_sha256"] = sha(run / f"transient_{steps}.pth")
            import torch
            aux = torch.load(run / f"transient_{steps}.pth", map_location="cpu", weights_only=True)
            a, b = torch.nn.Conv2d(384, 1, 1), torch.nn.Conv2d(384, 1, 1)
            a.load_state_dict(aux["head"], strict=True)
            b.load_state_dict(torch.load(run / f"transient_{steps}.pth", map_location="cpu", weights_only=True)["head"], strict=True)
            probe = torch.linspace(-1, 1, 384).reshape(1, 384, 1, 1)
            torch.testing.assert_close(a(probe), b(probe), rtol=0, atol=0)
            if aux["access"]["validation_masks_read"] != 0 or aux["access"]["protected_weight_violations"] != 0:
                raise RuntimeError("Training access/protection audit failed")
            record["uncertainty_reload"] = {"passed": True, "updates": aux["update_count"],
                 "masked_fraction_labeled_after_head_start": aux["masked_pixels"] / max(1, aux["seen_pixels"]),
                 "protected_weight_violations": 0, "validation_masks_read": 0}
        if args.mode == "full":
            commands = [
              ["organize_r1_renders.py", "--split", str(REPO / cfg["data"]["split"]), "--source-dir", str(run / "test/ours_30000/renders"), "--output-dir", str(run / "r1_validation_renders"), "--mapping-json", str(run / "r1_render_mapping.json")],
              ["evaluate_rgb.py", "--split", str(REPO / cfg["data"]["split"]), "--reference-dir", str(scene / "images"), "--prediction-dir", str(run / "r1_validation_renders"), "--per-view-csv", str(run / "r1_metrics_per_view.csv"), "--summary-json", str(run / "r1_metrics_summary.json"), "--lpips-device", "cuda"]]
            with (run / "evaluation.log").open("w") as log:
                for script, *arguments in commands:
                    subprocess.run([str(envroot / "bin/python"), str(REPO / "scripts" / script), *arguments], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        raise
    finally:
        record["frozen_parent_unchanged"] = all(sha(root / path) == expected for path, expected in record["frozen_parent_hashes"].items())
        record["ended_unix"] = time.time()
        record["elapsed_seconds"] = record["ended_unix"] - record["started_unix"]
        (run / "execution.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record))


if __name__ == "__main__":
    main()
