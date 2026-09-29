#!/usr/bin/env python3
"""Two fresh-process SEM335 replays. No target labels or calibration here."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_experts import load_expert, predict, verify_export, validate_probability
from src.quality_static_training import semantic_access


def replay(a, result):
    import numpy as np
    import torch
    from PIL import Image
    from src.semantic_cable_binary import read_rgb
    cfg = json.loads((a.run / "code/experiments/runs/SEM-335-SCAFFOLD-EXPERT-DURATION/config.yaml").read_text())
    branch = cfg["experts"][result["binding"]["expert"]]
    model_dir = a.artifact_root / cfg["model"]["relative_weights"]
    verify_hashes(model_dir, result["binding"]["model_hashes"])
    payload = torch.load(a.run / "expert.pt", map_location="cpu", weights_only=True)
    if payload["binding"] != result["binding"] or payload["access"] != result["access"] or payload["steps"] != cfg["training"]["steps"]:
        raise ValueError("Checkpoint provenance differs")
    model = load_expert(model_dir, cfg["seed"], branch.get("positive_class")).cuda()
    model.load_state_dict(payload["state_dict"])
    if model.config.num_labels != len(branch["class_ids"]):
        raise ValueError("Expert head has wrong class count")
    amp = torch.bfloat16 if result["amp"] == "torch.bfloat16" else torch.float16
    for name in result["prediction_views"]:
        actual = predict(model, read_rgb(a.rendered_root / "images" / name), crop=branch["crop"], stride=branch["stride"], amp_dtype=amp)
        with np.load(a.run / "probabilities" / (Path(name).stem + ".npz"), allow_pickle=False) as z:
            stored = z["probabilities"]
        validate_probability(stored, len(branch["class_ids"]), (989, 1320))
        with Image.open(a.run / "predictions" / name) as im:
            png = np.asarray(im)
        if not np.array_equal(actual, stored) or not np.array_equal(actual.argmax(0).astype(np.uint8), png):
            raise ValueError("Fresh expert inference differs from sealed native probabilities/PNG")
    write_new_json(a.output, {"status": "passed_fresh_expert_replay", "pid": os.getpid(),
        "views": result["prediction_views"], "probabilities_exact": True, "target_labels_decoded": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "rgb-run", "rendered-root", "artifact-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    result = json.loads((a.run / "result.json").read_text())
    if a.worker:
        replay(a, result)
        return
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        record = {"experiment_id": "SEM-335-SCAFFOLD-EXPERT-DURATION", "status": "running_expert_audit", "quality_metrics": None,
            "target_labels_decoded": False, "run_result_sha256": sha256(a.run / "result.json"),
            "audit_code_sha256": sha256(Path(__file__)), "binding": result["binding"], "access": result["access"],
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        write_new_json(a.output / "started.json", record)
        try:
            if result["status"] != "completed_expert_fit_pending_independent_audit" or result["trained_steps"] != 6000 or not result.get("prefix_3000", {}).get("exact"):
                raise ValueError("Requires complete registered6k expert and exact3k control")
            verify_hashes(a.run, result["artifacts"])
            verify_hashes(a.run / "code", result["code_sha256"])
            verify_hashes(ROOT, result["code_sha256"])
            verify_hashes(ROOT, result["binding"]["input_hashes"])
            if sha256(a.rgb_run / "result.json") != result["binding"]["parent_result_sha256"]:
                raise ValueError("Wrong RGB parent")
            parent = json.loads((a.rgb_run / "result.json").read_text())
            verify_hashes(a.rgb_run, parent["artifacts"])
            config = json.loads((a.run / "code/experiments/runs/SEM-335-SCAFFOLD-EXPERT-DURATION/config.yaml").read_text())
            if sha256(a.run / "code/experiments/runs/SEM-335-SCAFFOLD-EXPERT-DURATION/config.yaml") != result["binding"]["config_sha256"]:
                raise ValueError("Changed sealed training configuration")
            protocol = json.loads((ROOT / config["protocol"]).read_text())
            split = json.loads((ROOT / "experiments/splits/sem_001_scaffold_static_v1.json").read_text())
            oof = json.loads((ROOT / "experiments/splits/sem_005_oof_spatial_v1.json").read_text())
            access = semantic_access(split, oof, json.loads((a.rgb_run / "access.json").read_text()), result["access"]["partition"])
            if access != result["access"] or result["prediction_views"] != access["prediction"]:
                raise ValueError("Incomplete or leaking prediction partition")
            development = access["partition"] == "development"
            expected_export_hash = result["binding"]["target_export_sha256" if development else "train_export_sha256"]
            if sha256(a.rendered_root / "result.json") != expected_export_hash:
                raise ValueError("Replay export differs from training")
            verify_export(a.rendered_root, split["semantic_validation_labeled" if development else "semantic_train_labeled"],
                          result["binding"]["parent_result_sha256"], result["binding"]["identity"], "development" if development else "semantic-training")
            frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
            hashes = {i["path"]: i["sha256"] for i in frozen["artifacts"]}
            verify_hashes(a.artifact_root, hashes)
            replays = []
            for i in range(2):
                dest = a.output / f"replay-{i}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
                for key in ("run", "rgb_run", "rendered_root", "artifact_root"):
                    command += ["--" + key.replace("_", "-"), str(getattr(a, key).resolve())]
                command += ["--output", str(dest.resolve())]
                with (a.output / f"replay-{i}.log").open("x") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                replays.append(json.loads(dest.read_text()))
            if len({r["pid"] for r in replays}) != 2 or any(r["status"] != "passed_fresh_expert_replay" for r in replays):
                raise ValueError("Fresh process isolation failed")
            verify_hashes(a.artifact_root, hashes)
            verify_hashes(a.rgb_run, parent["artifacts"])
            verify_hashes(a.run, result["artifacts"])
            if sha256(Path(__file__)) != record["audit_code_sha256"]:
                raise ValueError("Audit code changed during replay")
            record.update(status="completed_independent_expert_replay", fresh_reloads=replays,
                frozen_verified=len(hashes), rgb_artifacts_unchanged=True)
        except BaseException as exc:
            record.update(status="expert_audit_failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            write_new_json(a.output / "result.json", record)
            print(json.dumps({"status": record["status"], "output": str(a.output)}), flush=True)


if __name__ == "__main__":
    main()
