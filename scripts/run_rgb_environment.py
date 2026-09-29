#!/usr/bin/env python3
"""Execute the preregistered RGB environment control/candidate, one run only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from package_submission_baseline import sha256, _copy_source_snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--arm", choices=("control", "candidate"), required=True)
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.add_argument("--attempt", default="r1")
    args = p.parse_args()
    root = args.artifact_root
    configpath = ROOT / "experiments/runs/RGB-101-ENVIRONMENT/config.yaml"
    config = json.loads(configpath.read_text())
    if shutil.disk_usage(root).free < 100*2**30:
        raise RuntimeError("Less than 100 GiB free")
    original = root / "r1/third_party/scaffold-gs"
    source = root / "r2/rgb_101/source/scaffold-gs"
    commit = subprocess.check_output(["git", "-C", str(original), "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "-C", str(original), "diff", "--binary"])
    assert commit == "59c833b56bbbf510f3f64d40f81721995caced66"
    assert hashlib.sha256(diff).hexdigest() == "0030e06dd814742a1c1537c6ff6b4ec6067492861bf9ca56304db0de162141ac"
    assert sha256(ROOT / "experiments/splits/rgb_r1_spatial_v1.json") == "141944ca14288536a73a55586763b3104493112069a536bc0dfcc2d9f0081847"
    assert sha256(ROOT / config["dataset_manifest"]) == "0a188d4b4cfa5ffc2a7f151018f1cb012782e8f9860a3338aed084478d675b3e"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        _copy_source_snapshot(original, source)
    for relative in ("train.py", "gaussian_renderer/__init__.py", "scene/gaussian_model.py"):
        assert sha256(source / relative) == sha256(original / relative)
    ident = "RGB-101-ENVIRONMENT" if args.arm == "candidate" else "RGB-100-ENV-CTRL"
    run = root / "r2/rgb_101/runs" / f"{ident}-{args.mode}-{args.attempt}"
    if args.mode == "full" and args.attempt != "r1":
        raise ValueError("New full attempt requires independent registration")
    run.mkdir(parents=True, exist_ok=False)
    steps = config["training"]["smoke_iterations"] if args.mode == "smoke" else config["training"]["iterations"]
    scene = root / "rgb_r1_spatial_v1/scene_400_eval"
    cuda = root / "r1/toolchain/cuda-12.8"
    pythonroot = root / "r1/envs/r1-cuda128"
    python = pythonroot / "bin/python"
    env = os.environ.copy()
    env.update({"CUDA_HOME": str(cuda), "PATH": f"{pythonroot}/bin:{cuda}/bin:"+env["PATH"], "LD_LIBRARY_PATH": str(cuda/"lib64")+":"+env.get("LD_LIBRARY_PATH", ""), "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "4", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TORCH_HOME": str(root / "r1/weights/torch"), "HF_HUB_OFFLINE": "1"})
    command = [str(python), str(ROOT / "scripts/train_scaffold_environment.py"), "--source-root", str(source), "--run-dir", str(run)]
    if args.arm == "candidate":
        command.append("--candidate")
    command += ["-s", str(scene), "-m", str(run), "--eval", "--resolution", "1", "--iterations", str(steps), "--save_iterations", str(steps), "--checkpoint_iterations", str(steps), "--test_iterations", str(steps), "--skip_metric_eval", "--quiet", "--port", "6019"]
    record = {"experiment_id": ident, "status": "running", "mode": args.mode, "seed": 0, "config_sha256": sha256(configpath), "source_commit": commit, "source_diff_sha256": hashlib.sha256(diff).hexdigest(), "code_sha256": {name: sha256(ROOT/name) for name in ("scripts/run_rgb_environment.py", "scripts/train_scaffold_environment.py", "src/scaffold_environment.py")}, "command": command, "started_unix": time.time()}
    path = run / "execution.json"
    path.write_text(json.dumps(record, indent=2)+"\n")
    try:
        with (run / "execution.log").open("w") as log:
            subprocess.run(command, env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((run / "environment_execution.json").read_text())
        assert report["environment_reload_exact"]
        assert report["environment_mean"] > 0 if args.arm == "candidate" else report["environment_max"] == 0
        record["environment"] = report
        record["checkpoint_sha256"] = sha256(run / f"chkpnt{steps}.pth")
        record["environment_sha256"] = sha256(run / "environment.pth")
        if args.mode == "full":
            commands = [
                ["organize_r1_renders.py", "--split", str(ROOT/"experiments/splits/rgb_r1_spatial_v1.json"), "--source-dir", str(run/"test/ours_30000/renders"), "--output-dir", str(run/"r1_validation_renders"), "--mapping-json", str(run/"r1_render_mapping.json")],
                ["evaluate_rgb.py", "--split", str(ROOT/"experiments/splits/rgb_r1_spatial_v1.json"), "--reference-dir", str(scene/"images"), "--prediction-dir", str(run/"r1_validation_renders"), "--per-view-csv", str(run/"r1_metrics_per_view.csv"), "--summary-json", str(run/"r1_metrics_summary.json"), "--lpips-device", "cuda"]]
            with (run/"evaluation.log").open("w") as log:
                for script, *arguments in commands:
                    subprocess.run([str(python), str(ROOT/"scripts"/script), *arguments], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        record["status"] = "completed"
    except Exception as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        record["elapsed_seconds"] = time.time()-record["started_unix"]
        path.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({"run": str(run), "status": record["status"]}), flush=True)


if __name__ == "__main__":
    main()
