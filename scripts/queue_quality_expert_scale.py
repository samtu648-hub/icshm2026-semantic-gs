#!/usr/bin/env python3
"""Bounded SEM368: three full fits, two fresh replays/fold, then CPU acceptance.

This is a single local dependency batch, not a recurring monitor. It ends before
quality calibration; no implicit promotion, retries, uploads or model changes.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_expert_scale import CODE, CONFIG, Context
from src.quality_protocol import sha256, verify_hashes, write_new_json
from scripts.run_quality_expert_scale import require_stability

QUEUE_CODE = tuple(sorted(set(CODE + (
    "scripts/audit_quality_expert_scale.py", "scripts/queue_quality_expert_scale.py"))))


def stages(batch, artifact_root, partitions, executable):
    """Exactly three independent fits and six replays, never a stability resume."""
    for part in partitions:
        full = batch / ("full-" + part)
        common = [executable, ROOT / "scripts/run_quality_expert_scale.py",
                  "--artifact-root", artifact_root, "--partition", part]
        yield "full-" + part, common + ["--mode", "full", "--output", full,
            "--stability-audit", batch / "stability-audit"], "completed_mask2former_fit_pending_replay"
        for n in (1, 2):
            stage = f"fresh-{part}-{n}"
            yield stage, common + ["--mode", "replay", "--run", full,
                "--output", batch / stage], "completed_mask2former_fresh_replay"
        stage = "audit-" + part
        yield stage, [executable, ROOT / "scripts/audit_quality_expert_scale.py",
            "--artifact-root", artifact_root, "--run", full,
            "--replay-1", batch / f"fresh-{part}-1", "--replay-2", batch / f"fresh-{part}-2",
            "--output", batch / stage], "completed_mask2former_fold_acceptance"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--detach", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args()
    batch = a.batch.resolve()
    batch.mkdir(parents=True, exist_ok=True)
    if not batch.is_relative_to(ROOT / "outputs/quality/expert_scale"):
        p.error("Use an isolated SEM368 quality batch")
    if a.detach and a.preflight_only:
        p.error("Preflight does not launch a detached process")
    cfg = json.loads((ROOT / CONFIG).read_text())
    common = [sys.executable, ROOT/'scripts/run_quality_expert_scale.py', '--artifact-root', a.artifact_root, '--partition', 'oof-0']
    planned = [('identity', [sys.executable, ROOT/'scripts/check_quality_expert_scale_identity.py', '--artifact-root', a.artifact_root, '--output', batch/'identity'], 'passed_scale_identity'),
               ('stability', common + ['--mode', 'stability', '--output', batch/'stability'], 'completed_mask2former_stability_pending_replay')]
    for n in (1, 2):
        planned.append((f'stability-fresh-{n}', common + ['--mode', 'replay', '--run', batch/'stability', '--output', batch/f'stability-fresh-{n}'], 'completed_mask2former_fresh_replay'))
    planned.append(('stability-audit', [sys.executable, ROOT/'scripts/audit_quality_expert_scale.py', '--artifact-root', a.artifact_root,
        '--run', batch/'stability', '--replay-1', batch/'stability-fresh-1', '--replay-2', batch/'stability-fresh-2', '--output', batch/'stability-audit'], 'completed_mask2former_stability_acceptance'))
    planned += list(stages(batch, a.artifact_root.resolve(), cfg['partitions'], sys.executable))
    for stage, _, _ in planned:
        if (batch / stage).exists():
            raise FileExistsError("Refusing to overwrite or resume: " + stage)
    if (batch / "queue").exists():
        raise FileExistsError("Existing batch queue is immutable")
    # Check each fold/access contract before paying for any full fit.
    for part in cfg["partitions"]:
        ctx = Context(ROOT, a.artifact_root, part)
    if shutil.disk_usage(batch).free < cfg['resource']['estimated_new_output_gib_upper_bound']*1024**3:
        raise OSError('Actual space does not cover planned new outputs; preserve old files')
    if a.preflight_only:
        print(json.dumps({"status": "passed_mask2former_batch_preflight", "stages": len(planned),
            "new_training": False, "free_gib": shutil.disk_usage(batch).free / 1024**3}))
        return
    if a.detach:
        with (batch / "queue-driver.log").open("x") as log:
            child = subprocess.Popen([sys.executable, Path(__file__).resolve(), "--batch", batch,
                "--artifact-root", a.artifact_root.resolve()], cwd=ROOT,
                env=dict(os.environ, OMP_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1"),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        rec = {"pid": child.pid, "unix": time.time(), "status": "launched_bounded_local_batch",
               "queue_script_sha256": sha256(Path(__file__)), "automatic_promotion": False}
        write_new_json(batch / "queue-launch.json", rec)
        print(json.dumps(rec), flush=True)
        return
    with (ROOT / "outputs/quality/expert_scale_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = batch / "queue"
        output.mkdir(exist_ok=False)
        start = time.monotonic()
        rec = {"experiment_id": cfg["experiment_id"], "status": "running_oof_acceptance_batch",
            "pid": os.getpid(), "seed": cfg["seed"], "git_commit": cfg["git_commit"],
            "config_sha256": sha256(ROOT / CONFIG), "code_sha256": {n: sha256(ROOT/n) for n in QUEUE_CODE},
            "events": [],
            "target_labels_decoded": False, "quality_metrics": None, "automatic_promotion": False}
        for name in (*QUEUE_CODE, CONFIG):
            dest = output / "code" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, dest)
        write_new_json(output / "started.json", rec)

        def guard():
            verify_hashes(ROOT, rec["code_sha256"])
            verify_hashes(ROOT, {CONFIG: rec["config_sha256"]})

        def event(stage, state, **extra):
            row = {"stage": stage, "state": state, "unix": time.time(), **extra}
            write_new_json(output / f"event-{len(rec['events']):03d}.json", row)
            rec["events"].append(row)
            print(json.dumps(row), flush=True)

        try:
            for stage, cmd, expected in planned:
                guard()
                event(stage, "started")
                with (output / (stage + ".log")).open("x") as log:
                    subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, check=True,
                        env=dict(os.environ, OMP_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1"))
                guard()
                result = json.loads((batch / stage / "result.json").read_text())
                if result["status"] != expected:
                    raise ValueError("Unaccepted stage: " + stage)
                event(stage, "completed", result_sha256=sha256(batch / stage / "result.json"))
            rec["status"] = "completed_mask2former_threefold_acceptance_pending_comparison"
        except BaseException as exc:
            rec.update(status="stopped_mask2former_batch_not_method_rejection", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            rec["seconds"] = time.monotonic() - start
            write_new_json(output / "result.json", rec)
            print(json.dumps({"status": rec["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
