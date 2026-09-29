#!/usr/bin/env python3
"""One-shot four-backbone static/2D expert batch; no silent promotion or retries."""
import argparse
from contextlib import ExitStack
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
from src.quality_protocol import sha256, verify_hashes, write_new_json
from scripts.run_quality_static import CODE as STATIC_CODE
from scripts.run_quality_expert import CODE as EXPERT_CODE


def resolve_source(value, artifact_root):
    kind, relative = value.split(":", 1)
    if kind not in ("repo", "artifact"):
        raise ValueError("Unknown source root")
    base = ROOT if kind == "repo" else artifact_root.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError("Source path escapes configured root")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("python", "artifact-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--attempt", required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not a.attempt.replace("-", "").isalnum():
        p.error("Use an alphanumeric/hyphen attempt ID")
    cfg_path = ROOT / "experiments/runs/SEM-331-QUALITY-EXPERTS/batch_config.yaml"
    cfg = json.loads(cfg_path.read_text())
    names = [arm["name"] for arm in cfg["arms"]]
    if names != ["Scaffold", "PGSR", "SOF-NVS", "SOF-GEO"]:
        raise ValueError("The first round must retain all four registered arms")
    if a.dry_run:
        print(json.dumps({"arms": names, "remaining_static_fits": 13, "expert_fits_total": 48,
            "first_expert_reuse": 1, "expert_steps_each": 3000, "static_steps_each": 10000,
            "max_concurrent_gpu_jobs": 1, "selection": None, "fusion_calibration": "pending separate runner"}))
        return
    root = ROOT / "outputs/quality"
    output = root / "queues" / a.attempt
    with ExitStack() as stack:
        # Serialize this batch against existing stage-A/static queues; children
        # acquire the common GPU lock themselves, so no nested deadlock.
        for name in ("stage_a_queue.lock", "static_queue.lock", "dualline_queue.lock"):
            lock = stack.enter_context((root / name).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        code_paths = sorted(set([*[ROOT / n for n in (*STATIC_CODE, *EXPERT_CODE, "src/quality_coverage.py", "src/quality_semantic_audit.py")],
            *[ROOT / "scripts" / n for n in ("queue_quality_dualline.py", "run_quality_static.py", "audit_quality_static.py",
                "run_quality_expert.py", "audit_quality_expert.py", "export_quality_rgb_inputs.py", "audit_quality_coverage.py", "audit_quality_diagnostic_evidence.py",
                "run_quality_rgb.py", "run_quality_stage_a.py", "probe_quality_adapter.py")],
            *[ROOT / "src" / n for n in ("semantic2d.py", "semantic_cable_binary.py", "scaffold_semantic_sidecar.py", "camera_adapter.py")]]))
        configs = [cfg_path, *[ROOT / f"experiments/runs/{n}/config.yaml" for n in
            ("QUAL-001-FOUNDATION", "QUAL-003-COVERAGE-INIT-DIAG", "SEM-330-QUALITY-STATIC", "SEM-331-QUALITY-EXPERTS")]]
        record = {"experiment_id": cfg["experiment_id"], "status": "running_serial_first_round", "events": [],
            "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in code_paths},
            "config_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in configs},
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "quality_selection": None, "fusion_calibration_completed": False, "arms": names}
        for path in [*code_paths, *configs]:
            dest = output / "code" / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        write_new_json(output / "started.json", record)
        def event(name, **kw):
            e = {"index": len(record["events"]), "name": name, "unix": time.time(), **kw}
            write_new_json(output / f"event-{e['index']:03d}.json", e)
            record["events"].append(e)
            print(json.dumps(e), flush=True)
        def unchanged():
            verify_hashes(ROOT, record["code_sha256"])
            verify_hashes(ROOT, record["config_sha256"])
            if shutil.disk_usage(root).free < 100 * 1024**3:
                raise OSError("100 GiB reserve reached; stopping without changing budgets")
        def execute(stage, argv, dest, status, env):
            unchanged()
            event("started", stage=stage, output=str(dest))
            with (output / f"{len(record['events']):03d}-{stage}.log").open("x") as log:
                subprocess.run([str(a.python), *map(str, argv), "--output", str(dest)], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            unchanged()
            r = json.loads((dest / "result.json").read_text())
            if r["status"] != status:
                raise ValueError(f"{stage}: child acceptance status differs")
            event("completed", stage=stage, output=str(dest), result_sha256=sha256(dest / "result.json"))
            return r
        def reuse(stage, dest, status):
            r = json.loads((dest / "result.json").read_text())
            if r["status"] != status:
                raise ValueError(f"Incomplete reusable stage: {stage}")
            if "artifacts" in r:
                verify_hashes(dest, r["artifacts"])
            event("reused", stage=stage, output=str(dest), result_sha256=sha256(dest / "result.json"))
            return r
        base_env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4")
        try:
            prepared = {}
            for arm in cfg["arms"]:
                name = arm["name"]
                rgb, rgb_audit, probe = (ROOT / arm[k] for k in ("rgb_run", "rgb_audit", "probe"))
                source = resolve_source(arm["source"], a.artifact_root)
                extras = [resolve_source(v, a.artifact_root) for v in arm["extra_sources"]]
                extension = "diff-plane-rasterization" if name == "PGSR" else "diff-gaussian-rasterization"
                env = dict(base_env, PYTHONPATH=os.pathsep.join(map(str, [source, source / "submodules" / extension, source / "submodules/simple-knn", ROOT, *extras])))
                extra_args = [item for path in extras for item in ("--extra-pythonpath", path)]
                static_common = [ROOT / "scripts/run_quality_static.py", "--run", rgb, "--audit", rgb_audit, "--probe", probe,
                    "--source", source, "--artifact-root", a.artifact_root, *extra_args]
                if arm["reuse_static"]:
                    sem, sem_audit = ROOT / arm["reuse_static"], ROOT / arm["reuse_static_audit"]
                    reuse(name+"-static-development", sem, "completed_static_fit_pending_independent_evaluation")
                    reuse(name+"-static-development-audit", sem_audit, "completed_independent_static_semantic_audit")
                else:
                    smoke = root / "semantic" / f"SEM-330-{name}-development-smoke-{a.attempt}"
                    sem = root / "semantic" / f"SEM-330-{name}-development-full-{a.attempt}"
                    sem_audit = root / "audits" / sem.name
                    execute(name+"-static-development-smoke", static_common+["--partition", "development", "--mode", "smoke"], smoke, "passed_static_semantic_smoke", env)
                    execute(name+"-static-development-full", static_common+["--partition", "development", "--mode", "full", "--smoke", smoke], sem, "completed_static_fit_pending_independent_evaluation", env)
                    execute(name+"-static-development-audit", [ROOT / "scripts/audit_quality_static.py", "--run", sem, "--rgb-run", rgb,
                        "--source", source, "--artifact-root", a.artifact_root, *extra_args], sem_audit, "completed_independent_static_semantic_audit", env)
                if arm.get("reuse_coverage"):
                    coverage_dest = ROOT / arm["reuse_coverage"]
                    coverage = reuse(name+"-coverage", coverage_dest, "completed_native_coverage_diagnosis")
                    if coverage["semantic_run_sha256"] != sha256(sem / "result.json"):
                        raise ValueError("Reused coverage has different static parent")
                else:
                    coverage_dest = root / "diagnostics" / f"QUAL-003-{name}-coverage-{a.attempt}"
                    execute(name+"-coverage", [ROOT / "scripts/audit_quality_coverage.py", "--rgb-run", rgb, "--semantic-run", sem,
                        "--semantic-audit", sem_audit, "--source", source, "--artifact-root", a.artifact_root, *extra_args],
                        coverage_dest, "completed_native_coverage_diagnosis", env)
                execute(name+"-coverage-recount", [ROOT / "scripts/audit_quality_diagnostic_evidence.py", "--run", coverage_dest,
                    "--semantic-run", sem, "--artifact-root", a.artifact_root], root / "audits" / f"QUAL-003-{name}-coverage-{a.attempt}",
                    "completed_independent_diagnostic_recount", base_env)
                exports = {}
                for role in ("semantic-training", "development"):
                    if role == "semantic-training" and arm["reuse_training_export"]:
                        dest = ROOT / arm["reuse_training_export"]
                        reuse(name+"-export-"+role, dest, "completed_native_expert_rgb_export")
                    else:
                        dest = root / "expert_inputs" / f"{name}-{role}-{a.attempt}"
                        execute(name+"-export-"+role, [ROOT / "scripts/export_quality_rgb_inputs.py", "--run", rgb, "--audit", rgb_audit,
                            "--probe", probe, "--source", source, "--artifact-root", a.artifact_root, "--partition", role, *extra_args],
                            dest, "completed_native_expert_rgb_export", env)
                    exports[role] = dest
                prepared[name] = (rgb, source, extra_args, static_common, env, exports)
            for arm in cfg["arms"]:
                name = arm["name"]
                rgb, source, extra_args, static_common, env, exports = prepared[name]
                for partition in ("oof-0", "oof-1", "oof-2", "development"):
                    if partition != "development":
                        smoke = root / "semantic" / f"SEM-330-{name}-{partition}-smoke-{a.attempt}"
                        sem_fold = root / "semantic" / f"SEM-330-{name}-{partition}-full-{a.attempt}"
                        execute(name+"-static-"+partition+"-smoke", static_common+["--partition", partition, "--mode", "smoke"], smoke, "passed_static_semantic_smoke", env)
                        execute(name+"-static-"+partition+"-full", static_common+["--partition", partition, "--mode", "full", "--smoke", smoke], sem_fold, "completed_static_fit_pending_independent_evaluation", env)
                        execute(name+"-static-"+partition+"-audit", [ROOT / "scripts/audit_quality_static.py", "--run", sem_fold, "--rgb-run", rgb,
                            "--source", source, "--artifact-root", a.artifact_root, *extra_args], root / "audits" / sem_fold.name, "completed_independent_static_semantic_audit", env)
                    for expert in ("fiveclass", "deck", "stay_cable"):
                        key = expert+"_"+partition.replace("-", "")
                        full = root / "experts" / f"SEM-331-{name}-{expert}-{partition}-full-{a.attempt}"
                        common = [ROOT / "scripts/run_quality_expert.py", "--rgb-run", rgb, "--rendered-root", exports["semantic-training"],
                                  "--artifact-root", a.artifact_root, "--expert", expert, "--partition", partition]
                        if partition == "development":
                            common += ["--target-root", exports["development"]]
                        if key in arm["reuse_expert"]:
                            full = ROOT / arm["reuse_expert"][key]
                            r = reuse(name+"-expert-"+key, full, "completed_expert_fit_pending_independent_audit")
                            if r["binding"]["expert"] != expert or r["access"]["partition"] != partition or r["binding"]["parent_result_sha256"] != sha256(rgb / "result.json"):
                                raise ValueError("Reused expert is a different arm/partition/branch")
                        else:
                            smoke = root / "experts" / f"SEM-331-{name}-{expert}-{partition}-smoke-{a.attempt}"
                            execute(name+"-"+key+"-smoke", common+["--mode", "smoke"], smoke, "passed_expert_smoke", base_env)
                            execute(name+"-"+key+"-full", common+["--mode", "full", "--smoke", smoke], full, "completed_expert_fit_pending_independent_audit", base_env)
                        if key in arm.get("reuse_expert_audit", {}):
                            ar = reuse(name+"-"+key+"-audit", ROOT / arm["reuse_expert_audit"][key], "completed_independent_expert_replay")
                            if ar["run_result_sha256"] != sha256(full / "result.json"):
                                raise ValueError("Reused expert audit has different fit parent")
                        else:
                            execute(name+"-"+key+"-audit", [ROOT / "scripts/audit_quality_expert.py", "--run", full, "--rgb-run", rgb,
                                "--rendered-root", exports["development" if partition == "development" else "semantic-training"], "--artifact-root", a.artifact_root],
                                root / "audits" / f"SEM-331-{name}-{key}-{a.attempt}", "completed_independent_expert_replay", base_env)
                event("arm_inputs_complete_no_promotion", arm=name)
            record["status"] = "completed_four_arm_inputs_fusion_calibration_pending"
        except BaseException as exc:
            record.update(status="stopped_first_round_batch_not_method_rejection", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            write_new_json(output / "result.json", record)
            print(json.dumps({"status": record["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
