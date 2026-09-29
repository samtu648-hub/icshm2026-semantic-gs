#!/usr/bin/env python3
"""Finish reporting after the already-running SEM386 train/inference pipeline.

This CPU-only follower neither starts training nor retries failed GPU jobs.
"""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EID = "SEM-386-RGB150-DEV270-CONTEXT50-MEMORY"
DIRECTORY = ROOT / "experiments/runs" / EID


def read(path):
    return json.loads(path.read_text())


def main():
    config = read(DIRECTORY / "config.yaml")
    batch = ROOT / config["output_path"]
    deadline = time.monotonic() + 3 * 3600
    while True:
        for stage in ["mask2former", "inference"]:
            path = batch / stage / "result.json"
            if path.exists() and read(path)["status"] == "failed_preserved":
                raise RuntimeError(f"{stage} failed; retained {path}")
        path = batch / "inference/result.json"
        if path.exists():
            if read(path)["status"] != "completed":
                raise RuntimeError("Unexpected inference status")
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Background follow-up timed out; no training was killed")
        time.sleep(30)
    subprocess.run([sys.executable, "scripts/build_sem386_comparison.py"], cwd=ROOT, check=True)
    train = read(batch / "mask2former/result.json")
    comparison = read(batch / "comparison/result.json")
    h = comparison["comparisons"]["holdout"]
    summary = (
        f"2026-09-17 SEM386训练与评价完成，待人工视觉复核。固定Dev30 270/30、seed20260805、6000步，"
        f"保留1025x768完整画面并启用Swin激活重计算。留出五类mIoU "
        f"{100*h['candidate_five_class_miou']:.6f}%（较SEM384 {h['five_class_delta_pp']:+.6f}pp），"
        f"前景mIoU {100*h['candidate_foreground_miou']:.6f}%（{h['foreground_delta_pp']:+.6f}pp）。"
        "结果为单划分单seed开发证据；未自动替换基线，未启动其他训练。"
        "下一步：人工检查对照页；据留出及局部表现判断是否保留。"
    )
    body = f"\n\n## Completed background run\n\n{summary}\n\n"
    body += f"Training: {train.get('training_seconds', train['seconds']):.2f} seconds; peak reserved {train['peak_reserved_mib']:.1f} MiB.\n\n"
    body += f"Independent stored-prediction confusion checks: {comparison['independent_confusion_checks']}. Frozen RGB equality: {train['frozen_rgb_hashes_exact']}.\n\n"
    body += "Comparison: `" + config['output_path'] + "/comparison/index.html`. Visual review remains pending; no promotion claim.\n\n"
    body += "Group and class changes:\n```json\n" + json.dumps(comparison['comparisons'], indent=2, ensure_ascii=False) + "\n```\n"
    with (ROOT / "outputs/quality/sem386-report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        p = DIRECTORY / "report.md"
        text = p.read_text()
        heading, rest = text.split("\n", 1)
        p.write_text(heading + body + "\n## Preparation history\n" + rest)
        p = ROOT / "experiments/registry.yaml"
        data = read(p)
        entry = next(x for x in data['experiments'] if x['experiment_id'] == EID)
        entry.update(status="completed_pending_visual_review", output_path=config['output_path'],
                     metrics=comparison['comparisons'], completed_date="2026-09-17",
                     quality_claim="Single Dev30 split and seed; no automatic promotion")
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        for name in ["PROJECT_STATUS.md", "TASKS.md", "ROADMAP.md"]:
            p = ROOT / name
            heading, rest = p.read_text().split("\n", 1)
            p.write_text(heading + "\n\n" + summary + "\n" + rest)
    with (batch / "repo-health-after-completion.log").open("w") as log:
        subprocess.run(["bash", "scripts/repo_health.sh"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    (batch / "background-completion.json").write_text(json.dumps({"status": "completed_pending_visual_review", "summary": summary}, indent=2, ensure_ascii=False) + "\n")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
