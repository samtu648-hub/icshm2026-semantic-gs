#!/usr/bin/env python3
"""Build the relocatable offline REPORT-003 progress presentation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PALETTE = np.array([[30, 30, 30], [80, 160, 245], [245, 210, 65], [220, 80, 80], [170, 100, 190]], dtype=np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def probe(path: Path) -> dict:
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,codec_name,pix_fmt:format=duration,size", "-of", "json", str(path)
    ], text=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/semantic_progress_20260923")
    args = parser.parse_args()
    output = args.output.resolve()
    for subdir in ["assets/videos", "assets/posters", "assets/hard", "assets/examples", "data"]:
        (output / subdir).mkdir(parents=True, exist_ok=True)

    manifest = []

    def copy(source: Path, relative: str) -> Path:
        target = output / relative
        shutil.copyfile(source, target)
        manifest.append({"path": relative, "sha256": sha256(target), "bytes": target.stat().st_size,
                         "source": str(source.relative_to(ROOT)), "source_sha256": sha256(source), "transform": "copy"})
        return target

    templates = ROOT / "scripts/report_assets"
    copy(templates / "semantic_progress.html", "index.html")
    copy(templates / "semantic-progress.css", "semantic-progress.css")
    copy(templates / "semantic-progress.js", "semantic-progress.js")

    videos = {
        "pre-sem001": ROOT / "outputs/project_progress_20260918/assets/videos/sem001.mp4",
        "pre-sem161": ROOT / "outputs/project_progress_20260918/assets/videos/sem161.mp4",
        "pre-m2f": ROOT / "outputs/project_progress_20260918/assets/videos/m2f.mp4",
        "full-sem380": ROOT / "outputs/project_progress_20260918/assets/videos/sem380.mp4",
        "backbones": ROOT / "outputs/quality/backbone_comparison/browser-r1/three-backbones-trajectory.mp4",
        "trajectory-best": ROOT / "outputs/quality/SEM-437/r2/review/videos/trajectory144_rgb_sem437_12k_side_by_side.mp4",
        "trajectory-sem": ROOT / "outputs/quality/SEM-437/r2/review/videos/trajectory144_sem437_12k.mp4",
        "unlabeled-best": ROOT / "outputs/quality/SEM-437/r2/review/videos/unlabeled301_400_rgb_sem437_12k_side_by_side.mp4",
        "unlabeled-sem": ROOT / "outputs/quality/SEM-437/r2/review/videos/unlabeled301_400_sem437_12k.mp4",
    }
    video_meta = {}
    for key, source in videos.items():
        target = copy(source, f"assets/videos/{key}.mp4")
        poster = output / f"assets/posters/{key}.jpg"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", "0.5", "-i", str(target),
                        "-frames:v", "1", "-q:v", "2", str(poster)], check=True)
        manifest.append({"path": str(poster.relative_to(output)), "sha256": sha256(poster),
                         "bytes": poster.stat().st_size, "source": str(target.relative_to(output)),
                         "transform": "ffmpeg frame at 0.5 seconds, JPEG q2"})
        video_meta[key] = probe(target)

    hard_sources = ROOT / "outputs/quality/SEM-437/r2/review/assets"
    fallback_sources = ROOT / "outputs/quality/SEM-439/r1/review/assets"
    for view in [350, 353, 356, 357]:
        copy(hard_sources / f"hard344_358-{view}-rgb.jpg", f"assets/hard/{view}-rgb.jpg")
        copy(hard_sources / f"hard344_358-{view}-sem437-12000.png", f"assets/hard/{view}-sem437.png")
    copy(fallback_sources / "356-sem439.png", "assets/hard/356-sem439.png")

    examples = ["049", "057", "146", "159"]
    roots = {
        "photo": ROOT / "outputs/quality/RGB-150/scene/images",
        "rgb": ROOT / "outputs/quality/RGB-170/full-r1/evaluation-r1/images/50000",
        "gt": ROOT / "outputs/quality/all_data/SEM-370-r1/labels",
        "pred": ROOT / "outputs/quality/SEM-437/r2/inference/predictions/12000",
    }
    for view in examples:
        for kind in ["photo", "rgb"]:
            source = roots[kind] / f"{view}.png"
            target = output / f"assets/examples/{view}-{kind}.jpg"
            with Image.open(source) as image:
                image.convert("RGB").save(target, quality=90, optimize=True)
            manifest.append({"path": str(target.relative_to(output)), "sha256": sha256(target),
                             "bytes": target.stat().st_size, "source": str(source.relative_to(ROOT)),
                             "source_sha256": sha256(source), "transform": "JPEG90 presentation copy"})
        for kind in ["gt", "pred"]:
            source = roots[kind] / f"{view}.png"
            labels = np.asarray(Image.open(source), dtype=np.uint8)
            if labels.ndim != 2 or int(labels.max()) > 4:
                raise ValueError(f"Invalid class-ID image: {source}")
            target = output / f"assets/examples/{view}-{kind}.png"
            Image.fromarray(PALETTE[labels]).save(target, optimize=True)
            manifest.append({"path": str(target.relative_to(output)), "sha256": sha256(target),
                             "bytes": target.stat().st_size, "source": str(source.relative_to(ROOT)),
                             "source_sha256": sha256(source), "transform": "fixed five-class presentation palette"})

    evidence = {
        "experiment_id": "REPORT-003-SEMANTIC-PROGRESS-WEB",
        "built_at": "2026-09-23",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "roles": {
            "historical74": "historical development metrics; not comparable to all300",
            "all300": "semantic training fit; all 300 labels used for training",
            "spatial_v1_v2": "previously observed development validation; no longer blind",
            "unlabeled301_400": "no-GT visual review only",
            "trajectory144": "no-GT visual review only",
        },
        "selected": {"reconstruction": "RGB-170-IMPROVEDGS-BUDGET15M-H50K",
                     "semantic": "SEM-437-RGB170-MASK2FORMER-CONTEXT50-LONG12K-ALL300",
                     "reconstruction_endpoint": 50000, "semantic_endpoint": 12000},
        "video_probes": video_meta,
        "sources": [
            "experiments/runs/RGB-170-IMPROVEDGS-BUDGET15M-H50K/report.md",
            "experiments/runs/SEM-434-RGB152-MASK2FORMER-LONG12K/report.md",
            "experiments/runs/SEM-435-RGB154-MASK2FORMER-LONG12K/report.md",
            "experiments/runs/SEM-437-RGB170-MASK2FORMER-CONTEXT50-LONG12K-ALL300/report.md",
            "experiments/runs/SEM-439-RGB170-DECK-ABSENCE-FALLBACK/report.md",
            "experiments/runs/SEM-440-RGB170-WEAK-STRUCTURE-TAIL14K/report.md",
            "outputs/quality/backbone_comparison/browser-r1/data.json",
            "outputs/project_progress_20260918/data/content.json",
        ],
        "claim": "Offline presentation only; no new training, hidden-test result or unlabeled accuracy claim."
    }
    write_json(output / "data/evidence.json", evidence)
    manifest.append({"path": "data/evidence.json", "sha256": sha256(output / "data/evidence.json"),
                     "bytes": (output / "data/evidence.json").stat().st_size, "transform": "curated evidence index"})
    write_json(output / "data/asset_manifest.json", {"schema_version": 1, "assets": manifest})
    (output / "README.txt").write_text(
        "ICSHM 2026 项目进展离线汇报 · 2026-09-23\n\n"
        "双击 index.html 即可离线打开。请保留整个目录结构。\n"
        "方向键可按章节翻页；右上角按钮进入全屏。所有视频可暂停、拖动和全屏。\n"
        "001-300 指标是训练拟合；空间 fold 是已观察开发验证；301-400 与 144 帧无 GT，只作视觉审阅。\n"
        "当前展示主方案：RGB170 50k + SEM437 12k；SEM439 仅作 355/356 的开发 fallback。\n",
        encoding="utf-8")
    files = {str(path.relative_to(output)): {"sha256": sha256(path), "bytes": path.stat().st_size}
             for path in sorted(output.rglob("*"))
             if path.is_file() and path.name not in {"package_manifest.json", "audit.json"}
             and "browser_qa" not in path.relative_to(output).parts}
    write_json(output / "package_manifest.json", {"schema_version": 1, "experiment_id": "REPORT-003-SEMANTIC-PROGRESS-WEB", "files": files})
    print(json.dumps({"status": "completed", "output": str(output), "files": len(files),
                      "bytes": sum(item["bytes"] for item in files.values())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
