#!/usr/bin/env python3
"""Audit REPORT-003 for offline completeness, hashes and evidence labels."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/semantic_progress_20260923"))
    args = parser.parse_args()
    root = args.output.resolve()
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    html = (root / "index.html").read_text()
    check("index exists", bool(html))
    check("no remote HTTP dependencies", not re.search(r'(?:src|href)=["\']https?://', html))
    check("no absolute home paths", "/home/" not in html)
    for phrase in ["全 300 训练拟合", "100 张无标注仅视觉审阅", "空间开发 fold", "357 · 停止点",
                   "RGB170 · ImprovedGS · 50k", "SEM437 · Mask2Former Swin-L · 12k", "stay_cable"]:
        check(f"required evidence wording: {phrase}", phrase in html)
    refs = re.findall(r'(?:src|href)="([^"#]+)"', html)
    for ref in refs:
        check(f"relative asset exists: {ref}", (root / ref).is_file())
    for css_ref in re.findall(r'url\(["\']?([^"\')]+)', (root / "semantic-progress.css").read_text()):
        if not css_ref.startswith("data:"):
            check(f"CSS asset exists: {css_ref}", (root / css_ref).is_file())

    package = json.loads((root / "package_manifest.json").read_text())
    for relative, expected in package["files"].items():
        path = root / relative
        check(f"manifest file exists: {relative}", path.is_file())
        check(f"manifest hash: {relative}", sha256(path) == expected["sha256"])
        check(f"manifest size: {relative}", path.stat().st_size == expected["bytes"])

    expected_frames = {"trajectory-best.mp4": 144, "trajectory-sem.mp4": 144,
                       "unlabeled-best.mp4": 100, "unlabeled-sem.mp4": 100}
    for name, frames in expected_frames.items():
        value = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                         "-show_entries", "stream=nb_frames", "-of", "default=nw=1:nk=1",
                                         str(root / "assets/videos" / name)], text=True).strip()
        check(f"video frame count {name}", int(value) == frames)

    evidence = json.loads((root / "data/evidence.json").read_text())
    check("selected RGB170 50k", evidence["selected"]["reconstruction_endpoint"] == 50000)
    check("selected SEM437 12k", evidence["selected"]["semantic_endpoint"] == 12000)
    check("unlabeled role no GT", evidence["roles"]["unlabeled301_400"].startswith("no-GT"))
    result = {"schema_version": 1, "status": "completed", "experiment_id": "REPORT-003-SEMANTIC-PROGRESS-WEB",
              "checks": len(checks), "failed": 0, "output": str(root)}
    (root / "data/audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
