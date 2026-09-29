#!/usr/bin/env python3
"""Build a UTF-8 interactive review for SEM396/432/433 and public100."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/quality/QUAL-019/review"
SUMMARY = ROOT / "outputs/quality/QUAL-019/summary.json"
COLORS = np.asarray(
    [
        [28, 28, 30],
        [103, 166, 255],
        [249, 220, 88],
        [222, 96, 91],
        [174, 103, 205],
    ],
    dtype=np.uint8,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def colorize(source: Path, destination: Path):
    label = np.asarray(Image.open(source), dtype=np.uint8)
    if label.shape != (989, 1320) or label.max() > 4:
        raise ValueError(f"Invalid semantic PNG: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(COLORS[label]).save(destination, optimize=True)


def difference(control: Path, candidate: Path, destination: Path):
    a = np.asarray(Image.open(control), dtype=np.uint8)
    b = np.asarray(Image.open(candidate), dtype=np.uint8)
    out = np.zeros((989, 1320, 3), dtype=np.uint8) + 28
    changed = a != b
    out[changed] = COLORS[b[changed]]
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(destination, optimize=True)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = read(SUMMARY)
    split = read(ROOT / "experiments/splits/spatial_confirmation_v1.json")["primary_fold"]
    datasets = {
        "validation30": sorted(split["semantic_validation_labeled"]),
        "public100": [f"{index:03d}.png" for index in range(301, 401)],
    }
    sources = {
        "validation30": {
            "baseline": ROOT / "outputs/quality/SEM-396/r1/inference/predictions",
            "control": ROOT / "outputs/quality/SEM-432/r1/inference/predictions",
            "candidate": ROOT / "outputs/quality/SEM-433/r1/inference/predictions",
            "gt": ROOT / "outputs/quality/all_data/SEM-370-r1/labels",
        },
        "public100": {
            "baseline": ROOT / "outputs/quality/SEM-378/r1/inference/predictions",
            "control": ROOT / "outputs/quality/QUAL-019/public100/control",
            "candidate": ROOT / "outputs/quality/QUAL-019/public100/candidate",
        },
    }
    for part, names in datasets.items():
        for name in names:
            for system, root in sources[part].items():
                colorize(root / name, OUT / "assets" / part / system / name)
            difference(
                sources[part]["control"] / name,
                sources[part]["candidate"] / name,
                OUT / "assets" / part / "difference" / name,
            )
    delta = summary["deltas"]
    gate_text = "通过，允许零重叠v2复验" if summary["all_promotion_gates_pass"] else "未通过，不进入v2训练"
    payload = json.dumps(datasets, ensure_ascii=False)
    root = "../../../../"
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SEM433 配对照片域监督审阅</title>
<style>
:root{{--bg:#101318;--panel:#181d25;--line:#303846;--text:#eef3fa;--muted:#a7b0bf;--good:#6ed6a0;--bad:#ff8b86}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,"Noto Sans SC",sans-serif}}
header{{position:sticky;top:0;z-index:5;background:#101318f2;border-bottom:1px solid var(--line);padding:14px 18px}}
h1{{font-size:20px;margin:0 0 6px}}.summary{{color:var(--muted)}}.summary b{{color:{'#6ed6a0' if summary['all_promotion_gates_pass'] else '#ff8b86'}}}
.tools{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px}}button,select{{background:#202733;color:var(--text);border:1px solid #3a4658;border-radius:6px;padding:7px 10px}}
.legend{{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;color:var(--muted)}}.sw{{display:inline-block;width:11px;height:11px;margin-right:4px;border-radius:2px}}
main{{padding:14px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}}
.card h2{{font-size:13px;margin:0;padding:9px 10px;border-bottom:1px solid var(--line)}}img{{width:100%;height:auto;display:block;background:#1c1c1e;aspect-ratio:1320/989;object-fit:contain}}
#detail{{font-variant-numeric:tabular-nums;color:var(--muted)}}@media(max-width:1100px){{main{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
</style></head><body><header><h1>SEM433 配对照片域监督：空间确认 + 100张无标签审阅</h1>
<div class="summary">固定主比较 SEM433−SEM432：五类 mIoU <b>{delta['five_class_miou_pp']:+.4f}pp</b>，前景 <b>{delta['foreground_miou_pp']:+.4f}pp</b>，索边界F1 <b>{delta['stay_cable_boundary_f1_pp']:+.4f}pp</b>；事前门：<b>{gate_text}</b>。无标签页只作视觉审阅，不当分数。</div>
<div class="legend"><span><i class="sw" style="background:#67a6ff"></i>deck</span><span><i class="sw" style="background:#f9dc58"></i>stay_cable</span><span><i class="sw" style="background:#de605b"></i>tower</span><span><i class="sw" style="background:#ae67cd"></i>foundation</span><span><i class="sw" style="background:#1c1c1e"></i>background/未变化</span></div>
<div class="tools"><select id="part"><option value="public100">301–400 无标签（RGB140测试输入）</option><option value="validation30">v1空间确认30张</option></select><button id="prev">上一张</button><select id="name"></select><button id="next">下一张</button><span id="detail"></span></div></header>
<main>
<section class="card"><h2>真实照片（仅人工对照，模型测试不读取）</h2><img id="photo"></section>
<section class="card"><h2>实际测试输入：重建RGB</h2><img id="render"></section>
<section class="card"><h2 id="baselineTitle">原二维基线</h2><img id="baseline"></section>
<section class="card"><h2>SEM432 · 双render匹配对照</h2><img id="control"></section>
<section class="card"><h2>SEM433 · render+photo训练候选</h2><img id="candidate"></section>
<section class="card"><h2 id="lastTitle">SEM433相对SEM432变化（黑色=不变）</h2><img id="last"></section>
</main>
<script>
const data={payload};let i=0;const part=document.querySelector('#part'),sel=document.querySelector('#name');
function rel(p,s,n){{return `assets/${{p}}/${{s}}/${{n}}`}}
function fill(){{sel.innerHTML=data[part.value].map(n=>`<option>${{n}}</option>`).join('');i=0;show()}}
function show(){{const p=part.value,names=data[p];i=(i+names.length)%names.length;sel.selectedIndex=i;const n=names[i],pub=p==='public100';
document.querySelector('#photo').src='{root}'+(pub?'outputs/quality/all_data/RGB-120-r2/scene/images/':'outputs/quality/RGB-152/scene/images/')+n;
document.querySelector('#render').src='{root}'+(pub?'outputs/quality/rgb140/full-r2/evaluation/images/':'outputs/quality/RGB-152/full-r2/evaluation/images/')+n;
for(const s of ['baseline','control','candidate'])document.querySelector('#'+s).src=rel(p,s,n);
document.querySelector('#last').src=rel(p,pub?'difference':'gt',n);document.querySelector('#lastTitle').textContent=pub?'SEM433相对SEM432变化（黑色=不变）':'人工真值（仅确认页）';
document.querySelector('#baselineTitle').textContent=pub?'SEM378 · 当前RGB140二维基线':'SEM396 · 单render空间基线';document.querySelector('#detail').textContent=`${{i+1}}/${{names.length}} · ${{n}}`;}}
part.onchange=fill;sel.onchange=()=>{{i=sel.selectedIndex;show()}};document.querySelector('#prev').onclick=()=>{{i--;show()}};document.querySelector('#next').onclick=()=>{{i++;show()}};document.onkeydown=e=>{{if(e.key==='ArrowLeft'){{i--;show()}}if(e.key==='ArrowRight'){{i++;show()}}}};fill();
</script></body></html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
