#!/usr/bin/env python3
"""Build a local interactive review for sealed SEM430 outputs."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/quality/SEM-430/r1"
result = json.loads((BASE / "result.json").read_text(encoding="utf-8"))
validation = result["validation_names"]
public = [row["name"] for row in result["public100"]["rows"]]
review = BASE / "review"
review.mkdir(exist_ok=False)
root = "../../../../../"
payload = json.dumps({"validation": validation, "public100": public})
document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>SEM430 桥面教师复验</title><style>
body{{font:14px system-ui;background:#11151b;color:#e8edf3;margin:0}}header{{position:sticky;top:0;background:#11151bf2;padding:12px;z-index:2;border-bottom:1px solid #344050}}button,select{{background:#202733;color:#fff;border:1px solid #4d5b70;padding:7px}}.summary{{color:#b8c6db;margin-top:7px}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;padding:10px}}figure{{margin:0;background:#1b222c;padding:6px}}img{{width:100%;display:block}}figcaption{{padding:5px}}@media(max-width:1000px){{.grid{{grid-template-columns:1fr}}}}</style></head><body>
<header><b>SEM430：只保留桥面照片教师</b> <select id='part'><option value='public100'>301–400无标签</option><option value='validation'>v2确认30图</option></select> <button id='prev'>上一张</button> <select id='name'></select> <button id='next'>下一张</button><div class='summary'>v2：4,631改变，2,847修正 / 1,527损伤，精度61.48%，未过75%门；候选只把render背景改为flip稳定的大块deck。</div></header>
<main class='grid'><figure><figcaption>真实照片（仅教师输入）</figcaption><img id='photo'></figure><figure><figcaption>重建RGB（学生/测试输入）</figcaption><img id='renderRgb'></figure><figure><figcaption>接受区域（白色）</figcaption><img id='mask'></figure><figure><figcaption>原render语义</figcaption><img id='base'></figure><figure><figcaption>photo teacher语义</figcaption><img id='teacher'></figure><figure><figcaption>桥面门控候选</figcaption><img id='candidate'></figure></main>
<script>const data={payload};const part=document.querySelector('#part'),sel=document.querySelector('#name');let i=0;function fill(){{sel.innerHTML=data[part.value].map(n=>`<option>${{n}}</option>`).join('');i=0;show()}}function show(){{const a=data[part.value];i=(i+a.length)%a.length;sel.selectedIndex=i;const n=a[i],p=part.value;document.querySelector('#photo').src='{root}'+(p==='public100'?'outputs/quality/all_data/RGB-120-r2/scene/images/':'outputs/quality/RGB-154/scene/images/')+n;document.querySelector('#renderRgb').src='{root}'+(p==='public100'?'outputs/quality/rgb140/full-r2/evaluation/images/':'outputs/quality/RGB-154/full-r1/evaluation/images/')+n;document.querySelector('#mask').src='../'+p+'/accepted-mask/'+n;document.querySelector('#base').src='../'+p+'/render-visual/'+n;document.querySelector('#teacher').src='../'+p+'/photo-teacher-visual/'+n;document.querySelector('#candidate').src='../'+p+'/candidate-visual/'+n}}part.onchange=fill;sel.onchange=()=>{{i=sel.selectedIndex;show()}};document.querySelector('#prev').onclick=()=>{{i--;show()}};document.querySelector('#next').onclick=()=>{{i++;show()}};document.onkeydown=e=>{{if(e.key==='ArrowLeft'){{i--;show()}}if(e.key==='ArrowRight'){{i++;show()}}}};fill();</script></body></html>"""
(review / "index.html").write_text(document, encoding="utf-8")
