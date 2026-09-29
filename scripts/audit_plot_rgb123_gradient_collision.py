#!/usr/bin/env python3
"""Independent NumPy reduction of saved gradient arrays; diagnostic review page."""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256
c=json.loads((Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'experiments/runs/RGB-123-GRADIENT-COLLISION-DIAG/config.yaml').read_text());o=ROOT/c['output'];r=json.loads((o/'result.json').read_text());b=json.loads((o/'binding.json').read_text());checks=0;summary=[]
for row in r['rows']:
 z=np.load(o/'arrays'/(Path(row['view']).stem+'-'+row['region']+'.npz'));s=z['signed_l1'];a=z['absolute_l1'];valid=a>1e-12;ratio=1-float(s[valid].sum(dtype=np.float64))/float(a[valid].sum(dtype=np.float64));assert abs(ratio-row['mass_cancellation'])<2e-6;assert np.isfinite(a).all() and np.isfinite(s).all() and np.all(a>=0) and np.all(s>=0);assert int(valid.sum())==row['count'];assert np.all(s<=a+2e-7)
 for key,stat in row['radius_strata'].items():
  lo,hi=map(int,key.split('-'));m=valid&(z['radii']>=lo)&(z['radii']<hi);assert int(m.sum())==stat['count']
  if m.any():assert abs(1-float(s[m].sum(dtype=np.float64))/float(a[m].sum(dtype=np.float64))-stat['mass_cancellation'])<2e-6
  checks+=1
 # Pixel counts and target background-vs-structure exclusion are auditable.
 assert int(z['source_mask'].sum())==row['source_pixels']
 if row['region'] in ['shore','near','far_bank_reference']:
  labels=np.asarray(Image.open(ROOT/c['semantic_batch']/'labels'/row['view']));assert not np.any(z['source_mask']&np.isin(labels,[1,2,3,4]))
 summary.append(dict(view=row['view'],region=row['region'],mass_cancellation=ratio,active=int(valid.sum())))
for k,h in b['parent_hashes'].items():assert sha256(ROOT/c['parent_run']/k)==h
(o/'audit.json').write_text(json.dumps(dict(status='passed',regions=len(summary),radius_strata=checks,source_masks_verified=True,triangle_bound=True,parent_hashes_exact=True),indent=2)+'\n')
html='<!doctype html><meta charset="utf-8"><title>RGB123 gradient cancellation</title><style>body{font:17px sans-serif;background:#f2f3f5;margin:30px;max-width:1500px}table{border-collapse:collapse;background:white}td,th{padding:8px 14px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}.bar{height:12px;background:#356d99}</style><h1>RGB123：像素梯度抵消诊断</h1><p>RGB120 / seed0 / 30k冻结端点；六训练视角，不更新模型。指标为 1−Σg(|Σp gx|+|Σp gy|)/Σgp(|gx|+|gy|)，越高表示抵消越多，不等于增密后会提高质量。区域为完整光度损失梯度的来源像素掩膜；区域面积、纹理与覆盖不同，非随机匹配对照。L1版本仅用于诊断，不冒充AbsGS完整训练实现。</p><table><tr><th>视角</th><th>区域</th><th>抵消比例</th><th>非零高斯数</th><th>比例示意</th></tr>'
for x in summary:html+=f'<tr><td>{x["view"]}</td><td>{x["region"]}</td><td>{x["mass_cancellation"]*100:.2f}%</td><td>{x["active"]}</td><td style="width:200px"><div class="bar" style="width:{x["mass_cancellation"]*100:.1f}%"></div></td></tr>'
html+='</table><h2>来源像素ROI定位</h2>'
for n in c['views']:
 im=Image.open(o/('native-'+n) if c.get('iteration',30000)!=30000 else ROOT/c['semantic_batch']/'export/images'/n).convert('RGB');draw=ImageDraw.Draw(im)
 for k,boxes in c['roi_boxes'].get(n,{}).items():
  for box in boxes:draw.rectangle(box,outline=(240,50,50),width=3);draw.text((box[0]+4,box[1]+4),k,fill='red')
 im.save(o/('roi-'+n));html+=f'<h3>{n}</h3><img src="roi-{n}">'
html=html.replace('30k冻结端点',str(c.get('iteration',30000))+'步冻结端点');(o/'index.html').write_text(html);print(json.dumps(summary))
