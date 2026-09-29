#!/usr/bin/env python3
"""Build fixed ROI/full-view RGB121 review panels from saved ablations."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
c=json.loads((ROOT/'experiments/runs/RGB-121-LOCAL-ISOLATION-DIAG/config.yaml').read_text());o=ROOT/c['output'];r=json.loads((o/'result.json').read_text());(o/'panels').mkdir(exist_ok=False)
html='<!doctype html><meta charset="utf-8"><title>RGB121 local isolation</title><style>body{font:18px sans-serif;margin:24px;background:#eee}img{max-width:100%}section{background:white;padding:14px;margin:20px 0}</style><h1>RGB121 局部隔离诊断</h1><p>全400训练内诊断 / seed0 / RGB120 30k。左照片，中原模型，右固定offset临时屏蔽；第四列差分放大8倍。保护判据是14视角标签扩张4px中的贡献，不等于未见视角安全。174原生图未见旧RGB110明显长线，天空屏蔽是阴性检查。</p>'
for group,s in c['rois'].items():
 n=s['view'];ims=[Image.open(ROOT/c['parent_run']/'scene/images'/n).convert('RGB'),Image.open(o/'images'/('base-'+n)),Image.open(o/'images'/(group+'-'+n))];diff=np.clip(np.abs(np.asarray(ims[2]).astype(float)-np.asarray(ims[1]))*8,0,255).astype('uint8');ims.append(Image.fromarray(diff))
 for j,box in enumerate(s['boxes']):
  w=440;h=round(w*(box[3]-box[1])/(box[2]-box[0]));p=Image.new('RGB',(w*4,h+32),'white');d=ImageDraw.Draw(p)
  for k,(im,title) in enumerate(zip(ims,['Photo','RGB120','Masked offsets','Difference x8'])):p.paste(im.crop(box).resize((w,h)),(k*w,32));d.text((k*w+8,8),title,fill='black')
  name=f'{group}-{j}.png';p.save(o/'panels'/name);html+=f'<section><h2>{group} ROI {box}</h2><p>IDs {r["accepted"][group]}</p><img src="panels/{name}"></section>'
for n in c['views']:
 ims=[Image.open(ROOT/c['parent_run']/'scene/images'/n),Image.open(o/'images'/('base-'+n)),Image.open(o/'images'/('union-'+n))];p=Image.new('RGB',(1980,527),'white');d=ImageDraw.Draw(p)
 for k,(im,title) in enumerate(zip(ims,['Photo','RGB120','Union masked'])):p.paste(im.resize((660,495)),(k*660,32));d.text((k*660+8,8),title,fill='black')
 p.save(o/'panels'/n);html+=f'<section><h2>{n} 同一固定ID集合跨视角</h2><img src="panels/{n}"></section>'
(o/'index.html').write_text(html)
