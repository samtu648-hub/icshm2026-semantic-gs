#!/usr/bin/env python3
"""Independent saved-image evaluation and quota accounting for RGB124."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256
c=json.loads((ROOT/'experiments/runs/RGB-124-MATCHED-DENSIFICATION-PF/config.yaml').read_text());o=ROOT/c['output'];parent=ROOT/c['parent_run'];batch=ROOT/'outputs/quality/all_data/SEM-370-r1';rows=json.loads((o/'evaluation.json').read_text());train=json.loads((o/'training.json').read_text());growth=train['growth'];assert len(growth)==5;tot=0;different=0
for i,g in enumerate(growth):
 assert g['iteration']==7100+i*100
 for l in g['levels']:
  q=l['quota'];assert q==min(c['level_caps'][l['level']],*l['available'].values());assert all(len(v)==q and len(set(map(tuple,v)))==q for v in l['selected'].values());tot+=q;different+=len(set(map(tuple,l['selected']['signed']))^set(map(tuple,l['selected']['absolute'])))
assert len(train['logs'])==1000
for i in range(500):assert train['logs'][2*i]['camera']==train['logs'][2*i+1]['camera'] and train['logs'][2*i]['iteration']==train['logs'][2*i+1]['iteration']==7001+i
assert train['anchors']['signed']==train['anchors']['absolute']
checks=0;summary=[]
for row in rows:
 n=row['view'];photo=np.asarray(Image.open(parent/'scene/images'/n)).astype(float);masks={'frame':np.ones(photo.shape[:2],bool)}
 if (batch/'labels'/n).exists():
  labels=np.asarray(Image.open(batch/'labels'/n));masks.update(deck=labels==1,stay_cable=labels==2,structure=binary_dilation(np.isin(labels,[1,2,3,4]),iterations=4))
 rc=json.loads((ROOT/'experiments/runs/RGB-122-LOCAL-OPACITY-PF/config.yaml').read_text())
 if n in rc['shore_boxes']:
  mask=np.zeros(photo.shape[:2],bool)
  for x0,y0,x1,y1 in rc['shore_boxes'][n]:mask[y0:y1,x0:x1]=True
  masks['shore']=mask
 for arm in c['arms']:
  im=np.asarray(Image.open(o/arm/'images'/n)).astype(float)
  for k,m in masks.items():
   if not m.any():continue
   err=im-photo;mae=np.abs(err)[m].mean();mse=(err[m]**2).mean();assert abs(mae-row['metrics'][k][arm]['mae'])<1e-9;assert abs(mse-row['metrics'][k][arm]['mse'])<1e-9;checks+=1
 a=row['metrics']['frame'];summary.append(dict(view=n,mae_delta=a['absolute']['mae']-a['signed']['mae'],psnr_delta=10*np.log10(a['signed']['mse']/a['absolute']['mse'])))
mean=lambda k:float(np.mean([x[k] for x in summary]));s=dict(status='passed_independent_audit',metric_regions=checks,total_new_anchors_each=tot,selected_symmetric_difference_total=different,matched500_camera_steps=True,mean_mae_delta=mean('mae_delta'),mean_psnr_delta=mean('psnr_delta'),improved=sum(x['mae_delta']<0 for x in summary),worse=sum(x['mae_delta']>0 for x in summary),same=sum(x['mae_delta']==0 for x in summary),rows=summary)
(o/'audit.json').write_text(json.dumps(s,indent=2)+'\n')
# Native full/crop panels for fixed cases plus worst five (explicitly exploratory).
fixed=['159.png','160.png','161.png','049.png','241.png','146.png','174.png'];worst=[x['view'] for x in sorted(summary,key=lambda x:-x['mae_delta'])[:5]];names=list(dict.fromkeys(fixed+worst));pan=o/'panels';pan.mkdir(exist_ok=False)
html='<!doctype html><meta charset="utf-8"><title>RGB124 matched densification</title><style>body{font:18px sans-serif;margin:28px;background:#eee}section{background:white;padding:12px;margin:18px 0}img{max-width:100%}</style><h1>RGB124：同新增预算的增密排序对照</h1><p>两臂同7k训练检查点、各500步；左照片，中signed控制，右absolute候选。训练内全400，不是新泛化。不是旧30k与7.5k的比较。</p>'
for n in names:
 ims=[Image.open(parent/'scene/images'/n),Image.open(o/'signed/images'/n),Image.open(o/'absolute/images'/n)]
 for suffix,box in [('full',(0,0,1320,989)),('crop',(0,620,1320,989) if n in ['159.png','160.png','161.png'] else (0,0,1320,989))]:
  w=660;h=round((box[3]-box[1])*w/(box[2]-box[0]));im=Image.new('RGB',(w*3,h+30),'white');d=ImageDraw.Draw(im)
  for i,(pic,title) in enumerate(zip(ims,['Photo','Signed / matched budget','Absolute / matched budget'])):im.paste(pic.crop(box).resize((w,h)),(i*w,30));d.text((i*w+7,7),title,fill='black')
  fn=Path(n).stem+'-'+suffix+'.png';im.save(pan/fn)
  if suffix=='crop':html+=f'<section><h2>{n} {"fixed" if n in fixed else "exploratory worst5"}</h2><img src="panels/{fn}"><p><a href="panels/{Path(n).stem}-full.png">完整图</a></p></section>'
(o/'index.html').write_text(html);print(json.dumps({k:v for k,v in s.items() if k!='rows'}))
