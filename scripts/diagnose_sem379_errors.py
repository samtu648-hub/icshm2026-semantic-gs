#!/usr/bin/env python3
"""All300, label-registered exposure and input-detail diagnostics; no training."""
import json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import uniform_filter,maximum_filter,minimum_filter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256
OUT=ROOT/'outputs/quality/SEM-379/r1';BASE=ROOT/'outputs/quality/SEM-378/r1';LABELS=ROOT/'outputs/quality/all_data/SEM-370-r1/labels';PHOTO=ROOT/'outputs/quality/all_data/RGB-120-r2/scene/images'
P=np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]],np.uint8)
def main():
 start=time.monotonic();out=OUT/'input-diagnostic';out.mkdir(exist_ok=False)
 audit=json.loads((OUT/'sampling-audit.json').read_text());names=[r['name'] for r in audit['per_view_exposure']]
 fixed=json.loads((ROOT/'experiments/runs/SEM-370-ALL300/config.yaml').read_text())['diagnostic_views']
 selected=list(dict.fromkeys(fixed+[names[i] for i in np.linspace(0,299,12,dtype=int)]))
 counts={k:0 for k in ['all_fn','photo_detail_render_weak','both_have_local_detail','both_low_detail','render_only_detail']};rows=[];panels=[]
 for i,n in enumerate(names):
  gt=np.asarray(Image.open(LABELS/n));pred=np.asarray(Image.open(BASE/'inference/predictions'/n));fn=(gt==2)&(pred!=2)
  photo=np.asarray(Image.open(PHOTO/n).convert('RGB'));rgb=np.asarray(Image.open(BASE/'export/images'/n).convert('RGB'))
  # Prespecified 5x5 local luminance range diagnostic, NOT a visibility classifier.
  detail=[]
  for im in [photo,rgb]:
   gray=im.astype(np.float32).mean(2);detail.append(maximum_filter(gray,size=5)-minimum_filter(gray,size=5))
  pd,rd=[v>=12 for v in detail]
  groups={'photo_detail_render_weak':pd&~rd,'both_have_local_detail':pd&rd,'both_low_detail':~pd&~rd,'render_only_detail':~pd&rd}
  row={'name':n,'fn':int(fn.sum()),'groups':{k:int((fn&m).sum()) for k,m in groups.items()}}
  counts['all_fn']+=row['fn']
  for k,v in row['groups'].items():counts[k]+=v
  rows.append(row)
  if n in selected:
   density=uniform_filter(fn.astype(np.float32),size=128,mode='constant');y,x=np.unravel_index(density.argmax(),density.shape);x=max(0,min(1064,int(x)-128));y=max(0,min(733,int(y)-128));box=(x,y,x+256,y+256)
   error=np.zeros_like(photo);error[fn]=[235,65,65];error[(gt!=2)&(pred==2)]=[200,70,220]
   canvas=Image.new('RGB',(256*5,284),'white');draw=ImageDraw.Draw(canvas)
   for j,(label,im) in enumerate([('photo',photo),('RGB140',rgb),('GT',P[gt]),('SEM378',P[pred]),('FN red / FP purple',error)]):
    draw.text((j*256+4,4),f'{n} {label}',fill='black');canvas.paste(Image.fromarray(im).crop(box),(j*256,28))
   canvas.save(out/n);panels.append({'name':n,'box':box,'selection':'fixed nine plus 12 evenly spaced sorted training names; highest 128px FN density patch within each'})
  if i%50==0:print('input diagnostic',i,flush=True)
 result={'status':'completed','experiment_id':'SEM-379-CABLE-REGION-SAMPLING','scope':'SEM378 all300 fit errors only','method':'local luminance range >=12/255 in 5x5; proxy for input detail, not physical cable visibility or causal attribution','warning':'GT stay_cable includes filled fan regions and inter-cable background; low contrast is not proof RGB geometry is missing','counts':counts,'per_view':rows,'panels':panels,'script_sha256':sha256(Path(__file__)),'seconds':time.monotonic()-start}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(counts)
if __name__=='__main__':main()
