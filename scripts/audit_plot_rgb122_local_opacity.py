#!/usr/bin/env python3
"""Independent saved-pixel checks and review panels for the fixed RGB122 run."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256
c=json.loads((ROOT/'experiments/runs/RGB-122-LOCAL-OPACITY-PF/config.yaml').read_text());o=ROOT/c['output'];r=json.loads((o/'result.json').read_text());b=json.loads((o/'binding.json').read_text());parent=ROOT/c['parent_run'];batch=ROOT/c['semantic_batch'];pan=o/'panels';pan.mkdir(exist_ok=False);checks=0;protect_rows=[]
html='<!doctype html><meta charset="utf-8"><title>RGB122 fixed opacity</title><style>body{font:18px sans-serif;background:#eee;margin:24px}section{background:white;margin:20px 0;padding:12px}img{max-width:100%}</style><h1>RGB122 / anchor4417 offset4 / opacity×0.5</h1><p>同RGB120 30k、seed0、全400训练内诊断。左照片、原样、固定0.5、零opacity、0.5差分×8。新相机无照片真值。旧模型保持；此页是局部实验，不代表整体拉丝已修复。</p>'
for row in r['rows']:
 n=row['view'];ims={k:np.asarray(Image.open(o/'images'/(k+'-'+n))) for k in ['base','half','zero']};photo=None if row['novel'] else np.asarray(Image.open(parent/'scene/images'/n));prot=None
 lp=batch/'labels'/n
 if not row['novel'] and lp.exists():prot=binary_dilation(np.isin(np.asarray(Image.open(lp)),[1,2,3,4]),iterations=4)
 sp=o/'support'/(Path(n).stem+'.npz');support=np.load(sp)['mass']>0 if sp.exists() else np.zeros(ims['base'].shape[:2],bool)
 for arm in ['half','zero']:
  diff=np.abs(ims[arm].astype(float)-ims['base']);changed=diff.max(2)>0;masks={'frame':np.ones(changed.shape,bool),'original_support':support,'changed':changed}
  if prot is not None:masks['protected']=prot
  if n in c['shore_boxes']:
   mask=np.zeros(changed.shape,bool)
   for x0,y0,x1,y1 in c['shore_boxes'][n]:mask[y0:y1,x0:x1]=True
   masks['shore']=mask
  for key,m in masks.items():
   actual=dict(pixels=int(m.sum()),changed_pixels=int(changed[m].sum()),max_delta=float(diff[m].max()) if m.any() else 0.,rgb_delta_mae=float(diff[m].mean()) if m.any() else 0.)
   if photo is not None and m.any():actual.update(photo_mae_before=float(np.abs(ims['base'].astype(float)-photo)[m].mean()),photo_mae_after=float(np.abs(ims[arm].astype(float)-photo)[m].mean()))
   assert all(abs(v-row['metrics'][arm][key][k])<1e-10 for k,v in actual.items()),(n,arm,key);checks+=1
  if arm=='half' and prot is not None:protect_rows.append(dict(view=n,changed_pixels=int(changed[prot].sum())))
 # Full-frame and full support (not an artificially clipped safety display).
 boxes=[('full',(0,0,1320,989))];sr=next(s for s in r['support'] if s['view']==n)
 if 'bbox' in sr:
  x0,y0,x1,y1=sr['bbox'];boxes.append(('support',(max(0,x0-24),max(0,y0-24),min(1320,x1+24),min(989,y1+24))))
 for tag,box in boxes:
  arrays=([photo] if photo is not None else [])+[ims[k] for k in ['base','half','zero']]+[np.clip(np.abs(ims['half'].astype(float)-ims['base'])*8,0,255).astype('uint8')];titles=(['Photo'] if photo is not None else [])+['RGB120','opacity x0.5','opacity zero','Difference x8'];w=420;h=round((box[3]-box[1])*w/(box[2]-box[0]));panel=Image.new('RGB',(w*len(arrays),h+28),'white');d=ImageDraw.Draw(panel)
  for i,(im,title) in enumerate(zip(arrays,titles)):panel.paste(Image.fromarray(im).crop(box).resize((w,h)),(i*w,28));d.text((i*w+4,7),title,fill='black')
  fn=Path(n).stem+'-'+tag+'.png';panel.save(pan/fn)
  if tag=='support' or not sr['support_pixels']:html+=f'<section><h2>{n} / {tag}</h2><p>Full support bbox {sr.get("bbox")}; labelled protected changed pixels {row["metrics"]["half"].get("protected",{}).get("changed_pixels","unavailable")}</p><img src="panels/{fn}"><p><a href="panels/{Path(n).stem}-full.png">完整画面</a></p></section>'
# Native sequence, synchronized animated triptych (presentation only).
seq=['159.png']+[f'novel-0-{j}.png' for j in range(4)]+['160.png']+[f'novel-1-{j}.png' for j in range(4)]+['161.png'];frames=[]
for n in seq:
 canvas=Image.new('RGB',(1500,600),'white');d=ImageDraw.Draw(canvas)
 for i,arm in enumerate(['base','half','zero']):
  im=Image.open(o/'images'/(arm+'-'+n));canvas.paste(im.crop((0,620,560,989)).resize((500,550)),(500*i,40));d.text((500*i+8,10),n+' / '+arm,fill='black')
 frames.append(canvas)
frames[0].save(o/'trajectory.gif',save_all=True,append_images=frames[1:],duration=350,loop=0)
html=html.replace('<h1>RGB122 / anchor4417 offset4 / opacity×0.5</h1>','<h1>RGB122 / anchor4417 offset4 / opacity×0.5</h1><p>159→160→161同步运动对照（展示动画经GIF量化，判定以原生PNG为准）</p><img src="trajectory.gif">');(o/'index.html').write_text(html)
for k,h in b['parent_hashes'].items():assert sha256(parent/k)==h
for k,h in b['copy_hashes'].items():assert sha256(o/'model'/k)==h
assert len([s for s in r['support'] if not s['novel']])==400
# Independent interpolation reconstruction vs saved official requests.
from scipy.spatial.transform import Rotation,Slerp
from src.quality_cameras import read_camera_requests
cams=read_camera_requests(parent/'scene/sparse/0');traj=json.loads((o/'trajectory.json').read_text());maxerr=0.
for t in traj:
 i=t['segment'];u=t['u'];keys=[cams[n] for n in c['trajectory']['keyframes']];cs=[x.pose.center for x in keys];p0,p1,p2,p3=cs[max(0,i-1)],cs[i],cs[i+1],cs[min(2,i+2)];center=.5*((2*p1)+(-p0+p2)*u+(2*p0-5*p1+4*p2-p3)*u*u+(-p0+3*p1-3*p2+p3)*u**3);rot=Slerp([0,1],Rotation.from_matrix(np.stack([keys[i].pose.rotation,keys[i+1].pose.rotation])))(u*u*(3-2*u)).as_matrix();error=max(np.max(np.abs(rot-np.array(t['rotation']))),np.max(np.abs(-rot@center-np.array(t['translation']))));maxerr=max(maxerr,float(error));assert error<1e-6
out=dict(status='passed',metric_regions=checks,rendered_views=len(r['rows']),official_support_count=400,affected_official=[s['view'] for s in r['support'] if not s['novel'] and s['support_pixels']],protected_pixel_changes=protect_rows,interpolation_max_abs_error=maxerr,parent_and_copy_exact=True)
(o/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
