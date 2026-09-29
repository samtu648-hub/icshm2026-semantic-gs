#!/usr/bin/env python3
"""CPU review of already-saved final6000 nine predictions while GPU is queued."""
import json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_sem378 import scores
from src.quality_protocol import sha256
OUT=ROOT/'outputs/quality/SEM-379/r1';BASE=ROOT/'outputs/quality/SEM-378/r1';LABELS=ROOT/'outputs/quality/all_data/SEM-370-r1/labels';PHOTO=ROOT/'outputs/quality/all_data/RGB-120-r2/scene/images';P=np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]],np.uint8)
def main():
 out=OUT/'nineview-review';out.mkdir(exist_ok=False);fit=json.loads((OUT/'mask2former/result.json').read_text());assert fit['trained_steps']==6000 and fit['status']=='completed'
 names=json.loads((ROOT/'experiments/runs/SEM-370-ALL300/config.yaml').read_text())['diagnostic_views'];regions=json.loads((ROOT/'outputs/quality/backbone_comparison/browser-r1/data.json').read_text())['images'];regions={r['name']:r for r in regions};rows=[];figures=[]
 for n in names:
  gt=np.asarray(Image.open(LABELS/n));old=np.asarray(Image.open(BASE/'inference/predictions'/n));new=np.asarray(Image.open(OUT/'mask2former/predictions'/n));error=np.zeros((*gt.shape,3),np.uint8);fix=(old!=gt)&(new==gt);harm=(old==gt)&(new!=gt);error[fix]=[40,190,100];error[harm]=[230,70,70];error[(old!=new)&~fix&~harm]=[170,100,210]
  images=[('Photo',Image.open(PHOTO/n).convert('RGB')),('RGB140',Image.open(BASE/'export/images'/n).convert('RGB')),('GT',Image.fromarray(P[gt])),('SEM378',Image.fromarray(P[old])),('SEM379 final6000',Image.fromarray(P[new])),('Fix green / Harm red',Image.fromarray(error))]
  row={'name':n,'corrected':int(fix.sum()),'harmed':int(harm.sum()),'sem378':scores(np.bincount((5*gt.astype(int)+old).ravel(),minlength=25).reshape(5,5)),'sem379':scores(np.bincount((5*gt.astype(int)+new).ravel(),minlength=25).reshape(5,5)),'prediction_sha256':sha256(OUT/'mask2former/predictions'/n)};rows.append(row)
  for kind,box in [('full',[0,0,1320,989]),('roi',regions[n]['box']),('structure',regions[n]['structure_box'])]:
   w=440;h=round(w*(box[3]-box[1])/(box[2]-box[0]));canvas=Image.new('RGB',(w*3,2*(h+24)),'white');draw=ImageDraw.Draw(canvas)
   for i,(label,im) in enumerate(images):
    x=(i%3)*w;y=(i//3)*(h+24);draw.text((x+4,y+4),n+' '+label,fill='black');canvas.paste(im.crop(box).resize((w,h),Image.Resampling.NEAREST),(x,y+24))
   filename=f'{Path(n).stem}-{kind}.png';canvas.save(out/filename);figures.append(f'<h2>{n} {kind} · {box}</h2><img src="{filename}" style="width:100%">')
 (out/'result.json').write_text(json.dumps({'status':'completed_final6000_nine_only_pending_full_inference','scope':'nine training views; no all300 conclusion','checkpoint_sha256':fit['artifacts']['model-6000.pt'],'rows':rows},indent=2)+'\n')
 (out/'index.html').write_text('<html><meta charset="utf-8"><title>SEM379 final6000 九图先行检查</title><body style="font:15px system-ui;max-width:1400px;margin:auto"><h1>固定九图：最终6000步端点</h1><p>仅九个训练视角的局部检查，全300和144轨迹仍待GPU推理。不调参、不筛选checkpoint。</p>'+''.join(figures))
 print({r['name']:{'miou_delta_pp':100*(r['sem379']['mIoU_with_background']-r['sem378']['mIoU_with_background']),'corrected':r['corrected'],'harmed':r['harmed'],'cable':r['sem379']['per_class']['stay_cable']} for r in rows})
if __name__=='__main__':main()
