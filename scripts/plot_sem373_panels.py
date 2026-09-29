#!/usr/bin/env python3
"""Native fixed-ROI contact sheets; image manipulation only for diagnostic layout."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
BASE=ROOT/'outputs/quality/SEM-373';ARMS=['baseline','control','joint'];PAL=np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]],np.uint8)

def main():
 c=json.loads((ROOT/'experiments/runs/SEM-373-STANDARD-JOINT-XYZ/config.yaml').read_text());rois=json.loads((ROOT/'experiments/runs/QUAL-005-PHOTO-INPUT-DIAG/config.yaml').read_text())['rois'];struct=json.loads((ROOT/'experiments/runs/QUAL-004-CURRENT-BOTTLENECK/config.yaml').read_text())['crops'];out=BASE/'panels-r1';out.mkdir(exist_ok=False)
 for name in c['diagnostic_views']:
  for kind,box in [('roi',rois.get(Path(name).stem,[0,0,1320,989])),('structure',struct.get(Path(name).stem,[0,0,1320,989]))]:
   w,h=box[2]-box[0],box[3]-box[1];result=Image.new('RGB',(w*2,(h+26)*4),'white');draw=ImageDraw.Draw(result)
   photos=[Image.open(ROOT/c['scene']/'images'/name),*[Image.open(BASE/('eval-'+arm+'-r1')/'rgb'/name) for arm in ARMS]]
   sems=[Image.fromarray(PAL[np.asarray(Image.open(ROOT/c['labels']/name))]),*[Image.fromarray(PAL[np.asarray(Image.open(BASE/('eval-'+arm+'-r1')/'semantic'/name))]) for arm in ARMS]]
   for i,(label,rgb,sem) in enumerate(zip(['photo / human GT',*ARMS],photos,sems)):
    y=i*(h+26);draw.text((5,y+5),name+' '+kind+' '+label,fill='black');draw.text((w+5,y+5),'RGB left / five-class right',fill='black');result.paste(rgb.crop(box),(0,y+26));result.paste(sem.crop(box),(w,y+26))
   result.save(out/(Path(name).stem+'-'+kind+'.png'))
 # Trajectory contact sheets retain identical frames and full image grids.
 for idx in [0,48,71,72,73,96,143]:
  name=f'frame{idx:04d}.png';w,h=1320,989;result=Image.new('RGB',(w*3,(h+26)*2),'white');draw=ImageDraw.Draw(result)
  for i,arm in enumerate(ARMS):
   draw.text((i*w+5,5),name+' '+arm,fill='black');result.paste(Image.open(BASE/('eval-'+arm+'-r1')/'rgb'/name),(i*w,26));result.paste(Image.fromarray(PAL[np.asarray(Image.open(BASE/('eval-'+arm+'-r1')/'semantic'/name))]),(i*w,h+52))
  result.save(out/(Path(name).stem+'.png'))
 print(out)

if __name__=='__main__':main()
