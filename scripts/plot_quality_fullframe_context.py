#!/usr/bin/env python3
"""Native corrected/harmed regions for a preregistered counterfactual, no deployment."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,write_new_json
P=np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]],np.uint8)
def im(p):
 with Image.open(p) as x:return np.array(x)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['evaluation','batch','artifact-root','output']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);r=json.loads((a.evaluation/'result.json').read_text());assert r['status']=='completed_fullframe_context_calibration_diagnostic';pc=json.loads((ROOT/'experiments/runs/SEM-340-MASK2FORMER-PREFLIGHT/config.yaml').read_text());cfg=json.loads((ROOT/'experiments/runs/SEM-346-FULLFRAME-CONTEXT-DIAG/config.yaml').read_text())
 fig,axes=plt.subplots(6,5,figsize=(15,17),squeeze=False)
 for row,e in enumerate(r['examples']):
  name=e['name'];raw=im(ROOT/cfg['comparison']/'calibration/mask2former_6k/raw'/name)
  with np.load(a.batch/f"full-oof-{e['fold']}"/(Path(name).stem+'.npz')) as z:full=z['probabilities'].argmax(0).astype(np.uint8)
  target=im(a.artifact_root/'r2/sem_001/labels'/name);mask=(raw==2)&(full==0);change=np.full((*raw.shape,3),245,np.uint8);change[mask&(target==0)]=[35,160,90];change[mask&(target==2)]=[220,55,55];change[mask&~np.isin(target,[0,2])]=[240,160,30]
  x0,y0,x1,y1=e['bbox_xyxy'];cx=(x0+x1)//2;cy=(y0+y1)//2;side=max(128,min(512,max(x1-x0,y1-y0)+64));left=max(0,min(cx-side//2,1320-side));top=max(0,min(cy-side//2,989-side))
  data=[im(ROOT/pc['rgb_export']/'images'/name),P[target],P[raw],P[full],change]
  for col,(data,title) in enumerate(zip(data,['Rendered RGB','Official GT','Old tiled raw','Whole-frame branch','Veto: green=correct/red=lost'])):
   ax=axes[row,col];ax.imshow(data[top:top+side,left:left+side],interpolation='nearest',extent=(left,left+side,top+side,top));ax.set_title(title,fontsize=8);ax.tick_params(labelsize=6)
  axes[row,0].text(0,1.15,f"{name} / fold{e['fold']} / {e['kind']} / component{e['pixels']}px",transform=axes[row,0].transAxes,fontsize=8)
 fig.suptitle('SEM346 fixed calibration counterfactual; no rule adopted; crops may truncate large components',fontsize=12);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(a.output/'fullframe_veto_cost.png',dpi=150);plt.close(fig)
 write_new_json(a.output/'result.json',{'evaluation_sha256':sha256(a.evaluation/'result.json'),'script_sha256':sha256(Path(__file__)),'artifacts':{'fullframe_veto_cost.png':sha256(a.output/'fullframe_veto_cost.png')}})
if __name__=='__main__':main()
