#!/usr/bin/env python3
"""SEM345 native diagnostic crops, no predicted-label edits."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json
P=np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]],np.uint8)
def im(p):
 with Image.open(p) as x:return np.array(x)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['source','artifact-root','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);r=json.loads((a.source/'result.json').read_text());assert r['status']=='completed_read_only_stay_cable_diagnostic';cfg=json.loads((ROOT/'experiments/runs/SEM-345-STAY-CABLE-BACKGROUND-DIAG/config.yaml').read_text());pc=json.loads((ROOT/cfg['preflight_config']).read_text());cases=r['calibration_examples']
 for role,examples in [('calibration',cases),('trajectory',r['trajectory_examples'])]:
  fig,axes=plt.subplots(len(examples),4,figsize=(13,len(examples)*3.4),squeeze=False)
  for row,e in enumerate(examples):
   x0,y0,x1,y1=e['bbox_xyxy'];left=max(0,x0-48);right=min(1320,x1+48);top=max(0,y0-48);bottom=min(989,y1+48)
   if role=='calibration':
    parent=ROOT/cfg['comparison'];name=e['name'];data=[im(ROOT/pc['rgb_export']/'images'/name),P[im(a.artifact_root/'r2/sem_001/labels'/name)],P[im(parent/'calibration/mask2former_6k/raw'/name)],P[im(parent/'calibration/mask2former_6k/calibrated'/name)]];titles=['Rendered RGB','Official GT polygon','Raw Mask2Former','Old calibrated fusion'];subtitle=f"{name} / {e['partition']} / {e['kind']} / {e['pixels']} pixels"
   else:
    base=ROOT/cfg['trajectory'];stem=Path(e['name']).stem+'.npz';fold=e['fold']
    with np.load(base/f'render-oof-{fold}'/stem) as rr,np.load(base/f'infer-oof-{fold}'/stem) as z:data=[rr['rgb'],P[rr['static']],P[z['raw']],P[z['fused']]]
    titles=['Rendered RGB','Static field','Raw Mask2Former','Old calibrated fusion'];subtitle=f"OOF-{fold} / {e['name']} / {e['pixels']} detached pixels; no GT"
   for col,(v,title) in enumerate(zip(data,titles)):
    ax=axes[row,col];ax.imshow(v[top:bottom,left:right],interpolation='nearest',extent=(left,right,bottom,top));ax.set_title(title,fontsize=10);ax.tick_params(labelsize=7)
   axes[row,0].text(0,1.14,subtitle,transform=axes[row,0].transAxes,fontsize=8)
  fig.suptitle('SEM345 diagnostic regions; ranking for display only; no cleanup applied',fontsize=12);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(a.output/f'{role}_crops.png',dpi=160);plt.close(fig)
 write_new_json(a.output/'result.json',{'source_sha256':sha256(a.source/'result.json'),'script_sha256':sha256(Path(__file__)),'artifacts':{p.name:sha256(p) for p in a.output.glob('*.png')}})
if __name__=='__main__':main()
