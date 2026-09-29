#!/usr/bin/env python3
"""Count image-level class-mask presence separately from pixel exposure."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256
OUT=ROOT/'outputs/quality/SEM-379/r1';BASE=ROOT/'outputs/quality/SEM-378/r1'
def main():
 old=[json.loads(l)['sample'] for l in (BASE/'mask2former/training.jsonl').read_text().splitlines()];new=[json.loads(l) for l in (OUT/'sampling-plan.jsonl').read_text().splitlines()];pairs={n:[] for n in sorted({x['name'] for x in old})}
 for a,b in zip(old,new):pairs[a['name']].append((a,b))
 counts={k:np.zeros(5,np.int64) for k in ['baseline','candidate']};presence={k:np.zeros(5,np.int64) for k in counts}
 for i,(n,ps) in enumerate(pairs.items()):
  gt=np.asarray(Image.open(ROOT/'outputs/quality/all_data/SEM-370-r1/labels'/n))
  for c in range(5):
   integral=np.pad((gt==c).cumsum(0).cumsum(1),((1,0),(1,0)))
   for col,k in enumerate(counts):
    for pair in ps:
     t=pair[col];y,x=t['top'],t['left'];v=integral[y+768,x+768]-integral[y,x+768]-integral[y+768,x]+integral[y,x]
     counts[k][c]+=v;presence[k][c]+=int(v>0)
  if i%50==0:print('class presence',i,flush=True)
 prior=json.loads((OUT/'sampling-audit.json').read_text())['class_pixel_exposure']
 assert all(counts[k].tolist()==prior[k] for k in counts)
 r={'status':'completed','class_order':['background','deck','stay_cable','tower','foundation'],'crop_class_presence':{k:v.tolist() for k,v in presence.items()},'pixel_exposure':{k:v.tolist() for k,v in counts.items()},'interpretation':'Presence affects which class masks enter author criterion; pixel exposure is not an exact loss weight because point sampling and per-mask normalization remain unchanged.','script_sha256':sha256(Path(__file__))}
 (OUT/'class-presence-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
if __name__=='__main__':main()
