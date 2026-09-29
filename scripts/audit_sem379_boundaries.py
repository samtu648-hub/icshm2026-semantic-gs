#!/usr/bin/env python3
"""Independent boundary metrics and sealed raw prediction hashes."""
import json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion,distance_transform_edt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256
from scripts.audit_sem378 import scores
OUT=ROOT/'outputs/quality/SEM-379/r1';BASE=ROOT/'outputs/quality/SEM-378/r1';LABELS=ROOT/'outputs/quality/all_data/SEM-370-r1/labels'
def main():
 start=time.monotonic();a=json.loads((OUT/'audit.json').read_text());names=[x['name'] for x in a['paired']['per_view']]
 result={};counts={arm:np.zeros(4,np.int64) for arm in ['sem378','sem379']};cm={arm:np.zeros((5,5),np.int64) for arm in counts};fn={arm:np.zeros(2,np.int64) for arm in counts}
 errors=OUT/'error-changes';errors.mkdir();rows=[];transition=np.zeros((5,5,5),np.int64)
 for i,n in enumerate(names):
  gt=np.asarray(Image.open(LABELS/n));g=gt==2;gb=g^binary_erosion(g);gd=distance_transform_edt(~gb)
  old=np.asarray(Image.open(BASE/'inference/predictions'/n));new=np.asarray(Image.open(OUT/'inference/predictions'/n));row={'name':n}
  transition+=np.bincount((25*gt.astype(np.int16)+5*old+new).ravel(),minlength=125).reshape(5,5,5)
  for arm,pred in [('sem378',old),('sem379',new)]:
   p=pred==2;pb=p^binary_erosion(p);pd=distance_transform_edt(~pb)
   v=np.array([np.sum(pb&(gd<=2)),np.sum(pb),np.sum(gb&(pd<=2)),np.sum(gb)],np.int64);counts[arm]+=v
   band=gd<=2;cm[arm]+=np.bincount(5*gt[band].astype(int)+pred[band],minlength=25).reshape(5,5)
   fn[arm]+=np.array([np.sum(g&~p&band),np.sum(g&~p&~band)],np.int64)
   row[arm]=v.tolist()
  im=np.zeros((*gt.shape,3),np.uint8);fix=(old!=gt)&(new==gt);harm=(old==gt)&(new!=gt);im[fix]=[40,190,100];im[harm]=[230,70,70];im[(old!=new)&~fix&~harm]=[170,100,210]
  Image.fromarray(im).save(errors/n);rows.append(row)
  if i%50==0:print('boundary',i,flush=True)
 for arm,v in counts.items():
  p=v[0]/v[1];r=v[2]/v[3];result[arm]={'tolerance_px':2,'boundary_precision':p,'boundary_recall':r,'boundary_f1':2*p*r/(p+r),'counts':v.tolist(),'fiveclass_gt_cable_boundary_band':scores(cm[arm]),'cable_fn_boundary_interior':fn[arm].tolist()}
 hashes={arm:{p.name:sha256(p) for p in sorted((path/'inference/predictions').glob('*.png'))} for arm,path in [('sem378',BASE),('sem379',OUT)]}
 assert all(len(v)==544 for v in hashes.values())
 nonc=[0,1,3,4]
 result['binary_cable_changes']={'fn_fixed':int(transition[2,nonc,2].sum()),'true_cable_lost':int(transition[2,2,nonc].sum()),'fp_removed':int(transition[np.ix_(nonc,[2],nonc)].sum()),'new_fp':int(transition[np.ix_(nonc,nonc,[2])].sum()),'background_fp_removed':int(transition[0,2,nonc].sum()),'new_background_fp':int(transition[0,nonc,2].sum())}
 c=result['binary_cable_changes'];old_m=a['metrics']['sem378']['per_class']['stay_cable'];new_m=a['metrics']['sem379']['per_class']['stay_cable']
 assert old_m['fn']-c['fn_fixed']+c['true_cable_lost']==new_m['fn']
 assert old_m['fp']-c['fp_removed']+c['new_fp']==new_m['fp']
 assert int(transition.sum())==300*989*1320
 result['gt_old_new_transition_cube']=transition.tolist()
 result.update(status='completed',per_view=rows,raw_prediction_hashes=hashes,seconds=time.monotonic()-start,script_sha256=sha256(Path(__file__)),scope='all300 training fit; boundary is annotated cable-region boundary, not individual physical wire boundary')
 (OUT/'boundary-audit.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
