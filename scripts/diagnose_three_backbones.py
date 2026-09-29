"""Reuse completed native exports for paired RGB regions; no new training."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from src.quality_protocol import sha256
def main():
 cpath=ROOT/'experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml';c=json.loads(cpath.read_text());out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 base=ROOT/'outputs/quality/backbone_comparison';parent=ROOT/'outputs/quality/all_data/RGB-120-r2';sem=ROOT/'outputs/quality/all_data/SEM-370-r1';rows=[];bindings={}
 for name in c['views']:
  photo=np.asarray(Image.open(parent/'scene/images'/name),dtype=float);gt=np.asarray(Image.open(sem/'labels'/name))
  errors={}
  for arm in ['scaffold','standard','mip']:
   path=base/arm/'export/images'/name;bindings[str(path.relative_to(ROOT))]=sha256(path);errors[arm]=np.abs(np.asarray(Image.open(path),dtype=float)-photo).mean(2)
  regions={'frame':np.ones(gt.shape,bool),**{n:gt==i for i,n in enumerate(['background','deck','stay_cable','tower','foundation'])}}
  for key,reg in c['regions'].items():
   if reg['view']!=name:continue
   mask=np.zeros(gt.shape,bool)
   for x0,y0,x1,y1 in reg['boxes']:mask[y0:y1,x0:x1]=True
   regions[key]=mask
  for region,mask in regions.items():
   if not mask.any():continue
   e={a:x[mask] for a,x in errors.items()};high={a:x>=np.quantile(x,.9) for a,x in e.items()};common=np.logical_and.reduce(list(high.values()));union=np.logical_or.reduce(list(high.values()))
   rows.append(dict(view=name,region=region,pixels=int(mask.sum()),mae_255={a:float(x.mean()) for a,x in e.items()},high_error_common_pixels=int(common.sum()),high_error_union_pixels=int(union.sum()),high_error_unique_pixels={a:int((h & ~np.logical_or.reduce([z for k,z in high.items() if k!=a])).sum()) for a,h in high.items()}))
 result=dict(experiment_id=c['experiment_id'],git_commit=c['git_commit'],seed=0,dataset_manifest=c['dataset_manifest'],split_id=c['split_id'],split_sha256=sha256(parent/'split.json'),config_sha256=sha256(cpath),code_sha256=sha256(Path(__file__)),source_hashes=bindings,status='completed_readonly_regions',seconds=time.monotonic()-start,rows=rows,scope='training-view descriptive RGB error; per-arm top decile overlap is not causal geometry attribution; labels only region diagnostics; Scaffold fixed UID0 with native UID supplement in QUAL007')
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('Completed',len(rows),'paired regions')
if __name__=='__main__':main()
