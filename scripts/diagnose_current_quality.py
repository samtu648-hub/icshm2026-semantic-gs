#!/usr/bin/env python3
"""Fixed-view saved-output development diagnosis; no models or prediction edits."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label
ROOT=Path(__file__).resolve().parents[1]
PAL=np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190],[220,220,220]],dtype=np.uint8)
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--artifact-root',type=Path,required=True);args=a.parse_args()
 c=read(ROOT/'experiments/runs/QUAL-004-CURRENT-BOTTLENECK/config.yaml');out=ROOT/c['output'];out.mkdir(exist_ok=True,parents=True)
 proofs={}; t=time.time()
 def image(p,expected=None,rgb=False):
  h=sha(p)
  if expected: assert h==expected,str(p)
  try: key='repo:'+str(p.relative_to(ROOT))
  except ValueError: key='artifact:'+str(p.relative_to(args.artifact_root))
  proofs[key]=h
  im=np.array(Image.open(p).convert('RGB') if rgb else Image.open(p))
  assert im.shape[:2]==(989,1320),(p,im.shape)
  return im
 config=read(ROOT/c['final_config']);src=config['final_sources'];records={k:read(ROOT/v/'result.json') for k,v in src.items()}
 seal=read(ROOT/'outputs/quality/final_comparison/SEM-350-predictions-r1/seal.json')['predictions'];finalroot=ROOT/'outputs/quality/final_transfer/SEM-363-transfer-r1';finalseal=read(finalroot/'predictions_sealed.json')['files']
 rows=[];thumbs=[]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
 def panel(images,titles,path,crop=None,cols=4,width=600):
  box=crop or (0,0,1320,989);height=round(width*(box[3]-box[1])/(box[2]-box[0]));cellh=height+34
  canvas=Image.new('RGB',(cols*width,((len(images)+cols-1)//cols)*cellh),'white');d=ImageDraw.Draw(canvas)
  for i,(im,title) in enumerate(zip(images,titles)):
   x=i%cols*width;y=i//cols*cellh;d.text((x+7,y+6),title,fill='black',font=font)
   tile=Image.fromarray(im).crop(box).resize((width,height),Image.Resampling.NEAREST)
   canvas.paste(tile,(x,y+34))
  canvas.save(path)
 for name in c['names']:
  gt=image(args.artifact_root/'r2/sem_001/labels'/name)
  real=image(ROOT/'outputs/quality/runs/RGB-110-SCAFFOLD-CTRL-full-r1/scene/images'/name,rgb=True)
  rgb=image(ROOT/'outputs/quality/expert_inputs/Scaffold-development-dualline-20260909-r1/images'/name,rgb=True)
  oldrgb=image(args.artifact_root/'r2/sem_102/validation_render/final-20260902-r1/images'/name,rgb=True)
  arms={}
  for k in ['old_sem161','new_raw','new_fused']:
   it=seal[name][k];arms[k]=image((ROOT if it['root']=='repo' else args.artifact_root)/it['path'],it['sha256'])
  arms['SEM363']=image(finalroot/'predictions'/name,finalseal['predictions/'+name])
  maps={}
  for key,rel in [('static','predictions/'),('b2_3k','step-3000/predictions/'),('b2_6k','step-6000/predictions/'),('deck','predictions/'),('stay_cable','predictions/')]:
   kind='b2' if key.startswith('b2') else key;path=rel+name
   v=image(ROOT/src[kind]/path,records[kind]['artifacts'][path]);assert np.isin(v,[0,1] if kind in ['deck','stay_cable'] else range(5)).all()
   maps[key]=np.where(v==1,1 if kind=='deck' else 2,5).astype(np.uint8) if kind in ['deck','stay_cable'] else v
  raw=arms['new_raw'];final=arms['SEM363'];correct=(raw==gt)
  for m in maps.values():correct|=m==gt
  err=final!=gt;common=err&~correct;unused=err&correct
  def n(m):return int(m.sum())
  stats={}
  for cl in range(5):
   target=gt==cl;lab,num=label(err&target);sizes=np.bincount(lab.ravel())[1:]
   stats[str(cl)]={'gt':n(target),'final_wrong':n(err&target),'common':n(common&target),'correct_alternative_unused':n(unused&target),'largest_error_component':int(sizes.max(initial=0)), 'branches_correct_on_final_error':{k:n(err&target&(m==gt)) for k,m in {'raw':raw,**maps}.items()}}
  cms={k:np.bincount((gt*5+v).ravel(),minlength=25).reshape(5,5).tolist() for k,v in arms.items()}
  # Cross-check selected current matrices against sealed historical counts.
  historical=next(r for r in read(finalroot/'per_view.json') if r['name']==name)
  assert cms['SEM363']==historical['confusions']['selector'];assert cms['new_raw']==historical['confusions']['new_raw']
  diag=np.full((*gt.shape,3),245,dtype=np.uint8);diag[common]=[210,45,155];diag[unused]=[0,155,160]
  statsrow={'name':name,'by_gt_class':stats,'confusions':cms,'final_wrong':n(err),'common':n(common),'unused':n(unused),'raw_correct_destroyed':n((raw==gt)&err),'raw_wrong_fixed':n((raw!=gt)&~err),'deck_to_background':{k:n((gt==1)&(v==0)) for k,v in arms.items()}}
  rows.append(statsrow)
  ims=[real,oldrgb,rgb,PAL[gt],PAL[arms['old_sem161']],PAL[raw],PAL[arms['new_fused']],PAL[final],PAL[maps['static']],PAL[maps['b2_3k']],PAL[maps['b2_6k']],diag]
  titles=['Real photo PINHOLE','RGB030 (old)','RGB110 (current)','GT','SEM161 (old)','SEM349 raw','SEM350 rule fusion','SEM363 learned fusion','SEM330 static','SEM362 B2 paired 3k','SEM362 B2 paired 6k','Wrong: magenta common / teal unused']
  panel(ims,[name[:3]+' | '+x for x in titles],out/(name[:3]+'_overview.png'))
  if name[:3] in c['crops']:
   box=c['crops'][name[:3]];panel([real,rgb,PAL[gt],PAL[raw],PAL[arms['new_fused']],PAL[final],diag,PAL[maps['deck']],PAL[maps['stay_cable']]],['Real photo','RGB110','GT','SEM349 raw','SEM350 rule fusion','SEM363 learned','Error source map','Deck positive (grey=abstain)','stay_cable positive (grey=abstain)'],out/(name[:3]+'_crop.png'),box,3,box[2]-box[0])
  print(name,statsrow['deck_to_background'],'wrong/common/unused',n(err),n(common),n(unused),flush=True)
 result={'experiment_id':c['experiment_id'],'config_sha256':sha(ROOT/'experiments/runs/QUAL-004-CURRENT-BOTTLENECK/config.yaml'),'code_sha256':sha(Path(__file__)),'git_commit':c['git_commit'],'seed':c['seed'],'manifest':c['dataset_manifest'],'split':c['split_id'],'seconds':time.time()-t,'views':rows,'inputs':proofs,'note':'Core candidates only: raw/static/B2 paired3k6k/binary positives. Binary negatives abstain, not background. Posthoc development diagnosis, no tuning, no model inference.'}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
