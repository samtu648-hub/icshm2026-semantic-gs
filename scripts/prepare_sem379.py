#!/usr/bin/env python3
"""CPU pre-registration, immutable bindings and paired 6000-crop audit."""
import json, sys, random, shutil, subprocess, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256, verify_hashes
from src.quality_mask2former_training import sampled_crop
from src.semantic_cable_sampling import CableRegionSampler
EID='SEM-379-CABLE-REGION-SAMPLING';OUT=ROOT/'outputs/quality/SEM-379/r1';BASE=ROOT/'outputs/quality/SEM-378/r1'
LABELS=ROOT/'outputs/quality/all_data/SEM-370-r1/labels'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 start=time.monotonic();config=read(ROOT/'experiments/runs'/EID/'config.yaml')
 prior=read(BASE/'mask2former/result.json');verify_hashes(ROOT,prior['code_sha256'])
 export=read(BASE/'export/result.json');verify_hashes(BASE/'export/images',export['image_hashes'])
 names=sorted(read(ROOT/config['split_path'])['semantic_train']);assert len(names)==300
 labels={n:np.asarray(Image.open(LABELS/n)) for n in names}
 class Lazy(dict):
  def __missing__(self,n):
   # One dummy RGB view suffices: sampling never examines its contents.
   return np.broadcast_to(np.zeros(3,np.uint8),(989,1320,3))
 images=Lazy();rng=random.Random(config['seed']);parent_rng=random.Random(config['seed'])
 sampler=CableRegionSampler(sampled_crop,config['seed'],config['mixture'])
 old=[json.loads(l)['sample'] for l in (BASE/'mask2former/training.jsonl').read_text().splitlines()]
 traces=[];counts={'baseline':np.zeros(5,np.int64),'candidate':np.zeros(5,np.int64)}
 zeros={k:0 for k in counts};fractions={k:[] for k in counts};examples=[]
 for i in range(6000):
  _,y,b=sampled_crop(parent_rng,names,images,labels);assert b==old[i]
  _,z,c=sampler(rng,names,images,labels);assert b['name']==c['name'] and b['flip']==c['flip']
  if c['source']=='uniform':assert np.array_equal(y,z) and all(b[k]==c[k] for k in b)
  else:
   if c['source']=='stay_cable_gt':
    assert labels[c['name']][c['anchor_y'],c['anchor_x']]==2 and np.any(z==2)
  for arm,label in [('baseline',y),('candidate',z)]:
   v=np.bincount(label.ravel(),minlength=5);counts[arm]+=v;zeros[arm]+=int(v[2]==0);fractions[arm].append(float(v[2]/label.size))
  traces.append(c)
  if c['source']=='stay_cable_gt' and len(examples)<18 and c['name'] not in [e[0]['name'] for e in examples]:examples.append((b,c))
  if i%1000==0:print('sample audit',i,flush=True)
 assert parent_rng.getstate()==rng.getstate()
 (OUT/'sampling-plan.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in traces))
 exposure=[]
 byname={n:[] for n in names}
 for b,c in zip(old,traces):byname[b['name']].append((b,c))
 for n in names:
  gt=labels[n];pred=np.asarray(Image.open(BASE/'inference/predictions'/n));row={'name':n,'gt_cable':int((gt==2).sum()),'fn':int(((gt==2)&(pred!=2)).sum()),'draws':len(byname[n])}
  for arm,col in [('baseline',0),('candidate',1)]:
   diff=np.zeros((990,1321),np.int32)
   for pair in byname[n]:
    t=pair[col];y,x=t['top'],t['left'];diff[y,x]+=1;diff[y+768,x]-=1;diff[y,x+768]-=1;diff[y+768,x+768]+=1
   cov=diff.cumsum(0).cumsum(1)[:989,:1320]
   row[arm]={}
   for key,mask in [('cable',gt==2),('cable_fn',(gt==2)&(pred!=2)),('background',gt==0)]:
    v=cov[mask];row[arm][key]={'pixels':len(v),'total_exposures':int(v.sum()),'never_seen':int((v==0).sum()),'mean':float(v.mean()) if len(v) else None}
  exposure.append(row)
 # Systematic first 18 distinct guided draws, not selected display views.
 panels=OUT/'sampling-panels';panels.mkdir()
 palette=np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]],np.uint8)
 for i,(b,c) in enumerate(examples):
  photo=Image.open(BASE/'export/images'/b['name']).convert('RGB');gt=Image.fromarray(palette[labels[b['name']]])
  canvas=Image.new('RGB',(1320,1037),'white');canvas.paste(photo,(0,24));canvas.paste(gt,(660,24)) if False else None
  # Two full-image rows, with the same crop boxes on RGB and GT.
  canvas=Image.new('RGB',(660,1040),'white');d=ImageDraw.Draw(canvas)
  for j,im in enumerate((photo,gt)):
   canvas.paste(im.resize((660,494),Image.Resampling.NEAREST),(0,j*520+24));d.text((4,j*520+4),b['name']+' uniform=cyan guided=magenta',fill='black')
   for t,color in [(b,'cyan'),(c,'magenta')]:
    x,y=t['left']/2,t['top']/2+j*520+24;d.rectangle((x,y,x+384,y+384),outline=color,width=2)
   d.ellipse((c['anchor_x']/2-3,c['anchor_y']/2+j*520+21,c['anchor_x']/2+3,c['anchor_y']/2+j*520+27),fill='red')
  canvas.save(panels/f'{i:02d}-{b["name"]}')
 result={'status':'completed','experiment_id':EID,'historical_control_source_hashes_exact':prior['code_sha256'],'parent_6000_trace_exact':True,'same_view_and_flip_sequence':True,'guided_draws':sum(t['source']=='stay_cable_gt' for t in traces),'uniform_draws':sum(t['source']=='uniform' for t in traces),'fallback_draws':sum(t['source']=='uniform_no_cable' for t in traces),'all300_views_seen':len({t['name'] for t in traces}), 'class_pixel_exposure':{k:v.tolist() for k,v in counts.items()},'zero_cable_crops':zeros,'cable_crop_fraction_quantiles':{k:np.quantile(v,[0,.1,.5,.9,1]).tolist() for k,v in fractions.items()},'per_view_exposure':exposure,'sampling_plan_sha256':sha256(OUT/'sampling-plan.jsonl'),'seconds':time.monotonic()-start}
 write(OUT/'sampling-audit.json',result)
 ex=OUT/'export';ex.mkdir();(ex/'images').symlink_to(Path('../../../SEM-378/r1/export/images'))
 # Read-only reuse of exact frozen renderings, with no RGB pipeline invocation.
 assert (ex/'images/159.png').exists();write(ex/'result.json',export)
 snapshot=OUT/'code';snapshot.mkdir()
 code=list(prior['code_sha256'])+['src/semantic_cable_sampling.py','scripts/prepare_sem379.py','scripts/run_sem379.py','src/semantic_cable_binary.py','src/semantic2d.py','src/quality_experts.py','src/quality_protocol.py','src/quality_semantic_audit.py']
 hashes={p:sha256(ROOT/p) for p in code}
 for p in code:
  dest=snapshot/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,dest)
 frozen={row['path']:row['sha256'] for row in read(ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json')['artifacts']}
 bind={'config_sha256':sha256(ROOT/'experiments/runs'/EID/'config.yaml'),'code_sha256':hashes,'label_hashes':{n:sha256(LABELS/n) for n in names},'label_manifest_sha256':sha256(LABELS.parent/'label_manifest.json'),'manifest_sha256':sha256(ROOT/config['dataset_manifest']),'split_sha256':sha256(ROOT/config['split_path']),'baseline_model_sha256':sha256(BASE/'mask2former/model-6000.pt'),'rgb140_artifacts':read(ROOT/config['rgb_run']/'result.json')['artifacts'],'frozen18':frozen}
 write(OUT/'binding.json',bind)
 print({k:v for k,v in result.items() if k not in ['per_view_exposure','historical_control_source_hashes_exact']})
if __name__=='__main__':main()
