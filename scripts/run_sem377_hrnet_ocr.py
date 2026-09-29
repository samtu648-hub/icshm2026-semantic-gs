#!/usr/bin/env python3
"""Single, auditable HRNet-W48+OCR fit on frozen RGB130 images (SEM377)."""
import argparse, fcntl, hashlib, json, os, random, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/quality/SEM-377-HRNET-OCR-RGB130-ALL300'
OUT = None
RGB = ROOT / 'outputs/quality/backbone_comparison/standard/export/images'
LABELS = ROOT / 'outputs/quality/all_data/SEM-370-r1/labels'
SPLIT = ROOT / 'outputs/quality/all_data/RGB-120-r2/split.json'
UPSTREAM = ROOT / 'third_party/hrnet-semantic-segmentation-official'
RUNTIME = ROOT / 'outputs/quality/mask2former/SEM-340/runtime'
WEIGHT = ROOT / 'outputs/quality/hrnet_ocr/weights/HRNet_W48_C_ssld_pretrained.pth'
CONFIG = ROOT / 'experiments/runs/SEM-377-HRNET-OCR-RGB130-ALL300/config.yaml'
SEED, CROP, UPDATES, ACCUM = 20260915, 768, 24000, 4

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def write(path, obj):
    with Path(path).open('x') as f: json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')

def setup():
    sys.path[:0]=[str(UPSTREAM/'lib'),str(RUNTIME)]
    import numpy as np
    np.int=int  # upstream NumPy-1.x compatibility alias, no algorithm change.
    import torch
    from config import config
    from config.default import update_config
    import argparse as ap
    update_config(config,ap.Namespace(cfg=str(UPSTREAM/'experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml'),opts=[]))
    config.defrost();config.DATASET.NUM_CLASSES=5;config.MODEL.PRETRAINED='';config.MODEL.NUM_OUTPUTS=2;config.LOSS.BALANCE_WEIGHTS=[.4,1.];config.freeze()
    import models.seg_hrnet_ocr as ocr
    # Upstream selects SyncBN for modern torch, which is DDP-only. Single-GPU BN is its valid local equivalent.
    ocr.BatchNorm2d=ocr.BatchNorm2d_class=torch.nn.BatchNorm2d
    return np,torch,config,ocr

def deterministic(np,torch):
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def normalized(np, arr):
    x=arr.astype(np.float32)/255.
    return (x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)

def crop_pair(np, names, images, labels, rng):
    n=names[rng.randrange(len(names))]; top=rng.randint(0,989-CROP); left=rng.randint(0,1320-CROP); flip=rng.random()<.5
    x=images[n][top:top+CROP,left:left+CROP]; y=labels[n][top:top+CROP,left:left+CROP]
    if flip:x=x[:,::-1].copy();y=y[:,::-1].copy()
    return x,y,dict(name=n,top=top,left=left,flip=flip)

def build(np,torch,config,ocr):
    model=ocr.get_seg_model(config)
    raw=torch.load(WEIGHT,map_location='cpu',weights_only=True)
    state=raw.get('state_dict',raw) if isinstance(raw,dict) else raw
    compatible={k:v for k,v in state.items() if k in model.state_dict() and tuple(v.shape)==tuple(model.state_dict()[k].shape)}
    missing,unexpected=model.load_state_dict(compatible,strict=False)
    if len(compatible)<1000: raise RuntimeError(f'Insufficient official pretrained coverage: {len(compatible)}')
    model.cuda();return model,dict(weight_sha256=sha(WEIGHT),loaded_tensor_count=len(compatible),missing_count=len(missing),unexpected_count=len(unexpected),parameter_count=sum(p.numel() for p in model.parameters()))

def loss(torch, outputs, target):
    import torch.nn.functional as F
    return .4*F.cross_entropy(F.interpolate(outputs[0],size=(CROP,CROP),mode='bilinear',align_corners=True).float(),target)+F.cross_entropy(F.interpolate(outputs[1],size=(CROP,CROP),mode='bilinear',align_corners=True).float(),target)

def predict(np,torch,model,rgb):
    import torch.nn.functional as F
    h,w=rgb.shape[:2]; tops=[0,h-CROP]; lefts=[0,512,w-CROP]; sums=np.zeros((5,h,w),np.float32); counts=np.zeros((h,w),np.float32)
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for top in sorted(set(tops)):
            for left in sorted(set(lefts)):
                x=torch.from_numpy(normalized(np,rgb[top:top+CROP,left:left+CROP]).transpose(2,0,1).copy())[None].cuda()
                p=F.interpolate(model(x)[-1].float(),size=(CROP,CROP),mode='bilinear',align_corners=True).softmax(1)[0].cpu().numpy()
                sums[:,top:top+CROP,left:left+CROP]+=p;counts[top:top+CROP,left:left+CROP]+=1
    return sums/counts[None]

def metric(np, cm):
    out={};vals=[]
    for i,n in enumerate(['background','deck','stay_cable','tower','foundation']):
        tp=float(cm[i,i]); fp=float(cm[:,i].sum()-tp); fn=float(cm[i,:].sum()-tp); den=tp+fp+fn
        out[n]=dict(IoU=tp/den if den else 0.,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.) ;vals.append(out[n]['IoU'])
    return dict(confusion=cm.tolist(),per_class=out,mIoU_with_background=float(np.mean(vals)))

def run(stage):
    global OUT
    OUT = BASE / (stage+('-r3' if stage=='train' else '-r2'))
    OUT.mkdir(parents=True,exist_ok=False)
    np,torch,config,ocr=setup();deterministic(np,torch)
    split=json.loads(SPLIT.read_text()); train=sorted(split['semantic_train']); allnames=sorted(split['rgb_train'])
    if len(train)!=300 or len(allnames)!=400 or len(set(train))!=300:raise RuntimeError('Unexpected RGB130/label role split')
    for n in allnames:
        if not (RGB/n).is_file():raise FileNotFoundError(RGB/n)
    for n in train:
        if not (LABELS/n).is_file():raise FileNotFoundError(LABELS/n)
    record=dict(experiment_id='SEM-377-HRNET-OCR-RGB130-ALL300',stage=stage,status='starting',started_unix=time.time(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),config_sha256=sha(CONFIG),upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip(),split_sha256=sha(SPLIT),weight_sha256=sha(WEIGHT),seed=SEED,classes=['background','deck','stay_cable','tower','foundation'],input_role=dict(rgb130_frozen_all400=True,labeled_train_views=300,unlabeled_qualitative=100,resolution=[1320,989]))
    start=time.monotonic()
    try:
      with (ROOT/'outputs/quality/gpu.lock').open('a') as lk:
       fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
       apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);record['gpu_processes_before']=apps
       if any(x.strip() and 'ToDesk' not in x for x in apps.splitlines()):raise RuntimeError('Another GPU compute task is active')
       torch.cuda.reset_peak_memory_stats();model,load=build(np,torch,config,ocr);record['pretrained_load']=load
       if stage=='preflight':
        from PIL import Image
        images={n:np.asarray(Image.open(RGB/n).convert('RGB')) for n in train}; labels={n:np.asarray(Image.open(LABELS/n),dtype=np.uint8) for n in train};rng=random.Random(SEED)
        opt=torch.optim.SGD([{'params':[p for n,p in model.named_parameters() if not any(k in n for k in ['cls','aux','ocr'])],'lr':1/300},{'params':[p for n,p in model.named_parameters() if any(k in n for k in ['cls','aux','ocr'])],'lr':1/30}],momentum=.9,weight_decay=5e-4)
        scaler=torch.amp.GradScaler('cuda'); rows=[]
        for step in range(1,9):
         x,y,t=crop_pair(np,train,images,labels,rng);opt.zero_grad(set_to_none=True)
         with torch.autocast('cuda',dtype=torch.float16):v=loss(torch,model(torch.from_numpy(normalized(np,x).transpose(2,0,1).copy())[None].cuda()),torch.from_numpy(y.astype('int64'))[None].cuda())
         if not torch.isfinite(v):raise FloatingPointError('nonfinite preflight loss')
         scaler.scale(v).backward();scaler.step(opt);scaler.update();rows.append(dict(step=step,sample=t,loss=float(v.detach()),scale=scaler.get_scale()))
        record.update(status='completed_numerical_preflight_not_quality',updates=8,rows=rows)
       elif stage=='train':
        from PIL import Image
        images={n:np.asarray(Image.open(RGB/n).convert('RGB')) for n in train};labels={n:np.asarray(Image.open(LABELS/n),dtype=np.uint8) for n in train}; rng=random.Random(SEED)
        bb=[p for n,p in model.named_parameters() if not any(k in n for k in ['cls','aux','ocr'])]; head=[p for n,p in model.named_parameters() if any(k in n for k in ['cls','aux','ocr'])]
        opt=torch.optim.SGD([{'params':bb,'lr':1/300},{'params':head,'lr':1/30}],momentum=.9,weight_decay=5e-4);scaler=torch.amp.GradScaler('cuda',init_scale=1024);model.train();journal=OUT/'training.jsonl'
        with journal.open('x') as f:
         opt.zero_grad(set_to_none=True)
         for step in range(1,UPDATES+1):
          factor=(1-(step-1)/UPDATES)**.9
          for g,base in zip(opt.param_groups,[1/300,1/30]):g['lr']=base*factor
          x,y,t=crop_pair(np,train,images,labels,rng)
          with torch.autocast('cuda',dtype=torch.float16):v=loss(torch,model(torch.from_numpy(normalized(np,x).transpose(2,0,1).copy())[None].cuda()),torch.from_numpy(y.astype('int64'))[None].cuda())/ACCUM
          if not torch.isfinite(v):raise FloatingPointError('nonfinite training loss')
          scaler.scale(v).backward()
          if step%ACCUM==0:
           scaler.unscale_(opt);norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=False));before=scaler.get_scale();scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
           if not torch.isfinite(torch.tensor(norm)) and scaler.get_scale() >= before:raise FloatingPointError('Nonfinite gradient without AMP backoff')
          else:norm=None
          row=dict(step=step,sample=t,loss=float(v.detach())*ACCUM,lr=factor/300,gradient_step=step%ACCUM==0,grad_norm=norm,scale=scaler.get_scale(),elapsed_seconds=time.monotonic()-start);f.write(json.dumps(row)+'\n')
          if step==1 or step%100==0:f.flush();print(json.dumps(dict(step=step,loss=row['loss'],seconds=row['elapsed_seconds'])),flush=True)
        ckpt=OUT/'model-24000.pt';torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'record_binding':record,'crop_steps':UPDATES,'optimizer_updates':UPDATES//ACCUM},ckpt)
        record.update(status='completed_training_pending_inference',crop_steps=UPDATES,optimizer_updates=UPDATES//ACCUM,training_seconds=time.monotonic()-start,checkpoint_sha256=sha(ckpt),checkpoint_mib=ckpt.stat().st_size/2**20)
       elif stage=='infer':
        from PIL import Image
        ckpt=BASE/'train-r3/model-24000.pt';state=torch.load(ckpt,map_location='cpu',weights_only=True)['model'];model.load_state_dict(state);model.cuda();predroot=OUT/'predictions';probroot=OUT/'probabilities';predroot.mkdir();probroot.mkdir();cm=np.zeros((5,5),np.int64);times=[]
        for i,n in enumerate(allnames):
         rgb=np.asarray(Image.open(RGB/n).convert('RGB'));torch.cuda.synchronize();t=time.monotonic();p=predict(np,torch,model,rgb);torch.cuda.synchronize();times.append(time.monotonic()-t);pred=p.argmax(0).astype(np.uint8);Image.fromarray(pred).save(predroot/n)
         if n in set(train):
          y=np.asarray(Image.open(LABELS/n),dtype=np.uint8);cm+=np.bincount((y.astype(np.int64)*5+pred).ravel(),minlength=25).reshape(5,5)
         if i in [0,99,199,299,399]:np.savez_compressed(probroot/(Path(n).stem+'.npz'),probabilities=p)
         if i%25==0:print(json.dumps(dict(inferred=i+1,total=400)),flush=True)
        record.update(status='completed_inference_training_view_metrics',checkpoint_sha256=sha(ckpt),metrics=metric(np,cm),labeled_count=300,unlabeled_count=100,inference_seconds=times,inference_median_seconds=float(np.median(times)),prediction_hashes={n:sha(predroot/n) for n in allnames})
       else: raise ValueError(stage)
    finally:
      record.update(seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
      if record['status']=='starting':record['status']='failed_preserved'
      write(OUT/'result.json',record)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight','train','infer'],required=True);a=p.parse_args();run(a.stage)
