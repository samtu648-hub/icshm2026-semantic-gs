#!/usr/bin/env python3
"""Matched RGB130/RGB137 evaluation, fixed local views and 144-pose trajectory."""
import argparse,fcntl,json,os,subprocess,sys,time
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();a.run=a.run.resolve();out=a.run/'evaluation';out.mkdir(exist_ok=False);start=time.monotonic()
 base=ROOT/'outputs/quality/backbone_comparison';parent=ROOT/'outputs/quality/all_data/RGB-120-r2';cfg=ROOT/'experiments/runs/RGB-137-STANDARD-ALL400-INIT400/config.yaml';fixed_cfg=json.loads((ROOT/'experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml').read_text());structure=json.loads((ROOT/'experiments/runs/QUAL-004-CURRENT-BOTTLENECK/config.yaml').read_text())['crops'];train=json.loads((parent/'split.json').read_text())['rgb_train'];baseline_eval=json.loads((base/'standard/rgb-evaluation/result.json').read_text());baseline_export=json.loads((base/'standard/export/result.json').read_text());run=json.loads((a.run/'result.json').read_text())
 r=dict(experiment_id='RGB-137-STANDARD-ALL400-INIT400',status='starting',seed=0,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),config_sha256=sha256(cfg),run_result_sha256=sha256(a.run/'result.json'),baseline_result_sha256=sha256(base/'standard-full-r1/result.json'),baseline_eval_sha256=sha256(base/'standard/rgb-evaluation/result.json'),code_sha256=sha256(Path(__file__)),quality_scope='all400 training-view fit; trajectory has no GT')
 try:
  assert run['status']=='completed_all400';verify_hashes(a.run,run['artifacts'])
  with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);r['gpu_processes_before']=apps
   if any(x.strip() and 'ToDesk' not in x for x in apps.splitlines()):raise RuntimeError('Other GPU compute process active')
   source=a.artifact_root/'r1/third_party/standard-3dgs';sys.path[:0]=[str(source),str(source/'submodules/diff-gaussian-rasterization'),str(source/'submodules/simple-knn')]
   import numpy as np,torch
   from PIL import Image,ImageDraw
   from plyfile import PlyData
   from gaussian_renderer import GaussianModel,render
   from arguments import PipelineParams
   from src.quality_cameras import read_camera_requests
   from src.quality_native_gaussian import make_native_view
   from scripts.evaluate_rgb import psnr,ssim,build_lpips_model,lpips_score
   torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
   ply=a.run/'model/point_cloud/iteration_30000/point_cloud.ply';m=GaussianModel(3);m.load_ply(str(ply));pp=PipelineParams(argparse.ArgumentParser());pipe=pp.extract(argparse.Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False,antialiasing=False));bg=torch.zeros(3,device='cuda')
   cams=read_camera_requests(parent/'scene/sparse/0');trajroot=a.artifact_root/'results/frozen_scaffold_sem001/flythrough_v1/cameras';traj=read_camera_requests(trajroot);req={**cams,**{n:replace(c,model='PINHOLE',radial_k=0.) for n,c in traj.items()}};assert sorted(cams)==sorted(train) and sorted(traj)==baseline_export['trajectory_names']
   r['trajectory_metadata_sha256']={n:sha256(trajroot/n) for n in ('images.txt','cameras.txt')};assert r['trajectory_metadata_sha256']==baseline_export['trajectory_metadata_sha256']
   def render8(model,cam):return np.rint(render(make_native_view(cam),model,pipe,bg)['render'].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
   (out/'images').mkdir();hashes={};times=[];lp,_,device=build_lpips_model('cuda');rows=[];baseline_rows={x['name']:x for x in baseline_eval['per_view']};regions=[];classes=['background','deck','stay_cable','tower','foundation'];class_err={k:[] for k in classes}
   with torch.inference_mode():
    for i,(n,cam) in enumerate(sorted(req.items())):
     torch.cuda.synchronize();t=time.monotonic();im=render8(m,cam);torch.cuda.synchronize();times.append(time.monotonic()-t);Image.fromarray(im).save(out/'images'/n);hashes[n]=sha256(out/'images'/n)
     if n in cams:
      photo=np.asarray(Image.open(parent/'scene/images'/n).convert('RGB'));ref=photo.astype(np.float32)/255;pred=im.astype(np.float32)/255;row=dict(name=n,PSNR=psnr(ref,pred),SSIM=ssim(ref,pred),LPIPS=lpips_score(ref,pred,lp,torch,device));row['delta_vs_rgb130']={k:row[k]-baseline_rows[n][k] for k in ('PSNR','SSIM','LPIPS')};rows.append(row)
      label=ROOT/'outputs/quality/all_data/SEM-370-r1/labels'/n
      if label.exists():
       gt=np.asarray(Image.open(label))
       for j,k in enumerate(classes):
        mask=gt==j
        if mask.any():class_err[k].append((float(np.abs(im.astype(float)-photo)[mask].mean()),float(np.abs(np.asarray(Image.open(base/'standard/export/images'/n)).astype(float)-photo)[mask].mean())))
      for key,reg in fixed_cfg['regions'].items():
       if reg['view']!=n:continue
       mask=np.zeros(im.shape[:2],bool)
       for x0,y0,x1,y1 in reg['boxes']:mask[y0:y1,x0:x1]=True
       old=np.asarray(Image.open(base/'standard/export/images'/n));regions.append(dict(region=key,view=n,pixels=int(mask.sum()),rgb130_mae_255=float(np.abs(old.astype(float)-photo)[mask].mean()),rgb137_mae_255=float(np.abs(im.astype(float)-photo)[mask].mean())))
     if i%50==0:print(json.dumps(dict(rendered=i+1,total=len(req))),flush=True)
   fresh=GaussianModel(3);fresh.load_ply(str(ply))
   with torch.inference_mode():
    for n in fixed_cfg['views']:assert np.array_equal(render8(fresh,cams[n]),np.asarray(Image.open(out/'images'/n)))
   metrics={k:float(np.mean([x[k] for x in rows])) for k in ('PSNR','SSIM','LPIPS')};paired={k:dict(mean_delta=float(np.mean([x['delta_vs_rgb130'][k] for x in rows])),improved=sum(x['delta_vs_rgb130'][k]>(0 if k!='LPIPS' else 0) for x in rows),degraded=sum(x['delta_vs_rgb130'][k]<(0 if k!='LPIPS' else 0) for x in rows)) for k in ('PSNR','SSIM','LPIPS')};paired['LPIPS']['improved']=sum(x['delta_vs_rgb130']['LPIPS']<0 for x in rows);paired['LPIPS']['degraded']=sum(x['delta_vs_rgb130']['LPIPS']>0 for x in rows)
   r.update(status='completed',metrics=metrics,baseline_metrics=baseline_eval['metrics'],paired=paired,per_view=rows,regions=regions,class_rgb_mae_255={k:dict(rgb137=float(np.mean([x[0] for x in v])),rgb130=float(np.mean([x[1] for x in v]))) for k,v in class_err.items()},image_hashes=hashes,train_count=len(rows),trajectory_count=len(traj),fresh_fixed9_exact=True,render_median_ms=float(np.median(times[1:])*1000),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,final_gaussian_count=int(len(PlyData.read(ply)['vertex'])),model_mib=ply.stat().st_size/2**20)
   del m,fresh,lp;torch.cuda.empty_cache()
  # CPU deliverables after releasing GPU lock.
  vis=out/'browser';vis.mkdir();panels=vis/'panels';panels.mkdir();tiles=[('真实照片',parent/'scene/images'),('RGB130：303视角初始化',base/'standard/export/images'),('RGB137：400视角初始化',out/'images')]
  for n in fixed_cfg['views']:
   boxes=[('full',(0,0,1320,989))];boxes+= [('crop',tuple(structure.get(Path(n).stem,[0,620,1320,989] if n in ('159.png','160.png') else [0,0,1320,380] if n=='174.png' else [0,0,1320,989])))]
   for suffix,box in boxes:
    w=660;h=round(w*(box[3]-box[1])/(box[2]-box[0]));canvas=Image.new('RGB',(w*3,h+28),'white');draw=ImageDraw.Draw(canvas)
    for j,(label,path) in enumerate(tiles):draw.text((j*w+6,7),label,fill='black');canvas.paste(Image.open(path/n).crop(box).resize((w,h)),(j*w,28))
    canvas.save(panels/(Path(n).stem+'-'+suffix+'.png'))
  frames=vis/'frames';frames.mkdir()
  for i in range(144):
   n=f'frame{i:04d}.png';canvas=Image.new('RGB',(2640,1018),'white');draw=ImageDraw.Draw(canvas)
   for j,(label,path) in enumerate(tiles[1:]):draw.text((j*1320+7,7),label,fill='black');canvas.paste(Image.open(path/n),(j*1320,28))
   canvas.save(frames/n)
  subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-framerate','12','-i',str(frames/'frame%04d.png'),'-c:v','libx264','-threads','4','-crf','18','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(vis/'trajectory-rgb130-vs-rgb137.mp4')],check=True)
  data=dict(metrics={'rgb130':r['baseline_metrics'],'rgb137':r['metrics']},paired=r['paired'],regions=r['regions'],fixed=fixed_cfg['views'],train=train,names=fixed_cfg['views']+[n for n in train if n not in fixed_cfg['views']]+sorted(traj),tiles=[dict(label=x,path=os.path.relpath(y,vis)) for x,y in tiles])
  write_new_json(vis/'data.json',data)
  html='''<!doctype html><meta charset="utf-8"><title>RGB137 全400初始化对照</title><style>body{font:15px/1.6 system-ui;margin:26px;background:#f4f5f7;color:#18202b}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.tile{background:#fff;padding:9px}canvas,video{width:100%}button,select{padding:7px;margin:4px}.metric{display:inline-block;background:#fff;padding:10px 18px;margin:6px}a{color:#075daf}</style><h1>RGB137：303视角初始化 vs 全400视角初始化</h1><p>两者均为Standard 3DGS、全400张RGB、30k、seed0；唯一主要变量为初始化来源。数值是训练视角拟合，144帧轨迹无GT，仅供视觉检查。</p><div id=m></div><select id=v></select><button id=full>完整图</button><button id=shore>岸线/近景</button><button id=sky>天空</button><span id=b></span><div class=grid id=g></div><h2>同一144帧轨迹</h2><video controls src=trajectory-rgb130-vs-rgb137.mp4></video><p><a href=data.json>完整指标与区域统计</a></p><script>const D=__D__;const F=(x,n=6)=>Number(x).toFixed(n);document.querySelector('#m').innerHTML=Object.entries(D.metrics).map(([k,x])=>`<div class=metric><b>${k}</b><br>PSNR ${F(x.PSNR)}<br>SSIM ${F(x.SSIM)}<br>LPIPS ${F(x.LPIPS)}</div>`).join('');const s=document.querySelector('#v');D.names.forEach(n=>s.add(new Option((D.fixed.includes(n)?'固定 · ':'')+n,n)));let box=[0,0,1320,989],items=[],version=0;function draw(){b.textContent=box.join(', ');for(const [im,c] of items){c.width=box[2]-box[0];c.height=box[3]-box[1];c.getContext('2d').drawImage(im,box[0],box[1],c.width,c.height,0,0,c.width,c.height)}}function load(){version++;const z=version;items=[];g.replaceChildren();const show=D.train.includes(s.value)?D.tiles:D.tiles.slice(1);for(const t of show){const d=document.createElement('div');d.className='tile';d.append(t.label);const c=document.createElement('canvas'),a=document.createElement('a');a.href=t.path+'/'+s.value;a.textContent=' 打开原生PNG';d.append(c,a);g.append(d);const im=new Image();im.onload=()=>{if(z==version){items.push([im,c]);draw()}};im.src=a.href}}s.onchange=()=>{box=[0,0,1320,989];load()};full.onclick=()=>{box=[0,0,1320,989];draw()};shore.onclick=()=>{box=[0,620,1320,989];draw()};sky.onclick=()=>{box=[0,0,1320,380];draw()};load();</script>'''.replace('__D__',json.dumps(data,ensure_ascii=False));(vis/'index.html').write_text(html)
  r.update(browser='browser/index.html',video='browser/trajectory-rgb130-vs-rgb137.mp4',video_sha256=sha256(vis/'trajectory-rgb130-vs-rgb137.mp4'));write_new_json(out/'result.json',r)
 except BaseException as e:r.update(status='failed_preserved',error=f'{type(e).__name__}: {e}');raise
 finally:r['seconds']=time.monotonic()-start;write_new_json(out/'result.json',r)
if __name__=='__main__':main()
