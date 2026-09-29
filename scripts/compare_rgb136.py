#!/usr/bin/env python3
"""Saved-image RGB136 comparison; all400 independent PSNR and protected-region audit."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json
read=lambda p:json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve();out=run/'comparison';out.mkdir(exist_ok=False);base=ROOT/'outputs/quality/backbone_comparison';parent=ROOT/'outputs/quality/all_data/RGB-120-r2';sem=ROOT/'outputs/quality/all_data/SEM-370-r1';c=read(ROOT/'experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml');split=read(parent/'split.json');names=split['rgb_train'];start=time.monotonic()
 arms={'scaffold_uid0':base/'scaffold/export/images','standard':base/'standard/export/images','mip':base/'mip/export/images','scaffold_noappearance':run/'export/images'};native=base/'scaffold-native-appearance/images';scores={k:read((run if k=='scaffold_noappearance' else base/('scaffold' if k=='scaffold_uid0' else k))/'rgb-evaluation/result.json') for k in arms};scores['scaffold_native_uid']=read(base/'scaffold-native-appearance/result.json');bindings={};rows=[];checks=0
 for k,path in arms.items():
  exp=read(path.parent/'result.json');assert exp['status']=='completed';assert exp['trajectory_metadata_sha256']==read(run/'export/result.json')['trajectory_metadata_sha256'];assert exp['trajectory_names']==read(run/'export/result.json')['trajectory_names'];verify_hashes(path,exp['image_hashes']);bindings[str((path.parent/'result.json').relative_to(ROOT))]=sha256(path.parent/'result.json')
 score_rows={k:{r['name']:r for r in s['per_view']} for k,s in scores.items()};assert all(set(v)==set(names) and len(v)==400 for v in score_rows.values())
 for i,n in enumerate(names):
  photo=np.asarray(Image.open(parent/'scene/images'/n),dtype=np.float64);masks={'frame':np.ones(photo.shape[:2],bool)}
  if (sem/'labels'/n).exists():
   gt=np.asarray(Image.open(sem/'labels'/n));masks.update({k:gt==j for j,k in enumerate(['background','deck','stay_cable','tower','foundation'])});bindings[str((sem/'labels'/n).relative_to(ROOT))]=sha256(sem/'labels'/n)
  for k,reg in c['regions'].items():
   if reg['view']==n:
    m=np.zeros(photo.shape[:2],bool)
    for x0,y0,x1,y1 in reg['boxes']:m[y0:y1,x0:x1]=True
    masks[k]=m
  available={**arms,**({'scaffold_native_uid':native} if (native/n).exists() else {})};rr={k:{} for k,m in masks.items() if m.any()}
  for arm,path in available.items():
   pred=np.asarray(Image.open(path/n),dtype=np.float64);err=pred-photo;psnr=-10*np.log10(np.mean(err**2)/255**2);assert abs(psnr-score_rows[arm][n]['PSNR'])<1e-5,(arm,n,psnr,score_rows[arm][n]['PSNR']);checks+=1;bindings[str((path/n).relative_to(ROOT))]=sha256(path/n)
   for k in rr:rr[k][arm]=dict(mae_255=float(np.abs(err)[masks[k]].mean()),mse_255=float((err[masks[k]]**2).mean()))
  rows.append(dict(view=n,regions=rr))
  if i%50==0:print(f'audit {i+1}/400',flush=True)
 means={};paired={}
 for region in ['frame','deck','stay_cable','tower','foundation']:
  means[region]={arm:float(np.mean([r['regions'][region][arm]['mae_255'] for r in rows if region in r['regions'] and arm in r['regions'][region]])) for arm in arms}
 for arm in ['scaffold_uid0','standard','mip','scaffold_native_uid']:
  dr=[score_rows['scaffold_noappearance'][n]['PSNR']-score_rows[arm][n]['PSNR'] for i,n in enumerate(names)];paired[arm]=dict(psnr_mean_delta=float(np.mean(dr)),improved=sum(x>0 for x in dr),worse=sum(x<0 for x in dr),same=sum(x==0 for x in dr))
 result=dict(experiment_id='RGB-136-SCAFFOLD-NOAPPEARANCE-ALL400',status='completed_saved_image_audit',roi_config_sha256=sha256(ROOT/'experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml'),native_uid_metrics_source_sha256=sha256(base/'scaffold-native-appearance/result.json'),script_sha256=sha256(Path(__file__)),config_sha256=read(run/'started.json')['config_sha256'],git_commit=read(run/'started.json')['git_commit'],seed=0,split_sha256=sha256(parent/'split.json'),manifest_sha256=read(run/'started.json')['manifest_sha256'],rgb_metrics={k:v['metrics'] for k,v in scores.items()},independent_psnr_checks=checks,region_mean_mae_255=means,paired_psnr=paired,regions=rows,source_hashes=bindings,seconds=time.monotonic()-start);write_new_json(out/'audit.json',result)
 panels=out/'panels';panels.mkdir();fixed=c['views'];panel_paths=[]
 tiles=[('Photo',parent/'scene/images'),('Scaffold UID0',arms['scaffold_uid0']),('Scaffold train UID',native),('Standard',arms['standard']),('Mip',arms['mip']),('Scaffold no appearance',arms['scaffold_noappearance'])]
 for n in fixed:
  for suffix,box in [('full',(0,0,1320,989)),('crop',(0,620,1320,989) if n in ['159.png','160.png'] else (0,0,1320,380) if n=='174.png' else (0,0,1320,989))]:
   w=660;hh=round(w*(box[3]-box[1])/(box[2]-box[0]));im=Image.new('RGB',(w*3,(hh+25)*2),'white');draw=ImageDraw.Draw(im)
   for j,(label,path) in enumerate(tiles):
    x=(j%3)*w;y=(j//3)*(hh+25);draw.text((x+5,y+5),label,fill='black');im.paste(Image.open(path/n).crop(box).resize((w,hh)),(x,y+25))
   dest=panels/(Path(n).stem+'-'+suffix+'.png');im.save(dest);panel_paths.append(dest.name)
 # Local interactive image browser; all image paths remain relative to this artifact.
 data=dict(metrics=result['rgb_metrics'],paired=paired,names=fixed+[n for n in names if n not in fixed]+read(run/'export/result.json')['trajectory_names'],train=names,fixed=fixed,tiles=[dict(label=l,path=os.path.relpath(p,out),native=l=='Scaffold train UID') for l,p in tiles],trajectory=read(run/'export/result.json')['trajectory_names'])
 html='''<!doctype html><meta charset="utf-8"><title>RGB136 Scaffold重训对照</title><style>body{font:16px system-ui;margin:24px;background:#f3f4f6;color:#182030}h1{font-size:26px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.tile,.metric{background:white;padding:12px;border-radius:8px}canvas{width:100%;image-rendering:auto}button,select,input{font:inherit;padding:6px;margin:5px}#metrics{display:flex;flex-wrap:wrap;gap:12px}.note{max-width:1100px;line-height:1.7}video{width:100%}</style><h1>RGB136：Scaffold 无逐照片外观重训</h1><p class=note>全400、30k、seed0；appearance_dim32→0。旧Scaffold固定UID0、训练UID附表和新模型分别列出。训练内拟合，不是隐藏视角排名；此页只比较RGB，不借用旧语义结果。下方照片/六图同步裁剪，训练UID原图仅保留历史固定九图。</p><div id=metrics></div><p id=conclusion>结论见同目录audit.json及实验report；先检查局部与轨迹，不能仅凭均值宣布最好。</p><select id=view></select><button id=full>完整图</button><button id=shore>岸边/近景</button><button id=sky>天空</button><span id=box></span><div id=grid class=grid></div><h2>统一轨迹</h2><p>旧Scaffold UID0 / Standard / Mip / 新Scaffold：同144相机。无照片真值，不报告伪造闪烁分数。</p><video controls preload=metadata src="trajectory.mp4"></video><script>const D=__DATA__;const labels={scaffold_uid0:'旧Scaffold UID0',scaffold_native_uid:'旧Scaffold 训练UID',standard:'Standard',mip:'Mip',scaffold_noappearance:'新Scaffold 无外观'};document.querySelector('#metrics').innerHTML=Object.entries(D.metrics).map(([k,v])=>`<div class=metric><b>${labels[k]}</b><p>PSNR ${v.PSNR.toFixed(4)} dB<br>SSIM ${v.SSIM.toFixed(6)}<br>LPIPS ${v.LPIPS.toFixed(6)}</p></div>`).join('');const sel=document.querySelector('#view');D.names.forEach(n=>sel.add(new Option(n,n)));let box=[0,0,1320,989],loaded=[],version=0;function redraw(){document.querySelector('#box').textContent=box.join(', ');loaded.forEach(([im,c])=>{c.width=box[2]-box[0];c.height=box[3]-box[1];c.getContext('2d').drawImage(im,box[0],box[1],c.width,c.height,0,0,c.width,c.height)})}function load(){version++;const v=version;loaded=[];const g=document.querySelector('#grid');g.replaceChildren();D.tiles.forEach(t=>{const p=document.createElement('div');p.className='tile';const title=document.createElement('p');title.textContent=t.label;p.append(title);g.append(p);if((t.native&&!D.fixed.includes(sel.value))||(t.label==='Photo'&&!D.train.includes(sel.value))){title.textContent+='（无已保存原图）';return}const c=document.createElement('canvas'),link=document.createElement('a');link.href=t.path+'/'+sel.value;link.textContent='原生PNG';link.target='_blank';p.append(c,link);const im=new Image();im.onload=()=>{if(v!==version)return;loaded.push([im,c]);redraw()};im.src=link.href;let start=null;c.onpointerdown=e=>{start=[e.offsetX/c.clientWidth,e.offsetY/c.clientHeight];c.setPointerCapture(e.pointerId)};c.onpointerup=e=>{if(!start)return;const end=[Math.max(0,Math.min(1,e.offsetX/c.clientWidth)),Math.max(0,Math.min(1,e.offsetY/c.clientHeight))];if(Math.abs(end[0]-start[0])>.02&&Math.abs(end[1]-start[1])>.02){const [x,y,x1,y1]=box;box=[Math.round(x+Math.min(start[0],end[0])*(x1-x)),Math.round(y+Math.min(start[1],end[1])*(y1-y)),Math.round(x+Math.max(start[0],end[0])*(x1-x)),Math.round(y+Math.max(start[1],end[1])*(y1-y))];redraw()}start=null}})}sel.onchange=()=>{box=[0,0,1320,989];load()};document.querySelector('#full').onclick=()=>{box=[0,0,1320,989];redraw()};document.querySelector('#shore').onclick=()=>{box=[0,620,1320,989];redraw()};document.querySelector('#sky').onclick=()=>{box=[0,0,1320,380];redraw()};load();</script>'''.replace('__DATA__',json.dumps(data,ensure_ascii=False))
 (out/'index.html').write_text(html);print(json.dumps({k:result[k] for k in ['rgb_metrics','paired_psnr','region_mean_mae_255','independent_psnr_checks']},indent=2))
if __name__=='__main__':main()
