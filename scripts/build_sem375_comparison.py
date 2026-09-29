#!/usr/bin/env python3
"""Extend the existing image comparison workflow with the matched RGB136 expert."""
import argparse,html,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from src.quality_protocol import write_new_json,sha256
PAL=np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]],np.uint8)
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);a=p.parse_args();batch=a.batch.resolve();out=batch/'comparison';out.mkdir(exist_ok=False)
 base=ROOT/'outputs/quality/backbone_comparison';oldpage=base/'browser-r1';audit=read(batch/'audit.json');export=read(batch/'export/result.json');assert audit['status']=='completed'
 arms=[('old_scaffold','Old Scaffold RGB120 / SEM370',base/'scaffold',oldpage/'scaffold'),('standard','Standard RGB130 / SEM371',base/'standard',oldpage/'standard'),('mip','Mip RGB131 / SEM372',base/'mip',oldpage/'mip'),('new_scaffold','New Scaffold RGB136 / SEM375',batch,out/'new_scaffold')]
 (out/'new_scaffold').mkdir();(out/'panels').mkdir()
 for n in export['image_hashes']:
  arr=np.asarray(Image.open(batch/'inference/predictions'/n));assert arr.shape==(989,1320) and arr.max()<5;Image.fromarray(PAL[arr]).save(out/'new_scaffold'/n)
 fixed=read(ROOT/'experiments/runs/SEM-370-ALL300/config.yaml')['diagnostic_views'];source=read(oldpage/'data.json');previous={r['name']:r for r in source['images']};names=fixed+[n for n in sorted(export['image_hashes']) if n not in fixed];items=[]
 def rel(p):return os.path.relpath(p,out)
 for n in names:
  record=dict(name=n,box=previous[n]['box'] if n in previous else [0,0,1320,989],structure_box=previous[n]['structure_box'] if n in previous else [0,0,1320,989],tiles=[])
  for arm,label,path,color in arms:record['tiles'].append(dict(label=label+' RGB',path=rel(path/'export/images'/n)))
  for arm,label,path,color in arms:record['tiles'].append(dict(label=label+' semantic',path=rel(color/n)))
  if n in previous:
   record['tiles'].append(dict(label='Photo',path=rel(ROOT/'outputs/quality/all_data/RGB-120-r2/scene/images'/n)))
   if (oldpage/'gt'/n).exists():record['tiles'].append(dict(label='GT',path=rel(oldpage/'gt'/n)))
  items.append(record)
 def panel(n,box=None):
  w,h=440,330;canvas=Image.new('RGB',(w*4,(h+26)*2),'white');draw=ImageDraw.Draw(canvas)
  for j,(arm,label,path,color) in enumerate(arms):
   for row,p in enumerate([path/'export/images'/n,color/n]):
    im=Image.open(p).convert('RGB')
    if box:im=im.crop(box)
    im=im.resize((w,h),Image.Resampling.LANCZOS if row==0 else Image.Resampling.NEAREST);y=row*(h+26);canvas.paste(im,(j*w,y+26));draw.text((j*w+5,y+6),label+(' RGB' if row==0 else ' semantic'),fill='black')
  return canvas
 for n in fixed:
  panel(n).save(out/'panels'/(Path(n).stem+'-full.png'));panel(n,previous[n]['box']).save(out/'panels'/(Path(n).stem+'-roi.png'));panel(n,previous[n]['structure_box']).save(out/'panels'/(Path(n).stem+'-structure.png'))
 traj=export['trajectory_names'];assert len(traj)==144
 cmd=['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1760x712','-r','12','-i','pipe:0','-an','-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'trajectory.mp4')]
 proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 try:
  for n in traj:proc.stdin.write(panel(n).tobytes())
 finally:proc.stdin.close()
 assert proc.wait()==0
 for idx in [0,48,71,72,73,96,143]:panel(traj[idx]).save(out/'panels'/('trajectory-'+str(idx)+'.png'))
 summaries=[dict(label=label,**audit['arms'][arm]['metrics']) for arm,label,_,_ in arms]
 data=dict(experiment_id=audit['experiment_id'],summaries=summaries,paired=audit['paired'],items=items,class_names=list(audit['arms']['new_scaffold']['metrics']['per_class']),provenance=audit['binding'])
 write_new_json(out/'data.json',data)
 doc='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>RGB136 匹配语义 SEM375</title><style>body{font:15px/1.6 system-ui;margin:24px;background:#f5f6f8;color:#17212a}h1{font-size:25px}table{border-collapse:collapse;background:white;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.controls{position:sticky;top:0;background:white;padding:10px;z-index:2}button,select{padding:7px;margin-right:8px}button:disabled{cursor:not-allowed;opacity:.45}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.tile{background:white}.tile p{margin:5px;font-size:12px}canvas{width:100%;display:block}video{width:100%}.note{max-width:1100px;color:#47515b}</style><h1>新 Scaffold RGB136 × 同配方语义 SEM375</h1><p class="note">同一 Mask2Former Swin-L 初始化、300 张训练标签、768 裁块、6000 次有效更新、seed 20260805；唯一主要变量为 RGB 输入。RGB136 无相机外观 UID。旧 Scaffold 为 SEM370 实际训练的固定 UID0 输入。所有指标为训练视角拟合，不能证明隐藏视角泛化；原始五类，不含历史多专家融合。</p><div id="scores"></div><p class="note">每列上方 RGB、下方语义；照片与 GT 在末行。黄色为 stay_cable，蓝色 deck，红色 tower，紫色 foundation，深灰 background。拖拽任意图选同一裁剪范围。</p><div class="controls"><button id="prev" type="button" aria-label="上一张视角">上一张</button><select id="views" aria-label="选择视角"></select><button id="next" type="button" aria-label="下一张视角">下一张</button><button id="full">完整图</button><button id="roi">困难区域</button><button id="structure">结构区域</button><span id="coords"></span></div><div class="grid" id="grid"></div><h2>同一官方坐标 144 帧轨迹</h2><p class="note">列顺序与上方一致，每列上 RGB、下语义。12fps；无GT，仅作视觉检查，不能据此给出轨迹准确率。</p><video controls preload="metadata" src="trajectory.mp4"></video><p><a href="data.json">逐视角对照数据</a> · <a href="../audit.json">独立核验记录</a></p><script>const D=__DATA__;const f=x=>(x*100).toFixed(4);document.querySelector('#scores').innerHTML='<table><tr><th>系统</th><th>mIoU %</th>'+D.class_names.map(n=>'<th>'+n+' IoU %</th>').join('')+'</tr>'+D.summaries.map(s=>'<tr><td>'+s.label+'</td><td>'+f(s.mIoU_with_background)+'</td>'+D.class_names.map(n=>'<td>'+f(s.per_class[n].IoU)+'</td>').join('')+'</tr>').join('')+'</table>';
let index=0,box=[0,0,1320,989],loaded=[],version=0;const sel=document.querySelector('#views'),prev=document.querySelector('#prev'),next=document.querySelector('#next');D.items.forEach((x,i)=>sel.add(new Option(x.name,i)));function draw(){document.querySelector('#coords').textContent=box.join(', ');for(const {im,c} of loaded){c.width=box[2]-box[0];c.height=box[3]-box[1];const ctx=c.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(im,box[0],box[1],c.width,c.height,0,0,c.width,c.height)}}function updateNavigation(){sel.value=String(index);prev.disabled=index===0;next.disabled=index===D.items.length-1}function load(){updateNavigation();const v=++version;loaded=[];const grid=document.querySelector('#grid');grid.replaceChildren();for(const tile of D.items[index].tiles){const div=document.createElement('div');div.className='tile';const p=document.createElement('p');p.textContent=tile.label;const c=document.createElement('canvas');div.append(p,c);grid.append(div);const im=new Image();im.onload=()=>{if(v!==version)return;loaded.push({im,c});draw()};im.src=tile.path;let start=null;c.onpointerdown=e=>{start=[e.offsetX/c.clientWidth,e.offsetY/c.clientHeight];c.setPointerCapture(e.pointerId)};c.onpointerup=e=>{if(!start)return;const end=[Math.max(0,Math.min(1,e.offsetX/c.clientWidth)),Math.max(0,Math.min(1,e.offsetY/c.clientHeight))];if(Math.abs(end[0]-start[0])>.02&&Math.abs(end[1]-start[1])>.02){const [x,y,r,b]=box;box=[Math.round(x+Math.min(start[0],end[0])*(r-x)),Math.round(y+Math.min(start[1],end[1])*(b-y)),Math.round(x+Math.max(start[0],end[0])*(r-x)),Math.round(y+Math.max(start[1],end[1])*(b-y))];draw()}start=null}}}function selectView(nextIndex){index=Math.max(0,Math.min(D.items.length-1,nextIndex));box=[0,0,1320,989];load()}sel.onchange=()=>selectView(+sel.value);prev.onclick=()=>selectView(index-1);next.onclick=()=>selectView(index+1);document.querySelector('#full').onclick=()=>{box=[0,0,1320,989];draw()};document.querySelector('#roi').onclick=()=>{box=[...D.items[index].box];draw()};document.querySelector('#structure').onclick=()=>{box=[...D.items[index].structure_box];draw()};load();</script></html>'''.replace('__DATA__',json.dumps(data,ensure_ascii=False))
 (out/'index.html').write_text(doc)
 for item in items:
  for tile in item['tiles']:assert (out/tile['path']).is_file(),tile
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(out/'trajectory.mp4')],text=True));v=probe['streams'][0];assert int(v['nb_frames'])==144 and v['width']==1760 and v['height']==712
 write_new_json(out/'result.json',dict(status='completed_artifacts_pending_visual_review',image_resources_checked=sum(len(i['tiles']) for i in items),frames=144,ffprobe=probe,script_sha256=sha256(Path(__file__))))
 print(out/'index.html')
if __name__=='__main__':main()
