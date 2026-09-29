#!/usr/bin/env python3
"""Build local full-resolution comparison browser and native trajectory videos."""
import argparse,html,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from src.quality_protocol import sha256,write_new_json
from src.quality_semantic_audit import independent_confusion,metrics
BASE=ROOT/'outputs/quality/backbone_comparison';SEM=ROOT/'outputs/quality/all_data/SEM-370-r1';PARENT=ROOT/'outputs/quality/all_data/RGB-120-r2'
PAL=np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]],np.uint8)
ARMS=['scaffold','standard','mip'];LABELS=['Scaffold RGB120 + SEM370','Standard RGB130 + SEM371','Mip RGB131 + SEM372']
def read(p):return json.loads(p.read_text())
def arr(p):return np.asarray(Image.open(p))
def size(path):return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
def vertex_count(ply):
 with ply.open('rb') as stream:
  for _ in range(200):
   line=stream.readline().decode('ascii').strip()
   if line.startswith('element vertex '):return int(line.split()[-1])
 raise ValueError('PLY vertex header missing')
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--skip-videos',action='store_true');a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False)
 names=read(ROOT/'experiments/runs/SEM-370-ALL300/config.yaml')['diagnostic_views'];rois=read(ROOT/'experiments/runs/QUAL-005-PHOTO-INPUT-DIAG/config.yaml')['rois'];structure_rois=read(ROOT/'experiments/runs/QUAL-004-CURRENT-BOTTLENECK/config.yaml')['crops']
 supplement=read(BASE/'scaffold-native-appearance/result.json');assert supplement['status']=='completed'
 summaries=[];images=[];roi_metrics=[];trajectories=[]
 for arm,label in zip(ARMS,LABELS):
  rgb=read(BASE/arm/'rgb-evaluation/result.json');sem=read(BASE/arm/'inference/result.json');exp=read(BASE/arm/'export/result.json')
  assert all(r['status']=='completed' for r in [rgb,sem,exp])
  assert len(rgb['per_view'])==400 and sem['labeled_count']==300
  run=PARENT if arm=='scaffold' else BASE/(arm+'-full-r1');train=read(run/'result.json')
  m2f=read((SEM if arm=='scaffold' else BASE/arm)/'mask2former/result.json')
  interruption=read(run/'interruption-r1.json') if (run/'interruption-r1.json').exists() else None
  trajectories.append(exp['trajectory_metadata_sha256'])
  plydir=run/'model/point_cloud/iteration_30000';modelpath=(SEM if arm=='scaffold' else BASE/arm)/'mask2former/model-6000.pt'
  summaries.append(dict(interruption=interruption,rgb_wall_without_pause_seconds=train['seconds']-(interruption['paused_seconds'] if interruption else 0),primitive_kind='anchor' if arm=='scaffold' else 'Gaussian',primitive_count=vertex_count(plydir/'point_cloud.ply'),arm=arm,label=label,rgb=rgb['metrics'],semantic=sem['metrics'],rgb_wall_seconds=train['seconds'],rgb_worker_seconds=train['worker']['seconds'],m2f_training_seconds=m2f['training_seconds'],m2f_wall_seconds=m2f['seconds'],rgb_peak_allocated_mib=train['worker']['peak_allocated_mib'],rgb_peak_reserved_mib=train['worker']['peak_reserved_mib'],m2f_peak_allocated_mib=m2f['peak_allocated_mib'],m2f_peak_reserved_mib=m2f['peak_reserved_mib'],render_median_ms=exp['render_median_seconds']*1000,inference_median_ms=sem['inference_median_seconds']*1000,rgb_deploy_mib=size(plydir)/2**20,semantic_checkpoint_mib=modelpath.stat().st_size/2**20,artifact_gib=size(BASE/arm)/2**30))
  color=out/arm;color.mkdir()
  for n in exp['image_hashes']:
   pred=arr(BASE/arm/'inference/predictions'/n);assert pred.shape==(989,1320) and pred.max()<5
   Image.fromarray(PAL[pred]).save(color/n)
 assert trajectories[0]==trajectories[1]==trajectories[2]
 (out/'gt').mkdir();region_boxes={**rois,'057':[0,0,1320,989],'064':[0,0,1320,989],'081':[0,0,1320,989],'140':[0,0,1320,989],'054':[0,0,1320,989]}
 split=read(PARENT/'split.json');labeled=set(split['semantic_train'])
 browse_names=names+[n for n in split['rgb_train'] if n not in names]
 for n in browse_names:
  gt=arr(SEM/'labels'/n) if n in labeled else None
  if gt is not None:Image.fromarray(PAL[gt]).save(out/'gt'/n)
  box=region_boxes.get(Path(n).stem,[0,0,1320,989]);x0,y0,x1,y1=box;reference=arr(PARENT/'scene/images'/n)
  tiles=[dict(label='真实照片（同PINHOLE网格）',path=os.path.relpath(PARENT/'scene/images'/n,out))]
  if gt is not None:tiles.append(dict(label='GT五类',path='gt/'+n))
  for arm,label in zip(ARMS,LABELS):
   tiles.extend([dict(label=label+' RGB',path=os.path.relpath(BASE/arm/'export/images'/n,out)),dict(label=label+' raw五类',path=arm+'/'+n)])
   if n not in names:continue
   cm=independent_confusion(gt[y0:y1,x0:x1],arr(BASE/arm/'inference/predictions'/n)[y0:y1,x0:x1]);pred=arr(BASE/arm/'export/images'/n)
   mae=float(np.abs(pred[y0:y1,x0:x1].astype(float)-reference[y0:y1,x0:x1]).mean()/255)
   roi_metrics.append(dict(name=n,arm=arm,box=box,rgb_mae=mae,**metrics(cm)))
  tiles=[tiles[i] for i in ([0,2,4,6,1,3,5,7] if n in labeled else [0,1,3,5,2,4,6])]
  images.append(dict(name=n,fixed=n in names,labeled=n in labeled,box=box,structure_box=structure_rois.get(Path(n).stem,[0,0,1320,989]),tiles=tiles))
 rgb_rows={arm:{x['name']:x for x in read(BASE/arm/'rgb-evaluation/result.json')['per_view']} for arm in ARMS}
 fixed_worst=sorted(rgb_rows['scaffold'],key=lambda n:rgb_rows['scaffold'][n]['PSNR'])[:40]
 pairwise={}
 for arm in ARMS[1:]:
  deltas=[dict(name=n,**{k:rgb_rows[arm][n][k]-rgb_rows['scaffold'][n][k] for k in ['PSNR','SSIM','LPIPS']}) for n in sorted(rgb_rows['scaffold'])]
  pairwise[arm]=dict(reference='scaffold',rgb_per_view_deltas=deltas,psnr_improved=sum(x['PSNR']>0 for x in deltas),psnr_degraded=sum(x['PSNR']<0 for x in deltas),lpips_improved=sum(x['LPIPS']<0 for x in deltas),lpips_degraded=sum(x['LPIPS']>0 for x in deltas),worst10_psnr_deltas=sorted(deltas,key=lambda x:x['PSNR'])[:10])
 fixed_worst_metrics={arm:{k:float(np.mean([rgb_rows[arm][n][k] for n in fixed_worst])) for k in ['PSNR','SSIM','LPIPS']} for arm in ARMS}
 data=dict(scaffold_native_appearance=supplement["metrics"],pairwise=pairwise,baseline_worst40_names=fixed_worst,baseline_worst40_metrics=fixed_worst_metrics,summaries=summaries,images=images,roi_metrics=roi_metrics,trajectory_hashes=trajectories[0],trajectory_frames=len(read(BASE/'scaffold/export/result.json')['trajectory_names']),class_names=['background','deck','stay_cable','tower','foundation'])
 write_new_json(out/'data.json',data)
 if not a.skip_videos:
  for arm in ARMS:
   command=['ffmpeg','-nostdin','-v','error','-n','-threads','4','-framerate','12','-i',str(BASE/arm/'export/images/frame%04d.png'),'-framerate','12','-i',str(out/arm/'frame%04d.png'),'-filter_complex','[0:v][1:v]vstack=inputs=2[v]','-map','[v]','-c:v','libx264','-threads','4','-crf','18','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(out/(arm+'-trajectory.mp4'))]
   subprocess.run(command,check=True)
  command=['ffmpeg','-nostdin','-v','error','-n']
  for arm in ARMS:command+=['-i',str(out/(arm+'-trajectory.mp4'))]
  command+=['-filter_complex','[0:v][1:v][2:v]hstack=inputs=3[v]','-map','[v]','-c:v','libx264','-threads','4','-crf','18','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'three-backbones-trajectory.mp4')];subprocess.run(command,check=True)
 html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>全数据三骨干比较</title><style>
body{font:15px/1.55 system-ui,sans-serif;margin:28px;background:#f6f7f9;color:#17202a}h1{font-size:26px}h2{font-size:21px;margin-top:32px}table{border-collapse:collapse;background:white;width:100%;font-variant-numeric:tabular-nums}td,th{padding:9px 12px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.note{max-width:1100px;color:#48515d}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.tile{background:white;overflow:hidden}.tile p{margin:5px;font-size:12px}.tile canvas{width:100%;display:block}.tools{position:sticky;top:0;padding:12px;background:#fff;border-bottom:1px solid #ccc;z-index:2}button,select{padding:7px;margin-right:8px}video{width:100%;background:#111}.legend span{display:inline-block;margin:8px;padding:3px 9px;border-left:10px solid}a{color:#075daf}.small{font-size:12px}#resources{overflow:auto}#crop{font-variant-numeric:tabular-nums}</style>
<h1>三种重建骨干 × 匹配 Mask2Former</h1><p class="note">全400张公开RGB / 全300张公开标签；所有量化结果均为训练视角拟合，不是隐藏视角成绩。相同原生1320×989、相机/初始化、RGB seed0 / 30k；M2F seed20260805 / 768采样 / 6k。只比较原始五类输出。</p>
<div id="metrics"></div>__REVIEW__<h2>固定九视角与全400浏览：照片、GT、RGB与原始语义</h2><p class="note">选择器前九项是固定诊断视角，其后可浏览其余公开图；100张无标注图省略GT。相同裁剪框应用于全部系统。照片及GT已用原协议转换到同PINHOLE网格；显示缩放仅用于浏览，训练/评价保持原生分辨率。可在任一图内拖拽绘制共同裁剪框。</p><div class="legend" id="legend"></div><div class="tools"><select id="view"></select><button id="full">完整图</button><button id="roi">固定困难区域</button><button id="structure">固定结构区域</button><span id="crop"></span></div><div class="grid" id="grid"></div>
<h2>同一144帧新视角轨迹</h2><p class="note">左Scaffold，中Standard，右Mip；每列上RGB、下原始五类。原生输出，12fps供慢速检查。无GT，不计算轨迹质量指标；相邻帧变化混有相机运动，不能直接解释为闪烁。</p><video controls preload="metadata" src="three-backbones-trajectory.mp4"></video><p><a href="scaffold-trajectory.mp4">Scaffold单独视频</a> · <a href="standard-trajectory.mp4">Standard单独视频</a> · <a href="mip-trajectory.mp4">Mip单独视频</a> · <a href="data.json">完整结构化结果与局部计数</a></p>
<h2>资源口径</h2><p class="note">相同30k不代表相同计算量。重建总时间包含加载与保存；M2F同时列训练循环与总时间。推理为同GPU顺序运行、单帧、剔除第一帧后的中位数（不含PNG编码），仍受GPU功耗与共享显示影响。显存为PyTorch峰值allocated；设备总显存采样包含桌面会话。Mip曾因另一任务进入GPU而暂停224秒后原进程恢复；暂停前存在短时共享，其训练耗时不能当作严格独占GPU基准。模型大小为30000端点目录及6000语义检查点，非全部历史产物。</p><div id="resources"></div><p class="small">本地比较页面；未上传、未替换旧系统。更多证据见各实验report.md和outputs/quality/backbone_comparison。</p><script>const D=__DATA__;
const f=(v,d=3)=>v===null?'—':Number(v).toFixed(d);const table=(heads,rows)=>'<table><thead><tr>'+heads.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
let h='<h2>全400 RGB拟合</h2>'+table(['系统','PSNR dB ↑','SSIM ↑','LPIPS ↓'],D.summaries.map(x=>[x.label,f(x.rgb.PSNR),f(x.rgb.SSIM,6),f(x.rgb.LPIPS,6)]));
h+='<p class=note>主表Scaffold使用SEM370真实训练输入的固定appearance uid0。RGB120训练还使用32维逐相机外观嵌入；按实际训练UID的补充RGB-only结果为 PSNR '+f(D.scaffold_native_appearance.PSNR)+' dB、SSIM '+f(D.scaffold_native_appearance.SSIM,6)+'、LPIPS '+f(D.scaffold_native_appearance.LPIPS,6)+'。补充口径不改变SEM370输入，不能与其语义结果混成另一套已训练系统。</p>';
h+='<h2>全300标签原始五类拟合</h2>'+table(['系统','五类mIoU %',...D.class_names.map(x=>x+' IoU %')],D.summaries.map(x=>[x.label,f(x.semantic.mIoU_with_background*100),...D.class_names.map(n=>f(x.semantic.per_class[n].IoU*100))]));
h+='<details><summary>各类precision / recall</summary>'+table(['系统','类别','precision %','recall %'],D.summaries.flatMap(x=>D.class_names.map(n=>[x.label,n,f(x.semantic.per_class[n].precision*100),f(x.semantic.per_class[n].recall*100)])))+'</details>';document.querySelector('#metrics').innerHTML=h;
document.querySelector('#resources').innerHTML=table(['系统','RGB总/扣暂停min','M2F训练/总min','RGB/M2F allocated (reserved) MiB','RGB渲染ms','M2F推理ms','RGB/语义MiB'],D.summaries.map(x=>[x.label,f(x.rgb_wall_seconds/60,1)+' / '+f(x.rgb_wall_without_pause_seconds/60,1),f(x.m2f_training_seconds/60,1)+' / '+f(x.m2f_wall_seconds/60,1),f(x.rgb_peak_allocated_mib,0)+' ('+f(x.rgb_peak_reserved_mib,0)+') / '+f(x.m2f_peak_allocated_mib,0)+' ('+f(x.m2f_peak_reserved_mib,0)+')',f(x.render_median_ms,2),f(x.inference_median_ms,2),f(x.rgb_deploy_mib,1)+' / '+f(x.semantic_checkpoint_mib,1)]));
const colors=['#1e1e1e','#50a0f5','#f5d241','#dc5050','#aa64be'];document.querySelector('#legend').innerHTML=D.class_names.map((n,i)=>'<span style="border-color:'+colors[i]+'">'+n+'</span>').join('');
let current=0,box=[0,0,1320,989],loaded=[],version=0;const sel=document.querySelector('#view');D.images.forEach((x,i)=>sel.add(new Option((x.fixed?'固定 · ':'')+x.name+(x.labeled?'':' · 无标注'),i)));function redraw(){document.querySelector('#crop').textContent='原生坐标 ['+box.join(', ')+']';loaded.forEach(({im,c})=>{c.width=box[2]-box[0];c.height=box[3]-box[1];const ctx=c.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(im,...box.slice(0,2),box[2]-box[0],box[3]-box[1],0,0,c.width,c.height);});}
function load(){version++;const v=version;loaded=[];const grid=document.querySelector('#grid');grid.replaceChildren();for(const [i,t] of D.images[current].tiles.entries()){const div=document.createElement('div');div.className='tile';if(!D.images[current].labeled&&i===4)div.style.gridColumn='2';const p=document.createElement('p');p.textContent=t.label;const c=document.createElement('canvas');const link=document.createElement('a');link.href=t.path;link.target='_blank';link.textContent='打开原图';link.className='small';div.append(p,c,link);grid.append(div);const im=new Image();im.onload=()=>{if(v!==version)return;loaded.push({im,c});redraw()};im.src=t.path;let start=null;c.onpointerdown=e=>{start=[e.offsetX/c.clientWidth,e.offsetY/c.clientHeight];c.setPointerCapture(e.pointerId)};c.onpointerup=e=>{if(!start)return;const end=[Math.max(0,Math.min(1,e.offsetX/c.clientWidth)),Math.max(0,Math.min(1,e.offsetY/c.clientHeight))];if(Math.abs(start[0]-end[0])>.02&&Math.abs(start[1]-end[1])>.02){const [x,y,x1,y1]=box;box=[Math.round(x+Math.min(start[0],end[0])*(x1-x)),Math.round(y+Math.min(start[1],end[1])*(y1-y)),Math.round(x+Math.max(start[0],end[0])*(x1-x)),Math.round(y+Math.max(start[1],end[1])*(y1-y))];redraw()}start=null;};}}
sel.onchange=()=>{current=Number(sel.value);box=[0,0,1320,989];load()};document.querySelector('#full').onclick=()=>{box=[0,0,1320,989];redraw()};document.querySelector('#roi').onclick=()=>{box=[...D.images[current].box];redraw()};document.querySelector('#structure').onclick=()=>{box=[...D.images[current].structure_box];redraw()};load();</script></html>'''.replace('__DATA__',json.dumps(data,ensure_ascii=False))
 review_path=BASE/'visual_review.md'
 review_block=''
 if review_path.exists():
  import html as html_api
  parts=review_path.read_text().split('\n\n')
  review_block='<section id=judgment><h2>复核与替换判断</h2>'+''.join('<p class=note>'+html_api.escape(x)+'</p>' for x in parts[1:3])+'<details><summary>展开困难区域、局部退化与轨迹复核</summary><div class=note style="white-space:pre-wrap">'+html_api.escape('\n\n'.join(parts[3:]))+'</div></details></section>'
 html=html.replace('__REVIEW__',review_block)
 (out/'index.html').write_text(html)
 write_new_json(out/'result.json',dict(status='completed_browser_and_videos_pending_visual_review',data_sha256=sha256(out/'data.json'),video_files=[p.name for p in out.glob('*.mp4')],fixed_views=names))
 print(out/'index.html')
if __name__=='__main__':main()
