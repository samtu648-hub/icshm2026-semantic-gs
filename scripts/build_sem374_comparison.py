#!/usr/bin/env python3
"""Local-only synchronized comparison of one preregistered Standard experiment."""
import argparse,html,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from src.quality_semantic_audit import independent_confusion,metrics
from scripts.run_sem373_joint import digest,write
BASE=ROOT/'outputs/quality/SEM-374';ARMS=['baseline','control','joint'];LABELS=['RGB130原始 + 冻结三维语义','控制：RGB→XYZ/SH；语义→logits','联合：标注多视角深度→XYZ'];PAL=np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]],np.uint8)

def read(p):return json.loads(p.read_text())
def load(p):return np.asarray(Image.open(p))

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--skip-videos',action='store_true');a=p.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False)
 c=read(ROOT/'experiments/runs/SEM-374-STANDARD-STRUCTURE-DEPTH/config.yaml');results={x:read(BASE/('eval-'+x+'-r2')/'result.json') for x in ARMS};assert all(v['status']=='completed' for v in results.values())
 assert results['baseline']['trajectory_hashes']==results['control']['trajectory_hashes']==results['joint']['trajectory_hashes']
 split=read(ROOT/c['split_path']);labeled=set(split['semantic_train']);fixed=c['diagnostic_views'];names=fixed+[n for n in sorted(split['rgb_train']) if n not in fixed];traj=results['baseline']['trajectory_names']
 rois=read(ROOT/'experiments/runs/QUAL-005-PHOTO-INPUT-DIAG/config.yaml')['rois'];struct=read(ROOT/'experiments/runs/QUAL-004-CURRENT-BOTTLENECK/config.yaml')['crops']
 for arm in ARMS:
  (out/arm).mkdir()
  for name in names+traj:Image.fromarray(PAL[load(BASE/('eval-'+arm+'-r2')/'semantic'/name)]).save(out/arm/name)
 (out/'gt').mkdir();(out/'inverse-depth').mkdir();images=[];audit=[]
 for name in names:
  tiles=[dict(label='真实照片 · PINHOLE',path=os.path.relpath(ROOT/c['scene']/'images'/name,out))]
  for arm,label in zip(ARMS,LABELS):tiles.append(dict(label=label+' · RGB',path=os.path.relpath(BASE/('eval-'+arm+'-r2')/'rgb'/name,out)))
  if name in labeled:
   gt=load(ROOT/c['labels']/name);Image.fromarray(PAL[gt]).save(out/'gt'/name);tiles.append(dict(label='人工GT · 五类',path='gt/'+name))
  for arm,label in zip(ARMS,LABELS):tiles.append(dict(label=label+' · 三维语义投影',path=arm+'/'+name))
  images.append(dict(name=name,fixed=name in fixed,labeled=name in labeled,box=rois.get(Path(name).stem,[0,0,1320,989]),structure_box=struct.get(Path(name).stem,[0,0,1320,989]),tiles=tiles))
  if name in fixed:
   rgbgt=load(ROOT/c['scene']/'images'/name).astype(np.float64)
   for arm in ARMS:
    rr=results[arm];pred=load(BASE/('eval-'+arm+'-r2')/'semantic'/name);row=next(v for v in rr['per_view'] if v['name']==name)
    assert independent_confusion(gt,pred).tolist()==row['semantic']['confusion']
    mae=float(np.abs(load(BASE/('eval-'+arm+'-r2')/'rgb'/name).astype(np.float64)-rgbgt).mean()/255)
    assert abs(mae-row['MAE'])<1e-7;audit.append(dict(name=name,arm=arm,independent_cm_and_mae=True))
 # Same inverse-depth display range for all arms and views; alpha-composited inverse-Z is not a surface GT.
 sampled=[]
 for name in fixed+traj:
  d=np.load(BASE/'eval-baseline-r2/depth'/(Path(name).stem+'.npy'));sampled.extend(d[::24,::24].ravel().tolist())
 lo,hi=np.quantile(np.array(sampled),[.01,.99]);depth_deltas=[]
 for name in fixed+traj:
  ds=[]
  for arm in ARMS:
   d=np.load(BASE/('eval-'+arm+'-r2')/'depth'/(Path(name).stem+'.npy'));ds.append(d)
   pic=np.clip((d-lo)/(hi-lo),0,1);Image.fromarray(np.rint(pic*255).astype(np.uint8)).save(out/'inverse-depth'/(arm+'-'+name))
  depth_deltas.append(dict(name=name,joint_control_mean_abs=float(np.abs(ds[2]-ds[1]).mean()),joint_baseline_mean_abs=float(np.abs(ds[2]-ds[0]).mean()),interpretation='change diagnostic only; no GT'))
 rows={arm:{v['name']:v for v in results[arm]['per_view']} for arm in ARMS};pairs={}
 for ref,cand in [('baseline','control'),('baseline','joint'),('control','joint')]:
  delta=[dict(name=n,**{k:rows[cand][n][k]-rows[ref][n][k] for k in ['PSNR','SSIM','LPIPS','MAE']}) for n in names]
  pairs[cand+'-minus-'+ref]=dict(mean={k:float(np.mean([x[k] for x in delta])) for k in ['PSNR','SSIM','LPIPS','MAE']},psnr_improved=sum(x['PSNR']>0 for x in delta),psnr_degraded=sum(x['PSNR']<0 for x in delta),lpips_improved=sum(x['LPIPS']<0 for x in delta),lpips_degraded=sum(x['LPIPS']>0 for x in delta),per_view=delta)
 summaries=[dict(arm=arm,label=label,rgb=results[arm]['rgb_metrics'],semantic=results[arm]['semantic_metrics'],class_rgb_mae=results[arm]['class_rgb_mae'],geometry=results[arm]['geometry'],fixed_regions=results[arm]['fixed_regions']) for arm,label in zip(ARMS,LABELS)]
 training=[read(BASE/(n+'-r2')/'result.json') for n in ['control','joint']]
 assert training[0]['initial_state']==training[1]['initial_state'] and training[0]['sampling_sha256']==training[1]['sampling_sha256']
 data=dict(experiment=c['experiment_id'],git_commit=c['git_commit'],seed=c['seed'],summaries=summaries,images=images,trajectory_names=traj,trajectory_hashes=results['baseline']['trajectory_hashes'],pairwise=pairs,classes=c['class_names'],depth_range=[lo,hi],depth_deltas=depth_deltas,training=[{k:x[k] for k in ['stage','successful_steps','label_visits','training_seconds','peak_allocated_mib','peak_reserved_mib']} for x in training],independent_fixed9_audit=audit)
 write(out/'data.json',data)
 if not a.skip_videos:
  for arm in ARMS:
   subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-threads','4','-framerate','12','-i',str(BASE/('eval-'+arm+'-r2')/'rgb/frame%04d.png'),'-framerate','12','-i',str(out/arm/'frame%04d.png'),'-filter_complex_threads','2','-filter_complex','[0:v][1:v]vstack=inputs=2[v]','-map','[v]','-c:v','libx264','-threads','4','-crf','18','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(out/(arm+'-trajectory.mp4'))],check=True)
  cmd=['ffmpeg','-nostdin','-v','error','-n']
  for arm in ARMS:cmd+=['-i',str(out/(arm+'-trajectory.mp4'))]
  cmd+=['-filter_complex_threads','2','-filter_complex','[0:v][1:v][2:v]hstack=inputs=3[v]','-map','[v]','-c:v','libx264','-threads','4','-crf','18','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'matched-trajectory.mp4')];subprocess.run(cmd,check=True)
 page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>SEM374 语义是否帮助重建</title><style>
body{font:15px/1.55 system-ui;margin:28px;background:#f6f7f9;color:#17202a}h1{font-size:26px}h2{font-size:21px;margin-top:30px}p{max-width:1150px}table{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.tile{background:white}.tile p{margin:5px;font-size:12px}.tile canvas{width:100%;display:block}.tools{position:sticky;top:0;background:white;padding:12px;z-index:2}button,select{padding:7px;margin:4px}video{width:100%}.note{color:#48515d}.small{font-size:12px}a{color:#075daf}.depth{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.depth img{width:100%}</style>
<h1>人工标注筛选的多视角深度能否帮助 Standard 重建？</h1><p class="note">SEM374 · 原始RGB130全400/30k。复用SEM373冻结6,000步语义分支；相同起点两臂各2,000步。语义NLL只更新logits。唯一变量：0.05标注筛选多视角稀疏深度损失能否影响XYZ。RGB更新XYZ/SH；尺度、旋转、透明度和高斯数量固定，增密/剪枝关闭。全400 RGB、全300语义均为训练视角拟合。三维语义投影分数不是三维几何真值；无GT轨迹只供视觉检查。Mask2Former和融合未改变。</p>
<div id="summary"></div><section id="conclusion">__REVIEW__</section>
<h2>同相机、同裁剪的RGB和五类对照</h2><p class="note">前九张为固定诊断。可浏览全400张；100张无标注图不展示GT。拖拽任一图选择共同裁剪。</p><div class="tools"><select id="view"></select><button id="full">完整图</button><button id="roi">岸线 / 天空ROI</button><button id="structure">结构ROI</button><span id="crop"></span></div><div id="grid" class="grid"></div>
<h2>原生逆深度诊断</h2><p class="note">左原始、中控制、右联合，全部共用显示范围。图中是原生 ∑Tα/z，受遮挡与透明度混合影响，不能解释成真值表面、几何精度或用倒数冒充深度。只为检查结构支持与变化。</p><select id="depthView"></select><div id="depth" class="depth"></div>
<h2>统一144帧官方COLMAP轨迹</h2><p class="note">左原始、中控制、右联合；每列上RGB、下三维语义。12fps。无GT，不报告隐藏视角提升或未补偿运动的闪烁分数。</p><video controls preload="metadata" src="matched-trajectory.mp4"></video><p><a href="baseline-trajectory.mp4">原始</a> · <a href="control-trajectory.mp4">控制</a> · <a href="joint-trajectory.mp4">联合</a> · <a href="data.json">完整指标、局部计数与资源</a> · <a href="../visual_review.md">视觉复核记录</a></p><h2>实际资源与参数变化</h2><div id="resources"></div><p class="note">世界坐标为COLMAP重建单位，不声称米制。参数位移表示发生了优化，不证明改善。单seed、固定短预算，无选参或自动替换。</p>
<script>const D=__DATA__;const f=(x,n=4)=>x==null?'—':Number(x).toFixed(n);const table=(h,r)=>'<table><tr>'+h.map(x=>'<th>'+x+'</th>').join('')+'</tr>'+r.map(a=>'<tr>'+a.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</table>';
let summary='<h2>全400 RGB训练拟合</h2>'+table(['系统','PSNR dB ↑','SSIM ↑','LPIPS ↓','MAE ↓'],D.summaries.map(s=>[s.label,f(s.rgb.PSNR),f(s.rgb.SSIM,6),f(s.rgb.LPIPS,6),f(s.rgb.MAE,6)]));summary+='<h2>全300三维语义投影拟合</h2>'+table(['系统','五类mIoU %',...D.classes.map(x=>x+' IoU %')],D.summaries.map(s=>[s.label,f(s.semantic.mIoU_with_background*100),...D.classes.map(n=>f(s.semantic.per_class[n].IoU*100))]));summary+='<details><summary>各类precision/recall及该类像素RGB误差</summary>'+table(['系统','类','Precision %','Recall %','RGB MAE'],D.summaries.flatMap(s=>D.classes.map(n=>[s.label,n,f(s.semantic.per_class[n].precision*100),f(s.semantic.per_class[n].recall*100),f(s.class_rgb_mae[n],6)])))+'</details>';document.querySelector('#summary').innerHTML=summary;
document.querySelector('#resources').innerHTML=table(['阶段','成功步数','人工语义访问','训练分钟','峰值allocated / reserved MiB'],D.training.map(t=>[t.stage,t.successful_steps,t.label_visits,f(t.training_seconds/60,2),f(t.peak_allocated_mib,0)+' / '+f(t.peak_reserved_mib,0)]))+table(['系统','高斯数','XYZ位移p50/p90/p99/max'],D.summaries.map(s=>[s.label,s.geometry.point_count,s.geometry.xyz_displacement_quantiles.slice(1).map(x=>f(x,6)).join(' / ')]));
let current=0,box=[0,0,1320,989],loaded=[],version=0;const sel=document.querySelector('#view');D.images.forEach((x,i)=>sel.add(new Option((x.fixed?'固定 · ':'')+x.name+(x.labeled?'':' · 无标注'),i)));
function redraw(){document.querySelector('#crop').textContent='原生坐标 ['+box.join(', ')+']';loaded.forEach(({im,c})=>{c.width=box[2]-box[0];c.height=box[3]-box[1];const ctx=c.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(im,box[0],box[1],c.width,c.height,0,0,c.width,c.height)})}
function loadView(){const v=++version;loaded=[];const grid=document.querySelector('#grid');grid.replaceChildren();for(const [i,t] of D.images[current].tiles.entries()){const tile=document.createElement('div');tile.className='tile';if(!D.images[current].labeled&&i===4)tile.style.gridColumn='2';const p=document.createElement('p');p.textContent=t.label;const c=document.createElement('canvas');const a=document.createElement('a');a.href=t.path;a.target='_blank';a.textContent='打开原图';tile.append(p,c,a);grid.append(tile);const im=new Image();im.onload=()=>{if(v!==version)return;loaded.push({im,c});redraw()};im.src=t.path;let start=null;c.onpointerdown=e=>{start=[e.offsetX/c.clientWidth,e.offsetY/c.clientHeight];c.setPointerCapture(e.pointerId)};c.onpointerup=e=>{if(!start)return;const end=[Math.max(0,Math.min(1,e.offsetX/c.clientWidth)),Math.max(0,Math.min(1,e.offsetY/c.clientHeight))];if(Math.abs(start[0]-end[0])>.02&&Math.abs(start[1]-end[1])>.02){const [x,y,x1,y1]=box;box=[Math.round(x+Math.min(start[0],end[0])*(x1-x)),Math.round(y+Math.min(start[1],end[1])*(y1-y)),Math.round(x+Math.max(start[0],end[0])*(x1-x)),Math.round(y+Math.max(start[1],end[1])*(y1-y))];redraw()}start=null}}}
sel.onchange=()=>{current=Number(sel.value);box=[0,0,1320,989];loadView()};document.querySelector('#full').onclick=()=>{box=[0,0,1320,989];redraw()};document.querySelector('#roi').onclick=()=>{box=[...D.images[current].box];redraw()};document.querySelector('#structure').onclick=()=>{box=[...D.images[current].structure_box];redraw()};loadView();
const depthNames=[...D.images.filter(x=>x.fixed).map(x=>x.name),...D.trajectory_names];const ds=document.querySelector('#depthView');depthNames.forEach(n=>ds.add(new Option(n,n)));function drawDepth(){const root=document.querySelector('#depth');root.replaceChildren();for(const arm of ['baseline','control','joint']){const div=document.createElement('div');const p=document.createElement('p');p.textContent=arm;const im=new Image();im.src='inverse-depth/'+arm+'-'+ds.value;div.append(p,im);root.append(div)}}ds.onchange=drawDepth;drawDepth();</script></html>'''.replace('__DATA__',json.dumps(data,ensure_ascii=False))
 review=BASE/'visual_review.md';block='<h2>复核结论</h2><p>固定结果已生成，人工视觉复核进行中；暂不作重建改善结论。</p>'
 if review.exists():block='<h2>复核结论</h2><div style="white-space:pre-wrap">'+html.escape(review.read_text())+'</div>'
 (out/'index.html').write_text(page.replace('__REVIEW__',block));write(out/'result.json',dict(status='browser_complete',audited_fixed_pairs=len(audit),data_sha256=digest(out/'data.json'),trajectory_frames=len(traj)))
 print(out/'index.html')

if __name__=='__main__':main()
