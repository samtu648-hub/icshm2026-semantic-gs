#!/usr/bin/env python3
"""Local 2x2 crossover comparison, sharing the established image viewer."""
import json,os,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from scripts.audit_sem376 import arm_paths,read
from src.quality_protocol import write_new_json,sha256
PAL=np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]],np.uint8)
def main():
 b=ROOT/'outputs/quality/SEM-376/r1';out=b/'comparison';out.mkdir(exist_ok=False);audit=read(b/'audit.json');assert audit['status']=='completed';paths=arm_paths(b);oldpage=ROOT/'outputs/quality/backbone_comparison/browser-r1';newpage=ROOT/'outputs/quality/SEM-375/r1/comparison';exp=read(ROOT/'outputs/quality/SEM-375/r1/export/result.json');labels={'standard_sem371':'Standard RGB + SEM371 (matched)','standard_sem375':'Standard RGB + SEM375 (cross)','scaffold_sem371':'Scaffold RGB + SEM371 (cross)','scaffold_sem375':'Scaffold RGB + SEM375 (matched)'};colors={'standard_sem371':oldpage/'standard','scaffold_sem375':newpage/'new_scaffold'}
 for arm in ['standard_sem375','scaffold_sem371']:
  colors[arm]=out/arm;colors[arm].mkdir()
  for n in exp['image_hashes']:Image.fromarray(PAL[np.asarray(Image.open(paths[arm][1]/'inference/predictions'/n))]).save(colors[arm]/n)
 previous={r['name']:r for r in read(oldpage/'data.json')['images']};fixed=read(ROOT/'experiments/runs/SEM-370-ALL300/config.yaml')['diagnostic_views'];names=fixed+[n for n in sorted(exp['image_hashes']) if n not in fixed];items=[]
 def rel(p):return os.path.relpath(p,out)
 for n in names:
  tiles=[dict(label=labels[arm]+' RGB',path=rel(rgb/'export/images'/n)) for arm,(rgb,sem) in paths.items()]+[dict(label=labels[arm]+' semantic',path=rel(colors[arm]/n)) for arm in paths]
  if n in previous:
   tiles.append(dict(label='Photo',path=rel(ROOT/'outputs/quality/all_data/RGB-120-r2/scene/images'/n)))
   if (oldpage/'gt'/n).exists():tiles.append(dict(label='GT',path=rel(oldpage/'gt'/n)))
  items.append(dict(name=n,tiles=tiles,box=previous[n]['box'] if n in previous else [0,0,1320,989],structure_box=previous[n]['structure_box'] if n in previous else [0,0,1320,989]))
 data=dict(experiment_id=audit['experiment_id'],summaries=[dict(label=labels[arm],**audit['arms'][arm]['metrics']) for arm in paths],class_names=list(audit['arms']['standard_sem371']['metrics']['per_class']),items=items,pairs=audit['pairs'],interaction_pp=audit['interaction_pp'],provenance=audit['binding'])
 write_new_json(out/'data.json',data)
 template=(newpage/'index.html').read_text();style=template.split('<style>',1)[1].split('</style>',1)[0];script=template.split('<script>',1)[1].split('</script>',1)[0];script=re.sub(r'^const D=.*?;const f=',lambda _: 'const D='+json.dumps(data,ensure_ascii=False)+';const f=',script,count=1,flags=re.S)
 pair=audit['pairs'];a=pair['swap_semantic_on_standard']['delta_pp'];c=pair['swap_semantic_on_scaffold']['delta_pp']
 doc='<!doctype html><html lang="zh"><meta charset="utf-8"><title>SEM376 冻结交叉对照</title><style>'+style+'</style><h1>同一 RGB 换语义模型：SEM376 交叉对照</h1><p class="note">固定 RGB130/新 Scaffold RGB136 与 SEM371/SEM375；本轮零训练，仅新增两组交叉推理。原匹配组直接复用。全300标签是训练视角拟合；100无标签图与144轨迹不计算GT指标。</p><p class="note">前两列使用完全相同的 Standard RGB，后两列使用完全相同的新 Scaffold RGB。每组分别用SEM371、SEM375，可直接观察更换语义模型的影响。</p><div id="scores"></div><p class="note">固定Standard RGB，SEM371→SEM375的mIoU变化为 '+f'{a:+.6f}'+'pp；固定Scaffold RGB，同一语义切换变化为 '+f'{c:+.6f}'+'pp。指标来自全局混淆，单seed，不能断言算法普遍优劣。</p><div class="controls"><select id="views"></select><button id="full">完整图</button><button id="roi">困难区域</button><button id="structure">结构区域</button><span id="coords"></span></div><div class="grid" id="grid"></div><h2>同一144帧官方坐标轨迹</h2><p class="note">列顺序与上方相同，每列上RGB、下语义。无GT；黄色stay_cable、蓝色deck、红色tower、紫色foundation、深灰background。</p><video controls preload="metadata" src="trajectory.mp4"></video><p><a href="data.json">全量对照数据</a> · <a href="../audit.json">独立评分与固定ROI计数</a></p><script>'+script+'</script></html>'
 (out/'index.html').write_text(doc);(out/'panels').mkdir()
 def panel(n,box=None):
  im=Image.new('RGB',(1760,712),'white');draw=ImageDraw.Draw(im)
  for j,(arm,(rgb,sem)) in enumerate(paths.items()):
   for row,p in enumerate([rgb/'export/images'/n,colors[arm]/n]):
    tile=Image.open(p).convert('RGB');tile=tile.crop(box) if box else tile;tile=tile.resize((440,330),Image.Resampling.LANCZOS if row==0 else Image.Resampling.NEAREST);im.paste(tile,(j*440,row*356+26));draw.text((j*440+5,row*356+5),labels[arm]+(' RGB' if row==0 else ' semantic'),fill='black')
  return im
 for n in fixed:
  for suffix,box in [('full',None),('roi',previous[n]['box']),('structure',previous[n]['structure_box'])]:panel(n,box).save(out/'panels'/(Path(n).stem+'-'+suffix+'.png'))
 cmd=['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1760x712','-r','12','-i','pipe:0','-an','-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'trajectory.mp4')];proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 try:
  for i,n in enumerate(exp['trajectory_names']):
   im=panel(n);proc.stdin.write(im.tobytes())
   if i in [0,48,71,72,73,96,143]:im.save(out/'panels'/f'trajectory-{i}.png')
 finally:proc.stdin.close()
 assert proc.wait()==0
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(out/'trajectory.mp4')],text=True));assert int(probe['streams'][0]['nb_frames'])==144
 for item in items:
  for t in item['tiles']:assert (out/t['path']).is_file()
 js=out/'page-check.js';js.write_text(script);subprocess.run(['node','--check',str(js)],check=True)
 write_new_json(out/'result.json',dict(status='completed_artifacts_pending_visual_review',script_sha256=sha256(Path(__file__)),template_sha256=sha256(newpage/'index.html'),resource_count=sum(len(i['tiles']) for i in items),frames=144,ffprobe=probe,javascript_syntax_passed=True,browser_interaction_tested=False))
 print(out/'index.html')
if __name__=='__main__':main()
