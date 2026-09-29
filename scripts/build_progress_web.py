#!/usr/bin/env python3
"""Build a relocatable, fully offline REPORT002 from exact saved research evidence."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256, verify_hashes
from src.quality_cameras import read_camera_requests
from src.camera_adapter import SimpleRadialCamera, redistort_rgb
from src.quality_external_camera import official_labels

PALETTE=np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]],np.uint8)
CLASSES=['background','deck','stay_cable','tower','foundation']


def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def clean(s):return re.sub(r'/(?:home|Users)/[^/\s"<>]+', '$USER_HOME',s)
def png(p):
    with Image.open(p) as im:return np.array(im)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifact-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/project_progress_20260918')
    ap.add_argument('--work',type=Path,default=ROOT/'outputs/progress_web_work/r1')
    ap.add_argument('--package',action='store_true')
    a=ap.parse_args();out=a.output.resolve();artifact=a.artifact_root.resolve();work=a.work.resolve()
    for n in ['assets/videos','assets/images','data','sources']:(out/n).mkdir(parents=True,exist_ok=True)
    records=read(ROOT/'experiments/registry.yaml')['experiments'];byid={e['experiment_id']:e for e in records}
    def record(prefix):
        if prefix in byid:return byid[prefix]
        found=[e for e in records if e['experiment_id'].startswith(prefix+'-')]
        if len(found)!=1:raise ValueError('Ambiguous experiment '+prefix)
        return found[0]
    def metric(prefix):
        r=record(prefix);return r.get('metrics',r.get('metrics_summary',{}))
    assets=[];source_data=[]
    def label(p):
        p=Path(os.path.abspath(p))
        if p.is_relative_to(ROOT):return str(p.relative_to(ROOT))
        if p.is_relative_to(artifact):return '$ARTIFACT_ROOT/'+str(p.relative_to(artifact))
        return p.name
    def source(p):
        item=dict(path=label(p),sha256=sha256(p));source_data.append(item);return item
    def copy(p,dest):
        p=Path(p);target=out/dest
        shutil.copyfile(p,target)
        assets.append(dict(path=dest,sha256=sha256(target),source=label(p),source_sha256=sha256(p),transform='copy'))
        return dest
    def derived(im,dest,inputs,transform):
        im.save(out/dest)
        assets.append(dict(path=dest,sha256=sha256(out/dest),sources=[source(p) for p in inputs],transform=transform))
        return dest
    for p in [ROOT/'experiments/runs/REPORT-002-PROGRESS-WEB/config.yaml',ROOT/'data/manifests/contest_dataset_v1.json',ROOT/'experiments/splits/sem_001_scaffold_static_v1.json']:
        source(p)
    cameras=read_camera_requests(artifact/'results/frozen_scaffold_sem001/flythrough_v1/cameras')
    names=[f'frame{i:04d}.png' for i in range(144)];assert sorted(cameras)==names
    for p in sorted((artifact/'results/frozen_scaffold_sem001/flythrough_v1/cameras').iterdir()):
        if p.is_file():source(p)
    trajectory_hashes={p.name:sha256(p) for p in (artifact/'results/frozen_scaffold_sem001/flythrough_v1/cameras').iterdir() if p.is_file()}
    oldroot=artifact/'results/frozen_scaffold_sem001/flythrough_v1/renders'
    old=read(oldroot/'submission_manifest.json');source(oldroot/'submission_manifest.json')
    assert old['cameras_txt_sha256']==trajectory_hashes['cameras.txt'] and old['images_txt_sha256']==trajectory_hashes['images.txt']
    for row in old['outputs']:
        verify_hashes(oldroot,{row[k]['path']:row[k]['sha256'] for k in ['rgb','semantic']})
    oldroot=ROOT/'outputs/vis006/full-20260908-r1/base'
    old=read(oldroot/'manifest.json');source(oldroot/'manifest.json')
    assert old['cameras_sha256']==trajectory_hashes['cameras.txt'] and old['images_sha256']==trajectory_hashes['images.txt']
    for row in old['outputs']:
        verify_hashes(oldroot,{v['path']:v['sha256'] for v in row['outputs'].values()})
    media=[
        ('sem001',artifact/'results/frozen_scaffold_sem001/flythrough_v1/renders/rgb',artifact/'results/frozen_scaffold_sem001/flythrough_v1/renders/semantic',False),
        ('sem161',ROOT/'outputs/vis006/full-20260908-r1/base/rgb',ROOT/'outputs/vis006/full-20260908-r1/base/semantic',False),
        ('b2',work/'render/official',work/'b2-r2/official',False),
        ('m2f',work/'render/official',work/'m2f/official',False),
        ('sem380',ROOT/'outputs/quality/SEM-380/r2/export/images',ROOT/'outputs/quality/SEM-380/r2/inference/predictions',True),
        ('sem383',ROOT/'outputs/quality/SEM-383/r2/export/images',ROOT/'outputs/quality/SEM-383/r2/inference-6000/predictions',True),
    ]
    # Check inputs against completed, sealed producers before deriving presentation media.
    for stage in ['render','b2-r2','m2f','score-r2']:
        p=work/stage/'result.json';result=read(p);assert result['status']=='completed';source(p)
        verify_hashes(work/stage,result['artifacts'])
    for prefix in ['SEM-380/r2','SEM-383/r2']:
        batch=ROOT/'outputs/quality'/prefix;export=read(batch/'export/result.json')
        verify_hashes(batch/'export/images',export['image_hashes']);source(batch/'export/result.json')
        assert export['trajectory_names']==names
        assert export['trajectory_metadata_sha256']=={k:trajectory_hashes[k] for k in ['cameras.txt','images.txt']}
    videos={}
    for key,rgbroot,semroot,warp in media:
        input_hashes={label(p):sha256(p) for n in names for p in [rgbroot/n,semroot/n]}
        fingerprint=hashlib.sha256(json.dumps(dict(inputs=input_hashes,warp=warp,palette=PALETTE.tolist(),fps=12,encoder='h264-crf18-native2640x990'),sort_keys=True).encode()).hexdigest()
        target=out/f'assets/videos/{key}.mp4';meta=work/f'video-{key}.json';poster=out/f'assets/videos/{key}.jpg'
        if not (meta.exists() and target.exists() and poster.exists() and read(meta)['fingerprint']==fingerprint and read(meta)['video_sha256']==sha256(target)):
            command=['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','2640x990','-framerate','12','-i','pipe:0','-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-threads','4',str(target)]
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                for index,n in enumerate(names):
                    rgb=png(rgbroot/n);sem=png(semroot/n)
                    assert rgb.shape==(989,1320,3) and sem.shape==(989,1320) and sem.dtype==np.uint8 and sem.max()<5
                    if warp:
                        cam=cameras[n];k=cam.matrix
                        radial=SimpleRadialCamera(cam.width,cam.height,float(k[0,0]),float(k[0,2]),float(k[1,2]),cam.radial_k)
                        rgb=redistort_rgb(rgb,radial);sem=official_labels(sem,cam)
                    canvas=Image.new('RGB',(2640,990),(30,30,30));canvas.paste(Image.fromarray(rgb),(0,0));canvas.paste(Image.fromarray(PALETTE[sem]),(1320,0))
                    if index==0:canvas.save(poster,quality=94)
                    process.stdin.write(canvas.tobytes())
                process.stdin.close();error=process.stderr.read().decode();code=process.wait()
                if code:raise RuntimeError(error)
            except Exception:
                process.kill();process.wait();raise
            save(meta,dict(fingerprint=fingerprint,video_sha256=sha256(target),poster_sha256=sha256(poster)))
        probe=read_probe(target);assert int(probe['streams'][0]['nb_frames'])==144 and float(probe['format']['duration'])==12
        videos[key]=dict(video=str(target.relative_to(out)),poster=str(poster.relative_to(out)))
        assets.append(dict(path=videos[key]['video'],sha256=sha256(target),input_hashes=input_hashes,transform='144 native frames; official SIMPLE_RADIAL presentation; RGB and nearest class-ID redistortion where needed; RGB left/semantic right; 1 bottom padding row; H264 CRF18; no model postprocessing',probe=probe))
        assets.append(dict(path=videos[key]['poster'],sha256=sha256(poster),transform='first composed frame, JPEG94'))
        print(json.dumps({'video':key,'bytes':target.stat().st_size}),flush=True)
    finalscore=read(ROOT/'outputs/quality/final_comparison/SEM-350-evaluation-r1/result.json')['evaluation']['metrics']
    source(ROOT/'outputs/quality/final_comparison/SEM-350-evaluation-r1/result.json')
    backbone=read(ROOT/'outputs/quality/backbone_comparison/browser-r1/data.json');standard=next(x for x in backbone['summaries'] if x['arm']=='standard');source(ROOT/'outputs/quality/backbone_comparison/browser-r1/data.json')
    b2=read(work/'score-r2/result.json')['metrics']
    def normalized(label_,id_,m,featured=False):
        row=dict(label=label_,id=id_,miou=m['mIoU_with_background'],featured=featured)
        for k in CLASSES[1:]:
            row[k]=m['per_class'][k]['IoU'] if 'per_class' in m else m['per_class_iou'][k] if 'per_class_iou' in m else m['IoU_'+k]
        return row
    history=[normalized('三维语义','SEM001',metric('SEM-001')),normalized('三维＋二维融合','SEM161',finalscore['old_sem161']),normalized('纯二维 SegFormer-B2','SEM331 · raw',b2),normalized('Mask2Former','SEM349 · raw',finalscore['new_raw'])]
    full=[normalized('Standard ＋ Mask2Former','RGB130 + SEM371',standard['semantic'])]+[normalized(title,ids,metric(pre),pre=='SEM-380') for pre,title,ids in [
        ('SEM-378','绝对梯度增密 ＋ Mask2Former','RGB140 + SEM378'),('SEM-380','结构区域 RGB ＋ Mask2Former','RGB148 + SEM380'),('SEM-381','DINOv3 ＋ SFP','RGB140 + SEM381'),('SEM-382','DINOv3 · 提高 SFP 学习率','RGB140 + SEM382'),('SEM-383','DINOv3 · 加入空间先验','RGB140 + SEM383')]]
    stages=[]
    specs=[
        ('sem001','三维语义基线','冻结三维表示','RGB030 + SEM001',history[0]['miou'],'历史74图','在冻结的 Scaffold-GS 上附着静态五类语义，建立三维语义起点。','桥面与桥塔可辨，stay_cable 在背景和水面有明显误报。'),
        ('sem161','三维＋二维融合','二维证据与类别修正','RGB030 + SEM161',history[1]['miou'],'历史74图','加入二维分割、拉索专家、基础复核与桥面恢复，保留三维类别保护。','相对 SEM001，历史74图五类 mIoU 提高21.777个百分点；仍有局部漏检和误补。'),
        ('b2','纯二维 SegFormer-B2','直接预测五类语义','RGB110 + SEM331 · raw',history[2]['miou'],'历史74图','对冻结 RGB110 的渲染结果直接做二维五类分割，不使用三维语义融合。','指标由保存的74图原始预测按原协议复算；不是融合版80.9177%。'),
        ('m2f','Mask2Former','Swin-L ＋ 掩码分类','RGB110 + SEM349 · raw',history[3]['miou'],'历史74图','升级为 Mask2Former-Swin-L，直接输出原始五类分割。','与 B2 使用同一 RGB110；B2训练3000步、Mask2Former训练6000步，不是同预算消融。'),
        ('sem380','全量训练最高分方案','结构区域 RGB ＋ Mask2Former','RGB148 + SEM380',metric('SEM-380')['mIoU_with_background'],'全300拟合','在400张 RGB、300张语义标签上训练，以结构区域 RGB 监督得到最高原始五类拟合分数。','较 SEM378 仅+0.0091个百分点，局部有得有失；尚未替换默认 RGB130＋SEM371。'),
        ('sem383','DINOv3 探索','冻结特征 ＋ 空间先验','RGB140 + SEM383',metric('SEM-383')['mIoU_with_background'],'全300拟合','将 Swin-L 替换为冻结 DINOv3，先提高 SFP 学习率，再加入轻量空间先验。','较 SEM382 +0.0361个百分点，但仍低于 SEM378 0.1387个百分点；不晋级。')]
    for key,title,short,id_,miou,scope,description,note in specs:stages.append(dict(title=title,short=short,id=id_,miou=miou,scope=scope,description=description,note=note,**videos[key]))
    rgbHistory=[]
    for pre,title in [('RGB-030-FULL','Scaffold · 三维与融合阶段'),('RGB-110','Scaffold · B2 / Mask2Former阶段')]:
        m=metric(pre);rgbHistory.append(dict(label=title,id=record(pre)['experiment_id'],PSNR=m.get('PSNR_dB',m.get('PSNR_dB_mean')),SSIM=m.get('SSIM',m.get('SSIM_mean')),LPIPS=m['LPIPS_Alex']))
    rgbFull=[dict(label='Standard',id='RGB130',**standard['rgb'])]+[dict(label=title,id=pre,**{k:metric(pre)[k] for k in ['PSNR','SSIM','LPIPS']}) for pre,title in [('RGB-140','Standard · 绝对梯度增密'),('RGB-148','Standard · 结构区域 RGB')]]
    views=[];batch=ROOT/'outputs/quality/SEM-380/r2';base=batch/'comparison';gallery=read(base/'data.json');source(base/'data.json')
    boxes={'159.png':[150,620,850,989],'160.png':[150,620,850,989],'146.png':[0,620,520,989],'057.png':[680,290,866,400]}
    captions={'159.png':'159 · 大倾角近景：桥面、索区边界与岸线伪影。','160.png':'160 · 斜视索区：关注语义区域的完整性及边缘外溢。','146.png':'146 · 复杂边界：观察桥面与 stay_cable 的局部混淆。','057.png':'057 · 小尺度基础：远处基础目标与背景的区分。'}
    for n,box in boxes.items():
        item=next(x for x in gallery['items'] if x['name']==n);tiles=[]
        for select,title in [('真实照片','真实照片'),('RGB148','RGB 重建'),('GT','GT 标注'),('SEM380','语义预测')]:
            p=base/next(x['path'] for x in item['tiles'] if x['label']==select);key=f'{Path(n).stem}-{len(tiles)}'
            fullpath=copy(p,f'assets/images/{key}.png');crop=derived(Image.open(p).crop(box),f'assets/images/{key}-crop.png',[p],f'display-only shared crop {box}')
            tiles.append(dict(label=title,path=fullpath,crop=crop))
        views.append(dict(name=n,box=box,caption=captions[n],tiles=tiles))
    trajectory=[]
    for f in [96,97,98]:
        n=f'frame{f:04d}.png';rgb=batch/'export/images'/n;sem=batch/'inference/predictions'/n
        canvas=Image.new('RGB',(1320,1978));canvas.paste(Image.open(rgb),(0,0));canvas.paste(Image.fromarray(PALETTE[png(sem)]),(0,989))
        dest=derived(canvas,f'assets/images/trajectory-{f}.png',[rgb,sem],'native PINHOLE RGB above / class palette below; no GT')
        trajectory.append(dict(frame=f,path=dest))
    refs=['SEM-001','SEM-111','SEM-141','SEM-151','SEM-161','SEM-331','SEM-349','SEM-350','SEM-371','SEM-378','SEM-380','SEM-381','SEM-382','SEM-383','SEM-384','SEM-386','RGB-030-FULL','RGB-110','RGB-130','RGB-140','RGB-148']
    sources=[]
    for pre in refs:
        r=record(pre);eid=r['experiment_id'];chunks=[]
        for field in ['report_path','config']:
            p=ROOT/r[field];source(p);chunks.append(f'<h2>{html.escape(p.name)}</h2><pre>{html.escape(clean(p.read_text()))}</pre>')
        dest='sources/'+eid+'.html'
        page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+html.escape(eid)+'</title><style>body{max-width:1040px;margin:40px auto;padding:20px;font:15px/1.8 system-ui;color:#152338}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fc;padding:20px}a{color:#2151d1}</style><a href="../index.html#sources">← 返回项目进展</a><h1>'+html.escape(eid)+'</h1><p>原始记录快照；历史措辞以当时阶段为准。个人路径已脱敏。</p>'+''.join(chunks)+'<h2>实验登记</h2><pre>'+html.escape(clean(json.dumps(r,ensure_ascii=False,indent=2)))+'</pre></html>'
        (out/dest).write_text(page);sources.append(dict(id=eid,link=dest,split=r['split_id'],parent=r.get('parent_experiment'),commit=r.get('git_commit'),config=r['config'],seed=r.get('seed'),manifest=r.get('dataset_manifest')))
    data=dict(stages=stages,history=history,full=full,rgbHistory=rgbHistory,rgbFull=rgbFull,views=views,trajectory=trajectory,palette=PALETTE.tolist(),classes=CLASSES,sources=sources)
    save(out/'data/content.json',data);save(out/'data/asset_manifest.json',assets)
    for f in ['progress.css','progress.js']:shutil.copyfile(ROOT/'scripts/report_assets'/f,out/f)
    template=(ROOT/'scripts/report_assets/progress.html').read_text()
    (out/'index.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')))
    provenance=dict(experiment_id='REPORT-002-PROGRESS-WEB',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),uncommitted_worktree=True,source_data=source_data,source_code={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'scripts/prepare_progress_media.py',*[ROOT/'scripts/report_assets'/f for f in ['progress.html','progress.css','progress.js']]]},scope='No training; frozen render/inference and independently counted saved raw B2; no new split or tuning',camera_scope='Same VIS001 official-reference 144 poses. Videos in SIMPLE_RADIAL official display grid; metrics/difficult-view photos in original PINHOLE evaluation grid.',video_codec='Lossy H264 visualization copy; native predictions remain scientific record')
    save(out/'data/build_provenance.json',provenance)
    # Bundle preparation receipts without checkpoints or raw predictions.
    for stage in ['render','b2-r2','m2f','score-r2']:
        receipt=clean((work/stage/'result.json').read_text());(out/'data'/f'{stage}-receipt.json').write_text(receipt)
    (out/'README.txt').write_text('ICSHM 2026 项目进展 · 2026-09-18\n\n解压整个目录，双击 index.html 即可离线使用。请保留 assets、data、sources 与 CSS/JS。\n点击里程碑切换视频；视频可暂停、拖动及全屏。困难视角可切换完整画面/局部放大，点击图片查看大图。\n历史74图开发评价、全300语义拟合和全400 RGB拟合分开；轨迹无GT。SEM380是全量拟合最高分，尚未替换默认方案。\n六段视频均为同一官方参考坐标系144帧轨迹，12fps/12秒，H.264。视频使用官方畸变网格，照片/GT对照与指标使用原生PINHOLE评价网格。\n所有素材仅用于离线研究展示，不含checkpoint；原始实验保持。\n')
    files={str(p.relative_to(out)):dict(sha256=sha256(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name!='package_manifest.json'}
    save(out/'package_manifest.json',dict(experiment_id='REPORT-002-PROGRESS-WEB',files=files))
    if a.package:
        with zipfile.ZipFile(str(out)+'.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file():z.write(p,str(Path(out.name)/p.relative_to(out)))
    print(json.dumps(dict(status='completed',output=str(out),files=len(files),assets=len(assets)),ensure_ascii=False))


def read_probe(p):return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,codec_name,pix_fmt:format=duration,size','-of','json',str(p)],text=True))


if __name__=='__main__':main()
