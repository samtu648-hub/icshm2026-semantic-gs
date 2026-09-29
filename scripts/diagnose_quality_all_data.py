#!/usr/bin/env python3
"""Fixed nine-view all-data fit diagnostics and frozen SEM363 transfer, no fit."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from src.quality_protocol import sha256,write_new_json,verify_hashes
from src.quality_learned_selector import neighbourhood,predict
PAL=np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]],dtype=np.uint8)

def read(p):return json.loads(p.read_text())
def arr(p):return np.asarray(Image.open(p))
def confusion(gt,p):return np.bincount((gt.astype(np.int64)*5+p).ravel(),minlength=25).reshape(5,5)
def panel(images,titles,out,box=None):
    box=box or (0,0,1320,989);w=550;h=round(w*(box[3]-box[1])/(box[2]-box[0]));cols=3
    canvas=Image.new('RGB',(cols*w,((len(images)+cols-1)//cols)*(h+32)),'white');d=ImageDraw.Draw(canvas)
    for i,(im,title) in enumerate(zip(images,titles)):
        x=i%cols*w;y=i//cols*(h+32);d.text((x+5,y+8),title,fill='black')
        canvas.paste(Image.fromarray(im).crop(box).resize((w,h),Image.Resampling.NEAREST),(x,y+32))
    canvas.save(out)

def main():
    p=argparse.ArgumentParser()
    for n in ['batch','rgb-run']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();out=a.batch/'diagnostics';out.mkdir(exist_ok=False);(out/'predictions').mkdir()
    cfg=read(ROOT/'experiments/runs/SEM-370-ALL300/config.yaml');names=cfg['diagnostic_views']
    old=ROOT/'outputs/quality/final_transfer/SEM-363-transfer-r1';head=old/'model.npz'
    assert sha256(head)=='06617fd19a55b674aa44ae45d7a33b6f426e91e4eb9f9a51579a185353cf2d45'
    with np.load(head,allow_pickle=False) as z:model={k:z[k] for k in z.files}
    records={k:read(a.batch/k/'result.json') for k in ['mask2former','static','b2','deck','stay_cable']}
    for k,r in records.items():verify_hashes(a.batch/k,r['artifacts'])
    oldcfg=read(old/'effective_config.json');oldfive=ROOT/oldcfg['final_sources']['five']
    rows=[];rois=read(ROOT/'experiments/runs/QUAL-005-PHOTO-INPUT-DIAG/config.yaml').get('rois',{})
    def probability(stage,rel,n):
        with np.load(a.batch/stage/rel/'probabilities'/(Path(n).stem+'.npz'),allow_pickle=False) as z:return z['probabilities'].copy()
    for n in names:
        rawp=probability('mask2former','',n);raw=rawp.argmax(0).astype(np.uint8)
        probs=[rawp,probability('b2','step-3000',n),probability('b2','step-6000',n)]
        static=arr(a.batch/'static/predictions'/n)
        binary=[probability(k,'step-3000',n) for k in ['deck','stay_cable']]
        maps=np.stack([static,*[v.argmax(0).astype(np.uint8) for v in probs[1:]],
            *[np.where(v.argmax(0)==1,cl,5).astype(np.uint8) for v,cl in zip(binary,[1,2])]])
        data=(raw,maps,probs,np.stack([v[1] for v in binary]),neighbourhood(raw))
        final=predict(data,model);Image.fromarray(final).save(out/'predictions'/n)
        gt=arr(a.batch/'labels'/n);oldpred=arr(old/'predictions'/n);oldraw=arr(oldfive/'predictions'/n)
        correct=(raw==gt)|np.any(maps==gt,axis=0);err=raw!=gt;recoverable=err&correct
        row=dict(name=n,confusions={k:confusion(gt,v).tolist() for k,v in [('old_raw',oldraw),('old_final',oldpred),('new_raw',raw),('new_final',final)]},
            raw_errors=int(err.sum()),recoverable=int(recoverable.sum()),recovered_by_fusion=int((recoverable&(final==gt)).sum()),
            common_failure=int((~correct).sum()),fusion_damages=int(((raw==gt)&(final!=gt)).sum()),
            deck_background={k:int(((gt==1)&(v==0)).sum()) for k,v in [('old_final',oldpred),('new_raw',raw),('new_final',final)]})
        rows.append(row)
        photo=arr(a.rgb_run/'scene/images'/n);oldrgb=arr(ROOT/'outputs/quality/diagnostics/QUAL-005-r1/render'/n) if (ROOT/'outputs/quality/diagnostics/QUAL-005-r1/render'/n).exists() else None
        # Use the exact old rendered expert input, not a historical screenshot.
        oldexport=read(ROOT/'experiments/runs/QUAL-005-PHOTO-INPUT-DIAG/config.yaml')['render_root']
        oldrgb=arr(ROOT/oldexport/n)
        images=[photo,oldrgb,arr(a.batch/'export/images'/n),PAL[gt],PAL[oldraw],PAL[oldpred],PAL[static],PAL[raw],PAL[final]]
        titles=[n+' Photo PINHOLE','RGB110 /303 training','RGB120 /400 training','GT (now training label)','SEM349 /219 labels','SEM363 old fixed final','SEM370 static /300','SEM370 M2F /300','SEM370 + frozen SEM363 head']
        panel(images,titles,out/(Path(n).stem+'.png'))
        if Path(n).stem in rois:panel(images,titles,out/(Path(n).stem+'-crop.png'),rois[Path(n).stem])
    write_new_json(out/'result.json',dict(experiment_id='SEM-370-ALL300',status='completed_fixed9_training_fit_diagnostic',
        head_sha256=sha256(head),new_fusion_fit=0,independent_holdout=False,rows=rows,
        caveat='New RGB and semantics trained on these views. Old/new comparison measures full-data fit, not generalization or isolated RGB causality.',
        artifacts={p.name:sha256(p) for p in out.glob('*.png')}))
    html='<!doctype html><meta charset="utf-8"><title>RGB120 / SEM370 全数据诊断</title><style>body{font:17px sans-serif;max-width:1700px;margin:30px auto;background:#eef1f5}img{width:100%}section{background:white;padding:16px;margin:20px 0}a{color:#1464a0}</style><h1>RGB120 / SEM370 全数据训练诊断</h1><p>400 张 RGB、300 张标签；固定 SEM363 融合头迁移。以下全部为训练内诊断，不是独立盲测；旧版 RGB110／SEM363 保留。</p><p>图中背景黑、桥面蓝、stay_cable 黄、桥塔红、基础紫。每行同位置，包含真实照片与准确版本。</p>'
    for row in rows:
        stem=Path(row['name']).stem;html+=f'<section><h2>{stem}</h2><p>新 raw 错误 {row["raw_errors"]}；可恢复 {row["recoverable"]}，融合修对 {row["recovered_by_fusion"]}、误改 {row["fusion_damages"]}；全部分支共同失败 {row["common_failure"]}。</p><a href="{stem}.png"><img src="{stem}.png"></a>'
        if (out/(stem+'-crop.png')).exists():html+=f'<a href="{stem}-crop.png">查看局部原图对照</a>'
        html+='</section>'
    (out/'index.html').write_text(html)

if __name__=='__main__':main()
