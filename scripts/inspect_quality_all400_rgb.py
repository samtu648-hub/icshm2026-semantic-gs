#!/usr/bin/env python3
"""Immediate fixed-view RGB inspection, reusing SEM370's native training export."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from src.quality_protocol import sha256,write_new_json,verify_hashes

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--rgb-run',type=Path,required=True);a=p.parse_args()
    cfg=json.loads((ROOT/'experiments/runs/QUAL-005-PHOTO-INPUT-DIAG/config.yaml').read_text())
    export=json.loads((a.batch/'export/result.json').read_text())
    assert export['status']=='completed_training_diagnostic_only'
    out=a.rgb_run/'inspection';out.mkdir(exist_ok=False);rows=[]
    font=ImageFont.truetype('DejaVuSans.ttf',22)
    for name in cfg['names']:
        paths=[a.rgb_run/'scene/images'/name,ROOT/cfg['render_root']/name,a.batch/'export/images'/name]
        assert sha256(paths[2])==export['image_hashes'][name]
        ims=[Image.open(p).convert('RGB') for p in paths];arrays=[np.asarray(i).astype(np.float64)/255 for i in ims]
        row=dict(name=name,input_hashes=[sha256(p) for p in paths],psnr={})
        for k,im in zip(['RGB110','RGB120'],arrays[1:]):row['psnr'][k]=float(-10*np.log10(np.mean((arrays[0]-im)**2)))
        rows.append(row)
        for suffix,box in [('',(0,0,1320,989)),('-crop',cfg['rois'].get(Path(name).stem,(0,0,1320,989)))]:
            w=660;h=round(w*(box[3]-box[1])/(box[2]-box[0]));canvas=Image.new('RGB',(w*3,h+42),'white');draw=ImageDraw.Draw(canvas)
            for i,(im,title) in enumerate(zip(ims,[name+' Photo / official PINHOLE','RGB110 / 303 photos / 30k','RGB120-r2 / 400 photos / 30k'])):
                draw.text((i*w+8,9),title,fill='black',font=font);canvas.paste(im.crop(box).resize((w,h),Image.Resampling.LANCZOS),(i*w,42))
            canvas.save(out/(Path(name).stem+suffix+'.png'))
    write_new_json(out/'result.json',dict(experiment_id='RGB-120-ALL400',status='completed_immediate_fixed9_rgb_inspection_assets',
        parent_result_sha256=sha256(a.rgb_run/'result.json'),export_result_sha256=sha256(a.batch/'export/result.json'),code_sha256=sha256(Path(__file__)),
        role='all-data reconstruction baseline; nine views now training, not held-out generalization',rows=rows))
    html='<!doctype html><meta charset="utf-8"><title>RGB120 全数据重建基线即时检查</title><style>body{font:18px sans-serif;margin:28px;background:#eef1f5}img{width:100%}section{background:white;padding:12px;margin:20px 0}</style><h1>RGB120 全数据重建基线：训练结束即时检查</h1><p>左：真实照片；中：RGB110 303张；右：RGB120-r2 400张。均同官方PINHOLE坐标、30k端点。九图现为训练视角，不能当独立泛化提升。复用SEM370输入导出，无重复GPU渲染。</p>'
    for r in rows:
        stem=Path(r['name']).stem;html+=f'<section><h2>{stem}</h2><img src="{stem}.png"><p><a href="{stem}-crop.png">局部放大</a></p></section>'
    (out/'index.html').write_text(html)
    print(json.dumps(rows))

if __name__=='__main__':main()
