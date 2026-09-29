#!/usr/bin/env python3
"""Native scientific crops: show BOTH largest corrected and harmed deck regions."""
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from src.semantic2d import read_json,write_json,sha256
from src.semantic_render_domain import semantic_rgb
from audit_sem260_error_budget import manifest_paths


def read(path):
    return np.asarray(Image.open(path))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('artifact-root','config','run-dir','output-dir'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    config=read_json(args.config)
    result=read_json(args.run_dir/'result.json')
    names={r['name'] for r in result['calibration']['target_access']}
    arms=config.get('arm_names',['coarse_control','fine_candidate'])
    predictions=read_json(args.run_dir/'prediction_manifest.json')
    baseline=manifest_paths(args.artifact_root/config['parents']['sem160'],'prediction_manifest.json','predictions','prediction','prediction_sha256')
    rgbpaths=manifest_paths(args.artifact_root/config['parents']['rgb'],'rendered_dataset_manifest.json','rendered_images','path','sha256')
    records=[]
    for name in config['visual_samples']:
        if name not in names:
            continue
        old=read(baseline[name][0])
        rgb=read(rgbpaths[name][0])
        gt=read(args.artifact_root/'r2/sem_001/labels'/name)
        arrays=[]
        for arm in arms:
            row=predictions[arm][name]
            if sha256(Path(row['path']))!=row['sha256']:
                raise ValueError('Sealed prediction changed')
            arrays.append(read(row['path']))
        new=arrays[1]
        corrected=(old!=new)&(new==gt)
        harmed=(old!=new)&(old==gt)
        relevant=(old==1)|(new==1)|(gt==1)
        error=np.zeros_like(rgb)
        error[corrected]=(40,220,80)
        error[harmed]=(255,50,50)
        for kind,mask in [('corrected',corrected),('harmed',harmed)]:
            mask=mask&relevant
            if not mask.any():
                continue
            scores=cv2.boxFilter(mask.astype(np.float32),-1,(224,128),normalize=False,borderType=cv2.BORDER_CONSTANT)
            y,x=np.unravel_index(scores.argmax(),scores.shape)
            x0=max(0,min(int(x)-112,old.shape[1]-224))
            y0=max(0,min(int(y)-64,old.shape[0]-128))
            columns=[rgb,semantic_rgb(gt),semantic_rgb(old),semantic_rgb(arrays[0]),semantic_rgb(new),error]
            titles=['Rendered RGB','GT','SEM-160 baseline',arms[0],arms[1],'Green corrected / red harmed']
            panel=Image.new('RGB',(448*6,288),'white')
            draw=ImageDraw.Draw(panel)
            for i,(array,title) in enumerate(zip(columns,titles)):
                tile=Image.fromarray(array[y0:y0+128,x0:x0+224]).resize((448,256),Image.Resampling.NEAREST)
                panel.paste(tile,(448*i,32))
                draw.text((448*i+4,6),name+' '+title,fill='black')
            path=args.output_dir/(Path(name).stem+'_'+kind+'_2x.png')
            panel.save(path)
            records.append({'view':name,'kind':kind,'roi_xywh':[x0,y0,224,128],
                            'resize':'2x nearest, identical for all columns; no filtering',
                            'path':path.name,'sha256':sha256(path)})
    write_json(args.output_dir/'manifest.json',{'experiment_id':config['experiment_id'],
        'selection':'Both maximum corrected and maximum harmed fixed-size regions per preregistered calibration view; visualization only',
        'result_sha256':sha256(args.run_dir/'result.json'),'panels':records})
    print(len(records),'native local panels')


if __name__=='__main__':
    main()
