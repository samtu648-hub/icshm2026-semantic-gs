#!/usr/bin/env python3
"""Photo geometry evidence browser; no synthetic reconstruction images."""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry-cache',type=Path,required=True);p.add_argument('--diagnostic',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    geo=json.loads(a.geometry_cache.read_text());diag=json.loads((a.diagnostic/'result.json').read_text())
    audit_path=a.diagnostic.parent/'photo-audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text());corrected={r['region']:r for r in audit['corrected_coverage']}
        for row in diag['summary']:
            c=corrected[row['region']];row['new_coverage_gt3px_from_old']=c['new_coverage_gt3px_from_retained_old'];row['old_distance_px']=c['distance_px']
        diag['coverage_correction']=audit['correction']
    photo=ROOT/'outputs/quality/all_data/RGB-120-r2/scene/images';examples=a.output/'proposals';examples.mkdir()
    records=[]
    for r in geo['proposals']:
        count=len(r['observations']);cols=min(4,count);rows=(count+cols-1)//cols
        im=Image.new('RGB',(cols*250,rows*310+45),'#101820');d=ImageDraw.Draw(im)
        for i,o in enumerate(r['observations']):
            x,y=o['xy'];x0,y0=int(x)-80,int(y)-80
            src=Image.open(photo/o['camera']).convert('RGB').crop((x0,y0,x0+160,y0+160)).resize((240,240))
            dd=ImageDraw.Draw(src);cx,cy=(x-x0)*1.5,(y-y0)*1.5;dd.ellipse((cx-4,cy-4,cx+4,cy+4),outline='yellow',width=2)
            col,row=i%cols,i//cols;im.paste(src,(col*250,row*310+28));role='BUILD' if i in r['build_indices'] else 'CHECK'
            d.text((col*250+3,row*310+5),o['camera']+' '+role,fill='white')
            d.text((col*250+3,row*310+275),f"err {r['reprojection_error_px'][i]:.3f}px; sigma {r['relative_depth_std'][i]*100:.2f}%",fill='white')
        d.text((10,rows*310+8),f"proposal {r['id']} / angle {r['max_angle_deg']:.2f}deg / training photos / no GT",fill='white')
        im.save(examples/f"{r['id']}.png")
        records.append({'id':r['id'],'views':[o['camera'] for o in r['observations']],'max_error_px':max(r['reprojection_error_px']),
            'max_relative_sigma':max(r['relative_depth_std']),'angle_deg':r['max_angle_deg'],'confidence_score':r['confidence']})
    with (a.output/'geometry_proposals.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['id','views','max_error_px','max_relative_sigma','angle_deg','confidence_score']);w.writeheader();w.writerows(records)
    (a.output/'diagnostic.json').write_text(json.dumps(diag,ensure_ascii=False,indent=2))
    data={'geometry_summary':geo['summary'],'proposals':records,'regions':diag['summary'],'layers':diag['depth_layers'],
          'photo_root':os.path.relpath(photo,a.output),'diagnostic_root':os.path.relpath(a.diagnostic,a.output),
          'overlay_root':os.path.relpath(a.diagnostic.parent/'retained-track-overlays',a.output),
          'fixed':['159','160','140','146','081','057','064','054','174']}
    (a.output/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    data['interpretations']=json.loads((a.diagnostic.parent/'diagnostic-interpretation.json').read_text())['regions']
    fixed=json.loads((ROOT/'experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml').read_text())
    data['boxes']={k:[min(b[0] for b in v['boxes']),min(b[1] for b in v['boxes']),max(b[2] for b in v['boxes']),max(b[3] for b in v['boxes'])] for k,v in fixed['regions'].items()}
    (a.output/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    html=(ROOT/'scripts/photo_geometry_review.html').read_text()
    (a.output/'index.html').write_text(html.replace('__DATA__',json.dumps(data,ensure_ascii=False)))


if __name__=='__main__':main()
