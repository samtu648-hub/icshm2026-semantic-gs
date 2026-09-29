#!/usr/bin/env python3
"""Auditable spatial/temporal distribution of actual photo-guided child placements."""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--geometry-cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    events=[json.loads(s) for s in (a.run/'placements.jsonl').read_text().splitlines()]
    rows=[q|{'iteration':e['iteration']} for e in events for q in e['placements'] if q['applied']]
    used={q['proposal_id'] for q in rows};assert len(used)==len(rows)
    cache=json.loads(a.geometry_cache.read_text());proposals={p['id']:p for p in cache['proposals']}
    fixed=json.loads((ROOT/'experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml').read_text());roi=fixed['regions'];regions={key:{'accepted':0,'guided':0} for key in roi}
    cameras={};classes={};all_xyz=np.array([p['xyz'] for p in proposals.values()]);xyz=np.array([q['photo_xyz'] for q in rows])
    for name in sorted({o['camera'] for p in proposals.values() for o in p['observations']}):
        obs=[(p,o) for p in proposals.values() for o in p['observations'] if o['camera']==name]
        cameras[name]={'accepted':len(obs),'guided':sum(p['id'] in used for p,o in obs)}
        label_path=ROOT/'outputs/quality/all_data/SEM-370-r1/labels'/name
        labels=np.asarray(Image.open(label_path)) if label_path.exists() else None
        for p,o in obs:
            x,y=o['xy'];active=p['id'] in used
            for key,r in roi.items():
                if r['view']==name and any(x0<=x<x1 and y0<=y<y1 for x0,y0,x1,y1 in r['boxes']):regions[key]['accepted']+=1;regions[key]['guided']+=active
            if labels is not None:
                key=f'{name}:{int(labels[int(y),int(x)])}';item=classes.setdefault(key,{'accepted':0,'guided':0});item['accepted']+=1;item['guided']+=active
        if name in fixed['views']:
            im=Image.open(ROOT/'outputs/quality/all_data/RGB-120-r2/scene/images'/name).convert('RGB');d=ImageDraw.Draw(im)
            for p,o in obs:
                x,y=o['xy'];active=p['id'] in used;d.ellipse((x-4,y-4,x+4,y+4),outline=(30,255,110) if active else (255,210,30),width=2)
            im.save(a.output/f'{Path(name).stem}-guided.png')
    with (a.output/'placements.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['iteration','parent_id','proposal_id','distance_sigma','displacement','photo_xyz','native_child_xyz']);writer.writeheader();writer.writerows({k:q[k] for k in writer.fieldnames} for q in rows)
    with (a.output/'view_coverage.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['view','accepted','guided']);writer.writeheader();writer.writerows({'view':k,**v} for k,v in cameras.items())
    def quant(key):return {str(q):float(np.quantile([r[key] for r in rows],q)) for q in [0,.25,.5,.75,.9,.95,.99,1]} if rows else {}
    result={'actual_guided':len(rows),'accepted_proposals':len(proposals),'unused_proposals':len(proposals)-len(rows),
            'regions':regions,'camera_coverage':cameras,'label_region_coverage':classes,'distance_sigma_quantiles':quant('distance_sigma'),
            'native_child_to_photo_displacement_quantiles_scene_units':quant('displacement'),
            'interpretation':'Proposal used once; 3sigma is relative to parent covariance, not a small absolute-motion bound. Label regions are diagnostic only and are not physical cable masks.',
            'run':str(a.run),'iteration_last_event':events[-1]['iteration']}
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(a.output/'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    axes[0,0].plot([e['iteration'] for e in events],np.cumsum([sum(q['applied'] for q in e['placements']) for e in events]));axes[0,0].set(title='Actual placements',xlabel='Training iteration',ylabel='Cumulative proposals used')
    axes[0,1].hist([q['distance_sigma'] for q in rows],bins=30);axes[0,1].set(title='Parent-to-proposal normalized distance',xlabel='Parent standard deviations',ylabel='Count')
    axes[1,0].scatter(all_xyz[:,0],all_xyz[:,1],s=5,c='#9aa4af',label='All accepted photo points')
    if len(xyz):axes[1,0].scatter(xyz[:,0],xyz[:,1],s=5,c='#148f79',label='Used proposals')
    axes[1,0].set(title='Official world XY, full range',xlabel='X (scene units)',ylabel='Y (scene units)');axes[1,0].legend()
    axes[1,1].hist(np.log10(np.maximum([q['displacement'] for q in rows],1e-12)),bins=40);axes[1,1].set(title='Native child to photo replacement displacement',xlabel='log10(scene units)',ylabel='Count')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('RGB146 / seed0 / training-photo geometry / no geometric ground truth');fig.savefig(a.output/'placement-distribution.png',dpi=160);plt.close(fig)
    print(json.dumps({k:result[k] for k in ['actual_guided','regions','distance_sigma_quantiles','native_child_to_photo_displacement_quantiles_scene_units']}))


if __name__=='__main__':main()
