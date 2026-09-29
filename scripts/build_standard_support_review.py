#!/usr/bin/env python3
"""Build a local, synchronized RGB142/143 review and independent PNG audit."""
import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import binary_dilation,binary_erosion
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/quality/rgb141/review-r1')
    p.add_argument('--support-run',type=Path,default=ROOT/'outputs/quality/rgb141/support-r1');a=p.parse_args()
    a.output.mkdir(exist_ok=False);start=time.monotonic()
    pairroot=ROOT/'outputs/quality/rgb141';parent=ROOT/'outputs/quality/all_data/RGB-120-r2/scene/images'
    roots={'RGB140':ROOT/'outputs/quality/rgb140/full-r2/evaluation','RGB142':pairroot/'control-r1/evaluation','RGB143':a.support_run/'evaluation'}
    results={k:json.loads((v/'result.json').read_text()) for k,v in roots.items()}
    assert all(v['status']=='completed' and v['train_count']==400 and v['trajectory_count']==144 for v in results.values())
    fixed=json.loads((ROOT/'experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml').read_text())
    names=sorted(x['name'] for x in results['RGB142']['per_view']);byname={k:{v['name']:v for v in r['per_view']} for k,r in results.items()}
    rows=[];regionrows=[];maxpsnrerror=0.;classes=['background','deck','stay_cable','tower','foundation'];classrows=[]
    for name in names:
        photo=np.asarray(Image.open(parent/name)).astype(float);images={k:np.asarray(Image.open(v/'images'/name)).astype(float) for k,v in roots.items()}
        row={'name':name}
        for model in roots:
            # Independent float64 PSNR from saved, quantized RGB.
            mse=np.square(images[model]/255-photo/255).mean();value=-10*np.log10(mse)
            maxpsnrerror=max(maxpsnrerror,abs(value-byname[model][name]['PSNR']))
            for metric in ['PSNR','SSIM','LPIPS']:row[model+'_'+metric]=byname[model][name][metric]
        for baseline in ['RGB140','RGB142']:
            for metric in ['PSNR','SSIM','LPIPS']:row['RGB143_minus_'+baseline+'_'+metric]=row['RGB143_'+metric]-row[baseline+'_'+metric]
        rows.append(row)
        for key,reg in fixed['regions'].items():
            if reg['view']!=name:continue
            mask=np.zeros(photo.shape[:2],bool)
            for x0,y0,x1,y1 in reg['boxes']:mask[y0:y1,x0:x1]=True
            rr={'region':key,'view':name,'pixels':int(mask.sum())}
            for model in roots:rr[model]=float(np.abs(images[model]-photo)[mask].mean())
            regionrows.append(rr)
        label=ROOT/'outputs/quality/all_data/SEM-370-r1/labels'/name
        if label.exists():
            gt=np.asarray(Image.open(label))
            for j,k in enumerate(classes):
                mask=gt==j
                if mask.any():classrows.append({'view':name,'class':k,**{m:float(np.abs(im-photo)[mask].mean()) for m,im in images.items()}})
            if name in fixed['views']:
                structure=gt!=0
                masks={'deck':gt==1,'stay_cable':gt==2,
                    'structure-label-boundary':binary_dilation(structure,iterations=2)&~binary_erosion(structure,iterations=2)}
                for key,mask in masks.items():
                    if mask.any():regionrows.append({'region':Path(name).stem+'-'+key,'view':name,
                        'pixels':int(mask.sum()),**{m:float(np.abs(im-photo)[mask].mean()) for m,im in images.items()}})
    assert maxpsnrerror<1e-4, maxpsnrerror
    with (a.output/'per_view.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    counts={b:{m:{'mean_delta':float(np.mean([r['RGB143_minus_'+b+'_'+m] for r in rows])),
        'improved':sum((r['RGB143_minus_'+b+'_'+m]>0) if m!='LPIPS' else (r['RGB143_minus_'+b+'_'+m]<0) for r in rows),
        'degraded':sum((r['RGB143_minus_'+b+'_'+m]<0) if m!='LPIPS' else (r['RGB143_minus_'+b+'_'+m]>0) for r in rows)} for m in ['PSNR','SSIM','LPIPS']} for b in ['RGB140','RGB142']}
    resources={}
    for arm,model in [('control','RGB142'),('support','RGB143')]:
        run=pairroot/'control-r1' if arm=='control' else a.support_run
        r=json.loads((run/'result.json').read_text());events=[json.loads(x) for x in (run/'densification.jsonl').read_text().splitlines()];numeric=[json.loads(x) for x in (run/'numeric.jsonl').read_text().splitlines()]
        assert len(events)==144 and len(numeric)==300 and numeric[-1]['iteration']==30000
        assert all(x['evidence_views']==100 and x['selected']<=2048 for x in events)
        assert all(np.isfinite(x['loss']) for x in numeric)
        gpu_samples=[json.loads(x)['gpu'] for x in (run/'resources.jsonl').read_text().splitlines()]
        sampled_memory=[float(x.split(',')[0]) for x in gpu_samples if x.strip()]
        resources[model]={'training_wall_seconds':r.get('training_wall_seconds_excluding_lock',r['seconds']),
            'worker_seconds':r['worker']['seconds'],'lock_wait_seconds':r.get('lock_wait_seconds',0),
            'peak_allocated_mib':r['worker']['peak_allocated_mib'],'peak_reserved_mib':r['worker']['peak_reserved_mib'],
            'sampled_whole_gpu_peak_mib':max(sampled_memory),
            'final_gaussians':results[model]['final_gaussian_count'],'model_mib':results[model]['model_mib'],
            'render_median_ms':results[model]['render_median_ms'],'fixed9_cuda_render_median_ms':results[model]['fixed9_cuda_render_median_ms'],
            'total_split_parents':sum(x['selected'] for x in events),'ranking_changed_parents':sum(x['different_from_plain'] for x in events),
            'events_budget_limited':sum(x['selected']<x['eligible'] for x in events),'events':events,
            'population_trace':[{'iteration':x['iteration'],'points':x['points']} for x in numeric]}
    os.environ.setdefault('MPLCONFIGDIR',str(a.output/'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,8),layout='constrained')
    for model,color in [('RGB142','#17688b'),('RGB143','#bc572e')]:
        trace=resources[model]['population_trace']
        axes[0].plot([x['iteration'] for x in trace],[x['points']/1000 for x in trace],label=model,color=color,linewidth=1.6)
    axes[0].set(title='Matched-budget training: Gaussian population',ylabel='Gaussians (thousands)',xlabel='Training iteration');axes[0].legend()
    ctrl=resources['RGB142']['population_trace'];cand=resources['RGB143']['population_trace']
    axes[1].plot([x['iteration'] for x in cand],[s['points']-c['points'] for s,c in zip(cand,ctrl)],color='#bc572e',label='RGB143 - RGB142')
    axes[1].axhline(0,color='0.5',linewidth=.6);axes[1].set(title='Population difference at the same logged iteration',ylabel='Gaussian count difference',xlabel='Training iteration');axes[1].legend()
    events=resources['RGB143']['events']
    axes[2].plot([x['iteration'] for x in events],np.cumsum([x['different_from_plain'] for x in events]),color='#17688b',label='Cumulative selection replacements')
    axes[2].set(title='Support ranking actually changes selected split parents',ylabel='Event-counted parents (not unique IDs)',xlabel='Training iteration');axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('RGB142 / RGB143 | seed 0 | all400 training-fit protocol',fontsize=13)
    fig.savefig(a.output/'budget-and-ranking.png',dpi=180);plt.close(fig)
    fontpath=next(iter(Path('/usr/share/fonts').rglob('DejaVuSans.ttf')),None);font=ImageFont.truetype(str(fontpath),18) if fontpath else ImageFont.load_default()
    panels=a.output/'panels';panels.mkdir();tiles=[('Photo',parent)]+[(k,v/'images') for k,v in roots.items()]
    def panel(name,box,dest,width=660):
        height=round(width*(box[3]-box[1])/(box[2]-box[0]));active=tiles if name in names else tiles[1:]
        canvas=Image.new('RGB',(width*len(active),height+31),'white');draw=ImageDraw.Draw(canvas)
        for i,(label,path) in enumerate(active):
            draw.text((i*width+5,5),label,fill='black',font=font);canvas.paste(Image.open(path/name).crop(box).resize((width,height)),(i*width,31))
        canvas.save(dest)
    for name in fixed['views']:panel(name,(0,0,1320,989),panels/(Path(name).stem+'-full.png'))
    inherited_crops=json.loads((ROOT/'experiments/runs/QUAL-004-CURRENT-BOTTLENECK/config.yaml').read_text())['crops']
    for stem,box in inherited_crops.items():
        panel(stem+'.png',box,panels/(stem+'-inherited-structure-crop.png'))
    for key,reg in fixed['regions'].items():
        for i,box in enumerate(reg['boxes']):panel(reg['view'],box,panels/(key+f'-{i}.png'),width=box[2]-box[0])
    # These frame indices are fixed before new renders, from the earlier reviews.
    for i in [0,24,48,71,72,73,96,97,98,120,143]:panel(f'frame{i:04d}.png',(0,0,1320,989),panels/f'frame{i:04d}.png')
    frames=a.output/'video_frames';frames.mkdir()
    for i in range(144):panel(f'frame{i:04d}.png',(0,0,1320,989),frames/f'frame{i:04d}.png',width=1320)
    contacts=a.output/'trajectory_contacts';contacts.mkdir()
    for start_frame in range(0,144,6):
        sheet=Image.new('RGB',(1320,6*360),'white');draw=ImageDraw.Draw(sheet)
        for row,i in enumerate(range(start_frame,start_frame+6)):
            draw.text((5,row*360+3),f'Frame {i:04d} | RGB140 / RGB142 / RGB143 | no GT',fill='black',font=font)
            strip=Image.open(frames/f'frame{i:04d}.png').resize((1320,340))
            sheet.paste(strip,(0,row*360+20))
        sheet.save(contacts/f'frames-{start_frame:03d}-{start_frame+5:03d}.jpg',quality=95)
    video=a.output/'trajectory-rgb140-rgb142-rgb143.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-framerate','12','-i',str(frames/'frame%04d.png'),'-c:v','libx264','-threads','4','-crf','18','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    data={'scope':'400 training images = fit; 144 trajectory frames have no GT',
        'preserved_interruption':{k:v for k,v in json.loads((pairroot/'support-r1/result.json').read_text()).items() if k in ['status','last_progress_iteration','training_seconds_until_last_resource_sample','lock_wait_seconds','retry']},
        'runs':{'RGB142':str((pairroot/'control-r1').relative_to(ROOT)),
                'RGB143':os.path.relpath(a.support_run,ROOT)},
        'metrics':{k:r['metrics'] for k,r in results.items()},'paired':counts,'rows':rows,'regions':regionrows,
        'class_rgb_mae_255':{cl:{m:float(np.mean([x[m] for x in classrows if x['class']==cl])) for m in roots} for cl in classes},
        'resources':resources,'fixed':fixed,'inherited_structure_crops':inherited_crops,
        'names':names,'trajectory':[f'frame{i:04d}.png' for i in range(144)],
        'tiles':[{'label':label,'path':os.path.relpath(path,a.output)} for label,path in tiles],
        'independent_float64_psnr_max_error':maxpsnrerror,'seconds':time.monotonic()-start}
    (a.output/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    template=(ROOT/'scripts/standard_support_review.html').read_text();(a.output/'index.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False)))
    print(json.dumps({'status':'built','paired':counts,'resources':{k:{x:y for x,y in v.items() if x!='events'} for k,v in resources.items()},'independent_psnr_max_error':maxpsnrerror}))

if __name__=='__main__':main()
