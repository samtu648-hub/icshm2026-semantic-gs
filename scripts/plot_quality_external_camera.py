#!/usr/bin/env python3
"""Scientific diagnostic panels and short videos from audited saved camera arrays."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json,CLASSES
PALETTE=np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]],np.uint8)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['trajectory','audit','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    audit=json.loads((a.audit/'result.json').read_text())
    if audit['status']!='passed_independent_saved_camera_audit':raise ValueError('Need successful array audit')
    for path,digest in audit['input_results'].items():
        if sha256(Path(path))!=digest:raise ValueError('Audited result changed')
        q=Path(path);verify_hashes(q.parent,json.loads(q.read_text())['artifacts'])
    output={'experiment_id':'SEM-344-EXTERNAL-CAMERA-PREFLIGHT','audit_sha256':sha256(a.audit/'result.json'),'script_sha256':sha256(Path(__file__)),
            'display_only':'Half-resolution overview/video, no inference resize. No GT. Changed pixels are proposals, not verified corrections.', 'artifacts':{}}
    names=[f['name'] for f in audit['folds'][0]['frames']]
    fig,axes=plt.subplots(3,4,figsize=(16,9),squeeze=False)
    for fold in range(3):
        rows=audit['folds'][fold]['frames'];cases=sorted(rows,key=lambda r:(-r['official_changes'],r['name']));name=cases[0]['name'];stem=Path(name).stem+'.npz'
        with np.load(a.trajectory/f'render-oof-{fold}'/stem) as r,np.load(a.trajectory/f'infer-oof-{fold}'/stem) as z:
            ims=[r['official_rgb'],PALETTE[z['official_raw']],PALETTE[z['official_selected']],PALETTE[z['official_fused']]]
            for col,(im,title) in enumerate(zip(ims,['Frozen RGB','Raw Mask2Former','Sealed 0.05','Calibrated fusion'])):
                ax=axes[fold,col];ax.imshow(im,interpolation='nearest');ax.set_title(f'{title}\nOOF-{fold} / {name}',fontsize=9);ax.axis('off')
            axes[fold,2].text(.02,-.04,f"{cases[0]['official_changes']} changed pixels; correctness unknown",transform=axes[fold,2].transAxes,fontsize=8)
    fig.suptitle('SEM344 official-camera trajectory: maximum proposal frame per fold (display only)',fontsize=13)
    fig.legend(handles=[Patch(facecolor=PALETTE[i]/255,label=n) for i,n in enumerate(CLASSES)],loc='lower center',ncol=5,frameon=False)
    fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(a.output/'trajectory_overview.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,9),squeeze=False)
    for fold in range(3):
        rows=audit['folds'][fold]['frames'];x=np.arange(len(rows))/24
        axes[fold,0].plot(x,[r['official_changes'] for r in rows],label='raw tower -> foundation proposals',color='#aa64be')
        axes[fold,0].set_ylabel(f'OOF-{fold}: pixels');axes[fold,0].legend(fontsize=8)
        for arm in ['raw','selected','fused']:
            axes[fold,1].plot(x,[r['class_counts'][arm][4] for r in rows],label=arm)
        axes[fold,1].set_ylabel('Foundation area (pixels)');axes[fold,1].legend(fontsize=8)
        for ax in axes[fold]:ax.set_xlabel('Time within fixed 24-frame clip (s)');ax.grid(alpha=.2)
    fig.suptitle('Descriptive camera-motion traces; no GT, no temporal-accuracy or flicker claim')
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(a.output/'trajectory_traces.png',dpi=160);plt.close(fig)
    for fold in range(3):
        path=a.output/f'trajectory_oof{fold}.mp4'
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','2640x520','-framerate','24','-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for name in names:
                stem=Path(name).stem+'.npz'
                with np.load(a.trajectory/f'render-oof-{fold}'/stem) as r,np.load(a.trajectory/f'infer-oof-{fold}'/stem) as z:
                    ims=[r['official_rgb'],PALETTE[z['official_raw']],PALETTE[z['official_selected']],PALETTE[z['official_fused']]]
                    canvas=Image.new('RGB',(2640,520),'white');draw=ImageDraw.Draw(canvas)
                    for col,(im,title) in enumerate(zip(ims,['Frozen RGB','Raw','Sealed 0.05','Calibrated fusion'])):
                        canvas.paste(Image.fromarray(im).resize((660,495),Image.Resampling.NEAREST),(col*660,25))
                        draw.text((col*660+8,5),f'OOF-{fold} / {name} / {title} / no GT',fill='black')
                    proc.stdin.write(np.asarray(canvas).tobytes())
        finally:
            proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('Video encoding failed')
    output['artifacts']={p.name:sha256(p) for p in a.output.iterdir() if p.is_file()}
    write_new_json(a.output/'result.json',output)

if __name__=='__main__':main()
