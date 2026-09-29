#!/usr/bin/env python3
"""SEM351 scientific display from accepted arrays, no model or GT access."""
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
    p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads((ROOT/'experiments/runs/SEM-351-FINAL-CAMERA-ACCEPTANCE/config.yaml').read_text())
    audit=json.loads((a.batch/'audit/result.json').read_text())
    if audit['status']!='passed_independent_final_camera_audit':raise ValueError('Accepted audit required')
    for stage,digest in audit['input_results'].items():
        path=a.batch/stage/'result.json'
        if sha256(path)!=digest:raise ValueError('Audit source changed')
        verify_hashes(path.parent,json.loads(path.read_text())['artifacts'])
    def arrays(name):
        stem=Path(name).stem+'.npz'
        with np.load(a.batch/'render'/stem) as r,np.load(a.batch/'infer'/stem) as z:
            return [r['official_rgb'],PALETTE[z['official_raw']],PALETTE[z['official_selected']],PALETTE[z['official_fused']]]
    titles=['Frozen RGB','Final Mask2Former raw','Sealed 0.05','Calibrated fusion']
    names=['frame0060.png','frame0071.png','frame0083.png']
    fig,axes=plt.subplots(3,4,figsize=(16,10),layout='constrained')
    for row,name in enumerate(names):
        for ax,im,title in zip(axes[row],arrays(name),titles):
            ax.imshow(im,interpolation='nearest');ax.set_title(f'{name} | {title}',fontsize=9);ax.axis('off')
    fig.suptitle('SEM351 | Final model on fixed official-camera trajectory\nNo GT; yellow regions are predictions, not verified errors',fontsize=13)
    fig.legend(handles=[Patch(facecolor=PALETTE[i]/255,label=n) for i,n in enumerate(CLASSES)],loc='outside lower center',ncol=5,frameon=False)
    fig.savefig(a.output/'trajectory_overview.png',dpi=160);plt.close(fig)
    rows={v['name']:v for v in audit['views']};x=np.arange(len(c['trajectory_names']))
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,cl in zip(axes.ravel()[:3],[2,3,4]):
        for arm,color in [('raw','#167aaf'),('selected','#268761'),('fused','#bf6924')]:
            ax.plot(x,[rows[n]['class_counts'][arm][cl] for n in c['trajectory_names']],label=arm,color=color)
        ax.set(title=CLASSES[cl]+' predicted area',xlabel='Frame offset from0060',ylabel='Pixels');ax.legend()
    axes[1,1].plot(x,[rows[n]['official_changes'] for n in c['trajectory_names']],color='#82689d')
    axes[1,1].set(title='Raw -> 0.05 proposals',xlabel='Frame offset from0060',ylabel='Pixels (correctness unknown)')
    fig.suptitle('SEM351 | Descriptive counts only; camera motion prevents a flicker/accuracy interpretation')
    fig.savefig(a.output/'trajectory_counts.png',dpi=160);plt.close(fig)
    for fps,suffix in [(24,'native_rate'),(6,'slow4x')]:
        path=a.output/f'trajectory_{suffix}.mp4'
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','2640x524','-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for name in c['trajectory_names']:
                canvas=Image.new('RGB',(2640,524),'white');draw=ImageDraw.Draw(canvas)
                for col,(im,title) in enumerate(zip(arrays(name),titles)):
                    canvas.paste(Image.fromarray(im).resize((660,495),Image.Resampling.NEAREST),(col*660,29))
                    draw.text((col*660+6,7),f'{name} | {title} | {suffix} | no GT',fill='black')
                proc.stdin.write(np.asarray(canvas).tobytes())
        finally:
            proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('Video failed')
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height,r_frame_rate','-of','json',str(path)],text=True))['streams'][0]
        if int(probe['nb_read_frames'])!=24:raise ValueError('Incomplete video')
    r={'experiment_id':c['experiment_id'],'config_sha256':audit['config_sha256'],'audit_sha256':sha256(a.batch/'audit/result.json'),
       'script_sha256':sha256(Path(__file__)),'fixed_display_names':names,'display_only':'Half-resolution video; native inference unchanged. Same24 frames at24fps and slowed6fps. No GT or quality selection.',
       'artifacts':{p.name:sha256(p) for p in a.output.iterdir() if p.is_file()}}
    write_new_json(a.output/'result.json',r);print(json.dumps(r))


if __name__=='__main__':main()
