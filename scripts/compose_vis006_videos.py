#!/usr/bin/env python3
"""Encode four faithful presentation videos from sealed VIS006 native PNGs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.semantic_render_domain import SEMANTIC_PALETTE


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(path.read_text())
def png(path):
    with Image.open(path) as im:return np.asarray(im).copy()
def check(path,h):
    if sha(path)!=h:raise ValueError('Sealed input differs: '+str(path))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--artifact-root',type=Path,required=True);args=p.parse_args()
    run=args.run_dir;result=read(run/'result.json');base=read(run/'base/manifest.json')
    if result['frame_count']!=144 or not result['three_original_probe_frames_exact']:raise ValueError('Incomplete render evidence')
    check(run/'base/manifest.json',result['base_manifest_sha256'])
    base_rows={r['pose']:r for r in base['outputs']};frames=[];old_exact=True
    for r in result['frames']:
        name=r['name'];record=base_rows[name]
        for value in record['outputs'].values():check(run/'base'/value['path'],value['sha256'])
        for value in r['outputs'].values():check(run/value['path'],value['sha256'])
        rgb=png(run/'base/rgb'/name);a=png(run/'base/semantic'/name);b=png(run/r['outputs']['sem326']['path']);c=png(run/r['outputs']['sem329']['path'])
        if rgb.shape!=(989,1320,3) or any(x.shape!=(989,1320) for x in (a,b,c)):raise ValueError('Native shape changed')
        if any(not np.array_equal(a==k,x==k) for k in (1,2,3) for x in (b,c)):raise ValueError('Protected class mismatch')
        old=args.artifact_root/'results/frozen_scaffold_sem001/flythrough_v1/renders/rgb'/name
        if not old.is_file() or not np.array_equal(rgb,png(old)):old_exact=False
        frames.append((name,rgb,a,b,c))
    rows=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    variants=[('rgb-reconstruction','纯RGB重建 · 冻结Scaffold',1),('sem329-semantic','SEM329 · 纯语义新视角',1),('rgb-sem329-overlay','RGB与SEM329叠加 · 同视角',2),('sem161-sem326-sem329','SEM161 / SEM326 / SEM329 · 同轨迹对比',3)]
    for kind,title,ncols in variants:
        head=0 if kind=='rgb-reconstruction' else 78;w=1320*ncols;h=989+head;h+=h%2
        video=run/(kind+'.mp4')
        if video.exists():raise FileExistsError('Refusing to overwrite encoded video')
        cmd=['ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate','24','-i','pipe:0','-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-threads','4',str(video)]
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            for i,(name,rgb,a,b,c) in enumerate(frames):
                if kind=='rgb-reconstruction':tiles=[rgb];titles=[]
                elif kind=='sem329-semantic':tiles=[SEMANTIC_PALETTE[c]];titles=['SEM329 | current research candidate']
                elif kind=='rgb-sem329-overlay':
                    overlay=rgb.copy();fg=c>0;overlay[fg]=np.rint(.45*rgb[fg]+.55*SEMANTIC_PALETTE[c[fg]]).astype(np.uint8)
                    tiles=[rgb,overlay];titles=['RGB-030-FULL | frozen Scaffold RGB','SEM329 | semantic overlay (alpha 0.55)']
                else:tiles=[SEMANTIC_PALETTE[a],SEMANTIC_PALETTE[b],SEMANTIC_PALETTE[c]];titles=['SEM161 | formal baseline','SEM326 | first-source review','SEM329 | second-source veto']
                canvas=Image.new('RGB',(w,h),(247,247,242));draw=ImageDraw.Draw(canvas)
                for j,tile in enumerate(tiles):
                    canvas.paste(Image.fromarray(tile),(1320*j,head))
                    if head:
                        draw.text((1320*j+18,9),titles[j]+f' / {i:03d}',font=font,fill=(31,55,46))
                        xx=1320*j+18
                        for label,color in zip(['background','deck','stay_cable','tower','foundation'],SEMANTIC_PALETTE):
                            draw.rectangle((xx,47,xx+14,61),fill=tuple(map(int,color)));draw.text((xx+21,43),label,font=small,fill=(31,55,46));xx+=int(draw.textlength(label,font=small))+48
                if i==0 or i==72 or i==143:
                    canvas.save(run/(kind+f'-frame{i:04d}.png'))
                if i==0:poster=run/(kind+'-poster.png');canvas.save(poster)
                process.stdin.write(canvas.tobytes())
            process.stdin.close();error=process.stderr.read().decode();code=process.wait()
            if code:raise RuntimeError(error)
        except Exception:
            process.kill();process.wait();raise
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,nb_frames,r_frame_rate,codec_name,pix_fmt','-show_entries','format=duration,size','-of','json',str(video)],text=True));stream=info['streams'][0]
        if int(stream['nb_frames'])!=144 or stream['r_frame_rate']!='24/1' or abs(float(info['format']['duration'])-6)>1e-9:raise ValueError('Video timing differs')
        rows.append({'title':title,'path':video.name,'poster':poster.name,'sha256':sha(video),'poster_sha256':sha(poster),'frame_count':144,'fps':24,'duration_seconds':6,'width':w,'height':h,'codec':'H.264','pixel_format':'yuv420p','bytes':video.stat().st_size})
        print(json.dumps({'encoded':kind,'bytes':video.stat().st_size}),flush=True)
    manifest={'experiment_id':'VIS-006-SEM329-FLYTHROUGH','render_result_sha256':sha(run/'result.json'),'composer_sha256':sha(Path(__file__)),'palette':SEMANTIC_PALETTE.tolist(),'frame_count':144,'videos':rows,'native_rgb_exact_to_VIS001_all144':old_exact,'native_tiles_unscaled':True,'protected_classes_exact':True,'target_gt_decodes':0,'quality_scores_generated':False,'padding':'bottom row only where needed for even H.264 height; native pixels untouched before lossy codec','codec_note':'MP4 is a lossy presentation copy; original RGB and integer semantic PNGs are the scientific record.'}
    (run/'media_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'native_rgb_exact_to_VIS001_all144':old_exact,'videos':len(rows)}),flush=True)


if __name__=='__main__':main()
