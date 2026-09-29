#!/usr/bin/env python3
"""Native-camera synchronized RGB trajectory, four arms, no synthetic content."""
import argparse,json,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve();out=run/'comparison';names=json.loads((run/'export/result.json').read_text())['trajectory_names'];assert len(names)==144
base=ROOT/'outputs/quality/backbone_comparison';arms=[('Scaffold UID0',base/'scaffold/export/images'),('Standard',base/'standard/export/images'),('Mip',base/'mip/export/images'),('Scaffold no appearance',run/'export/images')]
video=out/'trajectory.mp4';assert not video.exists();cmd=['ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1320x1038','-framerate','12','-i','-','-c:v','libx264','-crf','18','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(video)];proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
try:
 for n in names:
  frame=Image.new('RGB',(1320,1038),'white');d=ImageDraw.Draw(frame)
  for j,(label,path) in enumerate(arms):
   x=j%2*660;y=j//2*519;d.text((x+5,y+5),n+' '+label,fill='black');frame.paste(Image.open(path/n).resize((660,494)),(x,y+25))
  proc.stdin.write(frame.tobytes())
 proc.stdin.close();assert proc.wait()==0
except BaseException:
 proc.kill();proc.wait();raise
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(video)],text=True));s=probe['streams'][0];assert s['nb_read_frames']=='144' and s['r_frame_rate']=='12/1';(out/'video_audit.json').write_text(json.dumps(probe,indent=2)+'\n');print('144 synchronized frames; 12fps; 1320x1038')
