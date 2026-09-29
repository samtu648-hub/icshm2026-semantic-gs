#!/usr/bin/env python3
"""View synchronized native RGB124 trajectories and quantify unwarped frame changes."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];c=json.loads((ROOT/'experiments/runs/RGB-124-MATCHED-DENSIFICATION-PF/config.yaml').read_text());o=ROOT/c['output'];names=['159.png']+[f'novel-0-{i}.png' for i in range(4)]+['160.png']+[f'novel-1-{i}.png' for i in range(4)]+['161.png'];frames=[];rows=[];sheet=Image.new('RGB',(1400,130*len(names)),'white');d=ImageDraw.Draw(sheet)
for j,n in enumerate(names):
 arrays=[np.asarray(Image.open(o/k/'images'/n)) for k in ['signed','absolute']];diff=np.abs(arrays[1].astype(float)-arrays[0]);display=arrays+[np.clip(diff*4,0,255).astype('uint8')];d.text((5,130*j+2),n+' Signed / Absolute / Difference x4',fill='black')
 for i,im in enumerate(display):sheet.paste(Image.fromarray(im).crop((0,620,1320,989)).resize((466,110)),(466*i,130*j+20))
 frame=Image.new('RGB',(1320,211),'white');fd=ImageDraw.Draw(frame)
 for i,im in enumerate(arrays):frame.paste(Image.fromarray(im).crop((0,620,1320,989)).resize((660,185)),(660*i,26));fd.text((660*i+5,6),n+' '+['signed','absolute'][i],fill='black')
 frames.append(frame);rows.append(dict(view=n,between_arm_mae=float(diff.mean()),max_difference=float(diff.max())))
sheet.save(o/'trajectory-sheet.png');frames[0].save(o/'trajectory.gif',save_all=True,append_images=frames[1:],duration=350,loop=0);(o/'trajectory_summary.json').write_text(json.dumps(rows,indent=2)+'\n');p=o/'index.html';t=p.read_text();t=t.replace('</h1>','</h1><p>同步轨迹：GIF仅展示，原生PNG用于检查</p><img src="trajectory.gif">',1);p.write_text(t)
