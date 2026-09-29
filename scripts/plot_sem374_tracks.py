#!/usr/bin/env python3
"""Scientific overlay of human-verified SfM observations on fixed photos."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
c=json.loads((ROOT/'experiments/runs/SEM-374-STANDARD-STRUCTURE-DEPTH/config.yaml').read_text());base=ROOT/c['output'];out=base/'track-panels-final';out.mkdir(exist_ok=False);obs=json.loads((ROOT/c['structure_observations']).read_text())
for name in c['diagnostic_views']:
 fig,axes=plt.subplots(1,2,figsize=(14,5.6));photo=np.asarray(Image.open(ROOT/c['scene']/'images'/name));gt=np.asarray(Image.open(ROOT/c['labels']/name));pal=np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]],np.uint8)
 for ax,img in zip(axes,[photo,pal[gt]]):
  ax.imshow(img)
  for role,color,marker in [('fit','lime','+'),('check','cyan','x')]:
   xy=np.array([v['xy'] for v in obs[name] if v['role']==role])-.5
   if len(xy):ax.scatter(xy[:,0],xy[:,1],s=16,c=color,marker=marker,linewidths=.6,label=role+' depth targets ('+str(len(xy))+')')
  ax.set_xlim(0,photo.shape[1]);ax.set_ylim(photo.shape[0],0);ax.set_xlabel('image column (px)');ax.set_ylabel('image row (px)');ax.legend(loc='lower left',fontsize=7)
 axes[0].set_title(name+' photo + multiview SfM targets');axes[1].set_title('Human labels: deck / stay_cable / tower / foundation')
 fig.suptitle('SEM374 | labels select deck, tower, foundation only | SfM targets are not geometry GT',fontsize=11);fig.tight_layout();fig.savefig(out/(Path(name).stem+'.png'),dpi=120);plt.close(fig)
