#!/usr/bin/env python3
"""Plot recorded RGB141 evidence; no rendering, training, or model mutation."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'outputs/quality/rgb141/diagnostic-r1'
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from matplotlib.patches import Rectangle
    data=json.loads((out/'result.json').read_text())
    fig,axes=plt.subplots(8,2,figsize=(14,15),sharex=True,layout='constrained')
    for i,item in enumerate(data['summary']):
        rows=[r for r in data['rows'] if r['id']==item['id']]
        x=[int(Path(r['view']).stem) for r in rows]
        axes[i,0].scatter(x,[r['normalized_evidence'] for r in rows],s=5,color='#17688b')
        axes[i,1].scatter(x,[r['removal_mae_delta_255'] for r in rows],s=5,color='#bc572e')
        regions=[r['region'].removeprefix('159-') for r in data['selection'] if item['id'] in r['ids']]
        axes[i,0].set(ylabel=f"ID {item['id']}\n"+'\n'.join(regions),ylim=(-1.05,1.05))
        axes[i,0].yaxis.label.set_size(8)
        axes[i,1].set_yscale('symlog',linthresh=1e-4)
        axes[i,1].set_ylabel('Removal delta MAE\nsymlog; linear within ±1e-4',fontsize=8)
        for ax in axes[i]:
            ax.axhline(0,color='0.5',linewidth=.5);ax.grid(alpha=.15)
    axes[0,0].set_title('Normalized opacity-marginal evidence (positive: increasing opacity helps)')
    axes[0,1].set_title('Finite removal (positive: removal hurts RGB fitting)')
    for ax in axes[-1]:ax.set_xlabel('Training view ID (all400; not independent geometry evidence)')
    fig.suptitle('RGB141 frozen RGB140 | Selected on view 159 | Mixed cross-view effects',fontsize=15)
    fig.savefig(out/'cross-view-evidence.png',dpi=160);plt.close(fig)
    photo=np.asarray(Image.open(ROOT/'outputs/quality/all_data/RGB-120-r2/scene/images/159.png'))
    rendered=np.asarray(Image.open(ROOT/'outputs/quality/rgb140/full-r2/evaluation/images/159.png'))
    fixed=json.loads((ROOT/'experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml').read_text())
    fig,axes=plt.subplots(1,3,figsize=(18,5),layout='constrained')
    axes[0].imshow(photo);axes[0].set_title('159 Photo | fixed diagnostic regions')
    axes[1].imshow(rendered);axes[1].set_title('RGB140 | saved quantized RGB')
    heat=axes[2].imshow(np.abs(photo.astype(float)-rendered).mean(2),vmin=0,vmax=30,cmap='magma')
    axes[2].set_title('Absolute RGB residual | mean channels | clipped at 30')
    fig.colorbar(heat,ax=axes[2],label='MAE (0-255)',shrink=.8)
    for key,color in [('159-shore','#18ced2'),('159-far-occlusion','#7ff044'),('159-water-ghost','#fdb347')]:
        for x0,y0,x1,y1 in fixed['regions'][key]['boxes']:
            axes[0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=color,linewidth=1.4))
            axes[0].text(x0,y0-7,key.removeprefix('159-'),color=color,fontsize=8)
    for ax in axes:ax.axis('off')
    fig.savefig(out/'159-residual.png',dpi=180);plt.close(fig)
    print('Plotted existing frozen diagnostics; no new GPU work.')

if __name__=='__main__':main()
