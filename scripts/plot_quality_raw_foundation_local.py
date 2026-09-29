#!/usr/bin/env python3
"""SEM343 native local evidence figures; no inference or prediction edits."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch,Rectangle

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json,CLASSES
from src.quality_static_training import TrainingLabels
from scripts.diagnose_quality_raw_fusion import read_image

# Same semantic display palette as the existing research report (display only).
PALETTE=np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]],np.uint8)
COST=np.array([[245,245,245],[30,150,90],[200,35,35],[220,150,25]],np.uint8)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve()
    if not a.output.is_relative_to(ROOT/'outputs/quality/raw_fusion'):p.error('Use isolated output directory')
    a.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((a.source/'result.json').read_text());replay=json.loads((a.replay/'result.json').read_text())
    if replay['status']!='completed_fixed_check_independent_replay' or replay['replay_source_sha256']!=sha256(a.source/'result.json') or replay['evaluation']!=source['evaluation']:
        raise ValueError('Source has no matching independent replay')
    verify_hashes(a.source,source['artifacts'])
    cfg=json.loads((ROOT/'experiments/runs/SEM-343-RAW-FOUNDATION-GATE/config.yaml').read_text())
    pc=json.loads((ROOT/cfg['preflight_config']).read_text())
    parent_dir=ROOT/cfg['comparison'];parent=json.loads((parent_dir/'result.json').read_text())
    export_dir=ROOT/pc['rgb_export'];export=json.loads((export_dir/'result.json').read_text());rgb={v['name']:v['sha256'] for v in export['rendered_images']}
    manifest=json.loads((ROOT/pc['label_manifest']).read_text())
    audit={'experiment_id':'SEM-343-RAW-FOUNDATION-GATE','selected_margin':0.05,'source_sha256':sha256(a.source/'result.json'),
           'replay_sha256':sha256(a.replay/'result.json'),'script_sha256':sha256(Path(__file__)),
           'meaning':'post-hoc native local diagnosis; examples do not select thresholds or certify new-view/trajectory quality', 'roles':{}}
    for role in ('calibration','check'):
        views=json.loads((a.source/f'{role}_per_view.json').read_text())
        cases=sorted([{'view':n,'partition':v['partition'],**c} for n,v in views.items() for c in v['components']['harmed']],
                     key=lambda c:(-c['pixels'],c['view'],c['bbox_xyxy'][1],c['bbox_xyxy'][0]))[:6]
        if not cases:continue
        reader=TrainingLabels(a.artifact_root/'r2/sem_001/labels',list(dict.fromkeys(c['view'] for c in cases)),manifest,[1320,989])
        fig,axes=plt.subplots(len(cases),5,figsize=(13,2.55*len(cases)),squeeze=False)
        for row,c in enumerate(cases):
            name=c['view'];x0,y0,x1,y1=c['bbox_xyxy'];cx=(x0+x1)//2;cy=(y0+y1)//2
            left=max(0,min(cx-32,1320-64));top=max(0,min(cy-32,989-64));right=left+64;bottom=top+64
            verify_hashes(export_dir/'images',{name:rgb[name]})
            rel=f'{role}/mask2former_6k/raw/{name}';verify_hashes(parent_dir,{rel:parent['artifacts'][rel]})
            images=[read_image(export_dir/'images'/name),PALETTE[reader.read(name)],PALETTE[read_image(parent_dir/rel)],
                    PALETTE[read_image(a.source/role/'predictions'/name)],COST[read_image(a.source/role/'changes'/name)]]
            for col,(image,title) in enumerate(zip(images,('Frozen rendered RGB','Ground truth','Mask2Former raw','Sealed margin0.05','Change type'))):
                ax=axes[row,col];ax.imshow(image[top:bottom,left:right],interpolation='nearest',extent=(left,right,bottom,top))
                ax.add_patch(Rectangle((x0-1,y0-1),x1-x0+2,y1-y0+2,fill=False,edgecolor='black',linewidth=0.8))
                ax.set_title(title,fontsize=9);ax.set_xlabel('x (native pixels)',fontsize=7);ax.set_ylabel('y (native pixels)',fontsize=7);ax.tick_params(labelsize=7)
            axes[row,0].text(0,1.24,f"{name} / {c['partition']} / {c['pixels']} harmed tower pixels",transform=axes[row,0].transAxes,fontsize=9)
            c['display_crop_xyxy']=[left,top,right,bottom]
        handles=[Patch(facecolor=PALETTE[i]/255,label=n) for i,n in enumerate(CLASSES)]
        handles += [Patch(facecolor=COST[i]/255,label=n) for i,n in enumerate(('unchanged','corrected','harmed','still wrong')) if i]
        fig.legend(handles=handles,ncol=4,loc='lower center',fontsize=8,frameon=False)
        fig.suptitle(f'SEM343 | {role} | largest harmed components (8-connected)\nHistorical data; native64px crops; black boxes mark components. No threshold tuning.',fontsize=12,y=.994)
        fig.subplots_adjust(top=.94,bottom=.065,hspace=.72,wspace=.38)
        output=a.output/f'{role}_local_evidence.png';fig.savefig(output,dpi=160);plt.close(fig)
        audit['roles'][role]={'cases':cases,'decoded_label_views':sorted(reader.accessed),'figure_sha256':sha256(output)}
    write_new_json(a.output/'result.json',audit)
    print('Saved independent-replay-bound local evidence figures')


if __name__=='__main__':main()
