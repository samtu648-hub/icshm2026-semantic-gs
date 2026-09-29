#!/usr/bin/env python3
"""SEM352 scientific diagnostic figure from independently audited counts."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import CLASSES,sha256,write_new_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['diagnosis','audit','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    audit=json.loads((a.audit/'result.json').read_text());path=a.diagnosis/'evidence.json'
    if audit['status']!='passed_independent_saved_error_audit' or sha256(path)!=audit['evidence_sha256']:raise ValueError('Unaccepted evidence')
    e=json.loads(path.read_text());c=json.loads((ROOT/'experiments/runs/SEM-352-SAVED-ERROR-DIAG/config.yaml').read_text());rows={r['name']:r for r in e['per_view']}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    names=c['fixed_report_names'];matrix=np.array([[rows[n]['pairs']['old_sem161->new_raw']['classes'][cl]['delta_iou_pp'] for cl in CLASSES] for n in names])
    ax=axes[0,0];im=ax.imshow(matrix,cmap='RdBu',vmin=-60,vmax=60,aspect='auto')
    ax.set_xticks(range(5),CLASSES,rotation=15,ha='right');ax.set_yticks(range(len(names)),[n.removesuffix('.png') for n in names]);ax.set_title('New raw minus old SEM161: class IoU (pp)')
    for i in range(len(names)):
        for j in range(5):ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',color='white' if abs(matrix[i,j])>35 else 'black')
    fig.colorbar(im,ax=ax,shrink=.8,label='Percentage points')
    ax=axes[0,1];small=['057.png','058.png','064.png'];x=np.arange(3)
    for offset,arm,label,color in [(-.25,'old_sem161','Old SEM161','#727b86'),(0,'new_raw','New raw','#167aaf'),(.25,'new_fused','Fusion','#bf6924')]:
        bars=ax.bar(x+offset,[rows[n]['arms'][arm]['classes']['foundation']['tp'] for n in small],.25,label=label,color=color);ax.bar_label(bars,padding=2)
    support=[rows[n]['arms']['new_raw']['classes']['foundation']['support'] for n in small]
    ax.set_xticks(x,[f'{n[:3]}\nGT foundation: {s}px' for n,s in zip(small,support)]);ax.set(title='Tiny foundation: correctly recovered pixels',ylabel='True-positive pixels',ylim=(0,140));ax.legend()
    ax=axes[1,0];pairs=['old_sem161->new_raw','new_raw->new_fused'];vals=[e['pooled']['pairs'][p]['background_cable_churn'] for p in pairs]
    y=np.arange(2);persistent=np.array([v['persistent'] for v in vals])/1000;added=np.array([v['newly_added'] for v in vals])/1000
    ax.barh(y,persistent,label='Persists from reference',color='#bd9833');ax.barh(y,added,left=persistent,label='New relative to reference',color='#bd5b37')
    for i,v in enumerate(vals):ax.text(persistent[i]+added[i]+2,i,f"{v['persistent']+v['newly_added']:,}",va='center')
    ax.set_yticks(y,['New raw\n(reference: old SEM161)','Fusion\n(reference: new raw)']);ax.invert_yaxis();ax.set(xlabel='GT background predicted as stay_cable (thousands)',xlim=(0,415),title='Less yellow is not necessarily a correction');ax.legend(loc='lower right',fontsize=9)
    ax.text(.02,-.22,'Fusion removes 13,770 raw yellow-background pixels;\nonly 17 become correct background, 13,753 become another wrong class.',transform=ax.transAxes,fontsize=9)
    ax=axes[1,1];v=e['pooled']['pairs']['new_raw->new_fused'];counts=[sum(z['pixels'] for key in ['correction_types','harm_types','wrong_change_types'] for z in v[key] if z['old']=='stay_cable' and z['new']=='tower' and z['gt']==cl) for cl in CLASSES]
    bars=ax.barh(CLASSES,counts,color=['#727b86','#509ff5','#e3bd36','#ce5555','#aa64be']);ax.bar_label(bars,padding=4);ax.set(xlim=(0,42000),xlabel='Pixels grouped by saved GT class',title='Existing raw stay_cable -> fusion tower changes');ax.invert_yaxis()
    fig.suptitle('SEM352 | Historical74 saved-count diagnosis, seed20260805\nNo new predictions, calibration, metric changes or blind-test claim',fontsize=14)
    fig.savefig(a.output/'error_diagnosis.png',dpi=160);plt.close(fig)
    write_new_json(a.output/'result.json',{'experiment_id':c['experiment_id'],'config_sha256':audit['config_sha256'],'audit_sha256':sha256(a.audit/'result.json'),'evidence_sha256':sha256(path),'script_sha256':sha256(Path(__file__)),'figure_sha256':sha256(a.output/'error_diagnosis.png'),'display_only':True})


if __name__=='__main__':main()
