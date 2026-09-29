#!/usr/bin/env python3
"""SEM345 unchanged outputs: historical108 GT risk + saved no-GT trajectory."""
import argparse,json,shutil,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy import ndimage
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json
from src.quality_static_training import TrainingLabels
from src.quality_cable_diagnostic import evidence_masks,truth_counts,largest_component,joint_evidence,detached_cable
from src.quality_fusion import confusion
CONFIG='experiments/runs/SEM-345-STAY-CABLE-BACKGROUND-DIAG/config.yaml'
CODE=['scripts/diagnose_quality_cable_background.py','src/quality_cable_diagnostic.py','src/quality_static_training.py','src/quality_protocol.py','src/quality_fusion.py']
def read(p):return json.loads(p.read_text())
def image(p):
    with Image.open(p) as im:return np.array(im)
def checked(base,r,rel):
    verify_hashes(base,{rel:r['artifacts'][rel]});return base/rel

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output=a.output.resolve()
    if not a.output.is_relative_to(ROOT/'outputs/quality/cable_diagnostic'):p.error('Use new cable_diagnostic output')
    a.output.mkdir(parents=True,exist_ok=False);cfg=read(ROOT/CONFIG);start=time.monotonic()
    r={k:cfg[k] for k in ['experiment_id','git_commit','seed','dataset_manifest','split_id']};r.update(status='started',started_utc=datetime.now(timezone.utc).isoformat(),config_sha256=sha256(ROOT/CONFIG),code_sha256={n:sha256(ROOT/n) for n in CODE},new_training=False,new_model_inference=False,new_predictions=False,threshold_selection=False,check_labels_decoded=False,development_labels_decoded=False,artifacts={})
    try:
        verify_hashes(ROOT,cfg['input_hashes']);parent=read(ROOT/cfg['comparison']/'result.json');verify_hashes(ROOT,parent['input_result_hashes'])
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=cfg['git_commit']:raise ValueError('Commit mismatch')
        pc=read(ROOT/cfg['preflight_config']);export_root=ROOT/pc['rgb_export'];ex=read(export_root/'result.json');rgb={x['name']:x['sha256'] for x in ex['rendered_images']};frozen={x['path']:x['sha256'] for x in read(ROOT/cfg['frozen_manifest'])['artifacts']}
        def guard():
            if shutil.disk_usage(ROOT).free/2**30<cfg['minimum_free_gib']:raise OSError('Disk reserve reached')
            verify_hashes(export_root/'images',rgb);verify_hashes(a.artifact_root,frozen)
        guard();sources={}
        for rel in parent['input_result_hashes']:
            base=(ROOT/rel).parent;x=read(base/'result.json');kind=None
            if base.name.startswith('full-oof-'):kind='five'
            elif base.name.startswith('SEM-330-Scaffold-'):kind='static'
            elif base.name.startswith('SEM-331-Scaffold-stay_cable-'):kind='binary'
            if kind:sources.setdefault(x['access']['partition'],{})[kind]=(base,x)
        if len(sources)!=3 or any(set(x)!=set(['five','static','binary']) for x in sources.values()):raise ValueError('Missing source folds')
        names=parent['roles']['calibration'];labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',names,read(ROOT/pc['label_manifest']),[1320,989]);old=read(ROOT/cfg['comparison']/'calibration/per_view.json')
        for n in [CONFIG,*CODE]:
            q=a.output/'code'/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,q)
        write_new_json(a.output/'started.json',r)
        per_view=[];groups={};examples=[];joint=np.zeros((5,2,2,4,5),np.int64)
        for idx,name in enumerate(names):
            part=old[name]['partition'];src=sources[part]
            if any(name not in x['access']['prediction'] for _,x in src.values()):raise ValueError('Not own-fold OOF')
            maps={}
            for mode in ['raw','calibrated']:
                rel=f'calibration/mask2former_6k/{mode}/{name}';maps[mode]=image(checked(ROOT/cfg['comparison'],parent,rel))
            raw=maps['raw'];base,x=src['static'];static=image(checked(base,x,'predictions/'+name))
            probs={}
            for kind in ['five','binary']:
                base,x=src[kind]
                with np.load(checked(base,x,'probabilities/'+Path(name).stem+'.npz'),allow_pickle=False) as z:probs[kind]=z['probabilities']
            if not np.array_equal(raw,probs['five'].argmax(0)):raise ValueError('Raw mismatch')
            target=labels.read(name);binary=probs['binary'].argmax(0).astype(np.uint8);masks=evidence_masks(raw,static,binary);j=joint_evidence(raw,static,binary,probs['five'][2],target,cfg['probability_bins']);joint+=j
            group=groups.setdefault(part,{'joint':np.zeros_like(joint),'confusions':{k:np.zeros((5,5),np.int64) for k in ['raw','calibrated','static']}});group['joint']+=j
            cm={k:confusion(target,v) for k,v in {'raw':raw,'calibrated':maps['calibrated'],'static':static}.items()}
            for k in ['raw','calibrated']:
                if cm[k].tolist()!=old[name]['metrics']['mask2former_6k/'+k]['confusion']:raise ValueError('Historical confusion mismatch')
            for k in cm:group['confusions'][k]+=cm[k]
            counts={k:truth_counts(m,target).tolist() for k,m in masks.items()};lane=raw==2
            if j.sum()!=lane.sum() or not np.array_equal(j.sum(axis=(0,1,2,3)),truth_counts(lane,target)):raise ValueError('Joint audit mismatch')
            # Independently sum tensor partitions; this catches cross-field mask errors.
            expected={'detached':j[:,:,1].sum(axis=(0,1,2)),'static_background':j[0].sum(axis=(0,1,2)),
                      'binary_background':j[:,0].sum(axis=(0,1,2)),'both_background':j[0,0].sum(axis=(0,1)),
                      'detached_both_background':j[0,0,1].sum(axis=0)}
            if any(expected[k].tolist()!=counts[k] for k in counts):raise ValueError('Boolean/tensor partitions differ')
            distance=ndimage.distance_transform_edt(target!=2) if (target==2).any() else np.full(raw.shape,np.inf)
            distance_counts=[]
            for lo,hi in zip(cfg['distance_to_gt_cable_bins_pixels'][:-1],cfg['distance_to_gt_cable_bins_pixels'][1:]):
                distance_counts.append(int((lane&(target==0)&(distance>lo)&(distance<=hi)).sum()))
            er=largest_component(lane&(target==0));loss=largest_component(masks['detached_both_background']&(target==2))
            for kind,item in [('background_error',er),('true_cable_at_risk',loss)]:
                if item:examples.append({'kind':kind,'name':name,'partition':part,**item})
            per_view.append({'name':name,'partition':part,'raw_cable_truth':truth_counts(lane,target).tolist(),'evidence_truth':counts,
                'confidence_truth':j.sum(axis=(0,1,2)).tolist(),'background_error_gt_distance_bins':distance_counts,'confusions':{k:v.tolist() for k,v in cm.items()}})
            if idx%24==0:print(json.dumps({'calibration':idx+1,'total':len(names)}),flush=True)
        r['calibration']={'views':len(names),'joint':joint.tolist(),'per_fold':{k:{'joint':v['joint'].tolist(),'confusions':{n:cm.tolist() for n,cm in v['confusions'].items()}} for k,v in groups.items()},'raw_cable_truth':joint.sum(axis=(0,1,2,3)).tolist(),'evidence_truth':{k:np.sum([x['evidence_truth'][k] for x in per_view],axis=0).tolist() for k in cfg['evidence_masks']},'confidence_truth':joint.sum(axis=(0,1,2)).tolist(),'background_error_gt_distance_bins':np.sum([x['background_error_gt_distance_bins'] for x in per_view],axis=0).tolist()}
        r['calibration_examples']=sorted([e for e in examples if e['kind']=='background_error'],key=lambda e:(-e['pixels'],e['name']))[:3]+sorted([e for e in examples if e['kind']=='true_cable_at_risk'],key=lambda e:(-e['pixels'],e['name']))[:1]
        write_new_json(a.output/'calibration_per_view.json',per_view)
        ta=read(ROOT/cfg['trajectory_audit']);r['trajectory']=[];r['trajectory_examples']=[]
        for f in ta['folds']:
            fold=f['fold'];base=ROOT/cfg['trajectory'];rr=read(base/f'render-oof-{fold}/result.json');ir=read(base/f'infer-oof-{fold}/result.json');items=[]
            for entry in f['frames']:
                name=entry['name'];rel=Path(name).stem+'.npz'
                with np.load(checked(base/f'render-oof-{fold}',rr,rel),allow_pickle=False) as z:static=z['static']
                with np.load(checked(base/f'infer-oof-{fold}',ir,rel),allow_pickle=False) as z:
                    raw=z['raw'];fused=z['fused'];prob=z['five'][2]
                    if not np.array_equal(raw==2,z['selected']==2) or not np.array_equal(z['official_raw']==2,z['official_selected']==2):raise ValueError('Sealed0.05 changed stay_cable')
                    d=detached_cable(raw);cc=largest_component(d);items.append({'name':name,'raw_cable_pixels':int((raw==2).sum()),'detached_pixels':int(d.sum()),'detached_static_classes':np.bincount(static[d],minlength=5).tolist(),'detached_retained_by_fusion':int((d&(fused==2)).sum()),'detached_confidence_bins':np.histogram(prob[d],bins=cfg['probability_bins'])[0].tolist(),'largest_detached':cc})
            r['trajectory'].append({'fold':fold,'frames':items});best=sorted([x for x in items if x['largest_detached']],key=lambda x:(-x['largest_detached']['pixels'],x['name']))
            if best:r['trajectory_examples'].append({'fold':fold,'name':best[0]['name'],**best[0]['largest_detached']})
        guard();verify_hashes(ROOT,cfg['input_hashes']);verify_hashes(ROOT,r['code_sha256'])
        if sha256(ROOT/CONFIG)!=r['config_sha256']:raise ValueError('Config changed')
        r.update(status='completed_read_only_stay_cable_diagnostic',calibration_labels_accessed=sorted(labels.accessed),rgb_exports_verified=219,frozen_verified=18)
    except BaseException as exc:r.update(status='failed_cable_diagnostic',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        r['seconds']=time.monotonic()-start;r['artifacts']={p.name:sha256(p) for p in a.output.glob('*.json') if p.name!='result.json'};write_new_json(a.output/'result.json',r)
        print(json.dumps({'status':r['status'],'seconds':r['seconds']}),flush=True)

if __name__=='__main__':main()
