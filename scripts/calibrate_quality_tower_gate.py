#!/usr/bin/env python3
"""SEM353: fixed108 tower gate, all finite curves; no selection or deployment."""
import argparse,json,shutil,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json
from src.quality_mask2former_final import FinalContext
from src.quality_static_training import TrainingLabels,semantic_access
from src.quality_semantic_audit import metrics
from src.quality_fusion import confusion
from src.quality_tower_gate import apply_tower_gate
from scripts.diagnose_quality_raw_fusion import read_probability,read_image
CONFIG='experiments/runs/SEM-353-TOWER-GATE-CALIBRATION/config.yaml'


def exact_scores(cm):
    ious=[]
    for k in range(5):
        tp=int(cm[k][k]);den=int(sum(cm[k])+sum(row[k] for row in cm)-tp)
        if not den:raise ValueError('Expected all classes in pooled/fold matrix')
        ious.append(Fraction(tp,den))
    return sum(ious)/5,ious[2],ious[3]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['artifact-root','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--replay-from',type=Path)
    a=p.parse_args();a.output=a.output.resolve()
    if not a.output.is_relative_to(ROOT/'outputs/quality/raw_fusion'):p.error('Isolated raw_fusion output required')
    a.output.mkdir(parents=True,exist_ok=False);read=lambda p:json.loads(p.read_text());c=read(ROOT/CONFIG);start=time.monotonic()
    r={k:c[k] for k in ['experiment_id','git_commit','seed','dataset_manifest','split_id']}
    r.update(status='started',started_utc=datetime.now(timezone.utc).isoformat(),config_sha256=sha256(ROOT/CONFIG),
             code_sha256=c['code_sha256'],new_training=False,new_inference=False,new_prediction_outputs=False,
             check_labels_decoded=0,development_labels_decoded=0,selected_margin=None,automatic_adoption=False,
             pixel_replay=bool(a.replay_from))
    try:
        verify_hashes(ROOT,c['input_hashes']);verify_hashes(ROOT,c['code_sha256'])
        for name in [CONFIG,*c['code_sha256']]:
            dest=a.output/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        write_new_json(a.output/'started.json',r)
        ctx=FinalContext(ROOT,a.artifact_root,ROOT/c['final_batch'])
        parent=read(ROOT/c['comparison']/'result.json');info=read(ROOT/c['comparison']/'calibration/per_view.json')
        sel=read(ROOT/c['selected_calibration_counts']);seal=read(ROOT/c['selection'])
        names=[n for ns in c['views'].values() for n in ns]
        if len(names)!=108 or len(set(names))!=108 or set(names)!=set(parent['roles']['calibration']) or set(names)!=set(seal['roles']['calibration']) or set(names)&set(parent['roles']['check']) or set(names)&set(ctx.access['prediction']):raise ValueError('Wrong calibration role')
        labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',names,ctx.parent.label_manifest,[1320,989])
        sources={}
        rgb_access=read(ctx.parent.rgb_run/'access.json')
        for fold,fold_names in c['views'].items():
            expected=semantic_access(ctx.parent.split,ctx.parent.oof,rgb_access,fold)
            pair=[]
            for root,filename,kind in [(ROOT/c['batch']/('full-'+fold),'model.pt','five'),(ROOT/c['static_pattern'].format(fold=fold),'semantic_field.pt','static')]:
                src=read(root/'result.json');access=src['access']
                if access['partition']!=fold or any(access[k]!=expected[k] for k in ['train','prediction']) or set(fold_names)&set(access['train']) or not set(fold_names)<=set(access['prediction']):raise ValueError('Target not excluded from same-fold semantic training')
                identity=src['binding']['rgb_identity' if kind=='five' else 'identity']
                if identity!=ctx.binding['rgb_identity'] or src['binding']['seed']!=c['seed']:raise ValueError('Different RGB or seed')
                if kind=='five' and src['trained_steps']!=6000:raise ValueError('Wrong expert checkpoint step')
                verify_hashes(root,{filename:src['artifacts'][filename]});pair.append((root,src))
            if any(info[n]['partition']!=fold for n in fold_names):raise ValueError('Original partition differs')
            sources[fold]=pair
        groups={};per=[]
        for fold,fold_names in c['views'].items():
            (five_root,five_src),(static_root,static_src)=sources[fold]
            acc={b:{tag:np.zeros((5,5),np.int64) for tag in c['candidates']} for b in ['raw','selected']}
            counts={tag:np.zeros(5,np.int64) for tag in c['candidates']}
            for index,name in enumerate(fold_names):
                ctx.guard()
                raw_rel='calibration/mask2former_6k/raw/'+name
                verify_hashes(ROOT/c['comparison'],{raw_rel:parent['artifacts'][raw_rel]})
                raw=read_image(ROOT/c['comparison']/raw_rel)
                static_rel='predictions/'+name;verify_hashes(static_root,{static_rel:static_src['artifacts'][static_rel]});static=read_image(static_root/static_rel)
                five=read_probability(five_root,five_src,'probabilities/'+Path(name).stem+'.npz',5)
                if not np.array_equal(raw,five.argmax(0)):raise ValueError('Raw argmax differs')
                selected=raw.copy();selected[(raw==3)&(static==4)&(five[4]>=.1)&((five[3]-five[4])<=.05)]=4
                bases={'raw':raw,'selected':selected}
                # All candidates generated before decoding this target's GT.
                proposals={}
                for tag,margin in c['candidates'].items():
                    if a.replay_from:
                        ids=np.flatnonzero((raw.reshape(-1)==2)&(static.reshape(-1)==3))
                        gap=np.subtract(five[2].reshape(-1)[ids],five[3].reshape(-1)[ids],dtype=np.float32)
                        take=ids[:0] if margin is None else ids[gap<=margin]
                        outputs={b:base.copy() for b,base in bases.items()}
                        for out in outputs.values():out.reshape(-1)[take]=3
                    else:outputs={b:apply_tower_gate(raw,base,static,five,margin) for b,base in bases.items()}
                    proposals[tag]=outputs
                gt=labels.read(name);baseline={b:confusion(gt,x) for b,x in bases.items()}
                if baseline['raw'].tolist()!=info[name]['metrics']['mask2former_6k/raw']['confusion'] or baseline['selected'].tolist()!=sel[name]['curves'][1]['confusion']:raise ValueError('Historical raw/selected baseline not reproduced')
                row={'name':name,'fold':fold,'baseline_confusions':{b:m.tolist() for b,m in baseline.items()},'candidates':{}}
                for tag,outputs in proposals.items():
                    changed=outputs['raw']!=raw
                    if np.any(changed&~((raw==2)&(static==3)&(outputs['raw']==3))):raise ValueError('Wrong transition')
                    truth=np.bincount(gt[changed],minlength=5);counts[tag]+=truth
                    for b,base in bases.items():
                        out=outputs[b]
                        if not np.array_equal(out!=base,changed) or any(not np.array_equal(base==k,out==k) for k in [0,1,4]):raise ValueError('Protected classes or no-cascade violated')
                        cm=confusion(gt,out);expected=baseline[b].copy();expected[:,2]-=truth;expected[:,3]+=truth
                        if not np.array_equal(cm,expected):raise ValueError('Direct pixels differ from transition algebra')
                        acc[b][tag]+=cm
                    row['candidates'][tag]=truth.tolist()
                per.append(row)
                if index%20==0:print(json.dumps({'fold':fold,'views_done':index+1,'fold_total':len(fold_names)}),flush=True)
            groups[fold]={'truth_counts':{k:v.tolist() for k,v in counts.items()},'metrics':{b:{k:metrics(v) for k,v in d.items()} for b,d in acc.items()}}
        totals={b:{tag:metrics(sum(np.array(g['metrics'][b][tag]['confusion']) for g in groups.values())) for tag in c['candidates']} for b in ['raw','selected']}
        eligible={}
        for tag in c['candidates']:
            flags={}
            for fold,g in groups.items():
                for b in totals:
                    before=exact_scores(g['metrics'][b]['off']['confusion']);after=exact_scores(g['metrics'][b][tag]['confusion'])
                    flags[fold+'/'+b]=all(x>=y for x,y in zip(after,before))
            strict={b:exact_scores(totals[b][tag]['confusion'])[2]>exact_scores(totals[b]['off']['confusion'])[2] for b in totals}
            truth=np.sum([g['truth_counts'][tag] for g in groups.values()],axis=0).tolist()
            eligible[tag]={'per_fold_control_pass':flags,'pooled_tower_strict_gain':strict,'all_pass':all(flags.values()) and all(strict.values()),'nonzero':bool(sum(truth)),
                           'truth_counts':truth,'corrected':truth[3],'harmed':truth[2],'wrong_to_wrong':sum(truth[k] for k in [0,1,4]),'recall_loss_requires_user_review':bool(truth[2])}
        if sorted(labels.accessed)!=sorted(names):raise ValueError('Incomplete label access')
        outcome={'per_fold':groups,'metrics':totals,'eligibility':eligible,'calibration_views':sorted(labels.accessed)}
        if a.replay_from:
            original=read(a.replay_from/'result.json')
            if original['config_sha256']!=r['config_sha256'] or original['code_sha256']!=r['code_sha256'] or not original['status'].startswith('completed_'):raise ValueError('Unaccepted original run')
            if any(original[k]!=v for k,v in outcome.items()) or read(a.replay_from/'per_view.json')!=per:raise ValueError('Independent pixel replay differs')
            r.update(replay_exact=True,replay_source_sha256=sha256(a.replay_from/'result.json'))
        ctx.finish();verify_hashes(ROOT,c['input_hashes']);verify_hashes(ROOT,c['code_sha256'])
        if sha256(ROOT/CONFIG)!=r['config_sha256']:raise ValueError('Configuration changed during evaluation')
        write_new_json(a.output/'per_view.json',per)
        passed=[k for k,v in eligible.items() if v['all_pass'] and v['nonzero']]
        r.update(outcome,status='completed_fixed108_tower_calibration_no_selection',nonzero_candidates_passed=passed,rgb_verified=293,frozen_verified=18,
                 per_view_sha256=sha256(a.output/'per_view.json'),decision='pending_quality_review' if passed else 'keep_off_close_finite_tower_margin_family')
        if sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())>c['resource']['maximum_stage_mib']*2**20:raise OSError('Stage output budget exceeded; preserve results')
    except BaseException as exc:r.update(status='failed_tower_calibration_preserved',error=f'{type(exc).__name__}: {exc}');raise
    finally:r['seconds']=time.monotonic()-start;write_new_json(a.output/'result.json',r)


if __name__=='__main__':main()
