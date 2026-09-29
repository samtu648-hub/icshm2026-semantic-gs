#!/usr/bin/env python3
"""SEM343 fixed0.05: native historical111 diagnosis and selected108 local audit."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_static_training import TrainingLabels
from src.quality_semantic_audit import independent_confusion, metrics
from src.quality_raw_foundation_gate import apply_gate
from src.quality_raw_foundation_check import validate_selection, evidence_masks
from scripts.diagnose_quality_raw_fusion import read_probability, read_image
from scripts.calibrate_quality_raw_foundation_gate import compare

CONFIG = 'experiments/runs/SEM-343-RAW-FOUNDATION-GATE/audit_config.yaml'
CODE = ('scripts/evaluate_quality_raw_foundation_check.py','src/quality_raw_foundation_check.py',
        'src/quality_raw_foundation_gate.py','src/quality_protocol.py','src/quality_static_training.py',
        'src/quality_semantic_audit.py','src/quality_experts.py','scripts/diagnose_quality_raw_fusion.py',
        'scripts/calibrate_quality_raw_foundation_gate.py')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay',type=Path)
    a = p.parse_args(); output = a.output.resolve()
    if not output.is_relative_to(ROOT/'outputs/quality/raw_fusion'):
        p.error('Use a new raw_fusion output directory')
    output.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    protocol = json.loads((ROOT/CONFIG).read_text())
    record = {'experiment_id':protocol['experiment_id'],'git_commit':protocol['git_commit'],
              'seed':protocol['seed'],'config_sha256':sha256(ROOT/CONFIG),
              'code_sha256':{n:sha256(ROOT/n) for n in CODE},'selected_margin':0.05,
              'selection_sha256':protocol['selection_sha256'],'started_utc':datetime.now(timezone.utc).isoformat(),
              'new_training':False,'new_model_inference':False,'new_threshold_search':False,
              'automatic_promotion':False,'development_labels_decoded':0,
              'historically_reused_data':True,'whole_chain_spatial_validation':False,'replay':bool(a.replay)}
    try:
        verify_hashes(ROOT,protocol['input_hashes'])
        verify_hashes(ROOT,{protocol['selection']:protocol['selection_sha256']})
        selection = json.loads((ROOT/protocol['selection']).read_text())
        cfg = json.loads((ROOT/protocol['calibration_config']).read_text())
        verify_hashes(ROOT,cfg['input_hashes'])
        calibration = json.loads((ROOT/protocol['calibration_result']).read_text())
        verify_hashes(ROOT,calibration['code_sha256'])
        parent_dir = ROOT/protocol['comparison']
        parent = json.loads((parent_dir/'result.json').read_text())
        verify_hashes(ROOT,parent['input_result_hashes'])
        approved = validate_selection(selection,parent,calibration)
        pc = json.loads((ROOT/cfg['preflight_config']).read_text())
        export_dir = ROOT/pc['rgb_export']
        verify_hashes(export_dir,{'result.json':pc['rgb_export_result_sha256']})
        export = json.loads((export_dir/'result.json').read_text())
        rgb = {e['name']:e['sha256'] for e in export['rendered_images']}
        frozen = {e['path']:e['sha256'] for e in json.loads((ROOT/cfg['frozen_manifest']).read_text())['artifacts']}
        def guard():
            if shutil.disk_usage(ROOT).free < protocol['minimum_free_gib']*1024**3:
                raise OSError('100GiB reserve reached; preserve old artifacts')
            verify_hashes(export_dir/'images',rgb)
            verify_hashes(a.artifact_root,frozen)
        guard()
        sources = {}
        for part in ('oof-0','oof-1','oof-2'):
            base = ROOT/protocol['batch']/('full-'+part)
            sources[part]={'five':(base,json.loads((base/'result.json').read_text()))}
        for path in parent['input_result_hashes']:
            base=(ROOT/path).parent
            if base.name.startswith('SEM-330-Scaffold-'):
                r=json.loads((base/'result.json').read_text())
                sources[r['access']['partition']]['static']=(base,r)
        original = None
        if a.replay:
            original=json.loads((a.replay/'result.json').read_text())
            if original['status']!='completed_fixed_check_pending_independent_replay' or any(original[k]!=record[k] for k in ('code_sha256','config_sha256','selection_sha256')):
                raise ValueError('Unaccepted or differently bound replay source')
            verify_hashes(a.replay,original['artifacts'])
            record['replay_source_sha256']=sha256(a.replay/'result.json')
        for name in (CONFIG,protocol['selection'],*CODE):
            dest=output/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        write_new_json(output/'started.json',record)
        manifest=json.loads((ROOT/pc['label_manifest']).read_text())
        evaluation={}
        for role in protocol['roles']:
            names=selection['roles'][role]
            labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',names,manifest,[1320,989])
            rel=f'{role}/per_view.json';verify_hashes(parent_dir,{rel:parent['artifacts'][rel]})
            view_info=json.loads((parent_dir/rel).read_text())
            pooled={k:np.zeros((5,5),np.int64) for k in ('raw','calibrated','selected')}
            groups={part:{k:np.zeros((5,5),np.int64) for k in pooled} for part in sources}
            counts={part:np.zeros(5,np.int64) for part in sources};per_view={}
            for i,name in enumerate(names):
                part=view_info[name]['partition'];five_base,five_record=sources[part]['five']
                if name not in five_record['access']['prediction']:
                    raise ValueError('Wrong OOF prediction role')
                maps={}
                for key in ('raw','calibrated'):
                    rel=f'{role}/mask2former_6k/{key}/{name}'
                    verify_hashes(parent_dir,{rel:parent['artifacts'][rel]});maps[key]=read_image(parent_dir/rel)
                raw,fused=maps['raw'],maps['calibrated']
                five=read_probability(five_base,five_record,'probabilities/'+Path(name).stem+'.npz',5)
                if not np.array_equal(raw,five.argmax(0)):
                    raise ValueError('Raw argmax differs')
                if a.replay:
                    # Independent from apply_gate and saved fused proposals: static class4 + original gate.
                    base,r=sources[part]['static'];rel='predictions/'+name
                    verify_hashes(base,{rel:r['artifacts'][rel]});static=read_image(base/rel)
                    foundation_proposal=(static==4)&(five[4]>=parent['thresholds']['mask2former_6k']['foundation'])
                    if not np.array_equal(foundation_proposal,fused==4):
                        raise ValueError('Independent static foundation proposal differs')
                    candidate=raw.copy()
                    y,x=np.where((raw==3)&foundation_proposal)
                    keep=(five[3,y,x]-five[4,y,x])<=selection['selected_margin']
                    candidate[y[keep],x[keep]]=4
                else:
                    candidate=apply_gate(raw,fused,five,selection['selected_margin'])
                maps['selected']=candidate
                if any(not np.array_equal(candidate==k,raw==k) for k in (0,1,2)) or np.any(candidate[raw==4]!=4):
                    raise ValueError('Protected raw predictions changed')
                # Candidate is fixed before reading this target label.
                target=labels.read(name)
                matrices={k:independent_confusion(target,value) for k,value in maps.items()}
                for key in ('raw','calibrated'):
                    if matrices[key].tolist()!=view_info[name]['metrics']['mask2former_6k/'+key]['confusion']:
                        raise ValueError('Historical control confusion differs')
                changed=candidate!=raw;truth=np.bincount(target[changed],minlength=5)
                algebra=matrices['raw'].copy();algebra[:,3]-=truth;algebra[:,4]+=truth
                if not np.array_equal(algebra,matrices['selected']):
                    raise ValueError('Transition algebra and full recount differ')
                encoded,local=evidence_masks(raw,candidate,target)
                if (sum(v['pixels'] for v in local['corrected']),sum(v['pixels'] for v in local['harmed']))!=(int(truth[4]),int(truth[3])):
                    raise ValueError('Local costs differ')
                for kind,array in (('predictions',candidate),('changes',encoded)):
                    rel=f'{role}/{kind}/{name}'
                    if a.replay:
                        if not np.array_equal(array,read_image(a.replay/rel)):
                            raise ValueError('Native candidate or change evidence replay differs')
                    else:
                        path=output/rel;path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(array).save(path)
                counts[part]+=truth
                for k,m in matrices.items():pooled[k]+=m;groups[part][k]+=m
                per_view[name]={'partition':part,'selected_vs_raw':compare(matrices['selected'],matrices['raw']),
                                'selected_vs_calibrated':compare(matrices['selected'],matrices['calibrated']),
                                'changed_truth_counts':truth.tolist(),'corrected':int(truth[4]),'harmed':int(truth[3]),
                                'wrong_to_wrong':int(truth[:3].sum()),'components':local}
                if i%24==0:print(json.dumps({'role':role,'views':i+1,'total':len(names),'replay':bool(a.replay)}),flush=True)
            total=sum(counts.values())
            evaluation[role]={'views':sorted(labels.accessed),'metrics':{k:metrics(v) for k,v in pooled.items()},
                              'selected_vs_raw':compare(pooled['selected'],pooled['raw']),
                              'selected_vs_calibrated':compare(pooled['selected'],pooled['calibrated']),
                              'changed_truth_counts':total.tolist(),'corrected':int(total[4]),'harmed':int(total[3]),
                              'wrong_to_wrong':int(total[:3].sum()),
                              'per_partition':{part:{'selected_vs_raw':compare(m['selected'],m['raw']),
                              'selected_vs_calibrated':compare(m['selected'],m['calibrated']),
                              'changed_truth_counts':counts[part].tolist()} for part,m in groups.items()}}
            if role=='calibration' and (metrics(pooled['selected'])!=approved['metrics'] or total.tolist()!=approved['changed_truth_counts']):
                raise ValueError('Selected108 does not reproduce sealed calibration evidence')
            if a.replay:
                if per_view!=json.loads((a.replay/f'{role}_per_view.json').read_text()) or evaluation[role]!=original['evaluation'][role]:
                    raise ValueError('Independent per-view/local/fold metrics differ')
            write_new_json(output/f'{role}_per_view.json',per_view)
        guard()
        verify_hashes(ROOT,{CONFIG:record['config_sha256'],**record['code_sha256'],**protocol['input_hashes'],**cfg['input_hashes'],**parent['input_result_hashes']})
        record.update(status='completed_fixed_check_independent_replay' if a.replay else 'completed_fixed_check_pending_independent_replay',
                      evaluation=evaluation,rgb_exports_unchanged=len(rgb),frozen_artifacts_verified=len(frozen),
                      native_predictions_replayed=219 if a.replay else 0,
                      artifacts={str(f.relative_to(output)):sha256(f) for f in output.rglob('*') if f.is_file() and 'code' not in f.relative_to(output).parts})
    except BaseException as exc:
        record.update(status='failed_fixed_check_preserved',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['seconds']=time.monotonic()-start
        write_new_json(output/'result.json',record)
        print(json.dumps({k:record[k] for k in ('status','seconds')}),flush=True)


if __name__=='__main__':main()
