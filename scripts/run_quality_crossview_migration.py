#!/usr/bin/env python3
"""One bounded SEM338 batch: fresh new-RGB matches, unchanged review, audited scoring."""
import argparse
from contextlib import nullcontext
import fcntl
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
from PIL import Image
from run_sem320_dense_preflight import git_sources, offline_official_load, array_digest
from run_sem326_matching import independent_ranking
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_semantic_audit import independent_confusion, metrics
from src.quality_foundation_policy import changes
from src.quality_static_training import TrainingLabels
from src.quality_crossview_migration import (require_source, SourceLabels, class_map,
    migrated_prediction, evidence_trace, verify_trace)
from src.semantic_source_access import read_pose_only
from src.semantic_source_check import select_plan, summarize_access
from src.semantic_source_geometry import geometry_row, compare_rows
from src.semantic_second_source_veto import second_source
from src.geometry_prior.camera_io import read_cameras_txt
from src.semantic_protected_fusion import partition_training_names

CONFIG = 'experiments/runs/SEM-338-CROSSVIEW-MIGRATION/config.yaml'
EXP = 'SEM-338-CROSSVIEW-MIGRATION'
STAGES = ('prepare', 'smoke', 'replay', 'match', 'geometry', 'predict', 'audit', 'score')


def checked_json(path, digest):
    if sha256(path) != digest:
        raise ValueError('Changed input: '+str(path))
    return json.loads(path.read_text())


class Context:
    def __init__(self, args):
        self.args = args
        self.cfg = json.loads((ROOT/CONFIG).read_text()); c = self.cfg
        self.hashes = {CONFIG: sha256(ROOT/CONFIG), **c['immutable_core'], c['label_manifest']: c['label_manifest_sha256']}
        for key in ('matcher_config', 'geometry_config', 'old_rule_config'):
            self.hashes[c[key]] = c[key+'_sha256']
        self.matcher = checked_json(ROOT/c['matcher_config'], c['matcher_config_sha256'])
        self.geometry = checked_json(ROOT/c['geometry_config'], c['geometry_config_sha256'])
        if c['filters'] != self.geometry['filters'] or c['seed'] != self.matcher['seed']:
            raise ValueError('Inherited matcher or geometric recipe changed')
        protocol = json.loads((ROOT/c['protocol']).read_text())
        self.hashes.update(protocol['input_hashes'])
        self.base_root = ROOT/c['baseline_root']
        self.base = checked_json(self.base_root/'result.json', c['baseline_result_sha256'])
        audit = checked_json(ROOT/c['baseline_audit'], c['baseline_audit_sha256'])
        if (self.base['status'] != 'completed_paired_calibration_pending_independent_audit'
                or audit['status'] != 'completed_independent_paired_calibration_audit'
                or audit['run_result_sha256'] != c['baseline_result_sha256'] or self.base['selected']['3k'] != .8):
            raise ValueError('Accepted paired3k base not available')
        self.hashes.update(self.base['input_hashes'])
        self.render_root = ROOT/c['rendered_root']
        self.render = checked_json(self.render_root/'result.json', c['rendered_result_sha256'])
        if self.render['status'] != 'completed_native_expert_rgb_export' or self.render['pixel_space'] != 'PINHOLE' or self.render['parent_experiment'] != 'RGB-110-SCAFFOLD-CTRL':
            raise ValueError('Wrong new rendered RGB')
        self.hashes.update(self.render['code_sha256'])
        self.rgb = checked_json(ROOT/c['rgb_run']/'result.json', self.render['parent_result_sha256'])
        self.frozen = {x['path']: x['sha256'] for x in json.loads((ROOT/protocol['frozen_manifest']).read_text())['artifacts']}
        verify_hashes(ROOT, self.hashes); verify_hashes(args.data_root, c['camera_sha256'])
        self.poses = read_pose_only(args.data_root/'camera_parameters/images.txt')
        self.cameras = read_cameras_txt(args.data_root/'camera_parameters/cameras.txt')
        split = json.loads((ROOT/'experiments/splits/sem_001_scaffold_static_v1.json').read_text())
        self.folds = json.loads((ROOT/'experiments/splits/sem_005_oof_spatial_v1.json').read_text())['folds']
        self.roles = partition_training_names(sorted(split['semantic_train_labeled']), expected_count=219)
        if {k: len(v) for k,v in self.roles.items()} != {'calibration':108, 'check':111}:
            raise ValueError('Role split changed')
        self.role_of = {n:r for r,names in self.roles.items() for n in names}
        self.render_images = {r['name']: r for r in self.render['rendered_images']}
        if set(self.render_images) != set(self.role_of):
            raise ValueError('Missing or extra rendered targets')
        self.labels = json.loads((ROOT/c['label_manifest']).read_text())
        self.label_records = {r['image']:r for r in self.labels['records']}
        for role in self.roles:
            b = self.base['evaluation'][role]
            if sorted(b['views']) != sorted(self.roles[role]) or b['metrics']['adapted_3k'] != b['metrics']['fixed_3k']:
                raise ValueError('Paired3k baseline mismatch')
        self.runtime_args = argparse.Namespace(**{k:ROOT/c['local_matcher'][k] for k in ('roma_dir','dinov3_dir','weights')})
        self.finish()

    def base_path(self, name):
        rel = f'{self.role_of[name]}/adapted_3k/{name}'
        return self.base_root/rel, self.base['artifacts'][rel]

    def finish(self):
        verify_hashes(ROOT, self.hashes); verify_hashes(self.args.artifact_root, self.frozen)
        verify_hashes(self.args.data_root, self.cfg['camera_sha256'])
        for p,h in self.base['input_result_hashes'].items():
            if sha256(ROOT/p/'result.json') != h:
                raise ValueError('Paired baseline source result changed')
        for p,h in ((self.base_root/'result.json',self.cfg['baseline_result_sha256']),
                    (self.render_root/'result.json',self.cfg['rendered_result_sha256'])):
            if sha256(p) != h:
                raise ValueError('Model result changed')


def source_snapshot(output):
    paths = {CONFIG:sha256(ROOT/CONFIG)}
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            p = Path(name).resolve()
            if p.suffix == '.py' and p.is_relative_to(ROOT) and not p.is_relative_to(ROOT/'outputs'):
                paths[str(p.relative_to(ROOT))] = sha256(p)
    paths[str(Path(__file__).resolve().relative_to(ROOT))] = sha256(Path(__file__).resolve())
    for name in paths:
        dest = output/'source_snapshot'/name
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dest)
    return paths


def prepare(ctx, output):
    c = ctx.cfg
    plans = select_plan(ctx.folds, sorted(ctx.role_of), ctx.poses)
    independent_ranking(plans, ctx.folds, ctx.poses)
    if len(plans) != 219 or any(len(p['sources']) != 3 for p in plans):
        raise ValueError('Expected complete219 top3 source plan')
    # Check renderer camera metadata against official extrinsics/intrinsics before matches.
    sparse = ROOT/c['rgb_run']/'scene/sparse/0'
    render_poses = read_pose_only(sparse/'images.txt')
    render_cameras = read_cameras_txt(sparse/'cameras.txt')
    for name in ctx.role_of:
        a,b = ctx.poses[name],render_poses[name]
        ca,cb = ctx.cameras[a.camera_id],render_cameras[b.camera_id]
        if not np.allclose(a.rotation_world_to_camera,b.rotation_world_to_camera,atol=1e-10,rtol=0) or not np.allclose(a.tvec,b.tvec,atol=1e-10,rtol=0):
            raise ValueError('Renderer not in official coordinate frame')
        if ca.model != 'SIMPLE_RADIAL' or cb.model != 'PINHOLE' or (ca.width,ca.height) != tuple(c['resolution']) or (cb.width,cb.height) != (ca.width,ca.height):
            raise ValueError('Wrong native camera space')
        f,cx,cy,k = ca.params
        if not np.allclose(cb.params,(f,f,cx,cy),atol=1e-10,rtol=0):
            raise ValueError('PINHOLE intrinsics do not match official ideal projection')
    pairs, bases = [],{}
    for plan in plans:
        name = plan['target']; bp,bh = ctx.base_path(name)
        if sha256(bp) != bh: raise ValueError('Baseline PNG changed')
        bases[name] = {'path':str(bp.relative_to(ROOT)),'sha256':bh}
        target = ctx.render_root/'images'/name
        target_hash = ctx.render_images[name]['sha256']
        if sha256(target) != target_hash: raise ValueError('New RGB changed')
        for rank,src in enumerate(plan['sources'],1):
            require_source(ctx.folds,plan['fold'],name,src)
            source_path = ctx.args.data_root/'images'/src
            lp = ctx.args.artifact_root/c['labels']/src
            label_hash = ctx.label_records[src]['label_png_sha256']
            if sha256(lp) != label_hash: raise ValueError('Source label hash differs')
            pairs.append({'index':len(pairs),'fold':plan['fold'],'target':name,'source':src,'rank':rank,
                'target_path':str(target.relative_to(ROOT)),'target_sha256':target_hash,
                'source_path':'images/'+src,'source_sha256':sha256(source_path),'source_label_sha256':label_hash})
    calibration = set(ctx.roles['calibration'])
    probe_targets = [min(set(f['oof']) & calibration) for f in ctx.folds]
    probe_indices = [p['index'] for p in pairs if p['target'] in probe_targets and p['rank']==1]
    upstream = {kind:git_sources(getattr(ctx.runtime_args,kind+'_dir'),ctx.matcher['sources'][kind+'_commit']) for kind in ('roma','dinov3')}
    weights = ctx.runtime_args.weights
    if sha256(weights) != ctx.matcher['sources']['weight_sha256'] or weights.stat().st_size != ctx.matcher['sources']['weight_bytes']:
        raise ValueError('Author complete matcher weights differ')
    overlay = ROOT/c['local_matcher']['pillow_overlay']
    overlay_hashes = {str(p.relative_to(overlay)):sha256(p) for p in overlay.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    return {'status':'prepared_migration_source_roles_sealed','pairs':pairs,'plans':plans,'baseline':bases,
        'roles':ctx.roles,'probe_indices':probe_indices,'upstream':upstream,'overlay_sha256':overlay_hashes,
        'source_sha256':source_snapshot(output),'source_input_hashes':ctx.hashes,
        'camera_sha256':c['camera_sha256'],'render_camera_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in (sparse/'cameras.txt',sparse/'images.txt')},
        'identity':ctx.render['identity'],'render_result_sha256':c['rendered_result_sha256'],
        'label_manifest_sha256':c['label_manifest_sha256'],'target_labels_decoded':0,'source_labels_decoded':0,
        'observations_parsed':0,'independent_pose_ranking_exact':True,'official_projection_consistency':True}


def load_stage(ctx, stage):
    result = json.loads((ctx.args.batch/stage/'result.json').read_text())
    if result.get('experiment_id') != EXP or result.get('config_sha256') != sha256(ROOT/CONFIG) or 'failed' in result['status']:
        raise ValueError('Unaccepted '+stage+' result')
    return result


def load_plan(ctx):
    p = load_stage(ctx,'prepare')
    verify_hashes(ROOT,p['source_sha256']);verify_hashes(ctx.args.batch/'prepare/source_snapshot',p['source_sha256'])
    verify_hashes(ROOT,p['render_camera_sha256'])
    if p['identity'] != ctx.render['identity'] or p['roles'] != ctx.roles or p['source_input_hashes'] != ctx.hashes:
        raise ValueError('Cached plan is bound to another backbone/split/implementation')
    for pair in p['pairs']:
        require_source(ctx.folds,pair['fold'],pair['target'],pair['source'])
        if sha256(ROOT/pair['target_path']) != pair['target_sha256'] or sha256(ctx.args.data_root/pair['source_path']) != pair['source_sha256']:
            raise ValueError('Target/source RGB changed')
    return p


def match_stage(ctx, output, mode):
    import torch
    import PIL
    plan = load_plan(ctx)
    if mode == 'match':
        replay = load_stage(ctx,'replay')
        if not replay.get('fresh_bidirectional_probe_exact'): raise ValueError('Fresh probe required')
    reference = load_stage(ctx,'smoke') if mode != 'smoke' else None
    records_ref = {r['key']:r for r in reference['maps']} if reference else {}
    for kind in ('roma','dinov3'):
        if git_sources(getattr(ctx.runtime_args,kind+'_dir'),ctx.matcher['sources'][kind+'_commit']) != plan['upstream'][kind]:
            raise ValueError('Matcher source changed')
    verify_hashes(ROOT/ctx.cfg['local_matcher']['pillow_overlay'],plan['overlay_sha256'])
    if PIL.__version__ != '12.0.0' or not torch.cuda.is_available():
        raise RuntimeError('Exact isolated Pillow12/CUDA runtime required')
    if sha256(ctx.runtime_args.weights) != ctx.matcher['sources']['weight_sha256']:
        raise ValueError('Matcher checkpoint changed')
    sys.path.insert(0,str(ctx.runtime_args.roma_dir/'src'))
    from romav2 import RoMaV2
    from romav2.local_correlation import local_corr
    if local_corr is not None: raise ValueError('Different correlation backend')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(ctx.cfg['seed']);np.random.seed(ctx.cfg['seed']);torch.manual_seed(ctx.cfg['seed'])
    free,total=torch.cuda.mem_get_info()
    if free/2**20 < ctx.cfg['resource']['minimum_free_gpu_mib']: raise RuntimeError('Insufficient GPU headroom')
    torch.cuda.reset_peak_memory_stats();records=[];start=time.monotonic()
    with offline_official_load(ctx.runtime_args,ctx.matcher):
        model=RoMaV2(RoMaV2.Cfg(setting='fast',compile=False)).eval().requires_grad_(False)
        if (model.H_lr,model.W_lr,model.H_hr,model.W_hr,model.bidirectional)!=(512,512,None,None,False):
            raise ValueError('Matcher settings differ')
        load_seconds=time.monotonic()-start
        pairs=plan['pairs'] if mode=='match' else [plan['pairs'][i] for i in plan['probe_indices']]
        for pair in pairs:
            with Image.open(ROOT/pair['target_path']) as im: a=np.asarray(im.convert('RGB'))
            with Image.open(ctx.args.data_root/pair['source_path']) as im: b=np.asarray(im.convert('RGB'))
            if a.shape != (989,1320,3) or b.shape != a.shape: raise ValueError('Native RGB dimensions differ')
            inputs_hash=(array_digest(a),array_digest(b))
            for direction,x,y in (('ab',a,b),('ba',b,a)):
                key=f"pair{pair['index']}_{direction}"
                torch.cuda.synchronize();tick=time.monotonic()
                with torch.inference_mode(): pred=model.match(x,y)
                torch.cuda.synchronize()
                elapsed=time.monotonic()-tick
                warp=pred['warp_AB'].float().cpu().numpy()[0]
                overlap=pred['overlap_AB'].float().cpu().numpy()[0,...,0]
                if warp.shape!=(512,512,2) or overlap.shape!=(512,512) or not np.isfinite(warp).all() or not np.isfinite(overlap).all() or overlap.min()<0 or overlap.max()>1:
                    raise ValueError('Invalid correspondence arrays')
                row={'key':key,'pair_index':pair['index'],'direction':direction,'path':key+'.npz',
                    'warp_sha256':array_digest(warp),'overlap_sha256':array_digest(overlap),'seconds':elapsed,
                    'target_rgb_sha256':pair['target_sha256'],'source_rgb_sha256':pair['source_sha256']}
                if key in records_ref and any(row[k]!=records_ref[key][k] for k in ('warp_sha256','overlap_sha256')):
                    raise ValueError('Fresh matcher replay differs')
                np.savez_compressed(output/row['path'],warp=warp,overlap=overlap)
                row['sha256']=sha256(output/row['path']);records.append(row)
                if torch.cuda.max_memory_allocated()/2**20 > ctx.cfg['resource']['maximum_peak_allocated_mib']:
                    raise RuntimeError('GPU resource ceiling exceeded')
                if time.monotonic()-start > ctx.cfg['resource']['maximum_seconds_per_stage']:
                    raise RuntimeError('Registered stage timeout')
                if len(records)%20==0 or mode!='match':
                    print(json.dumps({'stage':mode,'maps_completed':len(records),'maps_total':len(pairs)*2,'seconds_last':elapsed}),flush=True)
            if inputs_hash!=(array_digest(a),array_digest(b)): raise ValueError('RGB arrays modified')
            if shutil.disk_usage(output).free < ctx.cfg['resource']['minimum_free_disk_gib']*2**30:
                raise RuntimeError('Disk reserve reached')
    return {'status':'completed_'+mode+'_matching','maps':records,'prepare_sha256':sha256(ctx.args.batch/'prepare/result.json'),
        'fresh_bidirectional_probe_exact':mode=='replay','peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'loading_seconds':load_seconds,
        'gpu':torch.cuda.get_device_name(),'versions':{n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','pillow','einops')},
        'target_label_decodes':0,'source_label_decodes':0,'real_target_slot_decodes':0,'training_steps':0}


def map_reader(ctx, matching):
    records={r['key']:r for r in matching['maps']}
    def read(index):
        maps=[]
        for direction in ('ab','ba'):
            r=records[f'pair{index}_{direction}'];p=ctx.args.batch/'match'/r['path']
            if sha256(p)!=r['sha256']:raise ValueError('Matching cache changed')
            with np.load(p,allow_pickle=False) as z:
                maps.append((z['warp'].astype(np.float64),z['overlap'].astype(np.float64)))
        return maps
    return read


def geometric_stage(ctx):
    import torch
    torch.set_num_threads(4)
    plan=load_plan(ctx);matching=load_stage(ctx,'match');read=map_reader(ctx,matching)
    if matching['prepare_sha256']!=sha256(ctx.args.batch/'prepare/result.json') or len(matching['maps'])!=2*len(plan['pairs']):
        raise ValueError('Incomplete or mismatched new cache')
    rows=[]
    for p in plan['pairs']:
        maps=read(p['index'])
        primary=geometry_row(p,maps,ctx.poses,ctx.cameras,ctx.geometry)
        independent=geometry_row(p,maps,ctx.poses,ctx.cameras,ctx.geometry,independent=True)
        compare_rows(primary,independent);rows.append(primary)
        if len(rows)%30==0: print(json.dumps({'stage':'geometry','pairs_checked':len(rows),'total':len(plan['pairs'])}),flush=True)
    summary=summarize_access(plan['plans'],rows)
    reranked=select_plan(ctx.folds,sorted(ctx.role_of),ctx.poses)
    if reranked!=plan['plans']: raise ValueError('Pose ranking changed')
    independent_ranking(reranked,ctx.folds,ctx.poses)
    return {'status':'completed_independent_migration_geometry','pairs':rows,'summary':summary,
        'prepare_sha256':sha256(ctx.args.batch/'prepare/result.json'),'matching_sha256':sha256(ctx.args.batch/'match/result.json'),
        'independent_geometry_exact':True,'target_label_decodes':0,'source_label_decodes':0}


def predict_stage(ctx, output, audit=False):
    import torch
    torch.set_num_threads(4)
    plan=load_plan(ctx);geo=load_stage(ctx,'geometry');matching=load_stage(ctx,'match')
    if geo['matching_sha256']!=sha256(ctx.args.batch/'match/result.json') or geo['prepare_sha256']!=sha256(ctx.args.batch/'prepare/result.json'):
        raise ValueError('Geometry bound to different cache')
    prior=load_stage(ctx,'predict') if audit else None
    if prior:
        if prior['geometry_sha256']!=sha256(ctx.args.batch/'geometry/result.json'): raise ValueError('Wrong prediction geometry')
        verify_hashes(ctx.args.batch/'predict',prior['artifacts'])
    read=map_reader(ctx,matching)
    pair_index={(p['fold'],p['target'],p['source']):p['index'] for p in plan['pairs']}
    readers=SourceLabels(ctx.folds,ctx.args.artifact_root/ctx.cfg['labels'],ctx.labels,ctx.cfg['resolution'])
    records=[]
    write_new_json(output/'source_roles_sealed.json',{'targets':geo['summary']['targets'],'folds':ctx.folds,
        'label_manifest_sha256':ctx.cfg['label_manifest_sha256'],'geometry_sha256':sha256(ctx.args.batch/'geometry/result.json'),
        'target_labels_decoded':False,'source_label_policy':'check fold.train before reading; historical target IDs in other folds are allowed source roles'})
    for t in geo['summary']['targets']:
        name=t['target'];pose=ctx.poses[name];camera=ctx.cameras[pose.camera_id]
        bp,bh=ctx.base_path(name);base=class_map(bp,bh,(989,1320))
        def source(src):
            if src is None:return None
            require_source(ctx.folds,t['fold'],name,src)
            sp=ctx.poses[src]
            return {'name':src,'pose':sp,'camera':ctx.cameras[sp.camera_id],
                'labels':readers.read(t['fold'],name,src),'maps':read(pair_index[(t['fold'],name,src)])}
        first,second=source(t['selected_source']),source(second_source(t))
        dense,veto,stats=migrated_prediction(base,t,first,second,pose,camera,ctx.cfg['filters'],independent=audit)
        trace=evidence_trace(base,dense,veto,first,second,pose,camera,ctx.cfg['filters'],independent=audit)
        outputs={}
        for arm,value in (('dense',dense),('veto',veto)):
            rel=arm+'/'+name
            if audit:
                stored=class_map(ctx.args.batch/'predict'/rel,prior['artifacts'][rel],base.shape)
                if not np.array_equal(stored,value):raise ValueError('Fresh native prediction differs')
                outputs[arm]=prior['artifacts'][rel]
            else:
                dest=output/rel;dest.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(value).save(dest);outputs[arm]=sha256(dest)
        rel='evidence/'+Path(name).stem+'.npz'
        if audit:
            with np.load(ctx.args.batch/'predict'/rel,allow_pickle=False) as z:verify_trace(dict(z),trace)
        else:
            dest=output/rel;dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**trace)
        row={'target':name,'fold':t['fold'],'role':ctx.role_of[name],'first_source':t['selected_source'],
            'second_source':second_source(t),'baseline_sha256':bh,'output_sha256':outputs,'stats':stats,
            'first_pair_index':pair_index.get((t['fold'],name,t['selected_source'])),
            'second_pair_index':pair_index.get((t['fold'],name,second_source(t))),
            'reason':'first reliable source background deletes baseline4; second reliable source4 vetoes only that deletion'}
        if audit and row!=prior['predictions'][len(records)]:raise ValueError('Independent inference counts differ')
        records.append(row)
        if len(records)%12==0: print(json.dumps({'stage':'audit' if audit else 'predict','views':len(records),'total':219}),flush=True)
    if audit and readers.access!=prior['source_label_access']:raise ValueError('Independent source access differs')
    return {'status':'completed_independent_migration_prediction_audit' if audit else 'completed_migration_predictions',
        'predictions':records,'source_label_access':readers.access,'target_label_decodes':0,
        'geometry_sha256':sha256(ctx.args.batch/'geometry/result.json'),
        'prediction_sha256':sha256(ctx.args.batch/'predict/result.json') if audit else None,
        'independent_native_reconstruction':audit,'protected_classes_exact':True,'no_baseline_background_added':True}


def score_stage(ctx, output):
    prediction=load_stage(ctx,'predict');audit=load_stage(ctx,'audit');geo=load_stage(ctx,'geometry')
    if audit['status']!='completed_independent_migration_prediction_audit' or audit['prediction_sha256']!=sha256(ctx.args.batch/'predict/result.json'):
        raise ValueError('Fresh native prediction audit required before target GT')
    verify_hashes(ctx.args.batch/'predict',prediction['artifacts'])
    records={r['target']:r for r in prediction['predictions']}
    comparisons={'dense_vs_baseline':('baseline','dense'),'veto_vs_dense':('dense','veto'),'veto_vs_baseline':('baseline','veto')}
    def empty():
        return {'matrices':{k:np.zeros((5,5),np.int64) for k in ('baseline','dense','veto')},
            'changes':{k:dict(changed=0,corrected=0,harmed=0,wrong_to_wrong=0) for k in comparisons}}
    def update(total,matrices,costs):
        for k in matrices:total['matrices'][k]+=matrices[k]
        for k in costs:
            for key in costs[k]:total['changes'][k][key]+=costs[k][key]
    def summarized(total):return {'metrics':{k:metrics(v) for k,v in total['matrices'].items()},'changes':total['changes']}
    write_new_json(output/'target_scoring_sealed.json',{'prediction_sha256':sha256(ctx.args.batch/'predict/result.json'),
        'audit_sha256':sha256(ctx.args.batch/'audit/result.json'),'roles':ctx.roles,'selection':False,'development_access':False})
    evaluation={};access={}
    for role,names in ctx.roles.items():
        labels=TrainingLabels(ctx.args.artifact_root/ctx.cfg['labels'],names,ctx.labels,ctx.cfg['resolution'])
        total=empty();groups={};views={}
        for name in names:
            bp,bh=ctx.base_path(name);maps={'baseline':class_map(bp,bh,(989,1320))}
            for arm in ('dense','veto'):
                rel=arm+'/'+name;maps[arm]=class_map(ctx.args.batch/'predict'/rel,prediction['artifacts'][rel],(989,1320))
            target=labels.read(name)
            matrices={k:independent_confusion(target,p) for k,p in maps.items()}
            costs={k:changes(maps[a],maps[b],target) for k,(a,b) in comparisons.items()}
            update(total,matrices,costs);fold=str(records[name]['fold']);update(groups.setdefault(fold,empty()),matrices,costs)
            views[name]={'fold':fold,'metrics':{k:metrics(v) for k,v in matrices.items()},'changes':costs}
        summary=summarized(total)
        if summary['metrics']['baseline']!=ctx.base['evaluation'][role]['metrics']['fixed_3k']:
            raise ValueError('Fixed3k metrics not reproduced')
        evaluation[role]={**summary,'views':names,'per_fold':{k:summarized(v) for k,v in groups.items()}}
        access[role]=sorted(labels.accessed);write_new_json(output/(role+'_per_view.json'),views)
        print(json.dumps({'stage':'score','role':role,'mIoU':{k:v['mIoU_with_background'] for k,v in summary['metrics'].items()}}),flush=True)
    return {'status':'completed_fixed_migration_quality_requires_human_review','evaluation':evaluation,
        'target_label_access':access,'prediction_sha256':sha256(ctx.args.batch/'predict/result.json'),
        'audit_sha256':sha256(ctx.args.batch/'audit/result.json'),'evidence_coverage':geo['summary'],
        'two_independent_confusion_counts':True,'threshold_search':False,'automatic_promotion':False}


def batch_stage(ctx, output):
    # Explicit one-shot continuation only; each stage is a new process and fails closed.
    plan=load_plan(ctx);replay=load_stage(ctx,'replay')
    if not replay.get('fresh_bidirectional_probe_exact'):raise ValueError('Smoke/replay gate not met')
    events=[]
    for stage in ('match','geometry','predict','audit','score'):
        verify_hashes(ROOT,plan['source_sha256'])
        argv=[sys.executable,str(Path(__file__).resolve()),'--mode',stage,'--batch',str(ctx.args.batch),
            '--data-root',str(ctx.args.data_root),'--artifact-root',str(ctx.args.artifact_root)]
        event={'stage':stage,'started_unix':time.time()}
        write_new_json(output/(stage+'-started.json'),event)
        with (output/(stage+'.log')).open('x') as log:
            completed=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=ctx.cfg['resource']['maximum_seconds_per_stage']+120)
        event.update(returncode=completed.returncode,ended_unix=time.time())
        write_new_json(output/(stage+'-finished.json'),event);events.append(event)
        if completed.returncode:raise RuntimeError('Batch stopped at '+stage+'; see immutable log/failure record')
        result=load_stage(ctx,stage)
        print(json.dumps({'completed':stage,'status':result['status']}),flush=True)
    return {'status':'completed_crossview_migration_batch_requires_review','events':events,
        'score_sha256':sha256(ctx.args.batch/'score/result.json'),'automatic_promotion':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=(*STAGES,'batch'),required=True)
    for name in ('batch','data-root','artifact-root'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    for k in ('batch','data_root','artifact_root'):setattr(args,k,getattr(args,k).resolve())
    output=args.batch/args.mode;output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();ctx=None
    record={'experiment_id':EXP,'config_sha256':sha256(ROOT/CONFIG),'seed':20260805,'mode':args.mode,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'status':'initializing','training':False,'development_target_labels':False,'whole_chain_spatial_validation':False}
    write_new_json(output/'started.json',record)
    try:
        ctx=Context(args)
        if shutil.disk_usage(output).free<ctx.cfg['resource']['minimum_free_disk_gib']*2**30:raise RuntimeError('Disk reserve too low')
        lock=(ROOT/'outputs/quality/gpu.lock').open('a') if args.mode in ('smoke','replay','match') else nullcontext()
        with lock as handle:
            if args.mode in ('smoke','replay','match'):fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.mode=='prepare':result=prepare(ctx,output)
            elif args.mode in ('smoke','replay','match'):result=match_stage(ctx,output,args.mode)
            elif args.mode=='geometry':result=geometric_stage(ctx)
            elif args.mode in ('predict','audit'):result=predict_stage(ctx,output,audit=args.mode=='audit')
            elif args.mode=='score':result=score_stage(ctx,output)
            else:result=batch_stage(ctx,output)
        ctx.finish()
        if args.mode!='prepare':load_plan(ctx)
        record.update(result,frozen_verified=len(ctx.frozen),rgb_source_hashes_unchanged=True,
            artifacts={str(f.relative_to(output)):sha256(f) for f in output.rglob('*') if f.is_file()})
    except BaseException as exc:
        record.update(status='migration_stage_failed_not_method_rejection',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['seconds']=time.monotonic()-started
        write_new_json(output/'result.json',record)
        print(json.dumps({'stage':args.mode,'status':record['status'],'seconds':record['seconds']}),flush=True)


if __name__=='__main__':main()
