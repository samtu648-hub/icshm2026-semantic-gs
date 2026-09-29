#!/usr/bin/env python3
"""Bounded RGB120 offset attribution and reversible, copied-endpoint ablation."""
import argparse,json,sys,time,shutil,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from src.quality_protocol import sha256

def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:run(a,cfg,out,start)
    except Exception as e:
        dump(out/'failure.json',dict(status='failed_preserved',error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise

def run(a,cfg,out,start):
    import torch
    from scipy.ndimage import binary_dilation
    from src.quality_scaffold import ScaffoldQualityRenderer
    from src.quality_cameras import read_camera_requests
    from src.quality_native_gaussian import make_native_view
    from src.scaffold_visibility_supervision import load_visibility_extension,render_frozen_contribution_diagnostics
    from src.scaffold_semantic_sidecar import _render_color_channels
    torch.set_num_threads(4);torch.manual_seed(0)
    parent=ROOT/cfg['parent_run'];batch=ROOT/cfg['semantic_batch'];src=a.artifact_root/'r1/third_party/scaffold-gs'
    pr=json.loads((parent/'result.json').read_text());before={k:sha256(parent/k) for k in pr['artifacts']};assert before==pr['artifacts']
    copy=out/'model';(copy/'point_cloud').mkdir(parents=True)
    shutil.copy2(parent/'model/cfg_args',copy/'cfg_args');shutil.copytree(parent/'model/point_cloud/iteration_30000',copy/'point_cloud/iteration_30000')
    copy_hash={str(p.relative_to(copy)):sha256(p) for p in copy.rglob('*') if p.is_file()}
    dump(out/'binding.json',dict(experiment_id=cfg['experiment_id'],git_commit=cfg['git_commit'],seed=0,config_sha256=sha256(a.config),split_sha256=sha256(parent/'split.json'),manifest_sha256=sha256(ROOT/cfg['dataset_manifest']),parent_artifacts=before,copy_artifacts=copy_hash,script_sha256=sha256(Path(__file__))))
    renderer=ScaffoldQualityRenderer(src,copy,30000,train_camera_count=400);model=renderer.model
    signature=renderer.rgb_signature();cameras=read_camera_requests(parent/'scene/sparse/0')
    ext=load_visibility_extension(out/'extensions')
    from gaussian_renderer import prefilter_voxel,generate_neural_gaussians
    bg=torch.zeros(3,device='cuda')
    def geometry(n):
        view=make_native_view(cameras[n]);visible=prefilter_voxel(view,model,renderer.pipe,bg)
        xyz,rgb,op,sc,rot,_,active=generate_neural_gaussians(view,model,visible,is_training=True)
        ids=(torch.nonzero(visible).flatten()[:,None]*model.n_offsets+torch.arange(model.n_offsets,device='cuda')).flatten()[active]
        return view,(xyz,op,sc,rot,active),rgb,ids
    def png(t):return np.rint(t.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    def protect(n):
        labels=np.asarray(Image.open(batch/'labels'/n));return binary_dilation(np.isin(labels,[1,2,3,4]),iterations=cfg['protection_dilation_px'])
    def roi_mask(spec):
        m=np.zeros((989,1320),bool)
        for x0,y0,x1,y1 in spec['boxes']:m[y0:y1,x0:x1]=True
        return m
    groups={};candidates={};baselines={};photos={};protections={};bindings={}
    (out/'images').mkdir();(out/'attribution').mkdir()
    with torch.inference_mode():
        for n in cfg['views']:
            protections[n]=protect(n);photos[n]=np.asarray(Image.open(parent/'scene/images'/n).convert('RGB'))
            baseline=renderer.render_pinhole(cameras[n]).rgb;baselines[n]=np.rint(baseline*255).astype(np.uint8)
            old=batch/'export/images'/n
            assert np.array_equal(baselines[n],np.asarray(Image.open(old))),n
            Image.fromarray(baselines[n]).save(out/'images'/('base-'+n))
            bindings[n]=dict(photo=sha256(parent/'scene/images'/n),label=sha256(batch/'labels'/n),export=sha256(old))
        for group,spec in cfg['rois'].items():
            n=spec['view'];view,g,rgb,ids=geometry(n)
            assert np.array_equal(png(_render_color_channels(g,rgb,view,bg)),baselines[n])
            stats=render_frozen_contribution_diagnostics(view,model,renderer.pipe,extension=ext)
            dom=stats['dominant_offset_index'].cpu().numpy();maximum=stats['max_contribution'].cpu().numpy();roi=roi_mask(spec)&~protections[n]
            residual=np.abs(baselines[n].astype(float)-photos[n]).mean(2)/255
            valid=roi&(dom>=0);scores=np.bincount(dom[valid],weights=maximum[valid]*residual[valid],minlength=len(ids))
            top=np.argsort(-scores)[:cfg['top_k_per_roi']];top=top[scores[top]>0];stable=ids.cpu().numpy();groups[group]=[int(stable[i]) for i in top]
            np.savez_compressed(out/'attribution'/(group+'.npz'),dominant_stable_id=np.where(dom>=0,stable[np.maximum(dom,0)],-1),maximum=maximum,roi=roi)
            for i in top:
                sid=int(stable[i]);sc=g[2][i].cpu().numpy()
                candidates.setdefault(sid,dict(id=sid,anchor_id=sid//model.n_offsets,offset_id=sid%model.n_offsets,source_rois=[],protection=[],contributions=[]))['source_rois'].append(dict(group=group,dominant_residual_score=float(scores[i]),xyz=g[0][i].cpu().tolist(),scale=sc.tolist(),anisotropy=float(sc.max()/sc.min()),opacity=float(g[1][i])))
        dump(out/'candidates_initial.json',dict(groups=groups,candidates=list(candidates.values())))
        # Exact T*alpha of each candidate, packed three at a time in RGB channels.
        for n in cfg['views']:
            view,g,rgb,ids=geometry(n);all_ids=list(candidates);prot=torch.as_tensor(protections[n],device='cuda')
            for offset in range(0,len(all_ids),3):
                chunk=all_ids[offset:offset+3];colors=torch.zeros_like(rgb)
                for ch,sid in enumerate(chunk):colors[:,ch]=(ids==sid).float()
                mass=_render_color_channels(g,colors,view,bg)
                for ch,sid in enumerate(chunk):
                    m=mass[ch];mp=float(m[prot].max());candidates[sid]['protection'].append(dict(view=n,max_Talpha=mp,sum_Talpha=float(m[prot].sum())))
                    candidates[sid]['contributions'].append(dict(view=n,total_Talpha=float(m.sum()),roi_Talpha={k:float(m[torch.as_tensor(roi_mask(s),device='cuda')].sum()) for k,s in cfg['rois'].items() if s['view']==n}))
            print(json.dumps(dict(stage='contribution',view=n,candidates=len(candidates))),flush=True)
        safe={sid for sid,c in candidates.items() if max(x['max_Talpha'] for x in c['protection'])<=cfg['protection_max_contribution']}
        accepted={k:sorted(set(v)&safe) for k,v in groups.items()};accepted['union']=sorted(safe)
        dump(out/'selection.json',dict(groups=groups,accepted=accepted,rejected=sorted(set(candidates)-safe),candidates=list(candidates.values())))
        rows=[]
        for n in cfg['views']:
            view,g,rgb,ids=geometry(n);base=baselines[n];photo=photos[n]
            for group,selected in accepted.items():
                # Apply same stable IDs in all views, no image-space clipping.
                op=g[1].clone();mask=torch.isin(ids,torch.tensor(selected,device='cuda',dtype=ids.dtype));op[mask]=0
                changed_g=(g[0],op,g[2],g[3],g[4]);candidate=png(_render_color_channels(changed_g,rgb,view,bg))
                Image.fromarray(candidate).save(out/'images'/(group+'-'+n))
                diff=np.abs(candidate.astype(float)-base.astype(float));b_err=np.abs(base.astype(float)-photo).mean(2);c_err=np.abs(candidate.astype(float)-photo).mean(2);pm=protections[n]
                masks={'frame':np.ones(pm.shape,bool),'protected':pm}|{k:roi_mask(s) for k,s in cfg['rois'].items() if s['view']==n}
                row=dict(view=n,group=group,selected_ids=selected,active_suppressed=int(mask.sum()),metrics={k:dict(pixels=int(m.sum()),changed_pixels=int(np.any(diff>0,2)[m].sum()),delta_rgb_mae=float(diff[m].mean()),max_rgb_delta=float(diff[m].max()),photo_mae_before=float(b_err[m].mean()),photo_mae_after=float(c_err[m].mean())) for k,m in masks.items()})
                rows.append(row)
            print(json.dumps(dict(stage='ablation',view=n)),flush=True)
        # Undo check on three targets; model parameters and on-disk copies must be exact.
        for n in ['159.png','160.png','174.png']:assert np.array_equal(png(torch.as_tensor(renderer.render_pinhole(cameras[n]).rgb,device='cuda').permute(2,0,1)),baselines[n])
    assert renderer.rgb_signature()==signature
    assert {k:sha256(parent/k) for k in before}==before
    assert {str(p.relative_to(copy)):sha256(p) for p in copy.rglob('*') if p.is_file()}==copy_hash
    result=dict(experiment_id=cfg['experiment_id'],status='completed_diagnostic_only',git_commit=cfg['git_commit'],seed=0,config_sha256=sha256(a.config),split_sha256=sha256(parent/'split.json'),view_input_hashes=bindings,baseline_export_exact=len(baselines),parent_and_copy_unchanged=True,undo_exact=3,candidate_count=len(candidates),accepted=accepted,rows=rows,seconds=time.monotonic()-start)
    dump(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','view_input_hashes']}),flush=True)
if __name__=='__main__':main()
