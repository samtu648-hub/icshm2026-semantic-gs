#!/usr/bin/env python3
"""Seal actual results and update project records only after visual review."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json
BASE=ROOT/'outputs/quality/backbone_comparison'
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--browser',type=Path,required=True);p.add_argument('--review',type=Path,required=True);a=p.parse_args()
 a.browser=a.browser.resolve();a.review=a.review.resolve()
 review=a.review.read_text();assert len(review)>100 and 'TODO' not in review
 data=read(a.browser/'data.json');assert len(data['images'])==400 and data['trajectory_frames']==144
 baseline=read(BASE/'baseline_snapshot.json');verify_hashes(ROOT,baseline['hashes'])
 frozen={x['path']:x['sha256'] for x in read(ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json')['artifacts']};verify_hashes(a.artifact_root,frozen)
 common=read(ROOT/'outputs/quality/all_data/RGB-120-r2/worker_train.json')['initialization']['canonical_ply_sha256']
 for arm in ['standard','mip']:
  r=read(BASE/(arm+'-full-r1')/'result.json');assert r['status']=='completed_all400' and len(r['worker']['actual_train'])==400 and not r['worker']['actual_evaluation']
  assert r['worker']['initialization']['canonical_ply_sha256']==common
  assert r['command'][r['command'].index('--iterations')+1]=='30000'
  fit=read(BASE/arm/'mask2former/result.json');assert fit['status']=='completed' and fit['trained_steps']==6000 and fit['sampling_matches_sem370'] and fit['resolved_recipe_matches_sem370']
  assert fit['pretrained_load']==read(ROOT/'outputs/quality/all_data/SEM-370-r1/mask2former/result.json')['pretrained_load']
  verify_hashes(BASE/arm/'mask2former',fit['artifacts'])
 assert read(BASE/'standard/mask2former/result.json')['initial_model_state_sha256']==read(BASE/'mip/mask2former/result.json')['initial_model_state_sha256']
 probes={}
 for video in a.browser.glob('*.mp4'):
  info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate','-of','json',str(video)],text=True))['streams'][0]
  assert int(info['nb_frames'])==144 and info['height']==1978 and info['width']==(3960 if video.name.startswith('three-') else 1320)
  probes[video.name]=info
 assert len(probes)==4
 summary='\n2026-09-15 实际完成三骨干全数据对照。全部指标为训练视角拟合，非隐藏视角。\n\n'
 for x in data['summaries']:
  rgb=x['rgb'];sem=x['semantic'];summary+=f'- {x["label"]}：RGB PSNR {rgb["PSNR"]:.6f} dB / SSIM {rgb["SSIM"]:.6f} / LPIPS {rgb["LPIPS"]:.6f}；五类mIoU {100*sem["mIoU_with_background"]:.6f}%。\n'
  for name,m in sem['per_class'].items():summary+=f'  - {name}：IoU {100*m["IoU"]:.6f}% / precision {100*m["precision"]:.6f}% / recall {100*m["recall"]:.6f}%。\n'
  summary+=f'  - RGB总时间 {x["rgb_wall_seconds"]:.3f}s；M2F训练/总时间 {x["m2f_training_seconds"]:.3f}/{x["m2f_wall_seconds"]:.3f}s；RGB/M2F峰值allocated {x["rgb_peak_allocated_mib"]:.1f}/{x["m2f_peak_allocated_mib"]:.1f}MiB；渲染/语义推理中位 {x["render_median_ms"]:.3f}/{x["inference_median_ms"]:.3f}ms；部署RGB/语义检查点 {x["rgb_deploy_mib"]:.3f}/{x["semantic_checkpoint_mib"]:.3f}MiB。Scaffold训练时间复用历史记录，推理本轮同机实测。\n'
 sup=data['scaffold_native_appearance'];summary+=f'\nScaffold主表沿用SEM370输入的uid0。补充训练UID的RGB-only：PSNR {sup["PSNR"]:.6f} / SSIM {sup["SSIM"]:.6f} / LPIPS {sup["LPIPS"]:.6f}，不改变SEM370输入或混作另一套语义结果。\n\n'+review
 summary+='\n\n验证：相同初始化、两次30k/400访问、两次6000采样序列/配方/预训练核对、三套fresh九概率重载、RGB哈希保持、18冻结及基线文件保持、4视频各144原生帧。完整明细及逐图/局部计数见对照页data.json和阶段result.json。回退：继续引用原RGB120+SEM370，原系统未被替换；无删除、上传或Git push。\n'
 report=ROOT/'experiments/runs/QUAL-007-BACKBONE-COMPARISON/report.md'
 if summary not in report.read_text():
  with report.open('a') as f:f.write(summary)
 mapping={'RGB-130-STANDARD-ALL400':'standard-full-r1','RGB-131-MIP-ALL400':'mip-full-r1','SEM-371-STANDARD-ALL300':'standard/mask2former','SEM-372-MIP-ALL300':'mip/mask2former'}
 for eid,rel in mapping.items():
  r=read(BASE/rel/'result.json')
  path=ROOT/'experiments/runs'/eid/'report.md'
  note=f'\n完整实际结果已核验：{r["status"]}，{r["seconds"]:.3f}s。逐项指标、局部退化、资源和不自动替换的判断见[QUAL007](../QUAL-007-BACKBONE-COMPARISON/report.md)。\n'
  if note not in path.read_text():
   with path.open('a') as f:f.write(note)
 reg=ROOT/'experiments/registry.yaml';registry=read(reg)
 for item in registry['experiments']:
  eid=item['experiment_id']
  if eid in mapping:item.update(status='completed_full_data_comparison',output_path=str((BASE/mapping[eid]).relative_to(ROOT)),result_sha256=sha256(BASE/mapping[eid]/'result.json'))
  if eid=='QUAL-007-BACKBONE-COMPARISON':item.update(status='completed_comparison_no_automatic_replacement',output_path=str(a.browser.relative_to(ROOT)),result_sha256=sha256(a.browser/'data.json'))
 reg.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n')
 note='2026-09-15 RGB130/131与SEM371/372、QUAL007完成：两套400/30k重建及匹配300/6k M2F、三系统400 RGB/300五类指标、固定九图及144帧统一轨迹已实际完成；RGB120/SEM370复用且不替换。结论与局部代价见experiments/runs/QUAL-007-BACKBONE-COMPARISON/report.md。本轮无剩余训练任务，后续是否切换由用户决定。'
 for name in ['PROJECT_STATUS.md','TASKS.md','ROADMAP.md']:
  path=ROOT/name;s=path.read_text();head,tail=s.split('\n',1);path.write_text(head+'\n\n'+note+'\n'+tail)
 with (ROOT/'DECISIONS.md').open('a') as f:f.write('\n\nQUAL007完成（2026-09-15）：'+review.rstrip()+'\n')
 with (ROOT/'RISKS.md').open('a') as f:f.write('\n\nQUAL007完成：全400/全300均为训练拟合、单seed与历史九图选择，不能替代隐藏视角泛化。Scaffold训练UID与SEM370固定uid0的输入口径明确分列；无GT轨迹只供视觉判断。旧成果保持，无自动替换。\n')
 write_new_json(BASE/'final_verification.json',dict(status='completed',baseline_files_unchanged=len(baseline['hashes']),frozen_unchanged=len(frozen),videos=probes,browser_data_sha256=sha256(a.browser/'data.json'),visual_review_sha256=sha256(a.review)))
 print('Finalized verified comparison; run repository health after record updates.')
if __name__=='__main__':main()
