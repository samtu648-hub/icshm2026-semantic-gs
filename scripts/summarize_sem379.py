#!/usr/bin/env python3
"""Write observed SEM379 quantities only, after all audits have completed."""
import json,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/quality/SEM-379/r1';BASE=ROOT/'outputs/quality/SEM-378/r1'
sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes

def read(p):return json.loads(p.read_text())
def pc(x):return '未定义' if x is None else f'{100*x:.6f}%'
def main():
 a=read(OUT/'audit.json');b=read(OUT/'boundary-audit.json');fit=read(OUT/'mask2former/result.json');inf=read(OUT/'inference/result.json');rgb=read(OUT/'fresh_rgb_audit.json');ck=read(OUT/'checkpoint_audit.json');cfg=read(ROOT/'experiments/runs/SEM-379-CABLE-REGION-SAMPLING/config.yaml');oldfit=read(BASE/'mask2former/result.json')
 assert all(r['status']=='completed' for r in [a,b,fit,inf,rgb,ck])
 assert fit['trained_steps']==6000 and fit['sampling6000_matches_plan'] and fit['initial_state_matches_sem378'] and fit['resolved_recipe_matches_sem378']
 assert inf['fresh_nine_probability_replay_exact'] and rgb['fresh_native_png_exact']==544
 verify_hashes(ROOT,read(OUT/'binding.json')['code_sha256'])
 verify_hashes(ROOT,read(OUT/'analysis-source-before-evaluation.json')['sha256'])
 old,new=[a['metrics'][k] for k in ['sem378','sem379']];c=b['binary_cable_changes'];paired=a['paired'];v0=old['per_class']['stay_cable'];v1=new['per_class']['stay_cable']
 lines=['# SEM379 定量结果（人工视觉结论另列）','',f"配置：`experiments/runs/{cfg['experiment_id']}/config.yaml`；Git `{cfg['git_commit']}`＋源码快照；seed {cfg['seed']}；manifest `{cfg['dataset_manifest']}`；split `{cfg['split_id']}`。",'',
 f"全300训练拟合 mIoU：{pc(old['mIoU_with_background'])} → {pc(new['mIoU_with_background'])}（{paired['global_mIoU_delta_pp']:+.6f} pp）。逐图{paired['improved_views']}改善/{paired['degraded_views']}退化/{paired['tied_views']}持平。",'','## 原始五类 IoU','']
 for name,v in new['per_class'].items():lines.append(f"- {name}：{pc(old['per_class'][name]['IoU'])} → {pc(v['IoU'])}（{100*(v['IoU']-old['per_class'][name]['IoU']):+.6f} pp）。")
 lines+=['','## stay_cable 收益与代价','',f"- precision：{pc(v0['precision'])} → {pc(v1['precision'])}；recall：{pc(v0['recall'])} → {pc(v1['recall'])}。",f"- FP：{v0['fp']:,} → {v1['fp']:,}（{v1['fp']-v0['fp']:+,}）；FN：{v0['fn']:,} → {v1['fn']:,}（{v1['fn']-v0['fn']:+,}）。",f"- 真实索漏检修复 {c['fn_fixed']:,}，原正确索损失 {c['true_cable_lost']:,}。",f"- 二值索误报移除 {c['fp_removed']:,}，新增误报 {c['new_fp']:,}；其中真背景移除/新增黄色误报 {c['background_fp_removed']:,}/{c['new_background_fp']:,}。二值移除误报不必然等于五类修对。",f"- 全五类修对 {paired['corrected_pixels']:,}、改错 {paired['harmed_pixels']:,}、错误类别互换 {paired['still_wrong_changed_pixels']:,}，净修对 {paired['corrected_pixels']-paired['harmed_pixels']:+,}。",'','各GT类别的五类修对/改错：','']
 for name,v in paired['per_class'].items():lines.append(f"- {name}：{v['corrected']:,} / {v['harmed']:,}（净{v['corrected']-v['harmed']:+,}）。")
 lines+=['','## 标注索区域边界（原生2px容差）','']
 for arm in ['sem378','sem379']:
  v=b[arm];lines.append(f"- {arm}：precision {pc(v['boundary_precision'])}、recall {pc(v['boundary_recall'])}、F1 {pc(v['boundary_f1'])}；GT索边界2px带/内部FN {v['cable_fn_boundary_interior'][0]:,}/{v['cable_fn_boundary_interior'][1]:,}。")
 lines+=['','这是GT区域边界，不是逐根物理索边界。','', '## 固定九视角困难框','']
 fixed=read(ROOT/'experiments/runs/SEM-370-ALL300/config.yaml')['diagnostic_views']
 for n in fixed:
  matches=[v for v in a['regions'] if v['view']==n and v['region'].startswith('difficult-')]
  for v in matches:lines.append(f"- {n}：局部mIoU {v['delta_pp']:+.6f}pp；修对/改错 {v['corrected']:,}/{v['harmed']:,}；真索修对/改错 {v['per_class']['stay_cable']['corrected']:,}/{v['per_class']['stay_cable']['harmed']:,}。")
 lines+=['','ROI mIoU按活跃类别计算；小框出现/消失类别会放大变化，须同时看像素计数。全部固定RGB/结构/困难区域见audit.json。','','## 无GT轨迹代理','']
 t=a['trajectory_proxy']
 for arm in ['sem378','sem379']:lines.append(f"- {arm}：相邻帧标签变化均值 {pc(t[arm]['mean_adjacent_label_change'])}；最大 {pc(t[arm]['max_adjacent_label_change'])}。")
 d=t['candidate_vs_baseline'];lines.append(f"两模型逐帧标签分歧均值 {pc(d['mean_label_disagreement'])}，最大 {pc(d['max_label_disagreement'])}（frame{d['max_frame_index']:04d}）。")
 lines+=['','这些量混合相机运动、遮挡与边界变化，不是准确率或独立闪烁指标；100无标签与144轨迹只定性。','','## 资源与核验','',f"- 训练本体 {fit['training_seconds']:.3f}s（{fit['training_seconds']/60:.3f}分钟），相对SEM378 {100*(fit['training_seconds']/oldfit['training_seconds']-1):+.3f}%。",f"- 成功更新6000；AMP重试{fit['amp_skips']}；allocated/reserved峰值 {fit['peak_allocated_mib']:.3f}/{fit['peak_reserved_mib']:.3f} MiB。",f"- 成功训练实例锁等待 {fit['queue_seconds']:.3f}s；之前终止/迁移等待另外保留，不计入训练本体。",f"- 544推理阶段 {inf['seconds']:.3f}s（含排队/加载/核验），单图中位 {1000*inf['inference_median_seconds']:.3f}ms；纯逐图推理时间和 {sum(inf['inference_seconds']):.3f}s。",f"- 600个逐图混淆独立复算；九图新进程概率精确；544 fresh RGB逐像素一致；两checkpoint各{ck['checkpoints'][0]['tensors']} tensor全部有限。",'- 300标签/544渲染/原RGB140端点/SEM378端点/正式18冻结项哈希保持；完整6000实际采样日志与事前计划一致；初始化和resolved配方一致。']
 size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())
 lines.append(f'- 本任务输出约 {size/2**30:.3f} GiB（不重复计算共享RGB目录）。')
 lines+=['','## 原始产物身份','']
 for rel in ['mask2former/model-6000.pt','inference/result.json','audit.json','boundary-audit.json','comparison/trajectory-rgb140-sem378-rgb140-sem379.mp4']:
  lines.append(f'- `{rel}`：`{sha256(OUT/rel)}`。')
 (OUT/'quantitative-report.md').write_text('\n'.join(lines)+'\n')
 print('\n'.join(lines[:25]))
if __name__=='__main__':main()
