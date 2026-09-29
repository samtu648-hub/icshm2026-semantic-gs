#!/usr/bin/env python3
"""Finalize RGB145/146 records only after saved metrics, audit and visual judgment exist."""
import argparse
import html
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'outputs/quality/rgb144');p.add_argument('--judgment',type=Path,required=True);a=p.parse_args();out=a.root
    data=json.loads((out/'review-r1/data.json').read_text());audit=json.loads((out/'full-pair-audit.json').read_text());assert audit['status']=='passed'
    judgment=json.loads(a.judgment.read_text());assert judgment['decision'] in ['drop_recipe_preserve_results','keep_research_candidate']
    placements=json.loads((out/'placement-distribution-r1/result.json').read_text());matching=json.loads((out/'matching-resources.json').read_text())
    geo=json.loads((out/'photo-audit.json').read_text());source=json.loads((out/'candidate-r1/started.json').read_text())
    metrics=data['metrics'];resources=data['resources'];paired=data['paired']['RGB145'];ctrl=resources['RGB145'];cand=resources['RGB146'];delta_points=(cand['final_gaussians']/ctrl['final_gaussians']-1)*100
    lines=['# RGB-146-STANDARD-GEOMETRY-SPLIT','',judgment['conclusion'],'',
        '## 配方与身份','',
        f"父RGB140；Git `{source['git_commit']}` 加run/code源码快照；seed0；manifest `data/manifests/contest_dataset_v1.json`；split `released_all400_rgb_all300_sem_v1`。400训练图、0留出，官方PINHOLE相机固定，1320×989，原72,292点初始化，30k。全400指标只称训练拟合；144官方参考轨迹无GT。",
        'RGB145是本轮实际重训的新控制，不是历史RGB140或RGB142结果的重命名。两臂使用RGB142相同绝对梯度排序、每事件最多2048父点和线性人口包络；原生signed克隆、opacity reset、剪枝、颜色/尺度/旋转优化与损失保持。唯一臂间变量是部分第一子高斯XYZ替换为照片三角化位置，未用RGB143支撑权重。',
        '两臂均读取同一缓存并关联：最近8个未用提议、父3σ椭球、至少3观测投影3σ椭圆、归一化距离/置信度/稳定ID排序，每次至多25%，每提议每臂最多用一次。先完整执行原生两个子点随机采样；候选再替换固定编号之一。两个子点其他属性和优化器初始化均为原生。无语义/渲染深度输入、相机调整、在线匹配、额外专家训练或参数扫描。','',
        '## 全400训练拟合','']
    for model in ['RGB140','RGB145','RGB146']:
        m=metrics[model];lines.append(f"- {model}：PSNR {m['PSNR']:.9f} dB；SSIM {m['SSIM']:.9f}；LPIPS {m['LPIPS']:.9f}。")
    lines+=['', '候选相对新控制：']
    for key,v in paired.items():lines.append(f"- {key}平均差 {v['mean_delta']:+.9f}；{v['improved']}图改善、{v['degraded']}图退化。")
    lines+=['',f"最终高斯：控制{ctrl['final_gaussians']:,}，候选{cand['final_gaussians']:,}，差{delta_points:+.3f}%。共同分裂上限和人口包络不保证后续克隆/剪枝路径完全等量；不能把历史RGB140容量差异单独归因于落点。",'', '## 局部收益与损伤','']
    for row in data['regions']:
        if row['region'] in ['159-shore','160-shore','159-near','160-near','159-far-occlusion','159-water-ghost','174-sky'] or row['region'].endswith(('stay_cable','deck','structure-label-boundary')):
            before,after=row['RGB145'],row['RGB146'];percent=(after/before-1)*100 if before else 0
            lines.append(f"- {row['region']}：RGB MAE（0–255） {before:.6f} → {after:.6f}（{percent:+.3f}%）；历史RGB140 {row['RGB140']:.6f}。")
    lines+=['','五类已有标签区域只用于RGB误差审计，不是语义IoU：']
    for cls,row in data['class_rgb_mae_255'].items():lines.append(f"- {cls}：控制{row['RGB145']:.6f} → 候选{row['RGB146']:.6f}；历史RGB140 {row['RGB140']:.6f}（255单位，逐图等权）。")
    lines+=['','人工同位置与轨迹观察：','']+judgment['visual_observations']+['', '## 几何证据和干预范围','',
        f"冻结LoFTR/Kornia0.8.1共1873照片对，{geo['raw_forward_matches']:,}前向匹配，{geo['accepted_pair_matches']:,}通过成对检查，2080可靠多视角提议。全部原始/拒绝缓存保留，来源、许可证、权重/包SHA256及Kornia上游commit见dependencies。照片来自本400训练集，与当前高斯模型独立构建，但不是独立验证。",
        '四相机/额外视角/扰动不确定性只限定提议适用性，不证明几何真值。159远端仅2条背景对应、159/160岸线2/3条、水面/近景ROI无通过点。固定九图Class ID2区域仅081/160各1条，标签扇区不等同逐根实体索。无对应不能当成空白、错误或否定细索。RGB144详细分类/贡献/50×400有限移除见其报告和geometry-review-r1。',
        f"完整候选实际引导{placements['actual_guided']:,}次，占{cand['total_split_parents']:,}次选中分裂父点的{placements['actual_guided']/cand['total_split_parents']*100:.3f}%，未用{placements['unused_proposals']:,}条。逐事件父ID/提议/原生子点与目标XYZ/位移及拒绝原因保留。空间分布见placement-distribution-r1；以下为固定ROI实际覆盖："]
    for key,row in placements['regions'].items():lines.append(f"- {key}：接受{row['accepted']}，实际使用{row['guided']}。")
    for key in ['081.png:2','160.png:2']:
        row=placements['label_region_coverage'][key];lines.append(f"- 固定九图stay_cable标签区域 {key}：接受{row['accepted']}，实际使用{row['guided']}；标签区域不等于实体索逐像素对应。")
    lines+=['', '只有这些对应覆盖的位置能讨论几何引导响应；其他区域的最终变化可能来自后续优化/克隆/剪枝耦合，不能说被直接几何约束。子点之后继续自由优化并可能被剪枝，引导次数不等同最终正确表面数。3σ是父协方差相对距离，不是绝对小位移保证；少数巨大父点存在位移长尾。不能凭空补回远离父高斯的整片缺失。',
        '位移分位（原生随机子点→照片位置，场景单位）：'+json.dumps(placements['native_child_to_photo_displacement_quantiles_scene_units'],ensure_ascii=False),'', '## 成本与验证','']
    for model,r in resources.items():lines.append(f"- {model}：训练{r['training_wall_seconds']/60:.3f}分钟；分裂hook累计（含筛选、关联、原生分裂与记录）{r['association_seconds']:.3f}秒；峰值allocated/reserved {r['peak_allocated_mib']:.1f}/{r['peak_reserved_mib']:.1f}MiB；整卡采样峰值{r['sampled_whole_gpu_peak_mib']:.1f}MiB；模型{r['model_mib']:.3f}MiB；固定九图GPU渲染中位{r['fixed9_cuda_render_median_ms']:.3f}ms。")
    lines+=['',f"匹配循环共{matching['total_matching_loop_seconds']/60:.3f}分钟，峰值allocated2691.270MiB；固定52对19.881秒外推剩余696.215秒，实际剩余1821对675.081秒。预检两臂合计251.189秒；冻结诊断328.501秒。这些附加成本没有隐藏在成功30k训练时间里。磁盘与最终健康检查另见delivery-audit.json。",
        '7项合成几何和4项关联测试通过。preflight-r2原生前向、同一参数梯度和绝对统计保持；零干预/控制拓扑、非XYZ属性、优化器初始化和CUDA随机状态精确相同，候选只改变指定第一子点XYZ。两臂1601步后候选实际1404次位置替换，数值正常，随后按固定配置从头30k；未以短期质量调参。',
        '完整pair audit核验源代码、配置、缓存、实际400相机/0留出、初始化、300个记录相机顺序、25%配额、3σ/至少3投影、提议不复用和0/实际应用。恢复状态含模型、优化器、曝光状态、所有随机状态、相机队列、增密统计和已用提议；每1000步保存，7k/15k/30k模型另存。恢复入口resume_standard_geometry.py保持原实际迭代预算并拒绝源码/config变更。',
        f"两臂1088评价PNG、同144相机元数据、九图fresh及历史RGB140九图重放均核验；1200图次独立float64 PSNR复算最大差{data['independent_float64_psnr_max_error']:.3g}dB。局部误差/五类区域由保存PNG复算。",
        '工程异常保留：初始相对路径规范化错误发生于匹配前；首预检两次原生CUDA反向的少量XYZ原子舍入差最大6.08e-6，随后改为同一梯度保持和原生统计精确重放检查，并未放宽科学几何筛选。旧覆盖混入769条无保留三维点观察，独立纠正覆盖图/计数，原GPU输出保留，匹配/提议/训练不变。','', '## 判断与回退','',judgment['conclusion'],
        '本判断针对一次固定配方/seed；未估计重复训练的随机波动或统计显著性，不新增逐类全胜硬门，也不证明所有几何引导方法无效或有效。保留旧模型、新模型、失败记录和恢复状态。默认RGB130+SEM371、研究系统候选RGB140+SEM378继续使用原路径；未覆盖模型，无需改写式回退。',
        '本次重建对照至此结束；DINOv3、细索语义采样、匹配语义重训、长轴分裂、动态水面和多模块组合均未启动。无Git提交或推送。',
        '浏览：outputs/quality/rgb144/review-r1/index.html；几何证据：outputs/quality/rgb144/geometry-review-r1/index.html；逐图CSV、144帧视频、全部接触表和固定放大图均在review-r1。']
    candidate='RGB-146-STANDARD-GEOMETRY-SPLIT';control='RGB-145-STANDARD-GEOMETRY-CTRL'
    (ROOT/f'experiments/runs/{candidate}/report.md').write_text('\n'.join(lines)+'\n')
    (ROOT/f'experiments/runs/{control}/report.md').write_text('# '+control+'\n\n完成本轮实际全400/30k新预算控制，不能冒充历史RGB140。父RGB140、seed0、官方相机、原72292点初始化与候选完全匹配；两臂读取同一照片几何缓存，控制只计算关联、实际XYZ替换0次。\n\n'+
        f"Git {source['git_commit']}加run/code源码快照；manifest contest_dataset_v1；split released_all400_rgb_all300_sem_v1；全部400为训练拟合，0留出。\n\n结果："+json.dumps(metrics['RGB145'],ensure_ascii=False)+f"；最终{ctrl['final_gaussians']}高斯；训练{ctrl['training_wall_seconds']/60:.3f}分钟。\n\n完整对照、局部损伤、资源、审计与保留判断见RGB146报告。旧默认与系统候选保持；本任务结束，无额外语义训练或参数扫描。\n")
    rp=ROOT/'experiments/registry.yaml';registry=json.loads(rp.read_text())
    for e in registry['experiments']:
        if e['experiment_id'] in [control,candidate]:
            model='RGB145' if e['experiment_id']==control else 'RGB146';e.update(status='completed_matched_control' if model=='RGB145' else 'completed_geometry_candidate',end_date='2026-09-16',metrics_summary=metrics[model]|{'quality_scope':data['scope']},resources=resources[model],source_snapshot=f"outputs/quality/rgb144/{'control' if model=='RGB145' else 'candidate'}-r1/code",review_path='outputs/quality/rgb144/review-r1/index.html',validation={'full_pair_audit':'passed','old_models_preserved':True},keep_or_drop='matched_control_retained' if model=='RGB145' else judgment['decision'])
            if model=='RGB146':e.update(paired_control=control,paired_changes=data['paired'],actual_guided_placements=placements['actual_guided'],conclusion=judgment['conclusion'])
    rp.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n')
    page=out/'review-r1/index.html';markup=page.read_text();start=markup.index('<!-- REVIEW_START -->');end=markup.index('<!-- REVIEW_END -->')+len('<!-- REVIEW_END -->')
    review='<p><strong>'+html.escape(judgment['conclusion'])+'</strong></p><details><summary>展开局部收益、损伤与轨迹复核记录</summary><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in judgment['visual_observations'])+'</ul></details>'
    page.write_text(markup[:start]+'<!-- REVIEW_START -->'+review+'<!-- REVIEW_END -->'+markup[end:])


if __name__=='__main__':main()
