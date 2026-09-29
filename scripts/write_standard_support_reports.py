#!/usr/bin/env python3
"""Write evidence-bound reports after paired metrics and explicit visual review."""
import json
import html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'outputs/quality/rgb141'
    data=json.loads((out/'review-r1/data.json').read_text())
    visual=json.loads((out/'review-r1/visual_review.json').read_text())
    audit=json.loads((out/'pair_audit.json').read_text())
    candidate_run=data['runs']['RGB143']
    assert audit['status']=='passed' and visual['decision'] in ['keep_research_candidate','drop_recipe']
    page=out/'review-r1/index.html'
    content=page.read_text()
    start,end='<!-- REVIEW_START -->','<!-- REVIEW_END -->'
    findings=visual['fixed_region_findings']+visual['trajectory_findings']
    review='<h2>复核结论</h2><p>'+html.escape(visual['conclusion'])+'</p><details><summary>收益、损伤和检查范围</summary><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in findings)+'</ul></details><p><a href="visual_review.json">视觉复核记录与抽查范围</a></p>'
    page.write_text(content.split(start)[0]+start+review+end+content.split(end)[1])
    models=['RGB140','RGB142','RGB143'];metrics=['PSNR','SSIM','LPIPS']
    lines=['# RGB-143-STANDARD-SUPPORT-DENSIFY — Standard 跨视角支撑引导分裂',
        '', '## 结论', '', visual['conclusion'], '',
        '两臂均完成全400 RGB、官方相机、原72,292点初始化、1320×989、seed0、30k。RGB142是共享预算下的新控制，不能冒充历史RGB140；RGB143只用连续光度边际支撑改变大高斯分裂排序。全部400图指标都是训练拟合，144帧轨迹无GT，不是未知视角准确率或闪烁改善证明。', '',
        '## 身份与源码', '',
        '- 父实验：RGB-140-STANDARD-ABSGRAD-ALL400；匹配控制：RGB-142-STANDARD-BUDGET-CTRL；只读诊断：RGB-141-STANDARD-SUPPORT-DIAG。',
        '- Git HEAD：49cdcdb1c9915a9544d568754af95ea20a85ec9c，工作区原有大量未提交修改。不能单靠commit复现，实际执行源码、配置和上游源码/扩展SHA绑定在各run的code/、started.json、result.json及executed_train.py。',
        '- manifest：data/manifests/contest_dataset_v1.json；split：released_all400_rgb_all300_sem_v1；实际400训练相机、0留出。',
        '- scene：outputs/quality/all_data/RGB-120-r2/scene；初始PLY SHA256：10a88fb018c27530b3bbfa0f7000408693ba75228067f5311e7775e4755e68f2。400照片哈希、官方相机文件与父PLY核验见input_identity.json、pair_audit.json。',
        '- Standard源码54c035f7834b564019656c3e3fcc3646292f727d加RGB140原绝对梯度补丁；Gaussian-Splatting research/evaluation license。828项原生源码/扩展与RGB140一致。实际PyTorch 2.11.0+cu128、CUDA12.8、RTX5070，双方与父训练原始worker记录一致。',
        f'- 正式输出：outputs/quality/rgb141/control-r1 和 {candidate_run}；浏览入口：outputs/quality/rgb141/review-r1/index.html。', '',
        '## 证据与方法', '',
        'RGB123的梯度抵消并非伪影专属；RGB124保留signed资格、500步/640新增的Scaffold排序负结果不能冒充本次Standard完整实验。RGB134/135显示贡献者兼有真实结构和伪影职责，局部opacity修复没有解决主要问题，本轮不重复固定ID处理。RGB141在RGB140上固定8个贡献者后完成3200次跨视角有限去除；81–106个贡献视角仍有相反效果，真实stay_cable贡献者也可能得到低于1的权重。没有证实伪影根因。', '',
        '读取原生光度损失L=0.8L1+0.2DSSIM的opacity logit导数g_iv，tau_v=median(nonzero |g_iv|)，r_iv=|g_iv|/(|g_iv|+tau_v)，e_iv=-g_iv/(|g_iv|+tau_v)。100个不同训练相机窗口内R=sum r、E=sum e，w=1+0.5E/(4+R)。控制分数G_abs，候选分数G_abs*w；无有效证据时w=1，范围[0.5,1.5]。少可见视角不会单独导致惩罚。', '',
        'alpha合成导数包含透射与后方颜色影响，是当前模型下的遮挡感知光度边际证据；弱导数也可能源于细索、小投影、饱和或拟合充分，r不是可见概率。未使用渲染深度/独立对应作为真实几何。100相机并非统计独立，窗口内模型参数也在更新；这不是冻结几何的独立对应检验。SH颜色、反射和动态水面仍会影响信号。连续权重不是伪影分类器，也不保证有限删除或分裂结果。', '',
        '共同预算：每100步原生增密事件（600–14900）最多分裂2048父点，同时按B(t)=ceil(72292+(790065-72292)*clip((t-500)/14400,0,1))限制分裂前可分配额度；K=min(合资格数,2048,max(0,B-克隆后点数))。资格仍是absolute>=0.0004和大尺度条件。小高斯signed克隆0.0002、剪枝、opacity reset、颜色优化、初始化、相机与分辨率不变。克隆/剪枝可使最终点数不同，B是分裂分配上包络而非全拓扑硬上限。', '',
        '两臂都计算支撑统计，只有候选应用权重；不增加训练前向/反向次数。每步O(N)数组/中位数，两个N长度float32累积，分裂事件O(M log M)排序。host统计计时不能冒充GPU耗时；实际资源如下。未加入语义损失、新专家、初始化筛选或其他模块。', '',
        '## 全400训练拟合结果', '', '| 模型 | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ |', '|---|---:|---:|---:|']
    for m in models:lines.append('| '+m+' | '+' | '.join(f'{data["metrics"][m][k]:.9f}' for k in metrics)+' |')
    lines += ['', '| RGB143相对 | 指标 | 平均差 | 改善图 | 退化图 |','|---|---|---:|---:|---:|']
    for baseline in ['RGB142','RGB140']:
        for k in metrics:
            r=data['paired'][baseline][k];lines.append(f'| {baseline} | {k} | {r["mean_delta"]:+.9f} | {r["improved"]} | {r["degraded"]} |')
    lines += ['', '完整400逐图数据保存在review-r1/per_view.csv及data.json；网页按PSNR差展示最差到最好。此排序为事后浏览，不是新的预注册验证集。单seed和CUDA原子非确定性限制对微小差值的因果解释。', '',
        '## 固定局部与真实结构', '', '| 固定区域 | RGB140 MAE | RGB142 MAE | RGB143 MAE | 候选−新控制 |', '|---|---:|---:|---:|---:|']
    for r in data['regions']:
        lines.append('| '+r['region']+' | '+' | '.join(f'{r[m]:.6f}' for m in models)+f' | {r["RGB143"]-r["RGB142"]:+.6f} |')
    lines += ['', 'MAE单位0–255，采用相同保存PNG。固定九图另报告deck、stay_cable和结构标签两像素边界带；边界带为结构并集膨胀2像素减腐蚀2像素，不是遮挡几何真值。该补充在新模型质量评价前确定。已有300标签只用于RGB区域审计，不是语义训练或语义IoU；类别汇总先逐图求该类MAE，再对存在该类的图等权平均，不是全像素池化。', '', '| 标签区域 | RGB140 MAE | RGB142 MAE | RGB143 MAE |', '|---|---:|---:|---:|']
    for cl,r in data['class_rgb_mae_255'].items():lines.append('| '+cl+' | '+' | '.join(f'{r[m]:.6f}' for m in models)+' |')
    lines += ['', *visual['fixed_region_findings'], '', '## 同一144帧轨迹', '', *visual['trajectory_findings'], '',
        '相同官方参考坐标系和相机元数据哈希、144个pose、原生RGB渲染。三路视频为RGB140/RGB142/RGB143；没有轨迹GT，不使用帧间变化宣称准确率或闪烁改善。固定抽查与完整视频的边界按visual_review.json记录。', '',
        '## 预算与资源', '', '| 项目 | RGB142控制 | RGB143候选 |', '|---|---:|---:|']
    for label,k in [('最终高斯','final_gaussians'),('模型MiB','model_mib'),('训练墙钟秒（排队除外）','training_wall_seconds'),('worker秒','worker_seconds'),('锁等待秒','lock_wait_seconds'),('峰值allocated MiB','peak_allocated_mib'),('峰值reserved MiB','peak_reserved_mib'),('整卡采样峰值MiB','sampled_whole_gpu_peak_mib'),('全544转图渲染中位ms','render_median_ms'),('固定九图三轮CUDA渲染中位ms','fixed9_cuda_render_median_ms'),('累计分裂父点','total_split_parents'),('受预算限制事件','events_budget_limited'),('相对同事件未加权排序改选父点','ranking_changed_parents')]:
        values=[data['resources'][m][k] for m in ['RGB142','RGB143']]
        lines.append('| '+label+' | '+' | '.join(str(v) if isinstance(v,int) else f'{v:.6f}' for v in values)+' |')
    delta=data['resources']['RGB143']['final_gaussians']/data['resources']['RGB142']['final_gaussians']-1
    lines += ['', f'候选相对控制最终点数差{delta*100:+.4f}%。事件累计改选不是最终独立高斯数量，拓扑ID不能跨事件直接对齐。各事件完整数量、权重分位和选择差异见densification.jsonl。共享预算控制与原生克隆/剪枝相互作用，不能声称严格等量或排除了所有容量差异。', '',
        'GPU与独立SEM379任务通过共享锁错峰。SEM任务模型和质量结果不进入本重建实验；实际排队记录单列。整卡显存含桌面进程，CUDA渲染计时与转图计时口径不同；单次墙钟测量不是独立纯算法开销测试。', '',
        '执行中断保留：support-r1在日志约20,830步处遭外部SIGTERM（工具退出143），终止来源未确认；GPU/主机内存健康，未见Python异常、OOM或非有限损失。最后资源采样距取得锁1652.666秒，整卡采样峰值6365MiB；无训练checkpoint，不能精确续训或把中断拓扑当最终模型。support-r2以相同科学配置、seed0和初始化从0重跑，控制不重跑，改用独立持久进程；不是参数扫描。失败现场、原始队列记录与新driver均保留。该额外约27.54分钟计算成本不隐藏在成功运行计时之外。', '',
        '## 验证、风险与回退', '',
        '前向差0，六组参数梯度及屏幕梯度差0，absolute有限非零；复用完全相同原生扩展的RGB140单像素oracle。支撑数组独立float64复算最大差9.54e-8，51,357个无证据高斯权重恰为1。两臂601步预检和正式30k均保留。预检首个事件因未截断而改选为0，不能冒充当时已经验证质量收益。',
        f'配对审计通过：共同原生源码、科学hook、实际执行train、配置/初始化/相机，300个日志位置相机顺序一致；全部保存高斯字段有限。两臂九图fresh重载精确；1200图次PSNR的独立float64 PNG复算最大差{data["independent_float64_psnr_max_error"]:.3g} dB，固定局部与标签区域由保存PNG重新计算。',
        '两臂共1088张评价PNG哈希核验通过，144相机元数据相同；历史RGB140在两个评价进程中的固定九图重渲染均与旧PNG逐像素相同。raw runner中的legacy method.all_other_densification_and_optimization_settings只描述继承的原生部分；registered_support和封存hook另外规定共享预算及排序，完整澄清见pair_audit.json。',
        '单seed、全400拟合、既有固定九图的历史选择、当前模型光度支撑的自我强化风险均保留。此实验不证明伪影根因、不证明未知视角提升，也没有要求每个类别都提升的新质量硬门。',
        '默认RGB130+SEM371、研究候选RGB140+SEM378与旧失败产物均保持。新模型仅在独立路径，回退直接继续使用旧路径，无需改写模型。未执行Git提交/推送，未删除旧模型。完成本次重建对照后停止；DINOv3、细索语义采样、匹配语义重训和组合不属于本任务。', '',
        '本次初始执行入口：scripts/run_standard_support_pair.py（--artifact-root 指定已有运行依赖）；中断后通过scripts/resume_standard_support_pair.py保留控制、重跑同配方候选并完成两臂评价，完成状态见pair_recovery-r1.json。单臂入口scripts/run_standard_support.py支持--run-name指定新的未占用目录；现有正式结果不可覆盖。浏览生成、独立审计和报告脚本均保留。']
    closure_path=out/'delivery_audit.json'
    if closure_path.exists():
        closure=json.loads(closure_path.read_text())
        if closure.get('status')=='completed':
            lines += ['', '## 最终交付核验', '', '最终repo_health.sh通过：532项测试运行、72项因可选环境依赖跳过，git diff --check通过；日志见outputs/quality/rgb141/repo_health_final.log。预检和GPU实测另有独立记录，未把跳过的测试写成通过。', '', '浏览器实测四路原生PNG、固定ROI同步裁剪与重置、拖选裁剪、400逐图跳转、轨迹下拉/滑块同步和三路显示通过；H264视频3960×1020、144帧、12fps/12秒加载及播放验证通过。页面加载无浏览器错误。视频播放检查仅验证交付功能，不增加几何或时序质量声明。', '', '最终实现/评价/报告/配置/文档快照及SHA256清单：outputs/quality/rgb141/evidence-code-r2/manifest.json；训练时快照保持原run/code，不能用收尾文档快照冒充事前训练源码。config.yaml中registered_not_started为事前登记字段，最终运行状态以registry、各run结果和delivery_audit.json为准。', '', '历史RGB140的790065高斯高于本次两臂约10.3%–10.4%；候选对历史三指标均退化同时伴随容量减少，不能将该差异单独归因于支撑权重。支撑排序的匹配推断以RGB143对RGB142为准。']
    (ROOT/'experiments/runs/RGB-143-STANDARD-SUPPORT-DENSIFY/report.md').write_text('\n'.join(lines)+'\n')
    ctrl=['# RGB-142-STANDARD-BUDGET-CTRL', '', '完整全400/30k新预算控制及配对评价已完成。它与RGB143共享预算、原生配方和支撑统计计算，唯一臂间差别是大高斯分裂排序是否乘连续权重。不能冒充历史RGB140。', '',
        '父RGB140；HEAD 49cdcdb1c9915a9544d568754af95ea20a85ec9c加封存源码；seed0；manifest data/manifests/contest_dataset_v1.json；split released_all400_rgb_all300_sem_v1。400指标均为训练拟合，144轨迹无GT。', '',
        '控制结果：'+json.dumps(data['metrics']['RGB142'],ensure_ascii=False)+'。高斯'+str(data['resources']['RGB142']['final_gaussians'])+'。', '',
        '完整方法、配对指标、固定区域、资源、视觉收益与损伤、审计和回退见[RGB143报告](../RGB-143-STANDARD-SUPPORT-DENSIFY/report.md)。控制完整训练记录、PLY与所有预检保持在outputs/quality/rgb141/control-r1。']
    (ROOT/'experiments/runs/RGB-142-STANDARD-BUDGET-CTRL/report.md').write_text('\n'.join(ctrl)+'\n')

if __name__=='__main__':main()
