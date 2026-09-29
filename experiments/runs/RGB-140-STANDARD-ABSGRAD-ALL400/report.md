# RGB-140-STANDARD-ABSGRAD-ALL400 — Standard 全400绝对梯度分裂对照

- 状态：completed_candidate_promoted
- 阶段：reconstruction
- 日期：2026-09-15 至 2026-09-16
- Git commit：`49cdcdb1c9915a9544d568754af95ea20a85ec9c`
- 父实验：RGB-130-STANDARD-ALL400
- 配置：`experiments/runs/RGB-140-STANDARD-ABSGRAD-ALL400/config.yaml`
- 正式运行：`outputs/quality/rgb140/full-r2`
- 浏览入口：`outputs/quality/rgb140/full-r2/evaluation/browser/index.html`
- 轨迹视频：`outputs/quality/rgb140/full-r2/evaluation/browser/trajectory-rgb130-rgb139-rgb140.mp4`

## 结论

RGB140 在与 RGB130 完全相同的初始化、相机、400 张训练 RGB、30k、seed0 和 Standard 配方下，只把**大高斯分裂资格**从 signed 屏幕坐标梯度改为 AbsGS 逐像素绝对（同向）梯度。小高斯克隆仍用 signed `0.0002`；absolute 分裂阈值为官方 `0.0004`。未引入相机优化、深度、语义、`use_reduce`、最大权重剪枝或其他 AbsGS 变量。

结果为明确但有限的重建候选晋级：相对 RGB130，400 张训练图 PSNR `+0.069664 dB`、SSIM `+0.002856`、LPIPS `-0.006928`，逐图改善分别覆盖 348、397、389 张；已有标签的五类 RGB 区域 MAE 全部改善。固定九图三项全图指标也全部改善，桥面与真实细索视觉清晰度/连续性略好，054/057/064 远桥全图指标方向一致。代价是最终高斯 `+21.12%`、训练墙钟 `+38.73%`、模型 `+21.12%`。

它没有消除全部几何伪影：159 远端遮挡 ROI 比 RGB130 恶化 `17.37%`，159 水面幽灵恶化 `1.06%`；轨迹仍可见水面宽带、雾块与少量天空斜线。无 GT 轨迹的时序二阶差分代理比 RGB130 高 `4.90%`，但该值受相机运动和更锐细节共同影响，只作为复核提示，不能单独证明闪烁变差。因此 RGB140 **值得进入一次严格匹配的语义训练/完整系统比较**，但在该比较完成前不覆盖冻结 RGB130，也不把训练图拟合增益写成独立视角泛化。

## 数据、初始化与变量隔离

- split：`released_all400_rgb_all300_sem_v1`
- scene：`outputs/quality/all_data/RGB-120-r2/scene`
- RGB：400 张，1320×989；全为训练视角，指标是拟合结果
- 官方相机：400 套内外参，固定，不优化
- 初始化：RGB130 原 303 视角 COLMAP 稀疏点云，72,292 点；SHA256 `10a88fb018c27530b3bbfa0f7000408693ba75228067f5311e7775e4755e68f2`
- RGB130 `worker.json`、RGB130 `model/input.ply` 与当前 scene PLY 三方哈希和点数一致
- 语义、LabelMe、伪标签、单目深度：均未输入
- Standard 3DGS：commit `54c035f7834b564019656c3e3fcc3646292f727d`
- AbsGS 方法参考：commit `87d7a5aac21c0b237a6af9e715e3a0c7df4902d3`
- 许可证：Gaussian-Splatting research/evaluation license

## 实现与预检

隔离源码位于 `outputs/quality/rgb140/source-r1`；可复现补丁为 `third_party/patches/standard-absgs-split-gradient.patch`，SHA256 `6022e88fc222b094e24815ca0d498cdacdb9664be11ee283d3618a5a82e14713`。CUDA 12.8 / PyTorch 2.9.1 环境中为 RTX 5070 编译的扩展 SHA256 为 `f1f07e0fd491dc40f668112bd2ee0af24cc7a7b2c1cd13571be3191ec6097c65`。

`preflight-r3` 通过：冻结 RGB130 模型在 001 视角的前向逐像素完全一致；signed 屏幕梯度最大差 `1.17e-8`；absolute 梯度有限且非零；八个单像素原生反向 oracle 最大误差 `3.55e-15`。CUDA 原子累加导致 xyz 梯度一个极端元素最大差 `3.43e-5`，相对 L2 `0.001182`、99.99 分位 `5.95e-9`；重复基线和候选运行确认这是少数强相消项的原子顺序尾部，不是系统性梯度改变。该冻结视角上 signed `0.0002` 候选 4,870 个，absolute `0.0004` 候选 12,166 个，证明新资格统计实际覆盖了被相消隐藏的大高斯。

正式命令：

```bash
ICSHM_ARTIFACT_ROOT=<artifact-root> python3 scripts/run_rgb140_absgs_all400.py --run-name full-r2 --iterations 30000
```

## 400 张训练图拟合指标

| 指标 | RGB130 | RGB140 | RGB140-RGB130 | 改善/退化视角 |
|---|---:|---:|---:|---:|
| PSNR | 23.217673 | 23.287337 | +0.069664 dB | 348 / 52 |
| SSIM | 0.893218 | 0.896074 | +0.002856 | 397 / 3 |
| LPIPS ↓ | 0.252782 | 0.245853 | -0.006928 | 389 / 11 |

次级参照 RGB139 为 23.210052 / 0.893514 / 0.252564；RGB140 相对它同样三项改善。指标口径、400 相机集合和 PNG 量化与 RGB130 固定评估一致；fresh 重载九图逐字节一致。

## 固定区域与结构观察

固定 ROI 的 RGB MAE/255 相对 RGB130：

| ROI | RGB130 | RGB140 | 相对变化 |
|---|---:|---:|---:|
| 159 岸线 | 6.9724 | 6.7634 | -3.00% |
| 159 近景 | 9.1638 | 7.9582 | -13.16% |
| 159 远端遮挡 | 4.9493 | 5.8092 | +17.37% |
| 159 水面幽灵 | 3.0639 | 3.0963 | +1.06% |
| 160 岸线 | 7.3056 | 6.6663 | -8.75% |
| 160 近景 | 9.7905 | 8.9235 | -8.85% |
| 174 天空 | 5.2300 | 4.6033 | -11.98% |

已有 300 张标签只作为 RGB 区域观察，不作训练或选择。background/deck/`stay_cable`/tower/foundation 的 RGB MAE 分别相对 RGB130 改善 `3.83% / 1.12% / 2.88% / 5.22% / 1.96%`。固定九图 054/057/064/081/140/146/159/160/174 的 PSNR、SSIM、LPIPS 均同向改善；细索不是被简单删除，桥面边缘和部分远桥轮廓更清楚。

局部不足同样保留：RGB139 的新初始化在 159/160 岸线与近景仍优于 RGB140，说明初始化覆盖与增密相消是互补而非同一问题；RGB140 在 159 远端遮挡和水面幽灵比 RGB130/RGB139 都差。视频抽查 frame 0/36/72/74–78/108 仍见水面宽带、局部雾块和少量斜线，未观察到全轨迹伪影消失。

无 GT 轨迹的二阶时序差分代理见 `evaluation/trajectory_temporal_proxy.json`：RGB140 全图/天空/下部场景均值较 RGB130 高 `4.90% / 5.28% / 4.81%`。最大响应集中在 75–78 附近，正好是相机运动和结构变化较大的段落；该代理不能分离运动、锐度与闪烁，不用于晋级门，只提示播放视频重点复核。

## 成本与产物

| 项目 | RGB130 | RGB140 | 变化 |
|---|---:|---:|---:|
| 初始点 | 72,292 | 72,292 | 0 |
| 最终高斯 | 652,298 | 790,065 | +21.12% |
| 训练墙钟 | 1,913.568 s | 2,654.670 s | +38.73% |
| PyTorch allocated 峰值 | 1,711.901 MiB | 1,998.660 MiB | +16.75% |
| PyTorch reserved 峰值 | 5,102 MiB | 5,842 MiB | +14.50% |
| `nvidia-smi` 10 秒采样峰值 | 未同口径登记 | 6,928 MiB | 仅记录 |
| 最终 PLY | 154.277 MiB | 186.861 MiB | +21.12% |

- 训练结果 SHA256：`16444f37c93e5739fc93febbdd820a4a9111b0f714f00598e6efd9284bfd0888`
- 评估结果 SHA256：`c4b7037a8f445a64899edc6d98a17733bc50d11df9fd6b08eb3b361179147dfa`
- 最终 PLY SHA256：`7f4b836160a377f4157dfbb3624dae5b9def561c1dd2fa0cf766150fd004c864`
- 144 帧、12 秒、3960×1018 H.264 视频 SHA256：`ac702a09d4082846f49c6c962baa3475471855752fd006a12d90c6d4b6031d99`
- 评估耗时 291.897 s；评估渲染中位数 17.013 ms；最终批次约 1.5 GiB（含模型、checkpoint、544 PNG、浏览页与视频）

## 失败与修复记录

- `preflight-r1`：脚本使用 sentinel 默认值，误把 `depths=None` 当成启用深度，因缺少 `depth_params.json` 在加载前退出；未改数据/模型，产物保留。
- `preflight-r2`：科学检查均正常，但最初 `2e-5` 的逐元素 xyz 梯度最大差门被一个强相消原子项触发；重复原版/候选反向并补 99.99 分位、相对 L2 与八像素独立 oracle 后，在不改变方法实现的情况下形成 `preflight-r3`。
- `full-r1`：手工登记时误抄初始化 SHA，在训练启动前被校验拦截，只留下空目录；用 RGB130 三方证据更正后运行 `full-r2`。没有删除失败现场。

## 保留决定与下一步

- keep_or_drop：`promote_reconstruction_candidate_keep_rgb130_frozen_until_matched_semantic_system_eval`
- RGB140 不覆盖 RGB130，RGB139 和所有失败产物继续保留。
- 值得下一步只做一个主要变量：冻结 RGB140 渲染，用与 SEM371 完全相同的数据、seed、Mask2Former 配方训练匹配语义模型，并与 RGB130+SEM371 做全300、九图和同144轨迹的完整系统比较。
- 本实验不自动启动上述语义训练，也不启动 HRNet、融合、相机优化或深度约束。

## 2026-09-16 后续研究计划更新（ADR-118）

第一优先：以本配方为父对照，准备跨视角支撑连续加权的增密单变量和相近高斯预算比较；匹配语义SEM378已完成。新实验尚未登记或启动。 本次仅更新计划，原指标、配置与保留决定不变。新配置和划分在执行准备时登记，不新增质量硬门；研究收益与新视角能力均待验证。
