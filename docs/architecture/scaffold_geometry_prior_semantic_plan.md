# Scaffold-GS + GeometryPrior + VC-Semantic 总体方案

> 状态：当前推荐总体方案；执行状态以主机的 PROJECT_STATUS、TASKS、ROADMAP、DECISIONS、RISKS、实验 registry 和各实验报告为准。
> 更新日期：2026-08-05。
> 当前分支的 GeometryPrior 方案整合冻结 commit：5af2941；其中准备流水线为 3cab499，测试为 a4c8ce2。
> GeometryPrior 统一评估协议 v2 为 38d645c；VGGT-Ω dry-run adapter 为 44019b2，测试为 3b9e3da。
> 已完成：冻结 Scaffold 证据登记、GeometryPrior 相机/图/航段准备、GEO-000 3k
> pilot，以及 MapAnything Apache P0 smoke、训练侧融合和新鲜 GEO-000/GEO-010
> 10k 对照。全量 MapAnything P0 初始化在10k退化，未进入30k；唯一额外预注册的
> additive MapAnything hybrid 3k stop-test 也已完成并因质量门失败淘汰。DA3-Large 已完成
> 真实 P0 smoke、训练侧离线几何、融合与一次独立 Scaffold 3k 工程/质量门；该 P0 初始化
> 明显落后匹配控制并已淘汰。VGGT-Ω 与 LingBot 仍待许可证/依赖和真实模型验证。
> 适用任务：IC-SHM 2026 Project 2，桥梁多视图语义三维重建。
> 术语：本文统一使用 `Scaffold-GS`；数据是 400 张多视图图片，不称为连续 400 帧视频。

## 1. 结论摘要

当前方案采用两条相互隔离、可独立回退的主线：

1. **RGB/几何主线**：冻结当前最优 `Scaffold-GS` 作为最终渲染骨干；前馈模型只在训练阶段提供几何补充、初始化和受约束位姿提议。
2. **语义主线**：冻结 Scaffold-GS 的位置、尺度、旋转、opacity、颜色和 RGB feature；依次验证静态五类语义、贡献感知可见性监督、多视图证据冲突、受限视角残差和半监督。

最终盲测不运行 MapAnything、Depth Anything 3、LingBot-Map 或 VGGT-Ω。提交 checkpoint 必须直接接收官方参考坐标系中的外部相机，输出 1320×989 RGB 与五类语义 ID 图。

当前推荐组合是：

```text
官方 COLMAP 位姿锚定
        +
官方 P0 原始 Scaffold 初始化
        +
已验证的静态语义 sidecar
        +
Scaffold-GS 最终 RGB/语义渲染
```

贡献门控和限定 Dirichlet 冲突正则均为已保留的负面证据，不是当前组合的一部分；任何新的语义 head 必须在解释错误来源后另行预注册。

DA3-Large 是已完成负面 P0 深度对照，而非当前可运行候选；它的 CC BY-NC 4.0 比赛/研究用途已经获得用户确认。VGGT-Ω是四模型统一评估协议 v2 的联合共视窗口候选，但仍未安装、未推理、未验证。

## 2. 已有证据与证据边界

### 2.1 已有 RGB 结果

用户提供的同条件 400 RGB、30k iteration 结果为：

| 骨干 | PSNR↑ | SSIM↑ | LPIPS↓ | 当前决定 |
|---|---:|---:|---:|---|
| Standard 3DGS | 15.0469 | 0.8032 | 0.4509 | 工程回退 |
| Mip-Splatting | 14.2259 | 0.7871 | 0.5036 | 淘汰 |
| Scaffold-GS | 20.5964 | 0.8502 | 0.3224 | 冻结 RGB 骨干 |

3DGS-MCMC 的 1k 预检也明显落后于 Standard 控制，因此当前不继续投入。

这些结果已经由主机 R1 审计、对应 config/report、registry、冻结的空间 split 与外部 Git 忽略 artifact 证据承载。冻结 Scaffold 上游 commit 为 59c833b56bbbf510f3f64d40f81721995caced66；历史 R1 主仓库快照有记录为未提交工作树的条目时，必须保留该边界，不能倒推不存在的精确 commit。

### 2.2 历史语义问题证据

以下数值是 registry 中 E2/E3 的用户提供历史证据，不替代已完成的 SEM-001 静态 Scaffold 基线；SEM-001 的实际指标和限制见第 11 节。

- 普通语义 3DGS：mIoU `0.579755`；
- `stay_cable` IoU：`0.138621`；
- 三个 seed 共同的背景→拉索误报约 517 万像素；
- 硬过滤虽然减少 93.43% 误报，但 mIoU 降至 `0.557671`，`stay_cable` IoU 降至 `0.062299`，真实拉索保留率只有 28.73%。

因此不再把硬标签清洗或独立 Proxy 作为主线；语义创新应直接建模可见贡献、多视图冲突和有限的观察方向依赖。

### 2.3 已实现且已验证的执行基础

- Scaffold-GS 已由 RGB-060 冻结；其 checkpoint、配置、空间 split、外部官方相机门和最小兼容 patch 已登记；
- 官方 COLMAP 文本读取、CAMERA_FROM_WORLD 到 cam2world/相机中心转换、SIMPLE_RADIAL 投影和反投影已实现；
- quaternion、w2c/c2w、投影/反投影和 Umeyama Sim(3) 单元测试已通过；
- 400 个官方相机构成的 camera graph 已生成；轨迹切分保留 33 个候选断点、接受 23 个断点，得到 24 个最小 4 图航段；
- MapAnything Apache 在隔离环境完成真实 P0 smoke、303训练图离线几何和融合；其全量 P0 初始化10k候选和受限添加式混合3k stop-test均已淘汰。DA3-Large 也已完成真实 P0 smoke、303训练图离线几何/融合和独立 Scaffold 3k工程门，但其质量明显落后匹配控制而淘汰；LingBot 仍只有 fail-closed 预检入口，未把 dry-run 当模型推理；
- GEO-000 已按冻结 Scaffold、400 RGB、303/74 空间 split、seed 0 和 3k 完成，checkpoint 重载及 74 视角固定评测通过。

### 2.4 尚未验证的内容

以下均是待运行假设：

- MapAnything、DA3 或 LingBot 的**其他预注册配方**能提高 Scaffold 最终指标；全量 MapAnything P0 xyz+原始RGB融合配方已在10k被否定，受限添加式混合配方也已在3k止损门否定；
- LingBot 位姿优于官方 COLMAP 位姿；
- 多前馈模型融合优于最佳单模型；
- CLAHE/Gamma 能改善前馈深度或匹配；
- 新的 Visibility、Dirichlet 冲突、视角残差或半监督配方能提高 mIoU；已运行的SEM-021/022、SEM-003和SEM-004受限一阶残差均未通过各自预注册门。

所有失败实验同样登记，不得静默删除。

## 3. 数据事实与验证划分

### 3.1 数据组成

- 300 张有 LabelMe 语义标注 RGB；
- 100 张无语义标注 RGB；
- 全部为 1320×989；
- 全部 400 张图片有官方 COLMAP 相机；
- 相机模型为 `SIMPLE_RADIAL`；
- `points3D.txt` 缺失是官方有意设置，可以自行三角化或采用其他重建方法；
- 正式类别为：
  - `0 background`
  - `1 deck`
  - `2 stay_cable`
  - `3 tower`
  - `4 foundation`

### 3.2 图片与 COLMAP frame 的区别

当前数据包中，400 张图片与 `frames.txt` 中 400 个单相机 capture record 一一对应；这不证明它们具有视频时间戳、固定 FPS 或严格时间连续性。

已完成的结构审计输出为：

- camera graph 含 400 个节点和 36,928 条共视边；
- 依据相邻中心位移、光轴夹角和共视下降，记录 33 个候选断点，接受 23 个断点；
- 输出 24 个航段，最短航段为 4 张图。

因此：

- MapAnything/DA3 使用共视图窗口；
- LingBot 使用连续航段；
- 文件编号只作为航段内部的弱顺序先验；
- 不能把 001–400 无条件输入成一条视频。

### 3.3 空间隔离 split

模型选择必须使用空间隔离 split，而不是随机图片划分：

1. 根据相机中心 PCA 主轴把桥梁划分成空间块；
2. 兼顾桥两侧、高度和远近视角；
3. 使用三个空间折；
4. 训练折和验证折之间保留相机邻域缓冲区；
5. 若无标注图与验证相机处于同一局部空间块，该图也从对应 RGB 训练中排除；
6. 140–146 和最差 10% 视角保留为困难区域诊断。

如果现有 20.5964 dB 来自训练使用过的同一批视角，它只能作为训练拟合指标；GeometryPrior 的保留决定必须来自统一的空间验证。

## 4. 总体架构

```text
400 RGB + 官方 COLMAP 相机
              │
      去畸变和坐标标准化
              │
     ┌────────┼─────────┐
     │        │         │
 MapAnything  DA3   LingBot-Map
 官方位姿几何  深度   分航段位姿/几何
     │        │         │
     └────────┼─────────┘
              │
   Sim(3) 对齐 + 置信度校准
              │
 多视图一致性过滤 + 3D 融合去重
              │
       Scaffold anchor 初始化
              │
     3k → 10k → 30k 短优化挑战
              │
        冻结全部 RGB 参数
              │
 Static → Visibility → Evidence/Conflict
              │
  bg/stay_cable 受限一阶视角残差
              │
       可选 2D–3D 一致性半监督
              │
 官方外部相机 → RGB + semantic class IDs
```

## 5. 坐标、相机与提交边界

### 5.1 COLMAP 转换

官方 `images.txt` 给出：

\[
T_{cw}=\mathrm{CAMERA\_FROM\_WORLD}
\]

转换为 OpenCV 风格 `cam2world`：

\[
R_{wc}=R_{cw}^{T},\qquad t_{wc}=-R_{cw}^{T}t_{cw}
\]

必须实现并测试：

- quaternion 解析；
- w2c→c2w→w2c 闭环；
- 相机中心恢复；
- 投影/反投影闭环；
- SIMPLE_RADIAL 去畸变；
- 原始与去畸变内参、分辨率和有效区域配对保存。

### 5.2 官方坐标锚定

任何前馈模型输出都先对齐到官方坐标，再进入融合。推荐保持最终 Scaffold 场景直接锚定官方参考系；如训练内部做归一化，保存：

```text
official_to_train.json
camera_model.json
undistortion.json
geometry_source.json
```

盲测输入只有官方相机，因此训练期的单图位姿残差不能作为隐藏相机必需输入。

## 6. 前馈几何候选

### 6.0 当前主机依赖状态与许可证边界

截至 2026-08-05，MapAnything Apache 与 DA3-Large 均已在隔离环境完成真实执行并固定来源、
commit、权重哈希和许可证证据。MapAnything 的两条 P0 初始化配方与 DA3 的 P0 初始化均为负面
结果；它们不能因换用更大显存设备而静默重跑。LingBot 仍未固定已批准 source/checkpoint；其 runner
只能生成确定性窗口 manifest。任何新候选真实执行前仍必须将来源、commit、代码和 checkpoint
许可证写入对应实验 config，且保持环境隔离。

| 候选 | 官方来源 | 代码许可证状态 | 当前主机状态 |
|---|---|---|---|
| MapAnything | https://github.com/facebookresearch/map-anything | 代码与 `facebook/map-anything-apache` 权重均为 Apache-2.0 | 官方 commit `9d1db2d`、权重 revision `00f9c24` 已验证；首个P0融合10k为负面结果 |
| DA3-Large | https://github.com/ByteDance-Seed/Depth-Anything-3 | 代码 Apache-2.0；官方 model-zoo 的 DA3-LARGE/1.1 权重为 CC BY-NC 4.0 | 用户已确认比赛/研究用途；官方 commit `3d835ec`、权重 SHA-256 已登记。真实 P0 smoke、303训练图几何和3k Scaffold 接入完成，但质量门失败而淘汰。 |
| LingBot-Map | https://github.com/robbyant/lingbot-map | Apache-2.0 | 未安装，未固定 commit/checkpoint |
| VGGT-Omega | https://github.com/facebookresearch/vggt-omega | 官方仓库要求查阅 LICENSE；尚未完成本项目许可证审查 | 未安装；仅保留为后续可选对照 |

这些状态不构成模型质量结论，也不授权下载。


### 6.1 MapAnything：主几何先验

定位：使用已知官方内参和相机位姿生成稠密深度、世界点与置信度，不替换最终相机。

官方实现支持 OpenCV `cam2world`、内参、深度和混合输入，并提供 memory-efficient inference：[MapAnything](https://github.com/facebookresearch/map-anything)。

RTX 5070 初始设置：

```text
memory_efficient_inference = true
minibatch_size = 1
AMP = bf16（不兼容时 fp16）
共视窗口 = 16 张
窗口重叠 = 4–8 张
```

窗口由共视图生成，不按固定编号切片。

### 6.2 Depth Anything 3 Large：深度竞争模型

定位：给定官方 extrinsics/intrinsics 的 pose-conditioned depth 对照，不使用其 GS 分支替换 Scaffold。

官方项目明确支持已知位姿输入和 pose-conditioned depth：[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)。具体 API、包括 `align_to_input_ext_scale` 是否可用，必须在获批且固定的 source/commit 上验证；当前不是已运行事实。

RTX 5070 初始设置：

```text
model = da3-large
窗口 = 8–16 张
align_to_input_ext_scale = candidate setting; verify after source pinning
infer_gs = false
```

不优先使用 Giant/Nested-Giant。

### 6.3 LingBot-Map：分航段位姿与几何挑战

定位：利用流式轨迹记忆生成分段几何和位姿提议；不能无条件替换官方相机。

官方实现提供流式/窗口模式、CPU offload、关键帧缓存和相机迭代：[LingBot-Map](https://github.com/robbyant/lingbot-map)。

RTX 5070 初始设置：

```text
checkpoint = lingbot-map-long
image_size = 518
mode = streaming
offload_to_cpu = true
use_sdpa = true（先保证兼容）
camera_num_iterations = 1（smoke）/ 4（正式候选）
keyframe_interval = 1（短航段）
mask_sky = true（仅几何）
save_predictions = true
```

航段切分同时考虑：

- 相邻相机中心跳变；
- 光轴夹角；
- COLMAP 共享观测；
- 图像匹配内点；
- 每段最小有效视图数。

跳变处重置状态；短于 8–12 张的孤立段优先回退到 MapAnything。

### 6.4 VGGT-Ω：可选小窗口补充

VGGT-Ω不依赖流式轨迹记忆，但 12GB 显存不适合 100–400 图联合输入。官方 A100 统计为 25 图约 7.80GB、50 图约 9.66GB、100 图约 13.37GB、400 图约 35.71GB：[VGGT-Ω](https://github.com/facebookresearch/vggt-omega)。

VGGT-Ω 与其他三种前馈候选一起进入统一评估协议 v2 的功能 smoke；同一时间只运行一个前馈模型。其 25 图→16 图的降显存顺序、输出 schema 和进入连续 10k 的条件由 [统一评估协议 v2](../experiments/geometry_prior_evaluation_v2.md) 固定。

## 7. LingBot 位姿提议与接受门控

建立三个明确模式：

| 模式 | 训练位姿 | 用途 |
|---|---|---|
| P0 | 官方 COLMAP | 主基线 |
| P1 | LingBot 位姿经逐段 Sim(3) 对齐 | 激进挑战 |
| P2 | 官方位姿 + LingBot 提议的受约束小残差 | 推荐挑战 |

逐航段对齐求解：

\[
C_i^{official}\approx sR C_i^{pred}+t
\]

使用 RANSAC + Umeyama，并联合检查相机前向轴和上方向。桥梁轨迹接近长条形，只使用相机中心可能导致 Sim(3) 退化，因此必须记录中心残差、方向角误差、内点比例和尺度稳定性。

位姿更新只有同时满足以下条件才保留：

- 特征重投影误差下降；
- 多视图深度一致性改善；
- 对齐尺度稳定；
- 相对官方位姿变化受限；
- 使用未经单图修正的官方验证相机渲染时，Scaffold 指标改善。

对齐失败或外部相机验证下降时，回退 P0。预期 LingBot 对几何初始化和官方位姿异常诊断的价值可能高于全面替换官方位姿。

## 8. CLAHE + Gamma

采用双路图像，避免破坏最终 RGB：

```text
原始 RGB：Scaffold 颜色、最终渲染和 RGB 指标
增强 RGB：几何匹配或独立 GeometryPrior 消融
```

第一轮所有前馈模型使用原始去畸变图片。只有低纹理诊断证明需要时，才比较：

- `RAW-GEO`：原始去畸变输入；
- `ENH-GEO`：仅低纹理图使用亮度 CLAHE + 轻度 Gamma。

初始搜索范围：CLAHE clip limit 1.5–2.0，tile 8×8，Gamma 0.8–1.2。保留依据是匹配内点、多视图深度一致性、浮点比例和 Scaffold 指标，而不是主观观感。

## 9. 置信度感知 3D 融合

### 9.1 点属性

每个候选点至少保存：

```text
xyz
原始 RGB 颜色
模型来源和 checkpoint
原始/校准 confidence
支持视角数
多视图深度一致性
局部法向
图像边缘或线结构分数
```

不同模型的 confidence 不可直接比较。先做单模型内部 percentile 归一化，再使用重投影深度一致性校准。

融合权重为：

\[
w=w_{model}\cdot w_{depth}\cdot w_{angle}\cdot w_{support}
\]

### 9.2 多视图检查

- 投影到 3–5 张强共视图片；
- 删除相机后方、有效图像区域外和低置信点；
- 初始相对深度容差为 3%–5%；
- 普通表面优先要求至少 3 视角支持；
- 细拉索可放宽到 2 视角，但要求不同基线和较强线结构证据；
- 天空不生成有限深度点；
- 水面与反射区域使用更严格一致性门槛；
- RGB 训练仍保留这些区域，不因几何过滤删除图像像素。

### 9.3 自适应体素

定义当前 Scaffold 初始化点的中位 5-NN 距离为 \(r_0\)：

- 普通表面：体素 \(r_0\)；
- 高置信细结构：\(r_0/2\)；
- 低置信区域：\(1.5r_0\) 或删除；
- 法向冲突的点不合并；
- 体素内使用稳健加权中心而不是简单均值。

第一轮只把融合位置和来自原始 RGB 的颜色传给 Scaffold，不接入 foundation feature，确保唯一主要变量是初始化几何。

## 10. Scaffold-GS 初始化与短优化

### 10.1 Anchor 初始化

- anchor position：融合点；
- color：支持视角原始 RGB 的稳健中位数；
- scale：局部 kNN 距离；
- offsets 与 feature：沿用冻结 Scaffold 实现；
- 不在第一轮增加新的 feature adapter。

### 10.2 训练阶段

| 阶段 | 建议预算 | 可训练内容 |
|---|---:|---|
| Warm-up | 500–1000 iter | 冻结 anchor 位置，训练外观、opacity、feature |
| Geometry release | 至 3000 iter | 小学习率开放 offset 和 scale |
| Short optimization | 10k iter | 正常 Scaffold 优化 |
| Final comparison | 30k iter | 与当前 Scaffold 公平比较 |

可选深度/法向先验只在早期启用，并在 3000–5000 iter 前衰减到 0，防止前馈几何错误长期限制 RGB 优化。

### 10.3 GeometryPrior v2 实验矩阵

| ID | 唯一主要变量 | 状态 |
|---|---|---|
| GEO-000 | 原始 Scaffold 初始化，3k pilot | 已完成；PSNR 20.06078、SSIM 0.83855、LPIPS-Alex 0.37803。它不是连续 10k 的恢复点。 |
| GEO-000-10K | 原始 Scaffold 初始化，连续 0→10k 控制 | 已完成；10k PSNR/SSIM/LPIPS-Alex 为 20.48798/0.85027/0.33167。 |
| GEO-010-10K | MapAnything 官方 P0 深度初始化，连续 0→10k | 已完成并淘汰；10k LPIPS +0.02587、最差10% PSNR -0.75640，相对控制退化。 |
| GEO-011-3K | MapAnything P0 添加式高置信补点，连续 0→3k | 已完成并淘汰；PSNR仅 +0.01864 dB、LPIPS +0.00437，最差10%/困难视角退化；不新建10k。 |
| GEO-020-10K | DA3-Large 官方 P0 深度初始化，连续 0→10k | 未启动：`GEO-020-3K-INT-R1` 工程门通过但相对匹配3k控制 PSNR −1.2184 dB、LPIPS +0.1012，按该接入门淘汰；不得以此 ID 续跑或重跑。 |
| GEO-025-10K | VGGT-Ω 联合共视集合官方 P0 深度初始化，连续 0→10k | 已预注册、未运行；先过 GEO-025-PF 功能/许可证门。 |
| GEO-030-10K | LingBot 分航段官方 P0 深度初始化，连续 0→10k | 已预注册、未运行；先过 GEO-030-PF 功能/许可证门。 |
| GEO-031 | LingBot 受约束位姿提议 | 仅在最终 Top1 几何有效且位姿诊断异常时运行。 |
| GEO-040 / GEO-050 | 多模型融合 / 早期深度先验 | 仅在 v2 单模型证实互补并另行预注册后运行。 |
| GEO-060 | 最佳方案 30k 完整确认 | 由 v2 Top2 与新 GEO-000-30K-V2 的连续 0→30k 比较替代。 |

### 10.4 晋级门槛

从 v2 开始，3k 不再是提前淘汰依据。所有通过功能、许可证和硬件门的候选与新的 GEO-000 控制均连续运行 0→10k，并在 1k/3k/5k/10k 记录结果。Top2 由硬门、Pareto、配对逐视角 bootstrap 和预注册近似平局补 seed 规则选择；Top2 与 GEO-000-30K-V2 再连续运行 0→30k。量化定义、OOM 顺序和结论边界见 [统一评估协议 v2](../experiments/geometry_prior_evaluation_v2.md)。

## 11. 冻结 RGB 后的语义架构

执行状态：SEM-001 已完成冻结 Scaffold RGB 上的五类静态 S0 基线，并验证 RGB 最大绝对像素差为 0、semantic checkpoint 重载和独立官方外部相机 RGB/五 logits/类别 PNG 输出。其本地空间验证 mIoU（含/不含 background）为 0.61610 / 0.56334，但 stay_cable IoU 仅 0.06876，background 到 stay_cable 误报为 13,166,331 像素；因此它是已冻结的比较基线，不是语义问题已解决的证据。

S1 Visibility 已完成主折筛选但因保护类退化登记为 drop；S2 Dirichlet Evidence/Conflict 已完成训练侧诊断、表示 CTRL 和匹配10k候选。候选相对 CTRL 的 mIoU、stay_cable IoU 均下降且 background→stay_cable 增加345,469像素，故按预注册门登记 drop。`SEM-004-DIAG` 随后仅对冻结 SEM-001 做了误报来源、LabelMe细线、贡献和训练半部方向的只读审计。基于该证据独立预注册的S3受限一阶残差已完成10k：相对SEM-001仅mIoU +0.0123pp、background→stay_cable −0.2222%，未过+1pp/1%门并登记drop；半监督仍未启动。


语义训练必须冻结：

```text
anchor position / offset / scale / rotation
opacity / color / RGB feature
```

每次语义消融验证 RGB 最大绝对像素差为 0。

### 11.1 S0：五类静态 Scaffold 语义

每个 `anchor-offset` 保存五类静态 logits，不直接复用视角相关 RGB 输出。像素概率采用冻结 alpha 贡献合成：

\[
P_p(c)=\sum_gT_{g,p}\alpha_{g,p}P_g(c)
+T_{end,p}\mathbf 1[c=background]
\]

静态强基线依次比较：

- CE；
- 截断平方根类权重 CE；
- WCE + Lovász；
- 仅在证据支持时增加边界课程训练。

### 11.2 S1：Contribution-aware visibility supervision

普通 alpha 渲染已有可见性权重，因此新增模块必须真正限制低贡献 Gaussian 的显式标签分配。

每个像素的边际贡献：

\[
r_{g,p}=T_{g,p}\alpha_{g,p}G_{g,p}
\]

比较累计贡献门控与相对最大贡献门控。门控权重 stop-gradient，语义损失不能改变 RGB 几何。

### 11.3 S2：Dirichlet 多视图证据与冲突

对 Gaussian 累计多视图类别证据：

\[
e_{g,c}=\sum_{v,p}r_{g,p}y_{p,c},\qquad
\alpha_{g,c}=\alpha_0+e_{g,c}
\]

记录类别后验、总证据、不确定性、有效支持视角和 bg/cable 冲突。低证据表示未知，不能与高证据冲突混为一类。

只有冲突统计达到预注册定位门槛，才进入 S3。

### 11.4 S3：受限一阶视角残差

只对高证据、高冲突 Gaussian 启用：

\[
\delta_g(v)=q_g a_g^T\phi_1(v)
\]

```text
cable_logit += delta
background_logit -= delta
deck/tower/foundation 不变
```

禁止相机 ID、每图 embedding 和无门控的大容量视角 MLP。先一阶 SH；一阶通过后才测试二阶。

### 11.5 拉索细结构辅助

只有错误分析显示真实拉索断裂时，才对高置信、低冲突的拉索核心测试 soft-clDice 或边界损失。不得在整个疑似包络区域无条件强化拓扑。

## 12. 半监督

100 张无标注图首先用于 RGB；语义伪标签是单独实验。

建议使用可复现的 2D 教师进行空间折 out-of-fold 校准，再与冻结 3D 语义渲染做双教师一致性：

1. 2D 教师输出软概率；
2. 当前 3D 模型从相同官方相机渲染概率；
3. 只接受逐类精度达到 95%、2D/3D 一致且 3D 不确定性低的像素；
4. `stay_cable` 达不到阈值时不生成该类伪标签；
5. 伪标签损失权重不超过 0.2–0.25；
6. 边界与高冲突 Gaussian 降权或排除。

实验顺序：

| ID | 配置 |
|---|---|
| SEM-U0 | 300 GT；100 图只用于 RGB |
| SEM-U1 | 2D 教师置信度软伪标签 |
| SEM-U2 | 2D 教师 + 3D 渲染一致性 |
| SEM-U3 | U2 + Dirichlet 不确定性门控 |

保留条件：相对 SEM-U0 平均 mIoU 至少 +0.5pp，至少两个空间折改善，任一类别下降不超过 0.5pp。

## 13. RTX 5070 / Ryzen 7 9700X 执行策略

该配置可以完成：

- 当前 Scaffold-GS 训练；
- 冻结 RGB 后的语义训练；
- MapAnything 小窗口推理；
- DA3-Large 小窗口推理；
- LingBot 分航段推理；
- 3D 融合与去重；
- VGGT-Ω约 25 图的可选窗口实验。

不适合：

- VGGT-Ω 100–400 图全局输入；
- DA3 Giant 大窗口训练或微调；
- 从头训练 3D 基础模型；
- 多个大型前馈模型同时驻留 12GB 显存。

建议至少 32GB 系统内存，优先 64GB，并预留 150–250GB NVMe 空间。

环境必须隔离：

```text
icshm-scaffold-frozen  # 不升级，保留当前最优复现
icshm-geometry         # MapAnything、DA3、融合
icshm-lingbot          # LingBot、Kaolin、FlashInfer/SDPA
```

## 14. 仓库与主机组织

当前已提交的 GeometryPrior 代码包括 `src/geometry_prior`、`src/fusion`、相机图/航段产物、
MapAnything P0 window runner、融合/Scaffold 初始化适配器、配置和单元测试。它们实现了一个
可审查的 P0 融合候选，不代表该候选已取得质量收益。


不新建第二个比赛项目仓库。使用同一项目仓库的新分支和独立实验目录：

```text
主项目仓库
├── src/geometry_prior/
├── src/fusion/
├── adapters/scaffold_gs/
├── configs/geometry/
└── experiments/runs/GEO-*/

third_party/（Git忽略）
├── Scaffold-GS
├── map-anything
├── Depth-Anything-3
└── lingbot-map
```

当前 Scaffold 上游修改较小时，以 adapter 或 patch 记录；只有需要长期维护大量 CUDA/renderer 改动时，才考虑独立的 Scaffold 私有 fork，并在主仓库固定其 commit。未经授权不推送。

## 15. 推荐执行顺序

执行状态：步骤 1–3 已完成并登记；MapAnything 完成步骤4–6的全量 P0 路径但10k未通过，
唯一添加式混合止损验证也未过3k gate，二者均不进入步骤7–9。DA3 已完成真实 smoke、训练侧
几何和3k接入，但同样在质量门淘汰；VGGT-Ω/LingBot仍待来源、许可证、checkpoint和真实 smoke。
SEM-001/S0已在当前冻结 Scaffold 上完成，S1/S2/S3均保留为负面证据；后续语义机制须另行解释并预注册。
在获得 A6000 后，先按 [A6000 GeometryPrior 交互准备矩阵](../experiments/a6000_geometryprior_interaction_readiness.md)
重新确认科学假设和唯一实验 ID，不能重用已完成或已淘汰的 GEO ID。

1. 封存当前最优 Scaffold checkpoint、commit、config、split、seed、环境和 SHA-256；
2. 建立空间 split、camera graph、航段和坐标单元测试；
3. 跑 GEO-000 3k 同预算基线；
4. 依次运行 MapAnything、DA3-Large、VGGT-Ω、LingBot 的独立功能 smoke；
5. 每个通过 smoke 的候选完成官方 P0 几何诊断与统一过滤；
6. GEO-000-10K 与所有可运行前馈候选连续 0→10k，保存 1k/3k/5k/10k；
7. 按硬门、Pareto、paired bootstrap 与近似平局补 seed 确定最多 Top2；
8. Top2 与 GEO-000-30K-V2 连续 0→30k；
9. 单模型互补得到证据并另行预注册后，才运行 GEO-040 融合；
10. 几何胜者 30k 正式超过当前 Scaffold 后，才替换语义父 checkpoint；
11. 将已完成的 SEM-001/S0 保留为冻结比较基线；后续按 S1、S2、S3 顺序验证；
12. 半监督最后独立验证；
13. 最终 checkpoint 在全新 Ubuntu 环境验证官方外部相机接口。

## 16. 最终提交契约

最终运行形式应保持简单：

```bash
python render.py \
  --checkpoint final_scaffold_semantic.ckpt \
  --camera-file hidden_test_cameras.txt \
  --output-dir outputs/
```

要求：

- 接受官方内参、外参、相机模型、宽和高；
- RGB 输出 1320×989；
- semantic 输出 1320×989 单通道整数 class ID；
- 输出文件名和目录规则配置化；
- MapAnything、DA3、LingBot 不属于盲测运行依赖；
- 保存并验证 official→train 的 Sim(3)；
- 已发布相机通过提交入口和本地入口渲染应一致。

## 17. 最终方法成立标准

一个新增模块只有同时满足以下条件才进入最终提交或论文核心论点：

1. 相对强 Scaffold/语义父实验带来独立收益；
2. 收益出现在空间隔离的未见视角；
3. 至少两个空间折改善，进入最终确认后 seed 方向一致；
4. GeometryPrior 不破坏官方相机坐标闭环；
5. 语义模块保持 RGB 最大绝对像素差为 0；
6. 受保护类别没有被牺牲；
7. 有逐视角、困难区域和机制证据；
8. 失败条件与回退方案明确；
9. 全新 Ubuntu 环境可从 checkpoint 复现；
10. 数据、第三方模型和许可证满足比赛复现要求。

如果所有 GeometryPrior 候选失败，回退为当前冻结的 Scaffold-GS；如果语义创新失败，回退为冻结 RGB 上的最佳静态五类语义基线。
