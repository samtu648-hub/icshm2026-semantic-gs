# GeometryPrior 统一评估协议 v2

> 状态：在前馈结果前冻结的 v2 预注册协议。随后 `GEO-000-10K` 控制与
> `GEO-010-10K` MapAnything P0 候选已完成；后者在 SSIM、LPIPS、最差 10% 和困难视角退化，
> 因此被淘汰且不进入 30k。随后 DA3-Large 完成真实 P0 smoke、训练侧几何和独立 3k 接入门，
> 但相对匹配控制明显退化，故没有启动 `GEO-020-10K`。VGGT-Ω、LingBot 尚无真实模型推理或质量结果。
> 生效范围：后续 GeometryPrior 真实实验。它不回写 RGB-060 的 Scaffold-GS 冻结，也不改写 GeometryPrior v1 的提交历史。
> 关联配置：[configs/geometry/evaluation_v2.json](../../configs/geometry/evaluation_v2.json)。

## 1. 目标与比较对象

本协议只比较训练期的几何初始化。最终盲测仍由冻结的 Scaffold-GS checkpoint 接收官方 COLMAP 外部相机并渲染；MapAnything、DA3-Large、VGGT-Ω 和 LingBot-Map 不进入盲测依赖。

主比较的五组为：

| 组 | 10k 实验 ID | 前馈输入组织 |
|---|---|---|
| 原始 Scaffold 控制 | GEO-000-10K | R1 训练期 DLT track 初始化 |
| MapAnything | GEO-010-10K | 16 图共视窗口，overlap 4 |
| DA3-Large | GEO-020-10K | 8 图共视窗口，overlap 4 |
| VGGT-Ω | GEO-025-10K | 16 或 25 图联合共视集合 |
| LingBot-Map | GEO-030-10K | 已切分航段；断点重置状态 |

GEO-000 保持已完成的 3k pilot/父实验，不覆盖、不续用 ID，也不以其 3k 指标淘汰候选。所有可运行组从 iteration 0 连续训练到 10k，保存 1k、3k、5k、10k 的指标与 checkpoint。

## 2. 三种冻结与共同输入

1. **RGB 冻结**：Scaffold-GS、其 R1 上游 commit、RGB 几何/颜色/opacity、外部官方相机接口不改变。
2. **GeometryPrior v1 冻结**：5af2941 及其相机、Sim(3)、camera graph、航段和既有 runner 的证据历史不改写。
3. **评估协议 v2 冻结**：本文件与机器可读配置固定于真实前馈结果出现前；改变任一字段须创建 v3，不能静默修改。

所有候选固定：

- 数据 manifest：data/manifests/contest_dataset_v1.json；
- 主 split：rgb_r1_spatial_v1 的 spatial-01，303 RGB 训练视角、74 标注验证视角和既有 guard band；
- 训练 seed：0；split generation seed：20260730；
- 官方 CAMERA_FROM_WORLD 位姿、SIMPLE_RADIAL 原始相机和 R1 无裁剪全帧 PINHOLE 训练输入；
- RGB 分辨率：1320×989；
- 同一官方相机集合、相同场景覆盖范围、同一 RTX 5070 12 GB 上限、同一 Scaffold 训练配置和同一 RGB 评测实现；
- 原始去畸变 RGB 作为第一轮所有前馈模型输入；CLAHE/Gamma 是未来单独变量。

“统一输入”不要求模型采用同一种窗口方式。每个 run manifest 必须记录图片 ID、顺序、窗口、overlap、state reset、有效覆盖、输入/输出分辨率和模型原生组织方式。

## 3. 官方坐标主路径

主比较只使用官方位姿 P0：

~~~text
模型每视角深度
  → 每视角尺度对齐
  → 官方内参 + 官方 cam2world 反投影
  → 官方参考坐标点云
  → 统一过滤、体素化、容量限制
  → Scaffold 初始化
~~~

对每张输入图，尺度 s_v 由该图可见的训练期官方稀疏轨迹计算：

\[
s_v=\operatorname{median}_{j\in M_v}\frac{z^{official}_{v,j}}{\max(z^{pred}_{v,j},10^{-6})}.
\]

M_v 是投影在有效输入网格内、预测深度有限且为正的对应点；少于 64 个匹配点时该视图没有有效尺度，必须在 manifest 中显式标记并从点生成中排除。尺度应用后深度必须表示相机 +Z 的 camera-space depth。不得利用验证 RGB、验证标签或隐藏数据估计尺度。

VGGT-Ω 和 LingBot-Map 的自由预测相机/世界点只用于诊断。若使用其世界点路径，必须先由预测与官方相机中心进行 RANSAC+Umeyama Sim(3)，保存尺度、旋转、平移、内点率、中心残差与方向误差。该路径至少要求 60% 内点、median 中心残差不超过归一化场景直径的 2%、median 光轴误差不超过 5°；任一条件失败即停止 world-point 路径并回退到官方 P0 深度反投影，不能静默替换官方位姿。

## 4. 统一输出、过滤与容量

每个模型的每视图/窗口几何输出遵循 geometry_prior_npz_v2：

- 必需：depth (float32, H×W, camera +Z)、confidence (float32, H×W)、valid_mask (uint8, H×W)；
- 必需元数据：来源模型/commit/checkpoint hash、图片名、输入尺寸、官方相机 ID、坐标 frame、深度尺度和窗口 manifest ID；
- 可选：world_points (H×W×3)、pred_cam2world (4×4)、normal、sky/reflection mask；
- 所有有限点都在官方坐标系，或明确标为待 Sim(3) 的诊断输出。

先在单模型内将 confidence 映射为该模型有效像素的经验 CDF percentile；仅以重投影深度一致性对其再校准。不同模型的原始 confidence 不得直接相加。

共同过滤规则：

- 去除 NaN、Inf、非正深度、相机后方、有效图像外和尺度无效点；
- 将点投影到其 3 个共享 POINT3D_ID 最多的有效邻居；相对深度误差为 |z_proj-z_pred|/max(z_proj,z_pred,10^-6)；
- 普通点要求至少 3 个邻居中 2 个误差不超过 5%；细线候选可要求 2 个不同基线邻居，但须在 manifest 标记 line_structure=true；
- 使用同一官方训练期初始化 PLY 的中位 5-NN 距离 r0 定义体素：普通点 r0、细线点 r0/2、低置信点 1.5r0 或删除；
- 每体素采用已校准 confidence 加权的稳健中位位置和原始 RGB 中位颜色；
- 最大初始化 anchor 数为 100,000。超预算时按 confidence 四分位分层、每层固定配额、图片名和体素坐标排序后确定性采样；点数不足时不得复制点。

每个节点记录原始点数、有限点数、过滤后点数、体素去重后点数、初始 anchor 数和 1k/3k/5k/10k/最终 anchor 数。

## 5. Phase 1：功能 smoke

每个候选先在其预注册的两个典型共视区域运行 smoke。只检查 checkpoint 与代码许可、checkpoint 许可、包/权重存在性、RTX 5070 12 GB 可运行性、NaN/Inf、输出 schema、深度尺度/坐标和 Scaffold 初始化接口。

此阶段不排名。若在当前 RTX 5070 和预注册的降显存顺序下仍不可运行，报告：

> 在当前 RTX 5070 和预注册降显存配置下不可评估。

这不是“性能较差”的结论。

OOM 处理顺序固定为：关闭无关 GPU 进程并记录状态 → batch=1 → bf16，不兼容时 fp16 → 缩小该模型原生窗口（MapAnything 16→12→8；DA3 16→12→8；VGGT-Ω 25→16；LingBot 16→12→8，保持航段断点重置）→ 仅模型支持时 CPU offload → 重新启动进程。每个预注册配置至多 1 次初始尝试和 2 次降显存重试；不得改变相机集合、评测或 Scaffold 配置来掩盖 OOM。

## 6. Phase 2：训练前几何诊断

没有真实 3D ground truth。以下均称为**几何一致性或初始化代理指标**，不得称为真实 3D 精度：

| 指标 | 确定性定义与报告 |
|---|---|
| 有效深度覆盖 | 每图无裁剪输入网格每 8 像素取样；分母为有效图像区域的样本数，分子为有限、正深度且 valid_mask=1 的样本数。报告每视图与均值/中位数。 |
| 多视图深度一致性 | 对第 4 节的邻居投影计算相对深度误差；报告支持数、支持率、median、p90 和误差≤5%的比例。 |
| 边界代理 | 灰度 RGB 以固定 Canny 100/200 获得边缘；深度相邻取样差超过 5% 为深度边缘。报告 RGB 边缘样本中 2 像素内命中深度边缘的比例。 |
| 孤立/浮点 | 体素化后，在半径 2r0 内少于 3 个邻居的点为孤立点；普通点少于 2 个多视图支持为浮点。报告数量和比例。 |
| 重复点 | 原始有限点映射至同一 r0 体素为重复；报告 1-unique_voxels/raw_finite_points。 |
| 点密度与空洞 | 报告每个占用 r0 体素的点数。重投影后的 8 像素网格中，被有效覆盖区域包围且面积至少 64 网格单元的无点连通域为结构空洞；报告每视图面积比例。 |
| 坐标闭环 | 报告官方中心/光轴误差、Sim(3) 内点率/尺度稳定性和提交入口与本地入口的像素差。 |
| 资源 | 记录推理秒数、峰值显存、磁盘字节、窗口/有效覆盖和初始 anchor 数。 |

NaN/Inf 比例必须为 0。若 world-point 路径未满足第 3 节 Sim(3) 门，诊断只允许报告其失败，不能把其点云送入 Scaffold。

## 7. Phase 3：连续 0→10k 主比较

所有通过 Phase 1 的候选，以及新的 GEO-000-10K 控制，均从 initialization 开始连续训练至 10,000 iteration；3k 不是独立重训，也不是提前淘汰门。每个 run 固定相同的 R1 Scaffold 超参数、数据、split、seed、渲染和 evaluate_rgb.py。`GEO-011-3K` 是本轮在 v2 外另行预注册的 MapAnything additive stop-test：它只决定该新配方是否值得新建 v2 风格的 10k 候选，不能改变本协议的既有排名或被表述为 10k 结果。该 stop-test 已完成但未达到其质量门，故不新建 MapAnything hybrid 10k 候选。

在 1k、3k、5k、10k 保存 checkpoint、评测、逐视角 CSV、均值/中位数、最差 10%、140–146 号困难视角及对应空间区域、训练时间、峰值显存、磁盘大小、anchor 数和外部官方相机门。真实运行若未通过许可证、功能或硬件门则保留失败记录，不虚构 10k。

GEO-000 的历史 3k checkpoint 不可作为严格连续 10k 的起点：它没有保存 Python/NumPy/PyTorch/CUDA RNG 状态。因此 GEO-000-10K 必须从 0 新运行，历史 pilot 只保留为父证据。

## 8. Phase 4：Top2 与连续 0→30k

Top2 只从通过 Phase 1 和 Phase 3 的前馈候选中选择；GEO-000-10K 始终是控制，不占 Top2 名额。候选必须满足功能、许可证、坐标、外部相机和 NaN/Inf 硬门，且不能在 PSNR、SSIM、LPIPS、最差10% PSNR、困难视角指标上被另一候选同时不差且至少一项严格更好（Pareto dominated）。

对每对候选的 74 个相同验证视角做 paired bootstrap：固定 seed 20260803，有放回抽样 10,000 次，报告平均 PSNR/SSIM/LPIPS 差的 95% percentile CI。若任一条件成立，则为近似平局：PSNR 差绝对值小于 0.1 dB、LPIPS 差绝对值小于 0.005、或任一主指标的 paired CI 跨 0。近似平局候选先增加第二训练 seed，再确定 Top2。

Top2 与 GEO-000-30K-V2 分别从 0 连续训练至 30k；不得用 10k checkpoint 接续来声称统一 30k 比较。若没有前馈候选通过，不强行构造 Top2，最终回退冻结 Scaffold，并将 GeometryPrior 记录为负面实验。

## 9. Phase 5–7：语义、空间折和位姿

Phase 5 仅比较 GEO-000-30K-V2 + Static Semantic、Top1 + Static Semantic、Top2 + Static Semantic。语义 seed、loss、optimizer、预算与 label split 相同，且 RGB 最大绝对像素差必须为 0；不得加入 Visibility、Dirichlet Conflict、视角残差或半监督。

Phase 6 不以单一 spatial-01 决定 Top1：Top2 分别在 spatial-01、spatial-02、spatial-03 运行固定 seed；随后 Top1 在主折额外补两种 seed。算力不足时必须在报告中写明缩减协议及其结论边界。

Phase 7 的主比较始终使用官方 P0。只有最终 Top1 的几何有效、且位姿诊断有明确异常证据时，才单独比较 Official Pose 与 Constrained Pose Proposal；之后才依序尝试 Static、Visibility、Dirichlet Conflict、受限视角残差和半监督。

## 10. 许可证、复现与结论边界

真实 run 前，每个候选 config 必须记录上游 URL、精确 source commit、代码许可证、checkpoint 名称/hash/许可证、环境摘要、命令和输出根；第三方完整仓库、checkpoint、PLY、NPZ、日志和数据不进入 Git。

若候选受限于许可证或硬件，报告只能称：

> All runnable candidates that passed the pre-registered functional, licensing, and hardware gates...

不得声称四个模型都已完成比较。任何改变本协议中尺度、过滤、容量、训练节点、排名或停止条件的实验均为新协议版本，而非本 v2 的结果。
