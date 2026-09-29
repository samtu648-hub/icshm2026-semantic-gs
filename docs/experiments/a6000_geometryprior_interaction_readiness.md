# A6000 GeometryPrior 交互实验准备矩阵

> 状态：`prepared_not_authorized_to_execute`。本文件只准备迁移到 A6000 后的审计边界、资源和配置入口；它不是已运行实验，也不授权 GPU 任务。
>
> 事实来源优先级：`experiments/registry.yaml`、各 `GEO-*` report/config、`PROJECT_STATUS.md`、`TASKS.md` 与 `DECISIONS.md`。本文件不改写既有实验的 ID、状态或结论。

## 1. 不可改变的共同协议

- 最终盲测仍只由冻结 Scaffold-GS 接收官方外部 COLMAP `CAMERA_FROM_WORLD` 相机；MapAnything 与 DA3 仅可作为离线训练期几何来源。
- 训练/验证边界固定为 `rgb_r1_spatial_v1` 的 `spatial-01`：303 RGB 训练视角、74 空间隔离验证视角与既有 guard band；官方 `SIMPLE_RADIAL` 必须配对 full-frame 去畸变 PINHOLE 内参。
- 每个未来 run 固定 seed 0、1320×989、Scaffold commit/minimal patch、训练损失与 `scripts/evaluate_rgb.py` 的本地协议；在 1k、3k、5k、10k 记录 checkpoint、PLY、逐视角指标、最差10%与140–146困难视角，并重载 checkpoint、以外部官方相机渲染。
- 任何候选都必须从 iteration 0 连续训练。10k checkpoint 不得接续为30k公平结果；30k只在新的、已批准选择门通过后另起连续训练。

## 2. 已有结果与迁移限制

| 历史实验 | 现有结论 | A6000 上允许的动作 |
|---|---|---|
| `GEO-000-10K` | 已完成原始 Scaffold 初始化控制，10k 为 20.48798 PSNR / 0.85027 SSIM / 0.33167 LPIPS-Alex。 | 可作为结果参照；若需要验证新机器一致性，必须新建唯一 `GEO-A6000-000-10K-*`，不得覆盖或改写历史控制。 |
| `GEO-010-10K` | MapAnything 全量 P0 替换初始化已完成并淘汰：LPIPS +0.02587、最差10% PSNR −0.75640。 | 不得仅因显存更大而复跑同一配方或使用同一 ID。需要不同窗口/过滤/容量假设时，必须另行预注册并建立新 ID。 |
| `GEO-020-3K-INT-R1` | DA3 P0 初始化工程门通过，但相对匹配3k控制 PSNR −1.2184 dB、LPIPS +0.1012，登记为 drop；没有 `GEO-020-10K`。 | 不得启动历史 `GEO-020-10K`。若 A6000 支持不同且有科学理由的 DA3 配方，先获独立 Goal 与新 ID。 |

因此，A6000 提供的是更充足的显存和更低 OOM 风险，不是推翻上述质量结论的理由。任何真正的 **Backbone × GeometryPrior** 审计必须预先说明新的主要变量，不能把硬件迁移包装成新模型收益。

## 3. 预备矩阵（尚不运行）

机器到位并获批后，先完成 `A6000-ENV-PF` 环境/接口预检。只有它通过，才可在新的 Goal 中选择下列行；星号后的 ID 是保留命名模式，当前不是 registry 中的运行实验。

| 目的 | 历史父证据 | 未来唯一 ID 模式 | 允许变量 | 禁止事项 |
|---|---|---|---|---|
| 新机器控制一致性 | `GEO-000-10K` | `GEO-A6000-000-10K-R1` | 仅硬件/环境迁移，完整记录编译与版本。 | 不与5070数字混为同一随机种子复现的质量提升；不得改 split、相机或损失。 |
| MapAnything 新假设 | `GEO-010-10K` 负面结果 | `GEO-A6000-010-<new-hypothesis>-R1` | 仅一个预注册的新变量，例如不同且可解释的过滤或窗口协议。 | 不重复原全量替换或 `GEO-011` 添加式配方；不得先看验证结果再改阈值。 |
| DA3 新假设 | `GEO-020-3K-INT-R1` 负面结果 | `GEO-A6000-020-<new-hypothesis>-R1` | 仅一个预注册的新变量，并保持 P0/xyz+原始RGB的边界，除非 Goal 明确批准更改。 | 不把预测位姿替换为官方 P0；不得续训历史3k或复用历史 ID。 |

本阶段不纳入 LingBot、VGGT-Ω、Standard、Mip-Splatting、MCMC 或新语义模块。它们需要独立的许可证、依赖与研究假设审查。

## 4. A6000 资源与磁盘门

- 目标硬件：NVIDIA RTX A6000，48 GiB VRAM。开始前保存主机 `nvidia-smi`、PyTorch/CUDA 探针、驱动、Python、PyTorch、CUDA、Scaffold/前馈环境与 CUDA extension build 证据。
- 不将 CPU RAM 与 GPU VRAM相加计算。CPU offload 仅是前馈模型的回退，不改变 GPU 显存门或冻结 Scaffold 环境。
- 5070 已观察到的峰值仅作下界：Scaffold 10k离散采样约7.10–7.71 GiB；MapAnything P0推理约6.61 GiB；DA3 P0推理约4.67 GiB。A6000 run 仍须在空闲显存下独立采样峰值，不能假定同比例缩放。
- 磁盘：历史 `GEO-000-10K`、`GEO-010-10K`、DA3 PF 与 DA3 3k产物分别约1.2、1.3、1.8、0.6 GiB。对每个未来10k run预留至少10 GiB；若同时保留前馈 NPZ、两个候选和30k追踪，开始前要求输出卷至少80 GiB可用，建议100 GiB。
- 时间：5070上控制/MapAnything 10k 的纯训练分别约549/622 s，且完整审计还包含渲染、LPIPS-Alex、重载和外部相机门。A6000的实际速度不能从这些数值外推；每个10k单任务申请按最多1 GPU小时预算，首轮仅在实际资源记录后估算批量30k。

## 5. 必过预检与执行顺序

1. `A6000-ENV-PF`：不训练；验证已固定 source commit、CUDA扩展、外部 `CAMERA_FROM_WORLD`、全帧去畸变、checkpoint重载及输出根的 Git 忽略。
2. 新的 P0 几何 smoke：只在存在新且获批的模型假设时运行；保存 source/checkpoint hash、window manifest、`geometry_prior_npz_v2` schema、坐标/尺度和训练侧边界证据。
3. 新鲜 `0→10k` 控制与一个候选：固定共同协议，单一主要变量，保存所有阶段 artifact 和配对74视角指标。
4. 只有候选通过其新预注册的质量、最差10%、困难视角、坐标、重载和外部相机门，才申请独立的30k批量预算。

所有原始 checkpoint、PLY、NPZ、渲染、日志及主机绝对路径仍只留在 Git 忽略的输出根。执行时必须更新 registry、config、report、manifest、split、seed、命令、环境和 SHA-256；失败同样登记。
