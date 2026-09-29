# SEM-001 — frozen Scaffold-GS static five-class semantic baseline

- 状态：completed
- 阶段：R2/S0；父实验：`RGB-030-FULL`、`RGB-060`、`SEM-001-PF`
- Git commit：working tree uncommitted（不提交）
- 配置：`experiments/runs/SEM-001/config.yaml`

## 假设

在完全冻结的 Scaffold-GS RGB 几何/颜色/opacity 上，按 anchor 附着的静态五类 semantic logits 能形成一个可审查的 S0 基线。它是后续可见性监督的比较对象，不包含伪标签、可见性筛选、冲突统计或视角条件残差。

## 预注册协议

- 219 张 R1 训练侧带标注图监督，74 张空间隔离带标注图验证；100 张无标注 RGB 不参与语义训练；
- R1 的 full-frame `SIMPLE_RADIAL → PINHOLE` 对齐策略用于 LabelMe 顶点；标签为单通道 uint8 PNG，ID 仅为 0–4；
- 每 anchor 训练五 logits，生成 Gaussian 继承其 anchor logits；训练图的像素频率确定加权 NLL 权重；
- 训练前后冻结 RGB artifact 的 SHA-256 相同，三张固定外部相机 RGB PNG hash 相同，最大绝对像素差必须为 0；
- 外部官方 `CAMERA_FROM_WORLD` 相机产生 RGB、5×H×W log-probability logits 和同尺寸单通道类别 PNG；identity Sim(3) 明确写入证据。

## 实际运行与环境

授权后的预注册 10,000 iteration 真实运行退出码为 0。命令如下；所有大体积输出只在 Git 忽略的 `$R1_OUTPUT_ROOT/r2/sem_001/full/` 下。

- 数据 manifest：`data/manifests/contest_dataset_v1.json`；语义 split：`experiments/splits/sem_001_scaffold_static_v1.json`；LabelMe label manifest：`experiments/runs/SEM-001/label_manifest.json`；
- 冻结 RGB 输入：`RGB-030-FULL` 的 Git 忽略 iteration-30000 checkpoint/PLY/MLP，固定 Scaffold-GS upstream `59c833b56bbbf510f3f64d40f81721995caced66`；其许可证、本地最小兼容补丁和 provenance 由 R1 记录保留。

```bash
export ICSHM_DATA_ROOT="$USER_HOME/data/Dataset-Project 2/Contest Dataset"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$R1_ENV_ROOT/bin/python" scripts/run_sem001.py \
  --config experiments/runs/SEM-001/config.yaml \
  --data-root "$ICSHM_DATA_ROOT" \
  --labels-root "$R1_OUTPUT_ROOT/r2/sem_001/labels" \
  --rgb-run-dir "$R1_OUTPUT_ROOT/r1/runs/RGB-030-FULL" \
  --source-root "$R1_THIRD_PARTY_ROOT/scaffold-gs" \
  --output-dir "$R1_OUTPUT_ROOT/r2/sem_001/full" --mode full
```

- Git：`b7d77a1094b7574a793bb6efb892430b5ac434e9`，运行时工作树为有意未提交状态；
- GPU：RTX 5070 12,227 MiB，driver 580.159.03；Python 3.12.3、PyTorch 2.11.0+cu128、CUDA 12.8；
- 训练：seed 20260731，10,000 iterations，3,076.649 s（0.8546 GPU h），PyTorch allocated 峰值 824 MiB；
- sidecar：189,073×5 anchor logits，checkpoint 3,783,912 B，SHA-256 `8e3dbe30e17f67081a72380018fc281c155ec00eed83cd5a19b3285bf4fb8aa3`；
- 损失：首/末/均值为 1.60849 / 0.14449 / 0.28305。训练使用的固定 class weights 为 background 0.06282、deck 1.0、stay_cable 0.63018、tower 2.75896、foundation 6.87625。

## 空间隔离验证结果

结果来自全部 74 个 `semantic_validation_labeled` 相机的像素累计混淆矩阵；这是本地固定协议，不宣称为主办方尚未发布的最终 mIoU 聚合规则。

| 指标 | 结果 |
|---|---:|
| mIoU（含 background） | 0.61610 |
| mIoU（不含 background） | 0.56334 |
| background IoU | 0.82717 |
| deck IoU | 0.90100 |
| stay_cable IoU / precision / recall | 0.06876 / 0.07796 / 0.36804 |
| tower IoU | 0.83487 |
| foundation IoU | 0.44872 |
| background → stay_cable 误报 | 13,166,331 px（真实 background 的 15.0076%） |

最差 10% 的 8 个视角是 061、059、062、060、063、064、058、057，其不含 background mIoU 均值为 0.42344。困难视角 140–146 的不含 background mIoU 均值为 0.60121。完整逐视角 CSV、混淆矩阵和所有 74 张预测均在外部输出，输出根的 `result.json` SHA-256 为 `abd82b7c21820c26e7cea4fa0d7b7ce155d833107b0388e9783c78ba9b81900d`。

定性检查的 140 视角显示 deck/tower 可用，但黄色 `stay_cable` 在水面和远景有明显扩散；它与上述大规模 background→stay_cable 误报相互印证。这是 S0 基线的真实负面证据，不是后续方法已取得的改善。

## 冻结 RGB、重载与外部相机门

- 训练参数白名单只有 `semantic_anchor_logits`；运行后冻结 Scaffold 的 RGB 可训练参数数为 0；
- RGB checkpoint、PLY、opacity/cov/color MLP 和 appearance embedding 的 6 个 SHA-256 训练前后逐项相同；
- 官方 001/054/140 三相机 RGB PNG hash 训练前后相同，最大绝对像素差严格为 0；
- semantic checkpoint 保存并 bit-identical 重载成功；
- 完整 runner 的外部 140 相机按 `CAMERA_FROM_WORLD`、identity Sim(3)、`SIMPLE_RADIAL` 输入输出 1320×989 RGB、`[5,989,1320]` log-probability logits 和单通道 uint8 0–4 PNG，且不读取参考 RGB/标签；
- 独立 `scripts/render_sem001_external_camera.py` 再从该 checkpoint 和外部 `cameras.txt`/`images.txt` 完成相同渲染。其 RGB、semantic PNG 与 logits 与完整 runner 的文件 SHA-256 均一致，证明接口不依赖训练 runner 内存或发布数据路径约定。

## 失败和限制

`SEM-001-PF` 的两次启动前失败（缺少固定 CUDA 子模块 import path、上游 GaussianModel 不是 nn.Module）均在任何语义迭代前发生，并已保留及记录在其报告。修复后预检与完整运行都通过。

SEM-001 只证明一个静态、空间隔离的可复现基线。其 `stay_cable` precision/IoU 和背景误报不适合作为最终提交语义方案；这为已预注册的可见性监督比较提供基线证据，但不等于可见性模块已经被证明有效。

## 保留决定

- keep_or_drop：keep_as_s0_baseline
- reason：完整工程门和真实空间验证均通过；静态语义的背景→拉索误报被量化并保留，作为 S1 及后续消融的对照，而不是被掩盖或当作最终语义方案。
