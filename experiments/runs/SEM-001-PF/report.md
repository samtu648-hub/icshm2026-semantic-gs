# SEM-001-PF — frozen Scaffold-GS semantic preflight

- 状态：completed_preflight_not_a_semantic_baseline
- 阶段：R2/S0；父实验：`RGB-030-FULL`、`RGB-060`
- Git commit：working tree uncommitted（运行时将记录状态；不提交）
- 配置：`experiments/runs/SEM-001-PF/config.yaml`

## 范围

这是低预算接口与参数隔离预检，不是语义准确率基线。它以 219 张 R1 空间训练侧的带标注图训练 8 次迭代；验证最多只抽少量视角确认 CUDA、标签、损失、重载和外部相机链路。

## 已冻结协议

- 类别：`0 background`、`1 deck`、`2 stay_cable`、`3 tower`、`4 foundation`；
- split：`sem_001_scaffold_static_v1`，继承 `rgb_r1_spatial_v1/spatial-01` 的空间隔离及缓冲区；219 train / 74 validation；
- 无标注 RGB、伪标签、验证 RGB 和验证标签绝不进入训练损失；
- LabelMe 多边形由释放的 `SIMPLE_RADIAL` 网格转换到 R1 训练 `PINHOLE` 网格，后写 shape 覆盖先写 shape；完整清单为 `../SEM-001/label_manifest.json`；
- RGB-030-FULL 的 anchor、offset、opacity、scaling、rotation、RGB MLP 和 appearance embedding 全部无梯度；只有每 anchor 五类 logits 可优化。

## 待产生证据

完整 mIoU 只属于后续 `SEM-001`，不得由本预检推断。

## 实际结果

第三次尝试在修复两项启动前工程问题后通过：8 次迭代训练用时 2.553 s（约 0.319 s/iteration），峰值 PyTorch allocated 为 719 MiB，训练损失从 1.60849 降至 1.36514。该短预检只评测前三个验证相机（054–056），其 0.58941 的不含背景 mIoU 是链路信号，绝不是 SEM-001 基线结果，也不用于比较或决策。

- 参数白名单：只有 `semantic_anchor_logits`；冻结 Scaffold 运行时仍有 0 个可训练 RGB 参数；
- artifact：checkpoint、PLY、opacity/cov/color MLP 和 appearance embedding 的 SHA-256 训练前后逐项相同；
- RGB：001/054/140 三张固定官方相机 PNG hash 完全相同，最大绝对像素差为 0；
- reload：3.7 MiB semantic sidecar checkpoint 保存后 bit-identical 重载通过；
- 外部官方 140 相机：identity Sim(3)、`CAMERA_FROM_WORLD`，输出 1320×989 RGB、`[5,989,1320]` log-probability logits 与 uint8 单通道 0–4 PNG；不读取该视角 RGB 或标签。

完整输出（包括两个保留的启动前失败 attempt）仅在 Git 忽略的 `$R1_OUTPUT_ROOT/r2/sem_001/` 下。按实测速度，预注册 10,000 iteration 完整预算约需 53 分钟 GPU 训练，另加模型加载和 74 视角评测；它等待明确资源授权。

## 失败和异常

首次尝试在任何语义迭代开始前失败：新 runner 只加入了 Scaffold 根目录，遗漏 R1 启动脚本显式加入的 `submodules/simple-knn` 与 `submodules/diff-gaussian-rasterization` 路径，因而出现 `ModuleNotFoundError: simple_knn`。没有加载完整 RGB 模型、没有写入 semantic checkpoint，亦未更改冻结 artifact。已将这两个固定上游子模块路径纳入 `src/scaffold_semantic_sidecar.py`，随后用新的、保留原失败输出目录的 attempt 重试。

第二次同样在训练开始前失败：固定上游的 `GaussianModel` 不是 `torch.nn.Module`，没有通用 `.parameters()` 方法。原先的冻结断言因而不完整。已改为显式枚举六个持久 anchor 张量及 opacity/cov/color/appearance（可选 feature-bank）模块全部参数，再逐个禁止梯度；这同时加强了参数白名单证据。失败输出同样保留，第三次仍使用完全相同的 8-iteration 配置。
