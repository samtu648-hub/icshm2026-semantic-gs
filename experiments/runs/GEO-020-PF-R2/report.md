# GEO-020-PF-R2 — DA3-Large official-P0 smoke retry

状态：`completed_smoke_and_geometry_generation`。

用户已明确确认 `DA3-LARGE-1.1` 的 CC BY-NC 4.0 权重允许用于本次比赛和相关研究。随后固定
官方源码 commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` 并下载目标权重。

已确认：RTX 5070 驱动可用（12,227 MiB），隔离环境 `icshm-geometry` 可用；权重
`model.safetensors` 为 1,643,843,860 bytes，SHA-256 为
`739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64`。

DA3-Large 在两个训练侧 8 图强共视窗口上真实推理成功，随后按 8 图、overlap=4 的
camera-graph 窗口覆盖 303 个训练视角，完成 70 个窗口、560 个 NPZ。输入为官方
`CAMERA_FROM_WORLD` 转换后的 P0 相机；输出 depth/confidence/valid_mask/world_points，
无 NaN/Inf，峰值 Torch 显存 4,671 MiB，AMP 为 bf16。DA3 预测位姿只保留为诊断。

DA3 P0 几何融合也已完成：普通表面至少 3 视角支持、线结构 2 视角例外、相对深度容差
0.05、r0=`0.05992645844916069`，输出 100,000 点 Scaffold PLY；PLY SHA-256 为
`5de866cfe1523b33c70dc8668f02def01dfcae092d7499b4d924113d8138d02f`。Scaffold 输入仅使用
xyz+原始 RGB，没有 DA3/语义 feature。

下一步为 `GEO-020-3K-INT`。首次启动因冻结环境缺少 `wandb` 失败；第二次发现已有 CUDA
扩展为 CPython 3.12，而实际 R1 Python 为 3.11。未改 Scaffold 算法，已用 R1 Python/CUDA
重新编译扩展并验证导入。最终重试命令被当前执行工具的 GPU 审批额度拒绝，尚无 3k checkpoint、
RGB 指标或外部相机结果；不得把工程接入写成完成，也不得启动 GEO-020-10K。
