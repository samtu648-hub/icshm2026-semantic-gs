# GEO-020-3K-INT — DA3 P0 + Scaffold engineering gate

状态：`blocked_runtime_approval_after_environment_fix`，不是 DA3 质量结论。

DA3-Large P0 smoke、训练侧 70 窗口几何生成、`geometry_prior_npz_v2` 输出、P0 融合和
100k 点 Scaffold 初始化 PLY 均已完成。初始化只改变外部场景的 `sparse/0/points3D.ply`，
使用官方坐标和原始 RGB。

Scaffold 接入先后发现两个工程问题：冻结环境缺少 `wandb` 导入；已有 CUDA 扩展为 CPython
3.12，而 R1 Python 为 3.11。没有修改 Scaffold 算法，已在相同上游 commit 和 CUDA 12.8
下重编 CPython 3.11 的 `diff_gaussian_rasterization`、`simple_knn`，导入验证通过。
随后正式 3k 命令被当前执行工具的 GPU 审批额度拒绝，命令没有执行。

因此当前没有 1k/3k checkpoint、RGB 指标、重载或外部相机结果；不得启动 GEO-020-10K。
恢复执行时应复用本 config、场景和 PLY，从 iteration 0 连续训练 3,000 iteration，并补齐
checkpoint reload、官方外部 `CAMERA_FROM_WORLD` 相机和 1320×989 输出证据。
