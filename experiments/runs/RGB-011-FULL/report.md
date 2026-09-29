# RGB-011-FULL — Standard 3DGS 400 RGB R1 full run

- 状态：failed（CUDA out-of-memory at 8,590/30,000）
- 阶段：R1；父实验：RGB-011；Git commit：working tree uncommitted
- 配置：`experiments/runs/RGB-011-FULL/config.yaml`

## 冻结条件

仅把 RGB-011 的500 iteration筛选预算提高到30,000 iteration。使用`rgb_r1_spatial_v1`、303 train / 74 validation、官方世界坐标和`CAMERA_FROM_WORLD`、同 K 全帧 PINHOLE输入、训练专用重三角化初始化、实际固定训练 seed=0、相同本地评测和显式`test.txt`。100张无标注RGB中只有空间隔离后仍合格的84张加入训练；不使用伪标签。

命令：`scripts/run_r1_rgb_candidate.sh RGB-011-FULL standard-3dgs "$R1_OUTPUT_ROOT/rgb_r1_spatial_v1/scene_400_eval" 30000`。输出、日志、checkpoint和渲染仅在 Git 忽略的`$R1_OUTPUT_ROOT/r1/runs/RGB-011-FULL`。

## 待产生证据

完成后才登记 checkpoint 加载、74相机渲染、PSNR/SSIM/LPIPS、最差10%、140–146、训练/渲染时间、GPU小时、显存、模型大小和外部相机门。缺失LPIPS时状态只能为 `completed_partial_metrics`，不得排名。

## 失败记录

训练在第 8,590/30,000 次迭代因 CUDA out-of-memory 退出码 1。RTX 5070 总显存 11.46 GiB，下一次 rasterization 需要 244 MiB，而当时仅剩 147.69 MiB。运行目录只有 `input.ply`、`cameras.json` 和 `cfg_args`，没有 checkpoint、PLY、渲染或固定评测摘要，因此不能后处理恢复。该失败是冻结协议下的真实结果；不得通过降低分辨率、改变 densification 或缩短迭代数冒充同条件 400-RGB 结果。
