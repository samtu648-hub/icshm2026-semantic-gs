# GEO-000-10K — Fresh continuous Scaffold control

状态：`completed_control`。这是从原始 R1 training-only DLT 初始化新鲜连续训练
0→10,000 的控制，绝不是历史 `GEO-000` 3k checkpoint 续训。

## 固定条件与运行证据

本运行使用 303 个训练 RGB、74 个空间隔离验证相机、官方 P0 去畸变 PINHOLE 相机、
seed 0、1320×989、冻结 Scaffold source/minimal patch、以及固定的本地
PSNR/SSIM/LPIPS-Alex 评测。训练 launcher revision 为
`7b7d2ebb5550183be865ae8e1ff12dd74f7bfbda`，实际 launch worktree HEAD 为
`c51e2289cd29341295db0380f62f12cd05b42421`。

训练进程从 2026-08-04 03:53:34 至 04:02:43+08:00 连续完成（约 549 s 的训练
阶段），并保存 1k/3k/5k/10k checkpoint 与 PLY。上游训练结束后错误地尝试其非协议
LPIPS-VGG 指标，因缓存 hash 失败；训练和 74-view 10k 渲染均已完成，结果未重训。
随后以 `23bfe97` 的恢复评测器对同一产物补齐四个 checkpoint 的固定 LPIPS-Alex
评测。运行 metadata 的完整外部 artifact SHA-256 为
`50bc0a4206b6eafbc38e09001db304efea01d043f03389e3ca6aaee08f38b2a4`；其 977 s
wall time 包含该失败和恢复，不能当成纯训练耗时。

| iteration | PSNR dB ↑ | SSIM ↑ | LPIPS-Alex ↓ | 最差10% PSNR ↑ |
|---:|---:|---:|---:|---:|
| 1k | 19.64309 | 0.82750 | 0.42780 | 16.55672 |
| 3k | 20.20079 | 0.83907 | 0.37339 | 17.56152 |
| 5k | 20.37993 | 0.84562 | 0.35153 | 17.76326 |
| 10k | 20.48798 | 0.85027 | 0.33167 | 17.89486 |

10k checkpoint SHA-256：
`7fe445cc7e464c56437dacdb303ddcf6e702c1b34077d7531a6d5c758a682d45`；10k PLY
SHA-256：`5ddc4c7afb16fed04dc3dd4a175dc590628bf0d32d862820d389bb868fb75e51`。
checkpoint 重载通过（176,561 anchors）；外部官方 SIMPLE_RADIAL、
`CAMERA_FROM_WORLD` 相机 `140.png` 以 identity Sim(3) 成功渲染 1320×989 RGB，
不读取参考 RGB/标签（PNG SHA-256
`7edc5e56860e863256efbb0a748a1cfe9612f548682ed803c0bb870c850bde71`）。

资源记录：RTX 5070 12GB；训练时离散采样的 Scaffold 进程最高为 7.10 GiB
（非连续峰值记录）。上游 74-view 渲染日志为 138.48 FPS，即名义单视图
7.22 ms；10k checkpoint/PLY 大小约 148/53 MiB。

结论：`GEO-000-10K` 是有效的 MapAnything P0 同预算控制，保留作为后续已批准
GeometryPrior 的共同基线。
