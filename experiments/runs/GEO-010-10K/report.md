# GEO-010-10K — Fresh continuous Scaffold + MapAnything Apache P0 initialization

状态：`completed_drop_at_10k`。MapAnything P0 融合初始化通过功能和接口门，但未通过
预注册的 RGB/工程入围门；不进入 30k，也不改变冻结 Scaffold RGB 骨干。

## 同预算运行

唯一主要变量是初始 PLY：`GEO-010-P0` 的 100,000 点、官方 P0 坐标、xyz+原始 RGB
融合产物（SHA-256
`f6a69c4d15bf49eaf8019a5c0cb31d4d796fd9261121a538c7110fd17832d889`）。训练侧相机、
303/74 split、seed=0、分辨率、Scaffold 损失、评测器与 `GEO-000-10K` 相同。训练
runner revision 为 `23bfe97e14e260688e05a42094f8a858bcfa962f`，launch worktree HEAD
为 `aa19b7077953307116bf23cce607bd9fa31ce322`。

单一 0→10k 进程于 2026-08-04 04:12:15–04:26:05+08:00 完成；其中训练阶段为
04:12:16–04:22:38（约 622 s）。该运行显式跳过 Scaffold 上游的非协议 VGG 指标，
随后对每个 checkpoint 以冻结 LPIPS-Alex 复算 74 个视图。完整 metadata SHA-256：
`173034cb9a100c0e3055ce9d1758e9ce2f3667f65a10b775248f7c889ee690c0`。

| iteration | PSNR dB ↑ | SSIM ↑ | LPIPS-Alex ↓ | 最差10% PSNR ↑ |
|---:|---:|---:|---:|---:|
| 1k | 19.52706 | 0.80713 | 0.52733 | 16.33709 |
| 3k | 19.77265 | 0.82547 | 0.43266 | 16.26038 |
| 5k | 20.28286 | 0.83651 | 0.39422 | 17.05374 |
| 10k | 20.48281 | 0.84441 | 0.35755 | 17.13846 |

## 与新鲜控制的 10k 配对比较

`compare_geometry_rgb_runs.py`（commit `ac5ab86`）在同一 74 个视图上以 seed
20260803、10,000 次 paired bootstrap 计算 candidate−control。比较 artifact SHA-256：
`674d5bff19201f9eedff8d594aedccdc5dd127d55fc2c7de43e9d9a3b635abec`。

| 指标 | candidate−control | 95% paired CI | 140–146 平均差 |
|---|---:|---:|---:|
| PSNR dB ↑ | -0.00518 | [-0.14922, +0.14568] | -0.14984 |
| SSIM ↑ | -0.00586 | [-0.00684, -0.00498] | -0.00586 |
| LPIPS-Alex ↓ | +0.02587 | [+0.02402, +0.02776] | +0.02279 |

最差10% PSNR 为 17.13846，相比控制 17.89486 下降 0.75640 dB。虽然平均 PSNR 接近，
LPIPS、SSIM、最差10%与困难 140–146 均退化，故不满足“PSNR +0.2 dB 或
LPIPS -0.01，且最差10%不得明显退化”的 10k 入围条件。

## 工程门与资源

10k checkpoint SHA-256：
`3da962d0ea658b2021118d5e98fe639f7c33528ca52264d189f767f9fabdf528`；10k PLY
SHA-256：`444504d1241de5c514c97fb0e6e3ac379fc2c11303076c04aecd8f08f9a3a621`。
checkpoint 重载通过（206,576 anchors）。外部官方 SIMPLE_RADIAL
`CAMERA_FROM_WORLD` 相机 `140.png` 用 identity Sim(3) 成功输出 1320×989 RGB，未读
参考 RGB/标签；PNG SHA-256 为
`d0668e30ddadd6567e1a8c59011d0f74e04ae3d1a0ea7e3a3889f5ec92e303b7`。

训练时离散采样的进程最高为 7.71 GiB（非连续峰值记录）；上游 74-view 渲染日志为
112.80 FPS，即名义单视图 8.87 ms。10k checkpoint/PLY 约为 173/63 MiB，均大于控制
的约 148/53 MiB。它以更多 anchors 和更慢渲染换来了更差的感知质量与困难视图表现。

结论：淘汰**该 MapAnything Apache P0 融合初始化配方**，回退原 Scaffold 初始化；
这不是“MapAnything 不可用”或其他模型/位姿/过滤配方都无效的结论。任何替代 MapAnything
消融都必须新建 experiment ID、在训练前预注册，不能复用本次 10k 作为正面结果。
