# GEO-020-PF-R1 — DA3-Large P0 license and environment retry audit

状态：`blocked_license_confirmation`，不是模型质量、坐标或显存失败。

## 审计范围

本记录只复核预注册的 `depth-anything/DA3-LARGE-1.1` 是否可以在隔离环境启动真实
P0 小窗口 smoke。它不下载源码或权重，不运行 DA3，不生成 NPZ/PLY，也不启动 Scaffold
训练、位姿优化、多模型融合或语义训练。

## 已核对事实

- 官方代码来源为 `https://github.com/ByteDance-Seed/Depth-Anything-3`，代码许可证为
  Apache-2.0；官方资料说明主系列支持已知位姿的 pose-conditioned depth；
- 预注册权重为 `depth-anything/DA3-LARGE-1.1`，官方 model-zoo 将该 Large-1.1 权重列为
  CC BY-NC 4.0；不得以 DA3-Base 的 Apache-2.0 权重静默替换；
- 独立 `icshm-geometry` 环境存在，CUDA 可用，且当前有足够空闲显存用于后续小窗口尝试；
  该环境中 `depth_anything_3`、`xformers` 和 `gsplat` 均不存在；
- 本机没有官方 DA3 源码 checkout 或目标 checkpoint；checkpoint hash、大小与 source commit
  因未下载而均不可填写；
- 项目记录中没有找到“CC BY-NC 4.0 的 DA3-Large-1.1 可用于本次比赛”的明确许可确认。

当前 `scripts/run_da3.py` 只能生成确定性 window manifest；其真实 adapter 在未固定并审查
官方 source API 前会显式退出。它是未来安装后的接口入口，不能被误报为 DA3 模型推理。

## 停止结论与回退

按 `GEO-020-PF-R1` 配置中的许可门，本次在下载、安装或模型执行前停止。没有真实窗口、
P0 深度、confidence、world points、`geometry_prior_npz_v2`、资源指标或质量结论；因此
不得启动 `GEO-020-10K`，也不得将本次阻塞称作 DA3 表现不佳。

回退保持为官方 P0 + 冻结 Scaffold 原始初始化。只有获得书面比赛用途许可确认后，才可固定
官方 source commit 和 checkpoint revision/hash，并在同一 `rgb_r1_spatial_v1` 训练侧用 8 图
共视窗口开始新的真实 P0 smoke；验证/guard-band/隐藏测试信息仍不得进入输入。
