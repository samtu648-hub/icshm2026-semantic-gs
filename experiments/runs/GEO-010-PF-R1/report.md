# GEO-010-PF-R1 — MapAnything Apache official-P0 real smoke retry

状态：`completed_passed`。这是对 `GEO-010-PF` 的新增真实 retry；原
`blocked_dependency` 历史保持不变。

## 固定来源与边界

- 主项目 P0 adapter/test commit：`b6756f969f11b4ccac940508321262a11d327ca7`。
- 官方源码：`facebookresearch/map-anything`，commit
  `9d1db2dd728bd8a10e74b15d2eb646e1bf933791`，Apache-2.0。
- 官方权重：`facebook/map-anything-apache` revision
  `00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a`，Apache-2.0，
  `model.safetensors` 为 4,914,062,480 bytes，SHA-256
  `fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201`。
- 隔离环境为 `icshm-geometry`：Python 3.12.13、PyTorch 2.11.0+cu128、
  CUDA 12.8、RTX 5070 12GB、driver 580.159.03。冻结 Scaffold 环境未安装
  或升级 MapAnything 依赖。
- 输入只取 `rgb_r1_spatial_v1` 的 303 个 `train_rgb_400` 相机；P0 为官方
  `CAMERA_FROM_WORLD` 转 OpenCV `cam2world`，全帧去畸变 PINHOLE RGB/内参。
  验证、guard-band、验证标签及盲测信息均未输入；预测位姿只记录诊断，不替换 P0。

## 真实 smoke

2026-08-04 完成两个 16 图官方共视窗口：`cov-w000`（根 `115.png`）与
`cov-w001`（根 `126.png`），overlap=4、batch=1、bf16、
`memory_efficient_inference=true`。窗口无 weak-context padding，第二窗口与第一
窗口有 4 图重叠；这不是把 400 张图片当连续视频。

- 模型加载 10.14 s，推理 88.53 s；峰值 PyTorch allocated 6,608 MiB
  （独立 `nvidia-smi` 采样约 7.10 GiB，非连续峰值记录）。
- 共输出 32 条 view record（28 个唯一训练图像）；无 OOM、NaN 或 Inf。
- 有效深度比例范围 0.57176–0.62697；P0 输入位姿最大绝对误差
  `1.176e-7`；深度尺度和官方稀疏轨迹对应关系均已写入 manifest。
- `geometry_prior_npz_v2` schema audit 无错误。window manifest SHA-256 为
  `c8e4ea79819c0dddefdfcd1d7898cce0b489d688eb3dfb23fdb1934f6f004e33`，audit
  SHA-256 为 `0865c62c31757d42b38f7d3687e9a009b2aa5f4c0eee4f0a69621ae3b90fc507`。

第一次 stdout 的 `tee` 包装目录错误后提前关闭，但模型子进程继续完成；该包装失败
和完整 manifest 都保留在 Git 忽略运行目录。它不被改写成一次无失败的运行。

结论：功能、许可证、显存、P0 坐标和 NPZ schema 门均通过，允许进入独立的
`GEO-010-P0` 全训练侧离线几何；这不是 RGB 质量结论。
