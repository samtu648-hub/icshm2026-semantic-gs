# RGB-001 — R1 spatial split and fair RGB backbone protocol

- 状态：completed（协议已冻结；候选预检完成，完整训练按单GPU串行运行）
- 阶段：R1
- 负责人：unassigned
- Git commit：working tree changes not committed
- 父实验：none
- 配置：`experiments/runs/RGB-001/config.yaml`

## 假设

在相同官方坐标、相机预处理、空间隔离验证集和训练预算下比较三个 RGB backbone，所得排序才可用于 R2 冻结决定。

## 与父实验的唯一主要差异

相对仓库已有的未定义留出视角，新增 `rgb_r1_spatial_v1`：基于官方 COLMAP 相机中心的确定性 5 空间簇，并对每个 holdout 簇应用 3×全局最近邻中位距离的训练排除缓冲区。

## 数据与划分

- Dataset manifest：`data/manifests/contest_dataset_v1.json`
- Split ID：`rgb_r1_spatial_v1`，见 `experiments/splits/rgb_r1_spatial_v1.json`
- 主验证折：`spatial-01`；74 张标注验证图，包含 140–146 全部历史困难视角。
- split 的确定性 k-means seed 为 20260730；三个上游的实际训练随机 seed 固定为 0，并已在所有 R1 screening/full 配置中如实记录。
- 400-RGB 训练：303 张；300-RGB 对照：219 张；验证及相邻缓冲视角不进入任一训练集。
- 是否使用100张无标注RGB：400-RGB 主比较使用剩余 84 张；300-RGB 对照不使用。
- 是否使用伪标签：否。
- 相机与畸变：官方 `CAMERA_FROM_WORLD` 保持不变；以 OpenCV 将 `SIMPLE_RADIAL` 全帧、无裁剪转换为 `PINHOLE`，并使用相同 K。
- 初始化：由 `images.txt` 现有共享 `POINT3D_ID` track 在去畸变像素空间 DLT 重三角化；均值重投影误差超过 3 px 的点剔除。
- 像素空间闭环：真实训练图像的“去畸变→回映射”RGB MAE 为 1.51/255，低于 4.0/255 门槛；语义回映射固定为最近邻插值。

## 命令与环境

```bash
export ICSHM_DATA_ROOT="/path/to/Contest Dataset"
export ICSHM_OUTPUT_ROOT="/path/outside/git"

python3 scripts/create_rgb_split.py \
  --dataset-root "$ICSHM_DATA_ROOT" \
  --write experiments/splits/rgb_r1_spatial_v1.json

python3 scripts/prepare_rgb_colmap_scene.py \
  --dataset-root "$ICSHM_DATA_ROOT" \
  --split experiments/splits/rgb_r1_spatial_v1.json \
  --output-root "$ICSHM_OUTPUT_ROOT/icshm2026/rgb_r1_spatial_v1/scene_400" \
  --training-key train_rgb_400
```

- GPU：主机上的 RTX 5070（12,227 MiB）和 NVIDIA 580.159.03 已由`nvidia-smi`验证可用。普通仓库沙箱会有意遮蔽`/dev/nvidia*`，所以训练和显存测量必须使用主机设备访问；这不是 Docker 迁移或驱动故障。
- Python/PyTorch/CUDA：系统 Python 3.12.3 没有本项目所需的 PyTorch、`nvcc` 或 COLMAP；R1 已在 Git 忽略位置建立隔离 Python 3.12 / PyTorch 2.11.0+cu128 / CUDA 12.8 环境。三个候选 CUDA 扩展均已编译并实际启动训练。
- 上游 commit：Standard `54c035f`、Mip `dda02ab`、Scaffold `59c833b`；均已克隆到 Git 忽略目录。最小补丁、许可证和 hash 将与运行日志一起归档；完整第三方仓库不进入 Git。
- 预算：四项 500 iteration screening 已完成（Scaffold 原始 checkpoint 序列化失败另行登记，修复重试成功）；四项 30,000 iteration完整比较已获授权且按单 GPU 串行执行。实际 GPU 小时、显存和最佳迭代只在真实运行结束后填写。

## 指标

本地公平比较固定使用`scripts/evaluate_rgb.py`：RGB经 Pillow 解码为[0,1]且不缩放或裁剪；PSNR基于每张图的全通道 MSE；SSIM使用11×11、σ=1.5的高斯窗口和反射边界；LPIPS固定为`lpips==0.1.4`的AlexNet v0.1，输入转为[-1,1]。每个候选必须输出同名验证PNG，保存逐视角CSV、均值、中位数和该指标的最差10%。这只是内部一致的本地实现，不能替代尚未发布的官方实现。

| 指标 | 结果 |
|---|---:|
| PSNR | not_available |
| SSIM | not_available |
| LPIPS | not_available |
| mIoU（含背景） | not_available |
| mIoU（不含背景） | not_available |
| `stay_cable` IoU | not_available |
| 背景→拉索误报 | not_available |

## 逐视角与困难视角

screening 已经为四个可运行候选产生 74 视角的 PSNR/SSIM CSV；LPIPS AlexNet 权重尚未缓存完整，所有这些条目均为明确的非排名 partial result。主验证折仍固定为 74 张标注图；全量、最差10%和140–146逐视角CSV是 RGB-010/011/020/030 的必交产物。

## 定性可视化

待填写低体积关键图及说明。原始渲染不进入Git。

## 失败和异常

- 先前的误判：普通沙箱中的`nvidia-smi`无法通信是设备节点隔离造成的；主机验证已排除此驱动问题。
- 所有全量训练仍在进行或排队，且 LPIPS 缓存尚未就绪；因此当前 screening 数值不能产生候选排序或冻结结论。
- Scaffold-GS 的上游 checkpoint capture/restore 调用已在迭代500暴露字段不一致；失败尝试 RGB-030 已保留，最小兼容补丁的独立重试 RGB-030-R1 已证明 checkpoint 可保存和加载。完整运行仍必须再次验证。

## 结论

数据 manifest、空间 split、困难视角归属、无泄漏缓冲审计、共同 PINHOLE 训练输入与固定本地RGB评测入口已经验证。因此 RGB-001 协议冻结完成。没有候选训练、评测或外部相机渲染结果，因而它尚不能用于冻结 RGB backbone。

## 保留决定

- keep_or_drop：keep_as_protocol
- reason：必须先完成 R1 公平训练及前两名语义冒烟测试，再作 R2 冻结决定。

## 下一步

完成四条授权的 30,000 iteration 运行，补全冻结 AlexNet LPIPS，对每个候选运行外部官方 COLMAP 相机门，并只让完整指标前两名进入五类语义渲染烟雾测试。
