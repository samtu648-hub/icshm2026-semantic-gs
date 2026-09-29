# 实验协议

## 不变量

公平比较中必须固定：

- 数据manifest；
- train/validation split ID；
- 输入分辨率和畸变策略；
- 相机坐标；
- 训练预算；
- seed集合；
- 硬件类型或明确记录差异；
- 指标实现和版本。

一次实验只改变一个主要变量，并通过`parent_experiment`说明基线。

## 验证划分

- 主要决策使用按相机中心、方向、高度或桥梁区域隔离的空间折；
- 相邻帧必须整体分配，防止随机抽帧导致虚高；
- 插值视角只用于调试，不能替代困难视角；
- 100张无标注图像落入验证空间块时，不得在该折的RGB训练中泄漏。

## 当前R1冻结划分

`experiments/splits/rgb_r1_spatial_v1.json`是 R1 的唯一主划分。它对官方 COLMAP 相机中心作确定性 5 簇分组，在每个 holdout 簇外再排除固定空间缓冲区。主折为`spatial-01`，包含 74 张标注验证图和全部 140–146 历史困难视角；其余训练图不得在 holdout 或缓冲区内。

该 split 的聚类 seed 是 20260730；R1 三个上游项目的 `safe_state()` 实际训练随机 seed 是固定的 0。两者分别记录，且所有 screening 与完整候选使用相同训练 seed=0；不得把 split seed 误报为训练 seed。

三个 RGB 候选和 Standard 的 300-RGB / 400-RGB 对照必须使用此文件的相同主折。因为空间隔离会排除验证和缓冲视角，`train_rgb_300`及`train_rgb_400`的实际张数分别小于 300 与 400；名称表示是否允许使用发布的 100 张无标注 RGB，而非绕过验证划分。

R1 的共同相机策略是：保留官方世界坐标与 `CAMERA_FROM_WORLD` 位姿，使用`prepare_rgb_colmap_scene.py`对`SIMPLE_RADIAL`图像进行全帧、无裁剪的 OpenCV 去畸变，并以相同 K 输出`PINHOLE`训练场景。缺失的`points3D.txt`由 released `images.txt`的共享 track 重三角化。此策略只定义本地公平比较输入；最终提交仍须把官方测试相机适配回其要求的像素空间。

本地 RGB 指标固定由[`scripts/evaluate_rgb.py`](../../scripts/evaluate_rgb.py)实现，具体解码、PSNR、SSIM、LPIPS和聚合参数记录在`RGB-001/config.yaml`与`METRICS.md`。候选不得自行更换指标实现；官方尚未发布的实现仍以最终协议为准。

## 报告

每个实验使用[实验报告模板](../../experiments/templates/report.md)，登记配置、命令、环境、逐视角结果、失败信息和保留决定。失败实验同样保留摘要。

## SEM-001 固定静态语义协议

`experiments/splits/sem_001_scaffold_static_v1.json`从 R1 主折 `spatial-01`确定性派生：只取`train_rgb_400`中有释放 LabelMe 标注的 219 张图作为语义监督，74 张`validation_labeled`为完整空间验证；R1 holdout、7 张标注 buffer 和全部100张无标注 RGB 不进入语义训练或损失。其 guard 检查继承 R1 最小 train-to-holdout 距离 0.5220235。

LabelMe JSON 由`prepare_semantic_labels.py`确定性转换到 R1 的无裁剪 PINHOLE 网格：顶点用同一 SIMPLE_RADIAL K 去畸变、四舍五入后填充，多边形按 LabelMe source-list 顺序覆盖；不存在释放的 ignore/void 像素。`experiments/runs/SEM-001/label_manifest.json`固定每个输入 JSON/输出 PNG hash、类别像素数和此规则。

SEM-001 只训练 per-anchor 五类 logits，生成 Gaussian 继承 anchor logits。所有 Scaffold anchor 几何、opacity、RGB MLP、covariance MLP 和 appearance embedding 都禁止梯度；任何完整运行必须以 RGB artifact hash 和固定三相机最大像素差 0 证明这一点。评测使用累计像素混淆矩阵的本地五类 IoU/mIoU，逐视角、最差10%和140–146记录在结果中；这不替代尚未发布的官方 mIoU 规则。

创新假设、消融矩阵、通过门槛和否定条件见[创新点验证与消融预注册](innovation_validation.md)。
