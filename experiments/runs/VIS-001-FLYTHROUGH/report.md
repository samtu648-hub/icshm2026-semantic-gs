# VIS-001-FLYTHROUGH — frozen Scaffold/SEM-001 novel-view video

- 状态：completed visualization-only
- 父实验：`SUBMIT-010-BUNDLE`、`RGB-030-FULL`、`SEM-001`
- 仓库 HEAD：`49cdcdb1c9915a9544d568754af95ea20a85ec9c`
- 运行时保留既有工作树；完成后 tracked diff SHA-256：`af0f925d89380a6b4ce6f2a614145a5ccef89e7c76b9c2a8aeaa93e2caa4bb90`
- 配置：`experiments/runs/VIS-001-FLYTHROUGH/config.yaml`

## 目的和数据边界

该结果用于证明冻结 Scaffold-GS 输出不是复制数据集 RGB，并提供不能由通用 SuperSplat 直接加载的 anchor+MLP 表示的忠实动态可视化。轨迹只读取官方 `cameras.txt`/`images.txt` 中的相机元数据，不读取任何参考 RGB 或标签，也不训练或修改 RGB/语义参数。

从 `140→146→154→161→169→176→183` 七个相机元数据关键帧构造六段轨迹。相机中心使用 Catmull-Rom，`CAMERA_FROM_WORLD` 四元数使用平滑 SLERP；每段24帧采用半步采样，故生成的144个 pose 均不等于数据集关键帧。最近生成中心到任一官方相机中心距离为0.00501774个官方坐标单位，最大相邻中心步长为0.124143。

轨迹 manifest SHA-256：`f5b9f43c93dcfa2a57238aac001571987fc60a04ca0a2a29acc7bf4b7fdde59e`。

## 真实渲染与视频

冻结提交包一次加载后渲染144个新 pose，耗时47.6127秒，峰值 PyTorch allocated 382 MiB；输入为官方参考坐标、`CAMERA_FROM_WORLD` 和 `SIMPLE_RADIAL`，输出为官方像素网格1320×989 RGB与单通道0–4 SEM-001类别图。包/输出审计全部通过，render manifest SHA-256 为 `2ec6df861b500d3c472ba4f94f3c074162060d285fe81fb755b819e1a3948669`。

FFmpeg使用固定24 fps、H.264、yuv420p和CRF 18编码：

| 视频 | 尺寸 | 帧数/时长 | 大小 | SHA-256 |
|---|---:|---:|---:|---|
| RGB novel-view | 1320×990 | 144 / 6.0 s | 3,131,390 B | `cfa4ab6a33929c6d233e1d50df9906295db93b6d2a2c164a8b584c9d27c9791b` |
| RGB + SEM-001 overlay | 2640×1030 | 144 / 6.0 s | 8,451,118 B | `3a8182de760ff594570000ddccb30048c5e9c9f8d78ce61dc8e9879743d44913` |

视频无音轨。右侧语义叠加颜色为 deck橙、stay_cable黄、tower蓝、foundation绿。中间视角可见训练覆盖不足处的模糊/浮点，这是冻结方案的真实局限，没有进行后处理掩盖。

## 结论

视频是真实 Scaffold anchor+MLP 动态神经高斯渲染，而不是把原始RGB编码成视频。它证明了冻结模型可以在不对应数据集图片的新相机上连续渲染，并同时输出SEM-001语义；它是结果可视化，不产生新的PSNR/mIoU结论，也不改变冻结模型决定。
