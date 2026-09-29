# GEO-010-P0 — MapAnything Apache P0 full offline geometry

状态：`completed_passed_for_candidate_generation`。本运行只生成训练侧离线几何；
它本身不等于 RGB 提升，RGB 结论见 `GEO-010-10K`。

## 全量 P0 几何

继 `GEO-010-PF-R1` 后，以同一官方 MapAnything Apache 源码/权重、P0
`CAMERA_FROM_WORLD`→OpenCV `cam2world` 和原始去畸变 RGB，完成 303 个训练侧
图像的 32 个共视窗口推理。验证 RGB/标签、guard-band 和盲测信息未被读取。

- 512 条 record、303 个唯一训练图像、32 个窗口；推理 1,257.48 s，峰值
  PyTorch allocated 6,611 MiB，无 schema error、OOM、NaN 或 Inf。
- 有效比例 min/mean/max 为 0.32432/0.50074/0.74599；深度尺度
  min/median/max 为 0.06408/0.10826/0.16492；P0 输入位姿最大绝对误差
  `2.372e-7`。
- 完整 window manifest SHA-256：
  `8a8d95ee0d52d30b9a088c94ccdc8bc8942fe53568ffed84bf46492ad88c4c16`；
  schema audit SHA-256：
  `4a67b34662109d7b4bd90d6a21c02a968d64ae4a3bfe340a6a6922e44040d73e`。

## 融合与 Scaffold 初始化

融合只向 Scaffold 导出 xyz 加原始 RGB，绝不混入深度、foundation 或语义 feature：

- 288,369 个候选点按普通表面至少 3 视角、细线至少 2 视角、相对深度阈值 0.05、
  stride 8 和由原 Scaffold 初始化定义的 `r0=0.05992646` 处理；最终为 100,000 点。
- 18,686 点使用细线结构规则；support 的 min/median/max 为 2/4/4。
- 输出 PLY SHA-256：
  `f6a69c4d15bf49eaf8019a5c0cb31d4d796fd9261121a538c7110fd17832d889`；
  provenance NPZ SHA-256：
  `6de4d03f822f3427f6eb0a3bd8901cca4313aa673d2a122968fb5bf3eb940933`；
  fusion manifest SHA-256：
  `bbc269edc9e12edcb95baa69e35e979b075befcb357bb3172037f202272dd6e6`。
- 首次 fusion 输出 PLY 后因 adapter 自身遗漏 `numpy` import 失败；该失败日志保留。
  修复 `ff2d464`、`76feffe` 后的第二次融合成功，未删除第一次输出。
- 冻结 Scaffold 的真实初始化加载检查通过：100,000 source points 变为 99,999
  anchors，全部有限；随后才启动 `GEO-010-10K`。

限制：本轮尚未验证独立的语义 sky/water mask；三视角支持是保守的几何代理，不能
把它表述为已证明的天空/反射完全清除。该限制和 P0 初始化的负面 RGB 结果均保留，
不得把本次融合复用为已验证的 GeometryPrior 胜者。
