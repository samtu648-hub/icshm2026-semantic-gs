# RGB-138-STANDARD-SORT-DIAG

状态：完成；零训练的渲染归因诊断，结果不支持继续 StopThePop 替换。

## 实际变量与来源

- 冻结 RGB130 的 652,298 个训练后高斯、官方相机和全部高斯参数，只比较同一 StopThePop 运行时内的全局中心深度排序与 hierarchical per-pixel depth 排序。两臂均关闭 rect/tight-opacity/tile/hierarchical culling、load balancing 和 proper EWA scaling；层次臂为 PTD_MAX、4/8/64 队列。因此执行时唯一主要变量是排序，未重新训练。
- 官方 `r4dl/StopThePop` commit `859b11bde9195e19a1c40536e1ab16765b64b365`，rasterizer submodule `d8cad09919ff49b11be3d693d1e71fa792f559bb`。Python 3.12 兼容补丁为 `third_party/patches/stopthepop-py312.patch`，SHA256 `e4182efc7d3ff476bb008519a77edb99056e2f561165e1926b3c974fccc499ad`；编译 sm120 二进制 SHA256 `c2b21d5e852693ec1cdd18122c44ae3f97020c9a27ead9320618f2d73c12c8ee`。
- 运行配置最初引用上游 `configs/hierarchical.json` 作为候选来源；正式执行按排序归因目的显式关闭该配置中的其他开关，精确设置已写入 `result.json`。这是执行配置的显式收窄，不是隐性质量调参。

## 结果

400 张训练图仍只是拟合口径：

| 冻结模型渲染方式 | PSNR dB | SSIM | LPIPS |
|---|---:|---:|---:|
| StopThePop 全局排序控制 | 23.217561 | 0.893216 | 0.252786 |
| StopThePop hierarchical 排序 | 22.952990 | 0.886531 | 0.256466 |
| hierarchical − global | −0.264571 | −0.006685 | +0.003680 |

逐图方向高度一致：PSNR 仅 2/400 改善，SSIM 0/400，LPIPS 33/400。全局控制与保存的 RGB130 指标仅相差 −0.000112 dB、−0.0000012 SSIM、+0.0000043 LPIPS，说明评价口径已基本复现；PNG 的平均绝对差为 0.00916/255，但不是逐像素完全相同，因此结论严格来自同一 StopThePop 运行时的两臂比较。

层次排序的中位渲染耗时为 60.32 ms，控制为 14.08 ms（约 4.29 倍）。全任务 386.94 秒；PyTorch 峰值 allocated/reserved 为 870.6/6626 MiB。固定九图重新加载后逐字节一致。

固定裁剪及 144 帧轨迹上，岸线拖影、真实细索断续和远端模糊没有得到稳定修复；多个视图反而出现一致的合成/色调差异。无 GT 的相邻帧非配准代理也略退，不能当成几何准确率。故当前伪影主要不是一个可由后处理排序单变量解释的问题，也不值得把 StopThePop 接入训练或语义模型。

## 交付

- 浏览页：`outputs/quality/rgb138/run-r1/browser/index.html`
- 四列固定九图：`outputs/quality/rgb138/run-r1/browser/panels/`
- 同一 144 帧轨迹：`outputs/quality/rgb138/run-r1/browser/trajectory-global-vs-hierarchical.mp4`
- 完整指标、逐图值、哈希、设置与资源：`outputs/quality/rgb138/run-r1/result.json`

首次向 pristine checkout 应用手写补丁时因 patch 格式损坏而失败，零科学运行；随后用独立 `source-r1` 修正并验证补丁可由 `git apply --check` 重放。原 RGB130、全部外部源码和失败痕迹均保留，无上传、删除、Git push 或语义训练。
