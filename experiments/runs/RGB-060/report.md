# RGB-060 — RGB Backbone 冻结决定

- 状态：completed
- 决定：冻结 Scaffold-GS（RGB-030-FULL）作为 R2 RGB 骨干。

## 同条件400-RGB证据

| 候选 | PSNR | SSIM | LPIPS Alex | 140–146 PSNR / SSIM / LPIPS | 工程门 |
|---|---:|---:|---:|---:|---|
| Standard 3DGS | 15.04685 | 0.80318 | 0.45093 | 14.73202 / 0.78243 / 0.45559 | checkpoint、外部相机、五类语义smoke通过 |
| Mip-Splatting | 14.22588 | 0.78709 | 0.50359 | 12.22683 / 0.74641 / 0.55130 | checkpoint、外部相机、五类语义smoke通过 |
| Scaffold-GS | 20.59640 | 0.85018 | 0.32237 | 21.33872 / 0.84849 / 0.31703 | checkpoint、外部相机、五类语义smoke通过 |

Scaffold 对 Standard 和 Mip 在全部三个RGB指标、最差10%和140–146困难视角均构成明显Pareto改善，远超过协议中的工程门槛；其训练时间也更短（1,893.20 s 对Standard的2,030.81 s），checkpoint与PLY更小。外部相机门输出1320×989，使用官方 `CAMERA_FROM_WORLD`、identity Sim(3)，且不读取参考RGB或标签。其已知风险是外部相机使用确定性训练appearance embedding index 0；这需要在最终提交接口中保留明确配置。

## 未完成控制与限制

Standard-300完成，指标为PSNR 14.41954、SSIM 0.79165、LPIPS 0.48452。原始 Standard-400 于8,590/30,000因桌面显存占用与allocator碎片OOM；关闭CloudCompare/Chrome并启用上游错误建议的 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后，RGB-011-R2-ALLOC 在其余冻结协议完全不变的条件下成功完成。400 RGB相对300 RGB使Standard的PSNR提高0.62731、SSIM提高0.01153、LPIPS降低0.03359，证明额外100张无标注RGB在该空间折中有正向RGB价值。

三条主候选均已得到同协议400-RGB可比较结果，最终前二名Standard与Scaffold也通过语义冒烟门。因此冻结Scaffold-GS，并允许后续R2仅在其冻结RGB几何、颜色和opacity的前提下开展五类语义基线；不得将RGB-050/RGB-050-R2的sidecar smoke误报为语义准确率。
