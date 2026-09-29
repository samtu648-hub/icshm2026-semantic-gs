# RGB-030-R1 — Scaffold-GS 400 RGB checkpoint-fix screening retry

- 状态：completed_partial_metrics（500 iteration screening；LPIPS待补）
- 父实验：RGB-030
- 唯一改动：修复 pinned upstream 的失效 checkpoint 序列化字段；数据、split、seed、模型超参数与预算保持不变。

RGB-030 在500 iter完成训练、原生 test PSNR 和 PLY/MLP 保存后，因`GaussianModel.capture()`读取不存在的`_local`而无法写入checkpoint。本重试在新的外部输出目录执行；`chkpnt500.pth`、PLY和 MLP 权重均已保存，74 个显式验证相机完成渲染并映射到官方文件名。

## 固定本地指标（暂不用于排名）

| 指标 | 全部74视角 | 困难视角140–146 |
|---|---:|---:|
| PSNR (dB) | 19.42492 | 19.80750 |
| SSIM | 0.81648 | 0.80153 |
| LPIPS Alex | deferred: frozen weight uncached | deferred |

- PSNR 最差10%均值：16.91802 dB（`158,157,161,159,160,175,148,156`）。
- SSIM 最差10%均值：0.72294（`159,161,160,158,157,148,149,150`）。
- 端到端 wall time：101.33 s；host max RSS：4,072,284 KiB；训练期 GPU 样本峰值：9,370 MiB；输出约308 MiB。

在 500 iter 的非排名筛选中，Scaffold 的 PSNR/SSIM 最高；由于 LPIPS 未补、只做了短预算且尚未通过外部相机和语义门，它没有获得冻结资格。
