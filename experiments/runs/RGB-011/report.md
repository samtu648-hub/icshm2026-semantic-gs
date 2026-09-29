# RGB-011 — Standard 3DGS 400 RGB R1 screening

- 状态：completed_partial_metrics（500 iteration screening；LPIPS待补）
- Git commit：working tree uncommitted
- Split：`rgb_r1_spatial_v1`，303 train / 74 explicit spatial validation
- Seed：0；GPU：RTX 5070（12,227 MiB）

## 已验证执行

在与 RGB-010 相同的相机、全帧去畸变、初始化、seed、分辨率及指标实现下，仅增加84个许可的无标注训练视角。500 iter 完成，checkpoint 和点云可用，74个固定验证视角的原生渲染被安全映射为官方文件名。

## 固定本地指标（暂不用于排名）

| 指标 | 全部74视角 | 困难视角140–146 |
|---|---:|---:|
| PSNR (dB) | 16.62121 | 15.91102 |
| SSIM | 0.79507 | 0.76791 |
| LPIPS Alex | deferred: frozen weight uncached | deferred |

- PSNR 最差10%均值：13.58025 dB（`143,144,145,146,176,148,147,149`）。
- SSIM 最差10%均值：0.70457（`160,161,159,158,157,144,143,145`）。
- 实测端到端 wall time：96.97 s；host max RSS：1,967,376 KiB；输出目录约177 MiB。

## 解释边界

500 iter 下此条件低于 RGB-010 的 PSNR/SSIM；这只说明短预算尚未体现额外视角的价值，不能替代30k公平对照或用来淘汰400-RGB配置。LPIPS尚未计算，因此不能进入候选排名。

## 下一步

补齐 LPIPS 后保留为300-vs-400控制证据；再由完整预算结果决定是否在最终训练中保留所有许可无标注视角。
