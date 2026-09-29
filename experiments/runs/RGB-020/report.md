# RGB-020 — Mip-Splatting 400 RGB R1 screening

- 状态：completed_partial_metrics（500 iteration screening；LPIPS待补）
- Git commit：working tree uncommitted
- Split：`rgb_r1_spatial_v1`，303 train / 74 explicit spatial validation
- Seed：0；GPU：RTX 5070（12,227 MiB）

## 已验证执行

Mip-Splatting 在同一 PINHOLE scene、训练专用初始化和明确的 `test.txt` 验证相机列表下完成500 iter；checkpoint、点云、74张渲染、名称映射和固定 PSNR/SSIM 都已写入 Git 忽略输出目录。

## 固定本地指标（暂不用于排名）

| 指标 | 全部74视角 | 困难视角140–146 |
|---|---:|---:|
| PSNR (dB) | 17.72945 | 17.07852 |
| SSIM | 0.80280 | 0.78064 |
| LPIPS Alex | deferred: frozen weight uncached | deferred |

- PSNR 最差10%均值：14.85657 dB（`144,145,143,146,148,176,147,149`）。
- SSIM 最差10%均值：0.70733（`160,159,161,158,157,148,149,147`）。
- 实测端到端 wall time：94.33 s；host max RSS：3,727,512 KiB；训练期 GPU 样本峰值：9,490 MiB；输出目录约181 MiB。

## 失败与重试

第一次启动在训练前退出（exit 2）：本仓库封装器错误传入了该 pinned 版本不支持的 `--disable_viewer`。移除参数后，以相同实验ID、配置、数据和seed重试并成功；失败日志保存在 Git 忽略的 `RGB-020-screening.log`，成功日志为 `RGB-020-screening-retry.log`。

## 解释边界

在500 iter的非排名 screen 中，Mip 的 PSNR 高于两个 Standard 对照，但 LPIPS缺失、预算极短且仍未完成外部相机/语义门，因此不能作为冻结结论。
