# RGB-010 — Standard 3DGS 300 RGB R1 screening

- 状态：completed_partial_metrics（500 iteration screening；LPIPS待补）
- Git commit：working tree uncommitted
- Split：`rgb_r1_spatial_v1`，219 train / 74 explicit spatial validation
- Seed：0；GPU：RTX 5070（12,227 MiB）

## 已验证执行

固定全帧 1320×989 PINHOLE 输入、训练专用 DLT 初始化和 `sparse/0/test.txt` 后，Standard 3DGS 完成 500 iter，保存 `chkpnt500.pth` 与 `point_cloud/iteration_500/point_cloud.ply`，并从全部 74 个指定验证相机原生渲染。渲染序列的顺序映射已保存为 Git 忽略的 `r1_render_mapping.json`。

命令以可移植环境变量记录在 `config.yaml`；实际日志、checkpoint、PLY、映射和图像均在 `$R1_OUTPUT_ROOT/r1/runs/RGB-010`，不进入 Git。

## 固定本地指标（暂不用于排名）

| 指标 | 全部74视角 | 困难视角140–146 |
|---|---:|---:|
| PSNR (dB) | 17.24618 | 17.39007 |
| SSIM | 0.80232 | 0.78808 |
| LPIPS Alex | deferred: frozen weight uncached | deferred |

- PSNR 最差10%均值：14.46844 dB（`159,158,145,144,146,147,157,071`）。
- SSIM 最差10%均值：0.70641（`160,161,159,158,157,148,149,147`）。
- 观测到的训练期 GPU 内存样本峰值：9,442 MiB；输出目录约166 MiB。

## 异常与限制

训练、checkpoint、验证相机渲染和 PSNR/SSIM 均成功。`evaluate_rgb.py` 的冻结 LPIPS AlexNet 权重从官方源下载极慢，完成渲染后被主动停止以免占用 GPU；因此本条是明确标注的非排名 partial result，不能用于 RGB-060。

## 下一步

在缓存权重完成后对现有同名 74 张渲染补算 LPIPS，无需重训；随后与 RGB-011 的完整 30k 对照一起判断100张无标注 RGB 的价值。
