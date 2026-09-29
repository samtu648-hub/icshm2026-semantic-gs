# RGB-020-FULL — Mip-Splatting 400 RGB R1 full run

- 状态：completed
- 阶段：R1；父实验：RGB-020；Git commit：working tree uncommitted
- 配置：`experiments/runs/RGB-020-FULL/config.yaml`

## 冻结条件

仅把成功的 RGB-020 500-iteration retry 提升到30,000 iteration。使用`rgb_r1_spatial_v1`、303 train / 74 validation、400-RGB主比较的共同相机/畸变/初始化/分辨率、实际固定训练 seed=0 和固定本地评测。唯一候选参数是已记录的 Mip kernel size=0.1；不使用伪标签。

命令：`scripts/run_r1_rgb_candidate.sh RGB-020-FULL mip-splatting "$R1_OUTPUT_ROOT/rgb_r1_spatial_v1/scene_400_eval" 30000`。输出只写到 Git 忽略的`$R1_OUTPUT_ROOT/r1/runs/RGB-020-FULL`。

## 待产生证据

完成后才登记 checkpoint 加载、74相机渲染、PSNR/SSIM/LPIPS、最差10%、140–146、训练/渲染时间、GPU小时、显存、模型大小和外部相机门。缺失LPIPS时状态只能为 `completed_partial_metrics`，不得排名。

## 已完成结果

30,000 iteration、checkpoint、PLY、74个固定验证相机渲染以及固定评测已完成（包装进程退出码 0）。LPIPS 在官方 AlexNet 权重完成缓存后，对同一组74张已保存的预测重算；未重训也未重新渲染。

| 指标 | 结果 |
|---|---:|
| PSNR 均值 | 14.22588 dB |
| PSNR 中位数 | 14.08830 dB |
| PSNR 最差10%均值 | 10.83529 dB |
| SSIM 均值 | 0.78709 |
| SSIM 中位数 | 0.80059 |
| SSIM 最差10%均值 | 0.67888 |
| LPIPS Alex 均值 | 0.50359 |
| LPIPS Alex 最差10%均值 | 0.64359 |

困难视角140–146：PSNR 12.22683 dB、SSIM 0.74641、LPIPS 0.55130。独立 checkpoint 重载通过（914,459 Gaussians）；外部官方 `SIMPLE_RADIAL` 相机门通过，输出1320×989 RGB、使用 `CAMERA_FROM_WORLD`、identity Sim(3)，且不读取参考RGB/标签。

资源摘要：wall 2,003.08 s（0.5564 GPU h）、采样峰值显存9,998 MiB（采样在训练启动后开始，非硬上限）、checkpoint 658,419,670 B、PLY 230,445,224 B。

## 结论

在400-RGB固定协议中，Mip-Splatting 相比 Scaffold-GS 的所有 RGB 指标和困难视角均明显落后；五类语义sidecar冒烟通过，但不进入冻结推荐。
