# RGB-030-FULL — Scaffold-GS 400 RGB R1 full run

- 状态：completed
- 阶段：R1；父实验：RGB-030-R1；Git commit：working tree uncommitted
- 配置：`experiments/runs/RGB-030-FULL/config.yaml`

## 冻结条件

仅把可保存和重新加载 checkpoint 的 RGB-030-R1 500-iteration retry 提升到30,000 iteration。使用`rgb_r1_spatial_v1`、303 train / 74 validation、400-RGB共同相机/畸变/初始化/分辨率、实际固定训练 seed=0 和固定本地评测。`r1_scaffold_checkpoint_v1`及延迟上游 VGG 指标补丁是唯一额外工程变量；它们的实际diff和hash在外部 provenance 中保存。不使用伪标签。

命令：`scripts/run_r1_rgb_candidate.sh RGB-030-FULL scaffold-gs "$R1_OUTPUT_ROOT/rgb_r1_spatial_v1/scene_400_eval" 30000`。输出只写到 Git 忽略的`$R1_OUTPUT_ROOT/r1/runs/RGB-030-FULL`。

## 待产生证据

完成后才登记 checkpoint 加载、74相机渲染、PSNR/SSIM/LPIPS、最差10%、140–146、训练/渲染时间、GPU小时、显存、模型大小和外部相机门。即使 RGB领先，Scaffold 仍必须通过完整指标后的语义烟雾测试才有冻结资格。

## 已完成结果

包装进程以退出码0完成30,000 iteration、checkpoint、PLY、74个显式验证相机渲染及冻结 PSNR/SSIM/LPIPS 评测。

| 指标 | 结果 |
|---|---:|
| PSNR 均值 / 中位数 / 最差10% | 20.59640 / 21.06947 / 17.73400 dB |
| SSIM 均值 / 中位数 / 最差10% | 0.85018 / 0.86680 / 0.75332 |
| LPIPS Alex 均值 / 中位数 / 最差10% | 0.32237 / 0.31705 / 0.44557 |

困难视角140–146：PSNR 21.33872 dB、SSIM 0.84849、LPIPS 0.31703。独立 checkpoint 重载通过（189,073 anchors）。外部官方 `SIMPLE_RADIAL` 相机门通过：输出1320×989 RGB、使用 `CAMERA_FROM_WORLD`、identity Sim(3)、不读取参考RGB/标签；外部appearance使用固定已知训练embedding index 0（303个训练embedding），此限制已记录为提交风险。

资源摘要：wall 1,893.20 s（0.5259 GPU h）、采样峰值显存9,755 MiB、checkpoint 165,864,386 B、PLY 59,749,301 B。

## 五类语义冒烟

冻结RGB后，per-anchor五类logits sidecar 在外部 `140.png` 相机上输出 `[5,989,1320]` logits和1320×989单通道类别PNG；仅优化sidecar logits。semantic checkpoint 保存并重新加载通过，RGB逐像素最大绝对差为0，冻结PLY哈希完全不变。证据在Git忽略的 `$R1_OUTPUT_ROOT/r1/semantic_smoke/RGB-030-FULL/evidence.json`。这是接口与参数隔离测试，不是语义mIoU结果。

## 结论

Scaffold-GS 在400-RGB共同协议中对 Mip-Splatting 形成PSNR、SSIM和LPIPS的Pareto改善，困难视角也显著更好，并通过外部相机、checkpoint与语义冒烟门；推荐作为 provisional RGB冻结候选，待更大显存机器上的 Standard-400 对照完成后转为最终冻结。
