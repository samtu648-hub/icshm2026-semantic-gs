# RGB-041-PF — Standard 3DGS 400 RGB matched 1k control

- 状态：completed
- 阶段：R1 extension；父实验：RGB-011-R2-ALLOC
- 配置：`experiments/runs/RGB-041-PF/config.yaml`

它只为 `RGB-040-PF` 提供相同 1,000 iteration、相同400 RGB和相同空间验证折的早期对照；不会取代已经完成的30k Standard-400结果，也不会改变 Scaffold-GS 冻结决定。

## 结果

首次真实启动失败时，Handy 桌面进程占用2,308 MiB，Standard 在加载相机时申请16 MiB失败，仅余约30 MiB可用。该失败输出保留在 Git 忽略的 `$R1_OUTPUT_ROOT/r1/runs/RGB-041-PF-attempt1-handy-vram-blocked`，不被覆盖，也不解释为模型质量。

用户关闭 Handy 后，以不变配置的唯一受控重试成功完成：PSNR 18.13317、SSIM 0.81465、LPIPS Alex 0.46033；140–146为19.04362 / 0.80829 / 0.44175。相对 MCMC-1k，它在所有比较指标上均显著更好，因而为 MCMC 的完整训练提供反证。
