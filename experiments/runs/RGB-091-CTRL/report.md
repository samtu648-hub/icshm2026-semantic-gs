# RGB-091-CTRL — Matched unweighted control

- 状态：completed_matched_control
- 父实验：RGB-030-FULL
- 配置：本目录 config.yaml；seed 0；split rgb_r1_spatial_v1。
- 2026-09-06 预注册并完成唯一30k；不覆盖冻结 RGB-030-FULL 或 SEM-161。
- 协议 ADR-048；最终74-view平均 PSNR 20.674209 dB、SSIM 0.851700781、LPIPS Alex 0.319847499；worst-10% PSNR 18.203331 dB。
- 严格checkpoint重载通过，195,876 anchors，完整执行（含重载与评估）1,751.724秒。checkpoint SHA256 `c4e20f52ffb1958d84f68d3d75948cae02937d907129d2fa912fce878bb5bcea`，PLY SHA256 `ecb47c1279bc821b4360eaefaca9ddfb22fec9ea1db5b740fcae1b9fdc4690c9`。
- 初始点云和相机列表分别与RGB-030及候选完全同哈希：`10a88fb018c27530b3bbfa0f7000408693ba75228067f5311e7775e4755e68f2`、`ff3afdf471a8393c17ee1150e405a1bfcf0c4aab65a3b156a865b39058dc7065`。cfg_args除输出路径外相同。
- 本次无降权重训比历史PSNR高0.07781 dB，不能把这部分变化归给瞬态机制，也不主张单seed重训逐位可复现。RGB-092必须相对本次匹配控制通过主门，并满足相对历史的保留门。
- 原始RGB-030全部保存产物哈希不变。运行记录保存在外部输出`r1/runs/RGB-091-CTRL/execution.json`。候选最终相对本控制PSNR −0.34963 dB、LPIPS +0.001509，未过门；见RGB-092报告。
- 本控制的140–146 PSNR为21.219144 dB，较历史下降0.119580 dB；不因均值略高而自动改用此控制提交，继续保留RGB-030-FULL + SEM-161。
- Git HEAD `49cdcdb1c9915a9544d568754af95ea20a85ec9c`加保留工作树；配置SHA256 `cdceece7e98de0d43bea5b45ed1a151cc1e1c1e9bb6f54fa7e40cb66336867c8`，训练模块SHA256 `9c9a021a04bd36df4a43fcb91849ea2a24094e39d2c588f0069425ba8e906614`；数据manifest/split哈希和唯一变量检查见RGB-092报告。30秒采样设备显存峰值7716 MiB。
