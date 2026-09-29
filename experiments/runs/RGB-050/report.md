# RGB-050 — 前两名冻结 RGB 的五类语义冒烟

- 状态：completed
- 候选：RGB-020-FULL（Mip-Splatting）、RGB-030-FULL（Scaffold-GS）
- 配置：`experiments/runs/RGB-050/config.yaml`

两者均从外部官方 `140.png` `CAMERA_FROM_WORLD` 相机输出五类 logits `[5,989,1320]` 与1320×989单通道类别PNG。Mip 使用固定 Gaussian 的 per-Gaussian logits；Scaffold 使用固定 anchor 的 per-anchor logits。语义训练步骤只优化sidecar logits；各自sidecar checkpoint 保存、重新加载通过。RGB参考渲染和冻结PLY的 SHA-256 前后一致，RGB最大绝对像素差为0。

证据位于Git忽略的 `$R1_OUTPUT_ROOT/r1/semantic_smoke/RGB-020-FULL/evidence.json` 与 `RGB-030-FULL/evidence.json`。这是可扩展性、接口和参数隔离冒烟测试；不含标注训练或mIoU结论。
