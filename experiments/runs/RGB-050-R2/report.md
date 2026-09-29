# RGB-050-R2 — 最终前二名冻结 RGB 的五类语义冒烟

- 状态：completed
- 候选：RGB-011-R2-ALLOC（Standard 3DGS）、RGB-030-FULL（Scaffold-GS）
- 配置：`experiments/runs/RGB-050-R2/config.yaml`

此前 RGB-050 在Standard-400可比较结果缺失时，对当时前两名Mip与Scaffold完成。Standard-400受控重试完成后，本实验按最终RGB排序补齐Standard与Scaffold的相同五类语义sidecar门；不重新训练RGB。

两者均通过：外部 `140.png` `CAMERA_FROM_WORLD` 相机输出五类logits `[5,989,1320]` 与1320×989单通道类别PNG；仅优化sidecar语义参数；semantic checkpoint保存和重新加载通过；RGB最大绝对像素差为0；冻结PLY哈希不变。Standard使用per-Gaussian、Scaffold使用per-anchor存储。证据仅在Git忽略的 `$R1_OUTPUT_ROOT/r1/semantic_smoke/`；不构成语义mIoU结论。
