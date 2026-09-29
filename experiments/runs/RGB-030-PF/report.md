# RGB-030-PF — Scaffold-GS CUDA 12.8 build preflight

- 状态：completed_with_followup_patch
- 阶段：R1；父实验：RGB-001；Git commit：working tree uncommitted
- 配置：`experiments/runs/RGB-030-PF/config.yaml`

## 已验证执行

固定上游 `59c833b56bbbf510f3f64d40f81721995caced66` 已在 Git 忽略的 R1 Python 3.12 / PyTorch 2.11.0+cu128 / CUDA 12.8 (`sm_120`) 环境中完成 CUDA 扩展编译。CUDA 12.8 头文件、显式验证 split、延迟 VGG LPIPS 初始化/跳过非冻结上游指标均有最小本地补丁；共同输入、相机和分辨率保持不变。

原始 RGB-030 在迭代500完成训练和原生测试渲染，但遗留 `capture()` 访问不存在的 `_local` 字段，无法写出所需 checkpoint。该失败保持独立登记。仅修正 capture/restore 的实际 anchor-model状态后，RGB-030-R1 完成500 iteration、保存并重新加载 checkpoint、渲染74个指定验证相机并计算固定 PSNR/SSIM。

## 结论

Scaffold-GS 已通过构建和修复后的完整最小运行门，可进入 RGB-030-FULL；checkpoint 与外部相机/语义门仍将在完整结果后独立验证。LPIPS 未就绪时不可排名。
