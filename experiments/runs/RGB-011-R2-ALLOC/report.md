# RGB-011-R2-ALLOC — Standard 3DGS 400 RGB allocator-only retry

- 状态：completed
- 父实验：RGB-011-FULL（第8,590/30,000次迭代CUDA OOM，保留为失败记录）
- 配置：`experiments/runs/RGB-011-R2-ALLOC/config.yaml`

该重试保持 RGB-011-FULL 的数据、空间split、官方相机、畸变策略、1320×989输入、训练专用初始化、seed=0和30,000 iteration完全不变。唯一环境变化是设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，它是原始PyTorch OOM信息建议的allocator碎片缓解措施；训练开始前用户关闭CloudCompare和Chrome，使空闲显存从约9.2GiB升至约10.3GiB。该变化不修改模型算法或公平协议，但必须单独记录。

输出、日志、checkpoint和渲染仅写入Git忽略的`$R1_OUTPUT_ROOT/r1/runs/RGB-011-R2-ALLOC`。若仍OOM，不降低分辨率、不改densification或迭代数；将以失败证据结束本机尝试。

## 已完成结果

30,000 iteration、checkpoint、PLY、74个显式验证相机渲染和冻结 PSNR/SSIM/LPIPS 评测均以退出码0完成。

| 指标 | 结果 |
|---|---:|
| PSNR 均值 / 中位数 / 最差10% | 15.04685 / 15.69938 / 10.99914 dB |
| SSIM 均值 / 中位数 / 最差10% | 0.80318 / 0.81578 / 0.68013 |
| LPIPS Alex 均值 / 中位数 / 最差10% | 0.45093 / 0.43144 / 0.65357 |

困难视角140–146：PSNR 14.73202 dB、SSIM 0.78243、LPIPS 0.45559。checkpoint独立重载通过（685,296 Gaussians）；外部官方 `SIMPLE_RADIAL` 相机门通过，输出1320×989 RGB，使用 `CAMERA_FROM_WORLD`、identity Sim(3)，且未读取参考RGB或标签。

资源摘要：wall 2,030.81 s（0.5641 GPU h）、重启后的采样峰值显存10,865 MiB、checkpoint 493,422,806 B、PLY 169,954,939 B。首段采样器意外只写出首个样本，峰值仅作采样下界；训练成功本身证明关闭桌面应用和 `expandable_segments:True` 后该12GB主机可完成该固定协议。

## 五类语义冒烟

per-Gaussian五类logits sidecar 在外部 `140.png` 相机上输出 `[5,989,1320]` logits和1320×989单通道类别PNG；仅优化sidecar logits。semantic checkpoint 保存/重新加载通过，RGB逐像素最大绝对差为0，冻结PLY哈希完全不变。证据为Git忽略的 `$R1_OUTPUT_ROOT/r1/semantic_smoke/RGB-011-R2-ALLOC/evidence.json`；该门不是语义mIoU结果。

## 结论

Standard-400成功完成了300-vs-400对照，但只相对Standard-300提升PSNR 0.62731、SSIM 0.01153、LPIPS降低0.03359；仍显著落后于同条件 Scaffold-400。其通过最终前二名语义冒烟门，但不是冻结推荐。
