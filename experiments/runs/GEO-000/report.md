# GEO-000 — Frozen Scaffold same-budget baseline

状态：`completed`

This is the 3,000-iteration control for all GeometryPrior experiments. It uses
the already prepared official-coordinate `scene_400_eval` and the same spatial
validation split as RGB-030-FULL. It must not overwrite the frozen 30,000-step
Scaffold checkpoint.

主机真实运行：2026-08-03 11:36:07–11:39:31，退出码为0。实际命令为：

```bash
export ICSHM_DATA_ROOT="<contest-dataset-root>"
export R1_OUTPUT_ROOT="<geometry-prior-output-root>"
export R1_ENV_ROOT="<r1-cuda128-environment-root>"
export R1_CUDA_HOME="<cuda-toolkit-root>"
export R1_THIRD_PARTY_ROOT="<scaffold-third-party-root>"
export R1_SKIP_LPIPS=1
bash scripts/run_r1_rgb_candidate.sh GEO-000 scaffold-gs \
  "$R1_OUTPUT_ROOT/../rgb_r1_spatial_v1/scene_400_eval" 3000
```

74个空间验证视角的固定评测结果：

| 指标 | 均值 | 中位数 | 最差10% |
|---|---:|---:|---:|
| PSNR | 20.06078 dB | 20.36941 dB | 16.99415 dB |
| SSIM | 0.83855 | 0.85130 | 0.74415 |
| LPIPS-Alex | 0.37803 | 0.36249 | 0.51891 |

证据位于由 R1_OUTPUT_ROOT 指向的 Git 忽略目录
$R1_OUTPUT_ROOT/r1/runs/GEO-000/：

- checkpoint：`chkpnt3000.pth`，SHA-256
  `96c98d0b6518611b7b79ca0d3e36be866708c4690cde04768d75190dd170d3e0`；
- PLY：`point_cloud/iteration_3000/point_cloud.ply`，SHA-256
  `89c5b51ccc0502146a388d8136040516c96c5c311ffe5b43ff8d38c46565b85a`；
- checkpoint 独立重载通过，iteration=3000，Scaffold anchor 数=94,059；
- RGB-030-FULL 的冻结30k checkpoint未被修改；
- 本次训练 wall time 204.2 s。该运行没有采样 GPU 峰值文件，因此峰值显存记为 `not recorded`，不猜测。

GEO-000 仅是后续 GeometryPrior 3k挑战的同预算控制，不与 RGB-030-FULL 的30k指标混写。

## GeometryPrior v2 续训审计（2026-08-03）

结论：`approximate_resume`，不得把本 checkpoint 作为严格连续 0→10k 的证据；v2 建立新的 `GEO-000-10K` 从初始化连续训练。

只读审计证据：

- checkpoint envelope 是 `(model_params, iteration)`；iteration=3000、format=`r1_scaffold_checkpoint_v1`，包含 anchor/offset/MLP/appearance 和 optimizer state；
- checkpoint 不包含 scheduler state 或 Python/NumPy/PyTorch/CUDA RNG state；训练循环用 Python `randint` 选择相机，重新启动会重置随机序列，故不能严格复现原连续 3001–10000 相机顺序；
- 学习率由 iteration 和固定训练参数重建；因此它可以技术性恢复模型和 optimizer，但随机状态边界仍使其仅为 approximate resume；
- run backup 与当前冻结 Scaffold checkout 的 train.py、dataset_readers.py、gaussian_model.py、两个 CUDA 补丁文件和 general_utils.py 的字节比较一致；上游 HEAD 仍为 `59c833b56bbbf510f3f64d40f81721995caced66`；
- input PLY SHA-256 为 `10a88fb018c27530b3bbfa0f7000408693ba75228067f5311e7775e4755e68f2`；checkpoint/iteration-3000 PLY hash 保持本报告上文记录；
- 只有 3000 checkpoint、3000 日志和 3000 固定评测；不存在 1000 checkpoint 或 1000 指标，故没有可审计的 1k 连续节点。

本审计未执行恢复、续训或重跑。
