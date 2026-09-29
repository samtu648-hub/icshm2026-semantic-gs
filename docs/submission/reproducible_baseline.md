# 当前可复现提交回退

当前回退由 `RGB-030-FULL Scaffold-GS + SEM-001` 组成。它接受外部 COLMAP `cameras.txt` 与 `images.txt`，按 `CAMERA_FROM_WORLD` 解释位姿，并输出同尺寸 RGB 与单通道五类语义 PNG；可选保存五通道 logits。Class ID 2 的唯一名称是 `stay_cable`。

这套入口已从搬运后的模型和源码副本真实渲染三个相机，但主办方尚未公布最终命令、命名和目录协议，因此仍是协议适配前的可复现回退，不是最终官方格式。

## 创建外部包

先设置本机 Git 忽略路径：

```bash
export R1_OUTPUT_ROOT="${ICSHM_OUTPUT_ROOT}/icshm2026"
export R1_THIRD_PARTY_ROOT="${R1_OUTPUT_ROOT}/r1/third_party"
export SUBMISSION_BUNDLE="${R1_OUTPUT_ROOT}/submission/sem001-baseline-v1"

python3 scripts/package_submission_baseline.py \
  --rgb-run-dir "${R1_OUTPUT_ROOT}/r1/runs/RGB-030-FULL" \
  --semantic-checkpoint "${R1_OUTPUT_ROOT}/r2/sem_001/full/checkpoints/semantic_sidecar.pth" \
  --source-root "${R1_THIRD_PARTY_ROOT}/scaffold-gs" \
  --output-dir "${SUBMISSION_BUNDLE}"
```

打包脚本拒绝覆盖非空目录，并验证固定 artifact、上游 commit 和 patch。包本体含 checkpoint、PLY 和第三方源码，不得提交 Git。

## 渲染外部相机

重复 `--image-name` 可选择若干 pose；省略时渲染 `images.txt` 中全部 pose。渲染入口没有参考 RGB 或标签参数。

```bash
"${R1_ENV_ROOT}/bin/python" "${SUBMISSION_BUNDLE}/code/scripts/render_submission.py" \
  --bundle "${SUBMISSION_BUNDLE}" \
  --cameras /path/to/official/cameras.txt \
  --images /path/to/official/images.txt \
  --output-dir /path/to/submission_outputs \
  --write-logits
```

输出目录为：

```text
submission_outputs/
  rgb/<pose-stem>.png
  semantic/<pose-stem>.png
  logits/<pose-stem>.npy       # 仅在 --write-logits 时生成
  submission_manifest.json
```

## 离线审计

```bash
"${R1_ENV_ROOT}/bin/python" "${SUBMISSION_BUNDLE}/code/scripts/audit_submission_bundle.py" \
  --bundle "${SUBMISSION_BUNDLE}" \
  --renders /path/to/submission_outputs
```

审计覆盖包内逐文件 hash、七个冻结模型 artifact、RGB/semantic/logits hash、尺寸、dtype、类别范围和“未读取参考 RGB/标签”契约。实际运行证据见 `experiments/runs/SUBMIT-010-BUNDLE/report.md`。

## 已知限制

- 外部相机 appearance 使用固定训练 embedding index 0，不读取测试 RGB；
- 当前源码快照保留 Scaffold-GS research-only license，分发时必须保留许可证并遵守其条款；
- 还未在全新 Ubuntu 环境重建 CUDA 扩展；该项属于 `SUBMIT-002`；
- 主办方最终协议发布后，需要在不改变模型结果的前提下适配最终文件名、批量相机格式和目录结构。
