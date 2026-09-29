# SEM-070-POST-CLEAN — frozen SEM-001 conservative anchor post-process

- 状态：**正式 full 已完成，候选按门登记为 show-only，最终回退 SEM-001**。冻结数据、RGB checkpoint、SEM-001 checkpoint、Python 3.12/CUDA 依赖和 CUDA 12.8 toolkit 均已核验。
- 父实验：`SEM-001`；最终回退：`RGB-030-FULL Scaffold-GS + SEM-001`。
- 配置：`experiments/runs/SEM-070-POST-CLEAN/config.yaml`
- 实现入口：`scripts/run_sem070_post_clean.py`、`scripts/render_sem070_external_camera.py`

## 目标和冻结边界

SEM-070 只增加推理层的一个变量：对跨训练视角稳定、但训练侧 dominant anchor 误报率高的 `stay_cable` anchor，把五类概率中的 `stay_cable` 质量转移到 `background`。Scaffold 的 RGB geometry、颜色、opacity、appearance、checkpoint 和 SEM-001 logits 都不训练、不写回、不覆盖。

SEM-001 原始输出始终保留。正式评测会在同一套冻结模型上产生 raw 与 cleaned 两套逐视角结果；可视化只复用通过冻结门的同一个 `postprocess_mask.npz`，没有更激进的展示版规则。

## 预注册选择协议

先只读取现有 219 张带标签语义训练视角，按 `sha256(image_name)[0]` 的 parity 固定分为 calibration half 和 check half。每张图使用 pinned rasterizer 的精确 dominant Gaussian，再映射回冻结 Scaffold anchor，累计支持视角数、dominant 像素数、GT background/stay_cable 支持、background→stay_cable 误报和真实 cable 命中。

候选必须满足 SEM-001 anchor softmax 的 argmax 为 `stay_cable`、calibration 支持至少 3 个视角、至少 128 个 dominant 像素；候选阈值固定为 `0.50/0.60/0.70/0.80/0.90` 的 calibration background→stay_cable 误报率。每个阈值只在不相交的 training check half 上评估，要求真实 `stay_cable` recall 相对 raw 至少保留 98%，deck/tower/foundation 每类 IoU 下降不超过 0.5 pp；在满足者中最大化误报下降，平局选择误删 anchor 少者。

选择完成后先写入只含一个 boolean `[anchor_count]` mask 的 `postprocess_mask.npz` 和 `mask_freeze.json`，然后才打开固定 74-view。脚本将该顺序写入结果元数据，并拒绝任何包含无标注图、隐藏测试或目标 RGB 的输入路径。

## 运行命令

```bash
export ICSHM_DATA_ROOT="/path/to/Contest Dataset"
export ICSHM_OUTPUT_ROOT="/path/to/ignored/outputs"
"$R1_ENV_ROOT/bin/python" scripts/run_sem070_post_clean.py \
  --config experiments/runs/SEM-070-POST-CLEAN/config.yaml \
  --data-root "$ICSHM_DATA_ROOT" \
  --labels-root "$ICSHM_OUTPUT_ROOT/r2/sem_001/labels" \
  --rgb-run-dir "$ICSHM_OUTPUT_ROOT/r1/runs/RGB-030-FULL" \
  --source-root "$R1_THIRD_PARTY_ROOT/scaffold-gs" \
  --sem001-checkpoint "$ICSHM_OUTPUT_ROOT/r2/sem_001/full/checkpoints/semantic_sidecar.pth" \
  --output-dir "$ICSHM_OUTPUT_ROOT/r2/sem_070/post_clean" \
  --extension-root "$ICSHM_OUTPUT_ROOT/torch_extensions/sem_070" \
  --mode full
```

独立官方相机门使用：

```bash
"$R1_ENV_ROOT/bin/python" scripts/render_sem070_external_camera.py \
  --config experiments/runs/SEM-070-POST-CLEAN/config.yaml \
  --checkpoint "$ICSHM_OUTPUT_ROOT/r2/sem_001/full/checkpoints/semantic_sidecar.pth" \
  --mask "$ICSHM_OUTPUT_ROOT/r2/sem_070/post_clean/postprocess_mask.npz" \
  --rgb-run-dir "$ICSHM_OUTPUT_ROOT/r1/runs/RGB-030-FULL" \
  --source-root "$R1_THIRD_PARTY_ROOT/scaffold-gs" \
  --cameras "$ICSHM_DATA_ROOT/camera_parameters/cameras.txt" \
  --images "$ICSHM_DATA_ROOT/camera_parameters/images.txt" \
  --image-name 140.png \
  --output-dir "$ICSHM_OUTPUT_ROOT/r2/sem_070/external_camera_reaudit"
```

## 当前结果状态

本次执行先完成两项准备、两次环境失败尝试和一次成功 preflight：

- `scripts/prepare_semantic_labels.py` 从释放的 300 个 LabelMe JSON 重建了 SEM-001 所需的 293 个标签 PNG（219 train + 74 validation），逐张与已登记 `label_manifest.json` 的 source/label SHA-256 一致，且 `source_rgb_used=false`。
- 冻结 SEM-001 checkpoint 位于外部输出目录，SHA-256 为 `8e3dbe30e17f67081a72380018fc281c155ec00eed83cd5a19b3285bf4fb8aa3`；冻结 RGB-030-FULL 运行目录和 pinned Scaffold 源码也已找到。
- 第一次 preflight 因调用绝对路径 Python 时未继承 `PATH`/`CUDA_HOME`，分别暴露了 Ninja 和 toolkit 环境变量缺口；补齐后，冻结 Python 3.12.3 / PyTorch 2.11.0+cu128、CUDA 12.8 和 RTX 5070 主机上下文的 8/4 preflight 全链路通过。该 preflight 只作工程烟雾检查，不作质量结论。

正式 full 使用 Git HEAD `49cdcdb1c9915a9544d568754af95ea20a85ec9c` 加当前工作树中的 SEM-070 实现；工作树原有未提交修改均保留。运行耗时 262.59 s，处理完整 219 train / 74 validation split。训练侧五阈值选择结果如下：

| 阈值 | 抑制 anchor | check recall 保留 | 保护类最大 IoU 下降 | check 误报下降 | 是否满足选择门 |
|---:|---:|---:|---:|---:|---|
| 0.50 | 12,165 | 30.33% | 0.067 pp | 90.61% | 否，recall |
| 0.60 | 11,353 | 35.49% | 0.061 pp | 87.03% | 否，recall |
| 0.70 | 10,431 | 44.23% | 0.084 pp | 81.54% | 否，recall |
| 0.80 | 9,359 | 50.48% | 0.077 pp | 77.20% | 否，recall |
| 0.90 | 7,957 | 61.47% | 0.060 pp | 63.01% | 否，recall |

五个阈值都无法达到预注册的 `stay_cable` recall 保留 98%，所以正式 mask 为 189,073 长度的全零 boolean mask（SHA-256 `55889520f3d857374b9985787be93d0eac94115035d58a876eae45c7bf3ef72f`），不删除任何 anchor。

74-view raw/cleaned 结果完全一致：含背景 mIoU `0.616103`，不含背景 mIoU `0.563337`，`stay_cable` IoU `0.068759`，recall `0.368036`，background→`stay_cable` `13,166,331` 像素；最差 10% mIoU `0.423440`，140–146 mIoU `0.601213`。正式门除“background→`stay_cable` 至少下降 1%”外全部通过；该门因全零 mask 为 0% 下降而失败。RGB 最大像素差为 0、RGB artifact hash 相等、checkpoint reload 通过、官方 `CAMERA_FROM_WORLD + SIMPLE_RADIAL` 外部相机门通过。

因此 SEM-070 证明了：仅凭训练侧 dominant-anchor 的跨视角误报率，在保留真实拉索的 98% 约束下没有可安全清除的 anchor。它作为负面/展示证据保留，不替代 `RGB-030-FULL Scaffold-GS + SEM-001`；不在本实验内继续调阈值或读取验证集反向改规则。

若任一正式保留门失败，结果状态必须为 `show_only_candidate_sem001_fallback`，只保留可视化对比，不替代 SEM-001；成功也必须保留 raw SEM-001 作为回退和审计对照。
