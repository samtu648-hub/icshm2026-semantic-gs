# VIS-003-POST-CLEAN — SEM-001 raw/cleaned semantic flythrough

- 状态：**已完成，作为 SEM-070 show-only 可视化证据保留**。
- 父实验：`SEM-070-POST-CLEAN`、`VIS-001-FLYTHROUGH`。
- 配置：`experiments/runs/VIS-003-POST-CLEAN/config.yaml`
- 入口：`scripts/render_vis003_post_clean.py`

该可视化严格复用 VIS-001 的 144 个 `CAMERA_FROM_WORLD`、官方参考坐标、`SIMPLE_RADIAL` pose。每一帧只使用相机元数据和冻结模型，输出 `SEM-001 raw` 与 `SEM-070 POST-CLEAN` 两个并排语义面板。语义清洗规则直接读取正式候选的 `postprocess_mask.npz`，不额外增加展示版阈值或形态学处理；RGB、标签和隐藏测试输入均不读取。

运行命令：

```bash
"$R1_ENV_ROOT/bin/python" scripts/render_vis003_post_clean.py \
  --config experiments/runs/VIS-003-POST-CLEAN/config.yaml \
  --sem070-config experiments/runs/SEM-070-POST-CLEAN/config.yaml \
  --checkpoint "$ICSHM_OUTPUT_ROOT/r2/sem_001/full/checkpoints/semantic_sidecar.pth" \
  --mask "$ICSHM_OUTPUT_ROOT/r2/sem_070/post_clean/postprocess_mask.npz" \
  --rgb-run-dir "$ICSHM_OUTPUT_ROOT/r1/runs/RGB-030-FULL" \
  --source-root "$R1_THIRD_PARTY_ROOT/scaffold-gs" \
  --trajectory-dir "$ICSHM_OUTPUT_ROOT/results/frozen_scaffold_sem001/flythrough_v1/cameras" \
  --output-dir "$ICSHM_OUTPUT_ROOT/results/frozen_scaffold_sem070_post_clean/flythrough_v1"
```

正式运行已生成 144 帧、24 fps、6.0 秒、2640×1032 的 H.264 yuv420p 视频，渲染耗时 75.91 s。轨迹 manifest SHA-256 为 `f5b9f43c93dcfa2a57238aac001571987fc60a04ca0a2a29acc7bf4b7fdde59e`，复用的 SEM-070 mask SHA-256 为 `55889520f3d857374b9985787be93d0eac94115035d58a876eae45c7bf3ef72f`，视频 SHA-256 为 `2e6000bd27efeb16e4b382656f2c1a99af704e53218bd2de2f7bbf195f44b8ef`。144 对 raw/cleaned 语义 PNG 逐帧 hash 均一致，符合全零 mask 的预期。

该视频只作为“raw vs show-only cleaned”审计证据，不改变最终回退：`RGB-030-FULL Scaffold-GS + SEM-001`。
