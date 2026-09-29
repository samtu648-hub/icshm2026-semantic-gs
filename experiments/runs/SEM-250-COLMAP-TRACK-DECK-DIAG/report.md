# SEM-250-COLMAP-TRACK-DECK-DIAG

状态：`completed_preflight_not_eligible`；不建立候选或运行新的OOF预测。

## 2026-09-07 评估范围勘误（原始产物不变）

代码审计确认：运行逐一读取并汇总了全部219个OOF目标标签，其中包括108个calibration和111个check目标。因此原文及原始result的`check_metrics_opened: false`不实；74-view developer确实未读。同折训练source标签与target/guard分离成立，但不等于calibration/check评估隔离。证据图只在内存中先构造，未形成逐图持久化封存清单，不能称作全部图封存后再评估。

保留原始result及其SHA-256，不重跑或改写历史数字。只产生2个可修改像素的直接稀疏点方案不具备实用覆盖；1/2的precision不能作为大量样本的精度估计，且这个直接点修正上界不能否定使用稀疏点训练稠密模型的潜力。以下历史表述凡与本勘误冲突者，以本节为准。

本实验是下一条跨视角语义研究线的零训练预检，而不是候选模型。它只问一个必要问题：从官方 `images.txt` 的特征轨迹重三角化得到的、独立于冻结Scaffold深度的稀疏对应，是否覆盖足够多的 `SEM-160` 桥面漏检，值得建设可部署的软对应表示。

每个OOF目标相机只接收同折训练相机的轨迹标签证据；目标及guard标签不能进入Beta(1,1)后验。目标标签只在证据图封存后用于训练侧覆盖测量。没有Scaffold渲染、深度、offset贡献、网络、阈值扫描、预测PNG或checkpoint。

历史预注册门：直接点修正的零假阳性oracle覆盖上界至少使deck IoU增加0.2pp，且固定多数证据precision至少94%。原计划不打开111-view check，但执行未遵守该评估边界，见上述勘误。此门只约束直接点修正配方，不能作为所有稠密学习方法的必要条件。

## 结果

冻结清单18/18项哈希一致。61,170条轨迹和三个固定空间OOF折完成读取；fold分别为114/95/10、128/84/7和177/40/2（train/OOF/guard）。每一target的标签在该图source-only证据计算后读取；全部219个target均参与评估，包含111-view check，74-view developer未读。

固定Beta(1,1)后验与Bayes多数在219张目标图只产生2个可由background补为deck的稀疏像素：其中1个真实deck、1个非deck。它只覆盖198,169个SEM-160 deck漏检中的1个（0.0005046%），即使假设该覆盖完全无误，deck IoU上界也仅增加0.00000635pp，远低于0.2pp；实际稀疏证据precision为50%，也低于94%。

运行5.888秒、peak RSS 623.875MiB；没有导入Scaffold模型或渲染器，因此冻结RGB产物重哈希一致且RGB输出按构造未变。没有输出预测PNG、checkpoint或外相机候选。结果JSON SHA256为`1ebffd248a9e9e43ea20326f656e33fa8c775a7f5a4e4cf805895d2698196b73`，外部根为`$ICSHM_OUTPUT_ROOT/r2/sem_250/colmap_track_deck_diag/full-20260907-r1/`。

结论：稀疏COLMAP轨迹既不能覆盖桥面漏检，也不能达到精度下限；禁止在SEM-250中引入splat、膨胀、阈值或第二个对应来源来挽救。`RGB-030-FULL + SEM-161`保持不变。
