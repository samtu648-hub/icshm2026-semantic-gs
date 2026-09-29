# RGB-073-RESIDUAL-SUPPORT-PRUNE-PF — 训练低支持高残差anchor保守屏蔽

- 状态：completed_engineering_gate_failed
- 父实验：`RGB-030-FULL`, `RGB-072-FLOATER-SUPPORT-DIAG`
- 唯一变量：推理时屏蔽训练侧固定的29个anchor

## 冻结规则

29个anchor完全由RGB-072的303训练视角统计决定：累计主导≥256像素、至少主导16像素的
视角数≤4、mean RGB MAE≥0.15、RGB MAE≥0.15的主导像素比例≥0.50。统计NPZ哈希和
预期anchor数已锁死；不读取74-view RGB/标签选点，不扫描阈值、scale、opacity或位置。

Checkpoint、PLY、MLP、appearance、相机和背景均冻结；mask只传给原生renderer的
`visible_mask`。先逐像素复现全部74个原始RGB-030-FULL渲染，再生成成对候选。

## 保留门

mean PSNR/SSIM不得下降，LPIPS不得升高且至少一项全局指标严格改善；最差10%和140–146
PSNR最多下降0.02 dB；artifact哈希必须相同。固定样例054/140/151/159/160仅检查是否
真实移除浮点且不伤结构，不用来改mask。

## 结果与决定

首次启动在054的候选渲染写出前被基线复现门拦截；第二次只增加差异记录后重复得到同一
失败。重放与保存的RGB-030-FULL只有25像素不同、最大差1/255、全图mean absolute
uint8 difference为`6.38335e-06`，但预注册要求逐像素完全一致，因此本ID不得放宽门或
继续质量评测。失败证据SHA256为
`2ba91d38d3433948d275698f7f6d2862364c5d1cd56b9ee3f6866f3747d0d85e`。

keep_or_drop：drop_engineering_gate_failed。29-anchor规则未被质量评测，RGB-030-FULL
保持。另立RGB-074显式注册最大1/255、mean uint8差≤1e-4、差异像素比例≤1e-4的严格
重放容差，并保存同次重放基线，不能把这一事后工程容差写回RGB-073。

