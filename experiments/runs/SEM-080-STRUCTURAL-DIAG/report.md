# SEM-080-STRUCTURAL-DIAG：训练侧桥梁拓扑语义清理可行性诊断

## 状态

已完成训练侧全量诊断；不进入74-view正式候选。经样例复核，规则还删除了正常主缆、吊索和部分斜拉索，故不保留为展示版；`SEM-001` 仍是唯一提交回退。

## 问题与假设

`SEM-070-POST-CLEAN` 证明，整颗 3D anchor 的 `stay_cable→background`
转移无法保留真实索线：同一 anchor 会同时贡献真实索线和天空/江水误报。本诊断只改变
后处理的作用域：在每个已渲染视角中，原始 `stay_cable` 像素必须同时满足“连续、近竖直、
一端接预测 tower、另一端接预测 deck”，否则将其概率质量转回 background。

这是一项**训练侧可行性诊断**，不是候选替代评测。它不打开74个验证视角，不产生
checkpoint、anchor mask 或外部相机结果，也不选择规则。训练标签仅在无标签的拓扑谓词
完成后用于报告官方指标；谓词本身只读取冻结 SEM-001 的五通道概率。

## 已冻结规则

- Hough threshold：32；最短线段：48 px；最大断裂：10 px；
- 竖直度 `|dy| / length` 至少 0.50；
- tower/deck 端点接触半径：40 px；保留线的半宽：3 px；
- 被拒绝的 raw `stay_cable` 像素执行
  `p_background += p_stay_cable; p_stay_cable = 0`；其余类别不变。

这套规则在本诊断内不调参。输出分别报告训练 calibration/check hash half 的 raw/cleaned
官方指标、有效 tower-to-deck 线数、保留像素数和 RGB=0 证据。它错误地假设所有正常索线
都直接连接 tower 与 deck：主缆可连接 tower↔tower/锚点，吊索可连接 deck↔主缆。因此它既
不允许进入74-view，也不允许作为展示版；提交回退仍为 `RGB-030-FULL Scaffold-GS + SEM-001`。

## 已完成的训练侧结果

正式输出：`external_gitignored:ICSHM_OUTPUT_ROOT/r2/sem_080/structural_diag/full-20260902-r4`。
冻结 Python 3.12.3 / PyTorch 2.11.0+cu128 / CUDA 12.8 在 RTX 5070 上用时 97.20 s；SEM-001
checkpoint SHA-256 为 `8e3dbe30e17f67081a72380018fc281c155ec00eed83cd5a19b3285bf4fb8aa3`，重载逐位一致。
访问日志确认只处理219张训练标签视角，74张验证、100张无标注和隐藏输入均为未读取。

谓词检测到12,351条满足 tower-to-deck 端点条件的 Hough 线；从52,588,100个 raw
`stay_cable` 像素中保留4,597,766个（8.743%），其余47,990,334个转回background。这说明
规则确实能抽取出一组桥梁连接线，而不是全零或失效实现。

但独立训练 check half 的官方标注指标明确否定它作为替代方案：

- background→`stay_cable` 从18,363,796降至1,019,147（−94.45%）；
- `stay_cable` precision 从38.48%升至57.90%，但 recall 从84.90%降至10.60%，仅保留12.49%；
- `stay_cable` IoU 从0.36019降至0.09842，含背景 mIoU 从0.67833降至0.63807（−4.03 pp）。

deck、tower、foundation 的 IoU 完全不变；RGB artifact hash、三个训练相机RGB PNG hash均相等，
逐像素最大差为0。样例复核进一步发现，`training_structural_triptych/001.png` 的红色被删区域中
包含正常主缆、吊索和部分斜拉索，不只是江水/天空的黄色包络。根因是 tower→deck 直连规则
不覆盖主缆的 tower↔tower/锚点连接及吊索的 deck↔主缆连接。因此该规则已经在视觉和训练侧
官方指标两方面失败：不打开74-view、不生成外部相机或视频、不再调 Hough/接触半径参数，也
不保留为物理结构展示。
