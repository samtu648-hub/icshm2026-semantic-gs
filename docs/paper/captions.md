# 图片与视频中英图注

伙伴交接版 · 证据截止 2026-09-23 · 整理 2026-09-29

图片共享图例：background 深灰、deck 蓝、stay_cable 黄、tower 红、foundation 紫。视频使用各自下列图例。

## F01 · 数据与评价角色

中文：300张标签的像素组成及不同证据角色。stay_cable不是最少的前景类；小构件长尾、尺度和标注形态需要单独讨论。

English: Label composition of 300 released views and distinct evaluation roles. Pixel frequency alone does not explain thin-structure difficulty.

证据角色：数据画像，不是模型成绩

样例选择：预定义主题及既有固定样例；未进行新质量筛选

来源：[QUAL-009](../../experiments/runs/QUAL-009-DATASET-PROFILE/report.md) · [SEM-001](../../experiments/runs/SEM-001/report.md) · [QUAL-012](../../experiments/runs/QUAL-012-DISJOINT-SPATIAL-BACKBONE-AUDIT/report.md)

素材：[F01_dataset_and_roles.svg](figures/F01_dataset_and_roles.svg) · [F01_dataset_and_roles.png](figures/F01_dataset_and_roles.png)

原尺寸单图：本组为可编辑SVG图表／方法示意，无原生照片单图。

## F02 · 三条系统角色，避免画成一条混合链

中文：历史专家融合、已有全量展示和未来待完成系统分开绘制。后期SEM437推理不包含早期SEM161专家链。

English: Separate the historical specialist-fusion system, existing full-data demonstration, and pending RGB166 semantic system.

证据角色：方法与完成状态

样例选择：预定义主题及既有固定样例；未进行新质量筛选

来源：[SEM-161](../../experiments/runs/SEM-161-DECK-BINARY-FINAL/report.md) · [SEM-437](../../experiments/runs/SEM-437-RGB170-MASK2FORMER-CONTEXT50-LONG12K-ALL300/report.md) · [QUAL-026](../../experiments/runs/QUAL-026-RGB170-SPATIAL-CONFIRMATION/report.md)

素材：[F02_system_roles.svg](figures/F02_system_roles.svg) · [F02_system_roles.png](figures/F02_system_roles.png)

原尺寸单图：本组为可编辑SVG图表／方法示意，无原生照片单图。

## F03 · 第一次大幅进步：三维基线到类别专家链

中文：历史74图五类mIoU：SEM001 61.610304%→SEM161 83.387017%（+21.776713pp）。054为既有固定样例；同视角图展示背景误报与构件变化。

English: On the same reused 74 development views, SEM001 to SEM161 increases five-class mIoU from 61.610304% to 83.387017%. View 054 is a pre-existing fixed example.

证据角色：历史74图开发；非盲测

样例选择：预定义主题及既有固定样例；未进行新质量筛选

来源：[SEM-001](../../experiments/runs/SEM-001/report.md) · [SEM-111](../../experiments/runs/SEM-111-PROTECTED-FUSION-FINAL/report.md) · [SEM-141](../../experiments/runs/SEM-141-CABLE-BINARY-FINAL/report.md) · [SEM-151](../../experiments/runs/SEM-151-FOUNDATION-GATE-FINAL/report.md) · [SEM-161](../../experiments/runs/SEM-161-DECK-BINARY-FINAL/report.md) · [SEM-329](../../experiments/runs/SEM-329-SECOND-SOURCE-VETO-FINAL/report.md)

素材：[F03_historical_progress.svg](figures/F03_historical_progress.svg) · [F03_historical_progress.png](figures/F03_historical_progress.png) · [F03_054_semantic_evolution.png](figures/F03_054_semantic_evolution.png)

原尺寸单图：[F03_054_reference.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F03_054_reference.png) · [F03_054_rgb.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F03_054_rgb.png) · [F03_054_gt.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F03_054_gt.png) · [F03_054_sem001.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F03_054_sem001.png) · [F03_054_sem111.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F03_054_sem111.png) · [F03_054_sem161.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F03_054_sem161.png)

## F04 · 类别专家的局部作用与代价

中文：160桥面/索区与183基础为既有固定样例；提供相同坐标裁剪及原尺寸单图。类别保护和单向修改有收益，也保留漏检与边界问题。

English: Matched crops from fixed views 160 and 183 illustrate class-specific changes. Protected updates improve selected errors while preserving some omissions and boundary failures.

证据角色：历史74图定性例；同视角、同裁剪

样例选择：预定义主题及既有固定样例；未进行新质量筛选

来源：[SEM-111](../../experiments/runs/SEM-111-PROTECTED-FUSION-FINAL/report.md) · [SEM-141](../../experiments/runs/SEM-141-CABLE-BINARY-FINAL/report.md) · [SEM-151](../../experiments/runs/SEM-151-FOUNDATION-GATE-FINAL/report.md) · [SEM-161](../../experiments/runs/SEM-161-DECK-BINARY-FINAL/report.md)

素材：[F04_160_class_specialists.png](figures/F04_160_class_specialists.png) · [F04_183_class_specialists.png](figures/F04_183_class_specialists.png)

原尺寸单图：[F04_160_reference.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F04_160_reference.png) · [F04_160_gt.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F04_160_gt.png) · [F04_160_sem111.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F04_160_sem111.png) · [F04_160_sem141.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F04_160_sem141.png) · [F04_160_sem151.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F04_160_sem151.png) · [F04_160_sem161.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F04_160_sem161.png) · [F04_183_reference.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F04_183_reference.png) · [F04_183_gt.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F04_183_gt.png) · [F04_183_sem111.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F04_183_sem111.png) · [F04_183_sem141.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F04_183_sem141.png) · [F04_183_sem151.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F04_183_sem151.png) · [F04_183_sem161.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F04_183_sem161.png)

## F05 · 强二维基线：总体提高，057仍退化

中文：同74固定评价中，Mask2Former raw为86.513086%，旧SEM161为83.387017%；68图提高、6图下降。B2一栏是融合版80.917724%，不是raw 75.324650%。RGB底座和训练预算不完全相同。

English: Fixed historical74 comparison: Mask2Former raw reaches 86.513086% versus 83.387017% for SEM161, with 68 improved and 6 regressed views. B2 denotes the fused system. Different RGB backbones and training budgets prevent a pure capacity attribution.

证据角色：历史74图开发；完整方法包比较

样例选择：159为既有固定困难图；057为SEM350登记的最大raw退化图（评分后诊断）

来源：[SEM-349](../../experiments/runs/SEM-349-MASK2FORMER-FINAL/report.md) · [SEM-350](../../experiments/runs/SEM-350-FIXED-FINAL-COMPARISON/report.md)

素材：[F05_mask2former_comparison.svg](figures/F05_mask2former_comparison.svg) · [F05_mask2former_comparison.png](figures/F05_mask2former_comparison.png) · [F05_159_m2f.png](figures/F05_159_m2f.png) · [F05_057_m2f.png](figures/F05_057_m2f.png)

原尺寸单图：[F05_159_Photo.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F05_159_Photo.png) · [F05_159_RGB110.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F05_159_RGB110.png) · [F05_159_GT.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F05_159_GT.png) · [F05_159_M2F_raw.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F05_159_M2F_raw.png) · [F05_057_Photo.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F05_057_Photo.png) · [F05_057_RGB110.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F05_057_RGB110.png) · [F05_057_GT.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F05_057_GT.png) · [F05_057_M2F_raw.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F05_057_M2F_raw.png)

## F06 · RGB骨干比较必须交代外观条件

中文：原始全量三骨干部署口径比较。Scaffold采用固定UID0，训练UID原生分数另有记录；随后RGB136无外观控制不能省略。相同30k也不是相同计算成本。

English: Full-data RGB comparison under the recorded deployment convention. Scaffold uses fixed UID0; its native training-UID score and later no-appearance control must be distinguished. Equal iterations do not imply equal compute.

证据角色：全400训练拟合；固定外观约定

样例选择：预定义主题及既有固定样例；未进行新质量筛选

来源：[QUAL-007](../../experiments/runs/QUAL-007-BACKBONE-COMPARISON/report.md) · [RGB-130](../../experiments/runs/RGB-130-STANDARD-ALL400/report.md) · [RGB-136](../../experiments/runs/RGB-136-SCAFFOLD-NOAPPEARANCE-ALL400/report.md)

素材：[F06_rgb_backbones.svg](figures/F06_rgb_backbones.svg) · [F06_rgb_backbones.png](figures/F06_rgb_backbones.png)

原尺寸单图：本组为可编辑SVG图表／方法示意，无原生照片单图。

## F07 · 跨空间fold与语义seed：方向重复，幅度不均

中文：每个fold展示两个语义seed的汇总分数均值差，而非集成预测分数。v1/v2均为已暴露开发数据，RGB只有seed0；部分区域/类别退化，v2训练和渲染成本增加。

English: Differences of pooled-score means across two semantic seeds per fold, not prediction ensembles. Both folds are historically exposed development evidence; RGB uses one seed. Regional regressions and compute premiums remain.

证据角色：空间开发确认；非盲测

样例选择：预定义主题及既有固定样例；未进行新质量筛选

来源：[QUAL-010](../../experiments/runs/QUAL-010-MATCHED-SPATIAL-BACKBONE/report.md) · [QUAL-011](../../experiments/runs/QUAL-011-SEMANTIC-SEED-REPLICATION-AUDIT/report.md) · [QUAL-012](../../experiments/runs/QUAL-012-DISJOINT-SPATIAL-BACKBONE-AUDIT/report.md) · [QUAL-013](../../experiments/runs/QUAL-013-SPATIAL-V2-SEED-REPLICATION/report.md)

素材：[F07_spatial_confirmation.svg](figures/F07_spatial_confirmation.svg) · [F07_spatial_confirmation.png](figures/F07_spatial_confirmation.png)

原尺寸单图：本组为可编辑SVG图表／方法示意，无原生照片单图。

## F08 · 训练长度：保留完整曲线

中文：RGB166使用同一次50k训练中的五端点；语义保留6k/8k/10k/12k波动。SEM437同时改变上下文和目标尺度；空间v1/v2长训最优端点不同，不能将12k写成泛化最优。

English: Saved endpoints from one RGB166 trajectory and the complete SEM436/437 curves. These are training-fit scores. Context mixing also changes object scale; spatial folds do not share one best semantic endpoint.

证据角色：全400/全300训练拟合

样例选择：预定义主题及既有固定样例；未进行新质量筛选

来源：[RGB-166](../../experiments/runs/RGB-166-IMPROVEDGS-FULL-ALL400-50K/report.md) · [SEM-434](../../experiments/runs/SEM-434-RGB152-MASK2FORMER-LONG12K/report.md) · [SEM-435](../../experiments/runs/SEM-435-RGB154-MASK2FORMER-LONG12K/report.md) · [SEM-436](../../experiments/runs/SEM-436-RGB170-MASK2FORMER-LONG12K-ALL300/report.md) · [SEM-437](../../experiments/runs/SEM-437-RGB170-MASK2FORMER-CONTEXT50-LONG12K-ALL300/report.md)

素材：[F08_training_lengths.svg](figures/F08_training_lengths.svg) · [F08_training_lengths.png](figures/F08_training_lengths.png)

原尺寸单图：本组为可编辑SVG图表／方法示意，无原生照片单图。

## F09 · 必须保留的反例：容量收益没有迁移

中文：严格配对RGB175/176只改变100万→150万预算；holdout排除于训练与初始化。150万在holdout30下降0.867008dB，26/30图退化。提供原尺寸照片/两臂单图；另附原审阅对照，其中RGB166/RGB170仅提供全量语境，不参与留出因果比较。

English: Matched RGB175/176 differ only in the 1.0M versus 1.5M budget. Training and initialization exclude holdout/buffer. The larger budget loses 0.867008 dB on holdout30. Native photo and paired outputs are included; RGB166/170 in the archived panels are full-data context, not holdout controls.

证据角色：历史暴露空间确认；严格配对容量比较

样例选择：159为既有困难视角；260为报告登记的最大holdout退化视角

来源：[QUAL-026](../../experiments/runs/QUAL-026-RGB170-SPATIAL-CONFIRMATION/report.md) · [RGB-175](../../experiments/runs/RGB-175-IMPROVEDGS-SPATIAL-BUDGET10M/report.md) · [RGB-176](../../experiments/runs/RGB-176-IMPROVEDGS-SPATIAL-BUDGET15M/report.md)

素材：[F09_capacity_tradeoff.svg](figures/F09_capacity_tradeoff.svg) · [F09_capacity_tradeoff.png](figures/F09_capacity_tradeoff.png) · [F09_159_saved_comparison.jpg](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F09_159_saved_comparison.jpg) · [F09_159_matched_capacity.png](figures/F09_159_matched_capacity.png) · [F09_260_saved_comparison.jpg](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F09_260_saved_comparison.jpg) · [F09_260_matched_capacity.png](figures/F09_260_matched_capacity.png)

原尺寸单图：[F09_159_saved_comparison.jpg](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F09_159_saved_comparison.jpg) · [F09_159_Photo.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F09_159_Photo.png) · [F09_159_RGB175.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F09_159_RGB175.png) · [F09_159_RGB176.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F09_159_RGB176.png) · [F09_260_saved_comparison.jpg](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F09_260_saved_comparison.jpg) · [F09_260_Photo.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F09_260_Photo.png) · [F09_260_RGB175.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F09_260_RGB175.png) · [F09_260_RGB176.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F09_260_RGB176.png)

## F10 · 当前表现与未解决视角并列

中文：159展示全300训练内照片/RGB/GT/预测；350和357为无标签视角，357仍无deck。无GT图不能计算准确率；不对原结果做修补、美化或生成式补画。

English: View 159 is an in-training photo/RGB/GT/prediction example. Unlabeled views 350 and 357 are qualitative only; deck remains absent in 357. No generative retouching or model rerun is used.

证据角色：159训练拟合；350/357无GT定性

样例选择：预定义主题及既有固定样例；未进行新质量筛选

来源：[SEM-437](../../experiments/runs/SEM-437-RGB170-MASK2FORMER-CONTEXT50-LONG12K-ALL300/report.md) · [SEM-439](../../experiments/runs/SEM-439-RGB170-DECK-ABSENCE-FALLBACK/report.md) · [SEM-440](../../experiments/runs/SEM-440-RGB170-WEAK-STRUCTURE-TAIL14K/report.md)

素材：[F10_350_unresolved.png](figures/F10_350_unresolved.png) · [F10_357_unresolved.png](figures/F10_357_unresolved.png) · [F10_159_current_fit.png](figures/F10_159_current_fit.png)

原尺寸单图：[F10_350_Photo.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F10_350_Photo.png) · [F10_350_RGB170.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F10_350_RGB170.png) · [F10_350_SEM437.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F10_350_SEM437.png) · [F10_357_Photo.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F10_357_Photo.png) · [F10_357_RGB170.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F10_357_RGB170.png) · [F10_357_SEM437.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F10_357_SEM437.png) · [F10_159_Photo.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F10_159_Photo.png) · [F10_159_RGB170.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F10_159_RGB170.png) · [F10_159_GT.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F10_159_GT.png) · [F10_159_SEM437.png](https://samtu648-hub.github.io/icshm2026-semantic-gs/native/F10_159_SEM437.png)

## V01a · 历史三维语义 SEM001

中文：历史三维语义 SEM001。保留真实伪影与原视频帧序。历史144相机视频单独播放；各阶段系统不同，不能作为单变量消融。

English: Historical static 3D semantics (SEM001): saved outputs only, no ground truth; qualitative evidence. Original frame order and appearance are retained.

证据角色：无GT固定轨迹，仅定性

图例：background 深灰 / deck 蓝 / stay_cable 黄 / tower 红 / foundation 紫

保存规格：144帧，12/1 fps，12.00秒。

[视频](https://samtu648-hub.github.io/icshm2026-semantic-gs/videos/V01a.mp4) · [SEM-001](../../experiments/runs/SEM-001/report.md)

## V01b · 历史类别专家 SEM161

中文：历史类别专家 SEM161。保留真实伪影与原视频帧序。历史144相机视频单独播放；各阶段系统不同，不能作为单变量消融。

English: Historical specialist fusion (SEM161): saved outputs only, no ground truth; qualitative evidence. Original frame order and appearance are retained.

证据角色：无GT固定轨迹，仅定性

图例：background 深灰 / deck 蓝 / stay_cable 黄 / tower 红 / foundation 紫

保存规格：144帧，12/1 fps，12.00秒。

[视频](https://samtu648-hub.github.io/icshm2026-semantic-gs/videos/V01b.mp4) · [SEM-161](../../experiments/runs/SEM-161-DECK-BINARY-FINAL/report.md)

## V01c · 历史 Mask2Former SEM349

中文：历史 Mask2Former SEM349。保留真实伪影与原视频帧序。历史144相机视频单独播放；各阶段系统不同，不能作为单变量消融。

English: Historical Mask2Former (SEM349): saved outputs only, no ground truth; qualitative evidence. Original frame order and appearance are retained.

证据角色：无GT固定轨迹，仅定性

图例：background 深灰 / deck 蓝 / stay_cable 黄 / tower 红 / foundation 紫

保存规格：144帧，12/1 fps，12.00秒。

[视频](https://samtu648-hub.github.io/icshm2026-semantic-gs/videos/V01c.mp4) · [SEM-349](../../experiments/runs/SEM-349-MASK2FORMER-FINAL/report.md)

## V02 · 原始三骨干同轨迹

中文：原始三骨干同轨迹。保留真实伪影与原视频帧序。Scaffold为原三骨干固定UID0部署口径；不是后期无外观Scaffold或ImprovedGS。

English: Three RGB backbones on the saved trajectory: saved outputs only, no ground truth; qualitative evidence. Original frame order and appearance are retained.

证据角色：无GT固定轨迹，仅定性

图例：纯RGB对比，无语义配色

保存规格：144帧，12/1 fps，12.00秒。

[视频](https://samtu648-hub.github.io/icshm2026-semantic-gs/videos/V02.mp4) · [QUAL-007](../../experiments/runs/QUAL-007-BACKBONE-COMPARISON/report.md)

## V03 · RGB166 同轨迹30k与50k

中文：RGB166 同轨迹30k与50k。保留真实伪影与原视频帧序。144个保存帧严格按相同文件名成对，12fps，未重新渲染。

English: RGB166 same-run 30k and 50k endpoints: saved outputs only, no ground truth; qualitative evidence. Paired by exact saved frame name at 12 fps.

证据角色：无GT固定轨迹，仅定性

图例：纯RGB对比，无语义配色

保存规格：144帧，12/1 fps，12.00秒。

[视频](https://samtu648-hub.github.io/icshm2026-semantic-gs/videos/V03.mp4) · [RGB-166](../../experiments/runs/RGB-166-IMPROVEDGS-FULL-ALL400-50K/report.md)

## V04 · 已有展示 RGB170 + SEM437 · 144帧

中文：已有展示 RGB170 + SEM437 · 144帧。保留真实伪影与原视频帧序。历史144相机视频单独播放；各阶段系统不同，不能作为单变量消融。

English: RGB170 and SEM437 on the 144-pose trajectory: saved outputs only, no ground truth; qualitative evidence. Original frame order and appearance are retained.

证据角色：无GT固定轨迹，仅定性

图例：background 黑 / deck 橙红 / stay_cable 黄 / tower 蓝 / foundation 橄榄绿

保存规格：144帧，12/1 fps，12.00秒。

[视频](https://samtu648-hub.github.io/icshm2026-semantic-gs/videos/V04.mp4) · [SEM-437](../../experiments/runs/SEM-437-RGB170-MASK2FORMER-CONTEXT50-LONG12K-ALL300/report.md)

## V05 · 已有展示 · 100张无标注视角

中文：已有展示 · 100张无标注视角。保留真实伪影与原视频帧序。301–400按视角编号播放，是图像序列，不是连续时间采集或独立测试集。

English: RGB170 and SEM437 on 100 unlabeled released views: saved outputs only, no ground truth; qualitative evidence. Original frame order and appearance are retained.

证据角色：无GT定性；100张图已用于全量RGB训练

图例：background 黑 / deck 橙红 / stay_cable 黄 / tower 蓝 / foundation 橄榄绿

保存规格：100帧，5/1 fps，20.00秒。

[视频](https://samtu648-hub.github.io/icshm2026-semantic-gs/videos/V05.mp4) · [SEM-437](../../experiments/runs/SEM-437-RGB170-MASK2FORMER-CONTEXT50-LONG12K-ALL300/report.md)

## V06 · 容量确认 · RGB170 / RGB175 / RGB176

中文：容量确认 · RGB170 / RGB175 / RGB176。保留真实伪影与原视频帧序。RGB170为全400语境；RGB175/176为匹配空间模型。视频没有GT，不承担holdout量化结论。

English: Capacity comparison: RGB170 context and matched RGB175/176: saved outputs only, no ground truth; qualitative evidence. Original frame order and appearance are retained.

证据角色：无GT固定轨迹，仅定性

图例：纯RGB对比，无语义配色

保存规格：144帧，12/1 fps，12.00秒。

[视频](https://samtu648-hub.github.io/icshm2026-semantic-gs/videos/V06.mp4) · [QUAL-026](../../experiments/runs/QUAL-026-RGB170-SPATIAL-CONFIRMATION/report.md)
