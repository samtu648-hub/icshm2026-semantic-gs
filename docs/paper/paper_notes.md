# 论文写作底稿

伙伴交接版 · 证据截止 2026-09-23 · 整理 2026-09-29

先读[作者必读](https://samtu648-hub.github.io/icshm2026-semantic-gs/author_guide.html)，完成任务、方法、证据边界与失败分析的理解，再使用以下段落。

以下为可改写段落，不是已提交英文论文。最终模型身份、作者贡献与参考文献由团队定稿。

## 研究问题与可支持的贡献

建议位置：Introduction / Contributions

本研究围绕桥梁场景的视角条件 RGB 与语义输出，研究重建表示、渲染域分割与空间验证之间的关系。可支持的贡献包括：建立官方相机坐标兼容且可追溯的完整实验流水线；通过静态三维语义、类别保护融合和强二维分割的演进识别主要质量来源；利用匹配空间fold、重复语义seed和容量反例，揭示训练拟合与空间留出表现的差异。贡献应落在经过验证的系统设计与实证分析，不将尚未实现的surface identity机制或采用已有方法本身声明为原创算法。

来源：[SEM-001](../../experiments/runs/SEM-001/report.md) · [SEM-161](../../experiments/runs/SEM-161-DECK-BINARY-FINAL/report.md) · [SEM-350](../../experiments/runs/SEM-350-FIXED-FINAL-COMPARISON/report.md) · [QUAL-013](../../experiments/runs/QUAL-013-SPATIAL-V2-SEED-REPLICATION/report.md) · [QUAL-026](../../experiments/runs/QUAL-026-RGB170-SPATIAL-CONFIRMATION/report.md)

## 数据集、标注与划分

建议位置：Dataset / Evaluation protocol

发布数据包括400张照片，其中300张有五类语义标签，类别为background、deck、stay_cable、tower和foundation。原生评价网格为1320×989。背景占85.5872%的评价像素；stay_cable虽占6.9745%，仍包含明显的小构件长尾和远景尺度变化。早期219/74/7语义划分、训练侧OOF、后期空间确认以及全300拟合各有不同用途。后期确认从RGB初始化起隔离holdout/buffer，但这些开发fold已有历史暴露。另100张无标签照片参与了全量重建，不能被称为完整系统未见测试集。

来源：[QUAL-009](../../experiments/runs/QUAL-009-DATASET-PROFILE/report.md) · [SEM-001](../../experiments/runs/SEM-001/report.md) · [QUAL-010](../../experiments/runs/QUAL-010-MATCHED-SPATIAL-BACKBONE/report.md) · [QUAL-012](../../experiments/runs/QUAL-012-DISJOINT-SPATIAL-BACKBONE-AUDIT/report.md)

## 完整方法与历史方法的界限

建议位置：Method

后期完整展示系统先按官方相机从冻结的RGB Gaussian模型渲染图像，再由匹配渲染域的Mask2Former预测五类语义。语义训练不更新RGB表示或相机。早期系统额外使用三维静态语义、二维类别专家及保护融合，这些模块应作为历史方法与消融介绍，不能画成SEM437推理时仍存在的组件。SEM437在固定RGB170上交替原生裁块与完整画面训练，继承固定12k轨迹和两阶段学习率；该修改同时改变上下文和目标尺度，不能归因于纯上下文。

来源：[SEM-161](../../experiments/runs/SEM-161-DECK-BINARY-FINAL/report.md) · [SEM-349](../../experiments/runs/SEM-349-MASK2FORMER-FINAL/report.md) · [SEM-436](../../experiments/runs/SEM-436-RGB170-MASK2FORMER-LONG12K-ALL300/report.md) · [SEM-437](../../experiments/runs/SEM-437-RGB170-MASK2FORMER-CONTEXT50-LONG12K-ALL300/report.md)

## 历史系统的主要进步

建议位置：Results / Development trajectory

在同一组历史74张开发图上，静态三维基线SEM001的五类mIoU为61.610304%。渲染域二维证据及类别保护融合将其提高到76.063477%，随后拉索专家、基础审核和桥面恢复组成的SEM161达到83.387017%。累计21.776713个百分点描述的是系统演进，并非贯穿全程的单变量因果效应。后续RGB110配合Mask2Former原始输出在同74图达到86.513086%；与旧系统比较时，RGB底座和语义方法同时变化，且存在少数视角退化和tower/foundation召回下降。

来源：[SEM-001](../../experiments/runs/SEM-001/report.md) · [SEM-111](../../experiments/runs/SEM-111-PROTECTED-FUSION-FINAL/report.md) · [SEM-141](../../experiments/runs/SEM-141-CABLE-BINARY-FINAL/report.md) · [SEM-151](../../experiments/runs/SEM-151-FOUNDATION-GATE-FINAL/report.md) · [SEM-161](../../experiments/runs/SEM-161-DECK-BINARY-FINAL/report.md) · [SEM-350](../../experiments/runs/SEM-350-FIXED-FINAL-COMPARISON/report.md)

## 匹配空间消融与区域差异

建议位置：Ablations / Spatial confirmation

在从初始化起隔离的两套空间开发fold中，AbsGS与Standard使用匹配训练设置和语义模型，并各重复两个语义seed。v1的平均前景mIoU优势为20.139420个百分点，v2为0.420671个百分点；索IoU和2px边界F1的总体方向也重复。增益幅度强烈依赖区域构成：v1由困难区域主导，v2部分区域及类别下降。两fold均有开发暴露历史，RGB仍只有seed0，因此结果支持重复的总体方向，不能支持统计显著性或普遍区域优势。

来源：[QUAL-010](../../experiments/runs/QUAL-010-MATCHED-SPATIAL-BACKBONE/report.md) · [QUAL-011](../../experiments/runs/QUAL-011-SEMANTIC-SEED-REPLICATION-AUDIT/report.md) · [QUAL-012](../../experiments/runs/QUAL-012-DISJOINT-SPATIAL-BACKBONE-AUDIT/report.md) · [QUAL-013](../../experiments/runs/QUAL-013-SPATIAL-V2-SEED-REPLICATION/report.md)

## 更长训练与更大容量的不同作用

建议位置：Ablations / Training and capacity

RGB166同一训练轨迹从30k延长到50k，全400照片PSNR提高0.062151dB，桥梁结构区域也改善，但细索缺失及水面伪影仍存在。增加至150万Gaussian预算使全量已见视角拟合进一步提高；然而QUAL026严格配对的空间确认表明，150万相对100万在30张holdout上PSNR下降0.867008dB、26张退化，四种桥梁类别均下降。该结果说明已见视角收益不足以支持未知视角选型；对错误深度与混合职责的解释应作为证据支持的机制判断，不表述成已隔离的唯一因果。

来源：[RGB-166](../../experiments/runs/RGB-166-IMPROVEDGS-FULL-ALL400-50K/report.md) · [RGB-170](../../experiments/runs/RGB-170-IMPROVEDGS-BUDGET15M-H50K/report.md) · [QUAL-026](../../experiments/runs/QUAL-026-RGB170-SPATIAL-CONFIRMATION/report.md)

## 代表性负结果

建议位置：Discussion / Supplement

简单结构清理和更强门控可降低误报，也会删除真实细索；跨视角一致可能重复同源错误；DINO与稠密对应的训练侧局部收益未必迁移到最终组合；视角条件三维语义虽超过静态控制，仍显著落后于匹配二维模型；照片监督改善大结构的同时可能增加背景索误报。应按待检验假设组织这些负结果，保留对照、评价角色和停止理由，避免仅列实验编号或把一次失败外推到整个方法家族。

来源：[SEM-080](../../experiments/runs/SEM-080-STRUCTURAL-DIAG/report.md) · [SEM-130](../../experiments/runs/SEM-130-ANCHOR-CROSSVIEW-CONSENSUS-OOF/report.md) · [SEM-314](../../experiments/runs/SEM-314-DINOV2-FOUNDATION-REVIEW-FINAL/report.md) · [SEM-329](../../experiments/runs/SEM-329-SECOND-SOURCE-VETO-FINAL/report.md) · [SEM-387](../../experiments/runs/SEM-387-RGB150-FEATURE-SPLATTING-DINOV3/report.md) · [SEM-428](../../experiments/runs/SEM-428-RGB152-DIR5-SPATIAL256/report.md) · [SEM-433](../../experiments/runs/SEM-433-PAIRED-PHOTO-DOMAIN/report.md)

## 代价、复现与局限

建议位置：Efficiency / Reproducibility / Limitations

质量提升伴随资源代价：v2的AbsGS相对Standard重建训练时间约增加42.78%、Gaussian/PLY规模增加21.19%、渲染时间增加46.36%。历史SEM161链完成同宿主GPU的干净Ubuntu容器复放，但不是跨GPU数值一致性证明，也不自动覆盖后来的模型。当前95.678649%只描述全300训练拟合；144帧轨迹无GT，不能用平滑程度证明几何或语义准确率。官方隐藏成绩、RGB多seed稳定性以及RGB166匹配最终语义均未完成。357视角桥面漏检、雾块、拉丝与细索缺口应在论文中明确展示。

来源：[QUAL-012](../../experiments/runs/QUAL-012-DISJOINT-SPATIAL-BACKBONE-AUDIT/report.md) · [SUBMIT-040](../../experiments/runs/SUBMIT-040-CLEAN-UBUNTU-REPRO/report.md) · [SEM-437](../../experiments/runs/SEM-437-RGB170-MASK2FORMER-CONTEXT50-LONG12K-ALL300/report.md) · [SEM-440](../../experiments/runs/SEM-440-RGB170-WEAK-STRUCTURE-TAIL14K/report.md) · [QUAL-026](../../experiments/runs/QUAL-026-RGB170-SPATIAL-CONFIRMATION/report.md)

## 写作核对

- 历史74图、OOF、空间确认与全量拟合分别报告；不能将61.61%与95.68%相减作为泛化提升。
- 五类mIoU与前景mIoU分列；累计混淆矩阵指标与逐图均值不能混用。百分点用pp，PSNR差用dB。
- v1/v2各有两个语义seed，但只有一个RGB seed；没有统计显著性或隐藏测试结论。
- 像素类别边界、宽stay_cable多边形、实体索线、深度与Gaussian表面职责不是同一概念。
- SEM439在看过无GT困难图后设计，355/356的视觉恢复不能用作准确率提升证据。
- 历史报告中“当前”“正式”“等待批准”均需结合日期与终态阅读；最新模型角色由ADR-175确定。
- 本包提供科研写作材料，不包含模型权重、完整数据集或可执行的最终比赛提交包。

第三方方法来源与许可证索引：[third_party 历史记录](https://samtu648-hub.github.io/icshm2026-semantic-gs/evidence/context/third_party__README.txt)。原始论文链接可从该记录追溯；本次未重新核验外部网页。
