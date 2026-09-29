# 完整开发历程

伙伴交接版 · 证据截止 2026-09-23 · 整理 2026-09-29

本项目面向桥梁场景的新相机 RGB 与五类语义输出。开发从统一官方 COLMAP 坐标和数据协议开始，经历 RGB 骨干比较、冻结三维语义、渲染域二维专家、几何与特征探索，再进入 Mask2Former、空间验证与重建细结构诊断。最可靠的成果不仅是分数提高，也包括识别了哪些改进只改善已见视角、哪些方向不能继续支持，以及如何让结论可以被独立追溯。

## P01 · 建立数据、相机和评价口径

项目初期 → 全程维护

**问题：** 不同实现的相机、图像网格和标签是否真正对应？

**尝试：** 统一官方 COLMAP 参考坐标、PINHOLE 训练与评价网格、五类标签和数据 manifest；区分重建照片、语义标签、隔离视角和无标签相机。外部相机独立测试 RGB 与语义输出。

**结果：** 后期 QUAL009 完成 300 标签和 400 相机的画像：391,644,000 个评价像素，background 占 85.5872%，foundation 仅 0.6026%。stay_cable 占 6.9745%，并非最少的前景类，但细小构件和远景尺度变化突出。

**决定与边界：** 论文应把尺度异质性、遮挡和二维多边形标注定义讲清；宽索面标签不等同于逐根物理索线。

**写作用途：** 正文：数据与实现；F01。

来源：[QUAL-009](../../experiments/runs/QUAL-009-DATASET-PROFILE/report.md) · [SUBMIT-030](../../experiments/runs/SUBMIT-030-SEM161-BUNDLE/report.md) · [SUBMIT-040](../../experiments/runs/SUBMIT-040-CLEAN-UBUNTU-REPRO/report.md)

## P02 · 早期 RGB 骨干选择与几何初始化

7–8 月

**问题：** 先获得能稳定渲染、坐标兼容的 RGB 底座，再讨论语义。

**尝试：** 在统一协议下比较 Standard、Mip 与 Scaffold；记录训练成本和困难视角。另探索 MapAnything、DA3、VGGT 系列几何/位姿提议与初始化，不直接替换冻结相机系。

**结果：** 早期 RGB060 对照后采用 Scaffold。几何先验出现工程可行、短程局部收益、坐标审计失败及未超过冻结底座等不同结果，不能合写为一个成功方法。后期 VGGT-Ω smoke 通过而全量 P0 坐标审计失败，说明预检可运行不代表坐标和质量合格。

**决定与边界：** RGB030 成为早期语义冻结基底。后续全量训练会重新比较骨干，早期 Scaffold 选择不代表全程最优。

**写作用途：** 正文简述技术起点；几何先验细节放补充材料。

来源：[RGB-060](../../experiments/runs/RGB-060/report.md) · [GEO-010-10K](../../experiments/runs/GEO-010-10K/report.md) · [GEO-020-3K-INT-R1](../../experiments/runs/GEO-020-3K-INT-R1/report.md) · [GEO-025](../../experiments/runs/GEO-025-PF/report.md) · [GEO-026](../../experiments/runs/GEO-026-P0-F64/report.md)

## P03 · 静态三维语义暴露投影冲突

7 月末—8 月

**问题：** 给冻结 Gaussian 表示附着类别，能否直接得到可靠桥梁语义？

**尝试：** SEM001 建立五类静态三维语义；继续尝试贡献门控、冲突正则、视角残差、伪标签资格审查和 Cable3D 几何先验。

**结果：** SEM001 在历史 74 图上五类 mIoU 为 61.610304%，stay_cable IoU 仅 6.875938%。一些门控降低背景误报，却破坏真实构件或没有达到质量门；几何及伪标签预检也未形成可保留升级。

**决定与边界：** 保留静态基线作为对照，转向利用渲染 RGB 中仍存在的二维辨识信息。失败仅限制实际测试的配方，不否定所有三维语义。

**写作用途：** 正文：动机及基线；F03。

来源：[SEM-001](../../experiments/runs/SEM-001/report.md) · [SEM-002](../../experiments/runs/SEM-002/report.md) · [SEM-003](../../experiments/runs/SEM-003/report.md) · [SEM-004](../../experiments/runs/SEM-004/report.md) · [SEM-052-PL-AUDIT](../../experiments/runs/SEM-052-PL-AUDIT/report.md) · [SEM-006](../../experiments/runs/SEM-006/report.md)

## P04 · 渲染域训练与类别保护带来第一次大幅提升

9 月 2–4 日

**问题：** 真实照片教师能否直接处理重建图？整图二维替换是否会损害原有结构？

**尝试：** 先比较照片/渲染域，再训练渲染域二维分割；SEM102 检验整图替换，SEM111 将二维证据限制在允许改变的类别区域，保护 deck、tower、foundation。

**结果：** 渲染域输入仍有索线线索，但存在明显域差。整图替换虽然改善部分指标，保护类和困难图退化；SEM111 保护融合使历史 74 图 mIoU 升到 76.063477%，背景→stay_cable 误报相对基底显著下降，保护类保持逐像素不变。

**决定与边界：** 重要贡献是找到可控的二维补充途径，不是证明任何二维模型都会更好。

**写作用途：** 正文：核心技术转折与直接父实验对照；F03、F04。

来源：[SEM-100](../../experiments/runs/SEM-100-RENDER-SEG-PF/report.md) · [SEM-101](../../experiments/runs/SEM-101-RENDERED-SEG-OOF/report.md) · [SEM-102](../../experiments/runs/SEM-102-RENDERED-SEG-FINAL/report.md) · [SEM-110](../../experiments/runs/SEM-110-PROTECTED-FUSION-OOF/report.md) · [SEM-111](../../experiments/runs/SEM-111-PROTECTED-FUSION-FINAL/report.md)

## P05 · 类别专家形成可复现的历史完整链

9 月 4–7 日

**问题：** 如何继续改善拉索、基础误报和桥面漏检，又不破坏冻结 RGB？

**尝试：** 加入原生分辨率 stay_cable 二分类专家 SEM141、foundation 单向审核 SEM151、deck 恢复 SEM161；训练侧 OOF 与 calibration/check 记录阈值来源。

**结果：** 历史 74 图 mIoU 依次达到 77.489877%、83.093383%、83.387017%。SEM001→SEM161 累计提高 21.776713 个百分点；这是系统演进，存在 seed 和配方变化，不是一个单变量消融。完成重载、RGB 精确不变和官方外部相机输出。

**决定与边界：** 保留 RGB030＋SEM161 为历史工程回退。报告逐类 precision/recall 和局部退化，保护类别并不意味着其预测已正确。

**写作用途：** 正文：方法演进；F03、F04、V01。

来源：[SEM-141](../../experiments/runs/SEM-141-CABLE-BINARY-FINAL/report.md) · [SEM-151](../../experiments/runs/SEM-151-FOUNDATION-GATE-FINAL/report.md) · [SEM-161](../../experiments/runs/SEM-161-DECK-BINARY-FINAL/report.md) · [SUBMIT-030](../../experiments/runs/SUBMIT-030-SEM161-BUNDLE/report.md) · [SUBMIT-040](../../experiments/runs/SUBMIT-040-CLEAN-UBUNTU-REPRO/report.md)

## P06 · 几何、特征与参考源纠错：局部收益难以完整迁移

9 月上旬

**问题：** 剩余边界、基础和桥面错误能否通过更精细三维表示或跨视角证据纠正？

**尝试：** 探索 XYZ/RGBD、offset 职责、稀疏轨迹、点级与边界训练；DINOv2 特征复核之后，引入 RoMa v2 稠密参考对应及第二参考源否决删除。

**结果：** DINOv2 训练侧 OOF 有正信号，但全219最终拟合迁移到74图时收益小且 foundation recall 有代价。SEM329 相对 SEM161 五类 mIoU +0.317985pp、foundation IoU +1.584083pp，但 recall −1.424762pp；第二源恢复也会返回背景误报。

**决定与边界：** 把来源覆盖、误删和迁移差异作为结论。早期 DINOv3 权重访问阻塞与后来实际完成的 DINOv3 实验分开；不能把访问失败写成质量失败。

**写作用途：** 补充材料：探索路线与负结果；F04 的可选原图。

来源：[SEM-260](../../experiments/runs/SEM-260-ERROR-BUDGET/report.md) · [SEM-263](../../experiments/runs/SEM-263-NESTED-POINT-OOF/report.md) · [SEM-310](../../experiments/runs/SEM-310-DINOV3-FROZEN-FEATURE-OOF/report.md) · [SEM-311](../../experiments/runs/SEM-311-DINOV2-FROZEN-FEATURE-OOF/report.md) · [SEM-314](../../experiments/runs/SEM-314-DINOV2-FOUNDATION-REVIEW-FINAL/report.md) · [SEM-320](../../experiments/runs/SEM-320-ROMAV2-DENSE-PREFLIGHT/report.md) · [SEM-325](../../experiments/runs/SEM-325-DENSE-FOUNDATION-REVIEW-OOF/report.md) · [SEM-326](../../experiments/runs/SEM-326-DENSE-FOUNDATION-REVIEW-FINAL/report.md) · [SEM-329](../../experiments/runs/SEM-329-SECOND-SOURCE-VETO-FINAL/report.md)

## P07 · 强二维 Mask2Former 改变完整系统的选择

9 月 9–15 日

**问题：** 提升渲染域分割模型容量后，还需要复杂的多专家融合吗？

**尝试：** 在新 RGB110 上评估 SegFormer-B2 与 Mask2Former-Swin-L，完成 OOF、全219拟合、同74固定比较，并探索 raw/fused/选择器及尺度变化。

**结果：** SEM350 同74比较中，Mask2Former raw 为 86.513086%，高于旧 SEM161 的 83.387017%；68 图提高、6 图下降。完整融合仅再增 0.029861pp，却有更多误改。另有 SEM363 学习式迁移达到历史74的86.596897%，后续变体并非继续稳定提高。

**决定与边界：** 采用更简洁的强二维路径作为后期质量锚。B2 3k 与 Mask2Former 6k 不是同预算容量消融；RGB030 与 RGB110 也不同，系统差不能归因于单一模块。

**写作用途：** 正文：强基线与完整系统比较；F05、V01。

来源：[SEM-341](../../experiments/runs/SEM-341-MASK2FORMER-OOF/report.md) · [SEM-349](../../experiments/runs/SEM-349-MASK2FORMER-FINAL/report.md) · [SEM-350](../../experiments/runs/SEM-350-FIXED-FINAL-COMPARISON/report.md) · [SEM-363](../../experiments/runs/SEM-363-FINAL-LEARNED-TRANSFER/report.md) · [SEM-366](../../experiments/runs/SEM-366-CLASS-CONDITIONAL-FINAL/report.md) · [SEM-368](../../experiments/runs/SEM-368-EXPERT-INPUT-SCALE/report.md)

## P08 · 全量训练重新比较 RGB 与语义输入域

9 月 15–18 日

**问题：** 早期选择在全400照片、全300标签条件下是否仍成立？

**尝试：** 比较 Standard、Mip、Scaffold 的全量渲染和匹配 Mask2Former；明确 Scaffold 训练 UID 与固定 UID0 的外观差异，再做无外观控制和跨 RGB 输入的交叉语义实验。探索绝对梯度增密、结构区域监督和 DINOv3 替换。

**结果：** Standard＋SEM371 与 AbsGS RGB140＋SEM378 的全300五类拟合分别为95.265719%和95.297405%，差仅0.031685pp。全量 RGB/语义排名受输入域影响；更高 RGB 分数不自动产生同等语义收益。结构监督和 DINOv3 变体有局部改进，却没有形成一致晋级。

**决定与边界：** 保留匹配 RGB—语义组合；把训练拟合与空间证据分开。原始三骨干比较必须标注 Scaffold UID0，后来的无外观控制另列，不能隐去。

**写作用途：** 正文：骨干及成本比较；F06、V02。

来源：[QUAL-007](../../experiments/runs/QUAL-007-BACKBONE-COMPARISON/report.md) · [RGB-130](../../experiments/runs/RGB-130-STANDARD-ALL400/report.md) · [RGB-136](../../experiments/runs/RGB-136-SCAFFOLD-NOAPPEARANCE-ALL400/report.md) · [RGB-140](../../experiments/runs/RGB-140-STANDARD-ABSGRAD-ALL400/report.md) · [SEM-375](../../experiments/runs/SEM-375-SCAFFOLD-NOAPPEARANCE-ALL300/report.md) · [SEM-376](../../experiments/runs/SEM-376-CROSS-RGB-INFERENCE/report.md) · [SEM-378](../../experiments/runs/SEM-378-RGB140-ABSGRAD-ALL300/report.md) · [SEM-380](../../experiments/runs/SEM-380-RGB148-GT-REGION-RGB-ALL300/report.md) · [SEM-381](../../experiments/runs/SEM-381-DINOV3-VITB-SFP-ALL300/report.md) · [SEM-382](../../experiments/runs/SEM-382-DINOV3-VITB-SFP-LR1E4-ALL300/report.md) · [SEM-383](../../experiments/runs/SEM-383-DINOV3-VITB-SPATIAL-SFP-ALL300/report.md) · [SEM-387](../../experiments/runs/SEM-387-RGB150-FEATURE-SPLATTING-DINOV3/report.md)

## P09 · 空间隔离与双语义 seed 给出更有价值的确认

9 月 18–21 日

**问题：** 全量拟合的微小差异能否迁移到训练排除的空间视角？

**尝试：** 从 RGB 初始化起排除 holdout/buffer，建立 Standard 与 AbsGS 严格匹配链；在零重叠 v1/v2 两空间 fold 上，各完成两个语义 seed 的配对复验。

**结果：** v1 双seed平均前景mIoU/索IoU/边界F1差为 +20.139420/+11.803546/+13.719429pp；v2 为 +0.420671/+1.928975/+3.436304pp。方向重复而幅度差异大；v1 大增益由困难区域主导，v2 有区域和 tower 代价。

**决定与边界：** 支持质量优先 AbsGS、效率回退 Standard 的阶段选择，不支持逐区支配或统计显著性。两fold均有开发暴露历史，RGB只有一个seed；不能写成全链多seed或隐藏测试结果。

**写作用途：** 正文优先：跨区域、跨seed证据和代价；F07。

来源：[QUAL-010](../../experiments/runs/QUAL-010-MATCHED-SPATIAL-BACKBONE/report.md) · [QUAL-011](../../experiments/runs/QUAL-011-SEMANTIC-SEED-REPLICATION-AUDIT/report.md) · [QUAL-012](../../experiments/runs/QUAL-012-DISJOINT-SPATIAL-BACKBONE-AUDIT/report.md) · [QUAL-013](../../experiments/runs/QUAL-013-SPATIAL-V2-SEED-REPLICATION/report.md)

## P10 · 边界、三维职责与照片教师的有限探索

9 月 19–21 日

**问题：** 二维结果剩余的细结构错误，是否可通过边界损失、Gaussian 分裂、射线分层或照片监督解决？

**尝试：** 测试边界Dice、小构件采样、语义方向分裂、跨视角可靠性、射线/深度职责、静态五logit与一阶视角条件；另测试照片教师门和真实GT双域训练。

**结果：** 部分方法通过工程门或改善局部，却未通过目标质量门。SEM428 的视角条件使前景mIoU较静态三维提高8.350061pp，但绝对仅50.296747%，仍低于匹配二维36.370630pp。照片分支改善部分大结构，却增加背景→stay_cable；小构件配额收益未在第二seed兑现总体改善。

**决定与边界：** 停止已测试的救援扫描。现有证据提示可见性及跨视角表面职责仍是瓶颈，但这是机制线索，不是已证明的唯一原因或已实现的新方法。

**写作用途：** 正文择一两个代表失败，其余补充材料；F10。

来源：[SEM-395](../../experiments/runs/SEM-395-RGB150-CROSSVIEW-RELIABILITY-DIAG/report.md) · [SEM-397](../../experiments/runs/SEM-397-RGB152-SPATIAL256-BOUNDARY/report.md) · [SEM-399](../../experiments/runs/SEM-399-RGB152-BOUNDARY-SPLIT-SEED2/report.md) · [SEM-401](../../experiments/runs/SEM-401-RGB152-RESPONSIBILITY-IDENTIFIABILITY-DIAG/report.md) · [SEM-406](../../experiments/runs/SEM-406-RGB152-DEPTH-MODE-PREFLIGHT/report.md) · [SEM-418](../../experiments/runs/SEM-418-COMPONENT-POINT-QUOTA/report.md) · [SEM-421](../../experiments/runs/SEM-421-COMPONENT-EXPLORATORY-CTRL/report.md) · [SEM-426](../../experiments/runs/SEM-426-RGB152-DIRECT5-SPATIAL256/report.md) · [SEM-428](../../experiments/runs/SEM-428-RGB152-DIR5-SPATIAL256/report.md) · [SEM-430](../../experiments/runs/SEM-430-PHOTO-TEACHER-DECK-REPLICATION/report.md) · [SEM-433](../../experiments/runs/SEM-433-PAIRED-PHOTO-DOMAIN/report.md)

## P11 · ImprovedGS 与延长训练改善桥梁拟合

9 月 21–23 日

**问题：** 重建是否仍欠训练，增加预算能否补齐细索？

**尝试：** 接入 ImprovedGS 方法包，在固定100万预算下保存同一训练轨迹30k/35k/40k/45k/50k端点；继续比较 Scaffold 50k、150万预算及结构修补。

**结果：** RGB166 同轨迹30k→50k 全400 PSNR 从23.428528升到23.490679dB，stay_cable 区域 +0.403523dB，但雾块和断索仍存在。RGB170 的150万预算在全400进一步小幅提高，后续局部分裂、职责克隆和EAS保护未稳定超过它。

**决定与边界：** 50k是已见拟合的停止点，不继续靠步数或容量宣称修复几何。ImprovedGS的方法包与此前AbsGS配方分开描述，不能混为同一个单变量。

**写作用途：** 正文：训练长度与拟合代价；F08、V03。

来源：[RGB-165](../../experiments/runs/RGB-165-IMPROVEDGS-FULL-ALL400/report.md) · [RGB-166](../../experiments/runs/RGB-166-IMPROVEDGS-FULL-ALL400-50K/report.md) · [RGB-168](../../experiments/runs/RGB-168-SCAFFOLD-GROWTH20K-ALL400-50K/report.md) · [RGB-170](../../experiments/runs/RGB-170-IMPROVEDGS-BUDGET15M-H50K/report.md) · [RGB-171](../../experiments/runs/RGB-171-IMPROVEDGS-BRIDGE-STRUCTURE-REPAIR/report.md) · [RGB-172](../../experiments/runs/RGB-172-IMPROVEDGS-EVIDENCE-DIRECTED-SPLIT/report.md) · [RGB-173](../../experiments/runs/RGB-173-IMPROVEDGS-RESPONSIBILITY-CLONE/report.md) · [RGB-174](../../experiments/runs/RGB-174-IMPROVEDGS-BRIDGE-PROTECTED-EAS/report.md)

## P12 · 后期语义优化与最终角色边界

9 月 22–23 日

**问题：** 全量语义的长训与上下文能否修复困难航段；最好展示模型是否也适合未知视角？

**尝试：** RGB170固定，SEM436做6k—12k同轨迹，SEM437交替原生裁块和完整画面上下文；多尺度与弱结构尾段作为独立尝试。QUAL026另用RGB175/176严格配对检查100万/150万预算。

**结果：** SEM437-12k全300五类mIoU为95.678649%，350—353部分桥面恢复，但357仍漏检。150万预算在353训练图PSNR仅+0.016242dB，30张holdout却−0.867008dB，26/30张下降；四类桥梁与索支持都退化。

**决定与边界：** RGB170＋SEM437保留为已有展示；未来未知视角回到RGB166配方，匹配语义尚未训练。SEM439只是355/356后验开发fallback，无GT准确率结论；不把它放成默认论文主方法。

**写作用途：** 正文：容量反例、局限与当前状态；F08—F10、V04—V06。

来源：[SEM-434](../../experiments/runs/SEM-434-RGB152-MASK2FORMER-LONG12K/report.md) · [SEM-435](../../experiments/runs/SEM-435-RGB154-MASK2FORMER-LONG12K/report.md) · [SEM-436](../../experiments/runs/SEM-436-RGB170-MASK2FORMER-LONG12K-ALL300/report.md) · [SEM-437](../../experiments/runs/SEM-437-RGB170-MASK2FORMER-CONTEXT50-LONG12K-ALL300/report.md) · [SEM-439](../../experiments/runs/SEM-439-RGB170-DECK-ABSENCE-FALLBACK/report.md) · [SEM-440](../../experiments/runs/SEM-440-RGB170-WEAK-STRUCTURE-TAIL14K/report.md) · [QUAL-026](../../experiments/runs/QUAL-026-RGB170-SPATIAL-CONFIRMATION/report.md)
