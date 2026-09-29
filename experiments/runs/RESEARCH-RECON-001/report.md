# RESEARCH-RECON-001 — 质量优先的桥梁重建选型深度调研

- 状态：completed_research_only；日期：2026-09-08。
- 用户问题：PGSR／2DGS、GOF及后续SOF是否适合本桥；已有方案哪些值得组合；以质量和拿奖为目标重新考虑重建。
- 父实验：RGB-060、RGB-030-FULL、RGB-071、RGB-090、SEM329、VIS006，完整ID见[config](config.yaml)。
- 研究工作树：`49cdcdb1c9915a9544d568754af95ea20a85ec9c`加保留的未提交修改；不是早期被引实验的历史代码版本。
- seed：not_applicable_read_only_research。manifest与RGB／语义split及26份关键实验config/report的SHA256封存在config中。没有新读取GT像素或重算历史质量指标。

## 交付与证据边界

交付为8页中文PDF：`outputs/reconstruction_review_20260908/reconstruction-review.pdf`。同目录保存唯一规范内容源`report-source.md`、逐结论证据账本`claim-ledger.json`、PDF结构／链接检查`reconstruction-review.qa.json`及逐页PNG。媒体与研究中间产物被Git忽略；本登记不表示其已经备份或上传。

- PDF SHA256：`c598297bead16f443196255d0d50ed153da25f155acaeb6ca5bdd395f3a6aaa8`。
- 内容源 SHA256：`f2c73567a69faecce86d9592ab356f39fc606279cb87bad3bca92bc2938eb931`。
- PDF技能用于中文字体嵌入、逐页渲染和内容／链接验证；深度调研技能用于一手文献、反证与证据账本流程。无新模型安装、训练、推理或质量评分。

## 调研结论

1. 重建值得重新选型，但“桥面漏检主要由RGB高误差造成”没有被证明。RGB071支持细索外观与漏检的关联，RGB090提供桥面反例；插值帧072无同视角真实RGB／GT，不能单图证明物理缺失或真实桥梁损伤。
2. 推荐顺序是研究判断：保留Scaffold控制；PGSR优先检验局部表面／多视角约束；SOF检验排序／深度场一致性。2DGS、GOF作为机制对照，Zip-NeRF作为异构RGB备选。不是“已有新方法赢过本桥Scaffold”。
3. 作者论文及代码存在重要配方差异：PGSR承认几何／RGB折中；SOF公开NVS脚本将三项几何相关损失设为0，不能把其RGB基准解释成全部几何约束的成绩。
4. 优先复用渲染域训练、SEM141／151／161类别专家、OOF协议、官方相机接口及SEM329保守参考源复核。旧高斯关联、渲染缓存、概率、专家权重和阈值不能无验证迁移。
5. 既有MapAnything／DA3初始化、深度重投影和补洞的负结果保留；不把弱短程证据扩大为“方法永远无效”，也不把失败产物当作改进先验。

## 一手来源与关键定位

- [官方任务／评分PDF](https://www.ntu.edu.sg/media/docs/librariesprovider117/cee-events/2026/the-4th-international-competition-for-shm_2026_0527.pdf?sfvrsn=b2ecb394_1)：打印页10、13；表示形式不限于高斯，准确性和最终评审不是网格单项比赛。
- [PGSR论文](https://arxiv.org/html/2406.06521v2)、[参数代码](https://raw.githubusercontent.com/zju3dv/PGSR/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/arguments/__init__.py)、[render代码](https://raw.githubusercontent.com/zju3dv/PGSR/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/render.py)：多视角表面约束、坐标尺度、曝光与GT依赖分离。
- [2DGS论文](https://arxiv.org/html/2403.17888v2)、[参数代码](https://raw.githubusercontent.com/hbb1/2d-gaussian-splatting/f3e3b9fa67bbd1c75e05167ff37391d8dab2a678/arguments/__init__.py)：表面表示及默认lambda_dist=0。
- [GOF论文](https://arxiv.org/html/2404.10772v2)：方法、增密消融与反射局限。
- [SOF论文](https://arxiv.org/html/2506.19139v2)、[官方NVS脚本](https://raw.githubusercontent.com/r4dl/SOF/main/scripts/run_nvs.py)：表1／3与三项loss=0；main按研究日核对，采用时重新固定文件。
- [Zip-NeRF作者实现](https://github.com/jonbarron/camp_zipnerf)：测试与外观视频配置的区别。
- [空间先验高斯论文](https://www.sciencedirect.com/science/article/pii/S0952197625014472)、[ISENet](https://www.sciencedirect.com/science/article/pii/S0926580525004054)：仅核对可访问一手摘要；正文受限，未宣称完整实现审计。

## 建议的下一轮边界

正式冻结的RGB030-FULL＋SEM161和SEM329研究候选保持不变。本轮仅提出候选，不改变技术决定或启动新实验，因此未新增ADR或改写既有RISKS。新候选需另行登记，在统一训练输入和官方相机接口下完成充分训练，再冻结RGB、重建语义，分别评价两任务及固定困难区域。至少为有竞争力的最终候选规划多seed；既有74开发集保留历史复用身份，不改称盲测。

剩余风险：候选本桥成绩、完整依赖许可、GPU扩展兼容、二维标签与物理表面不一致、未观测结构覆盖和隐藏泛化均未解决。当前不存在可承诺的得分提升，也没有保证获奖的方案。

## 验证与回退

- PDF初版发现相邻Markdown链接解析吞并问题，已修复构建器并重建；最终所有正文行都能在提取文本中找到，24个链接注释无缺失或意外URL。
- 8页逐页视觉检查完成；链接修正后的4／7／8页重新渲染复检，未见文字裁切、遮挡或缺字，正文与来源链接完整。
- `bash scripts/repo_health.sh`通过：上传安全、秘密扫描、Markdown链接、138份配置、Shell语法及273项单元测试；其中53项因可选依赖跳过。当前默认数据路径不存在，数据检查按既定逻辑跳过，不声称重新验证数据集。
- `git diff --check`通过；registry中本研究ID恰好一条。既有大量未提交修改保持原样，本次仅追加研究记录和报告入口。
- 无冻结模型、源码、配置或预测产物被修改，无Git提交或推送。若不采纳调研建议，继续既有冻结方案即可，无模型回滚操作。

## 2026-09-08追问：有没有具体的一体化解决方案

用户追问“更可靠重建＋重新适配语义专家＋跨视角复核”的具体实现。本节补充公开方案筛查与组合判断；原8页PDF、规范内容源及封存config不改，本节不属于原PDF内容。研究ID／manifest／split／seed不适用属性沿用本记录，没有新训练、安装、GT像素读取或质量评分，也不改变冻结决定。

### 可组装验证的主线

建议先对照验证PGSR或SOF，再冻结胜出的RGB、重建五类语义关联，并在该渲染域重训已有SegFormer-B2专家；随后重新生成RoMaV2来源证据，逐项验证来源可达性、双向对应、极线、遮挡及第二源一致性，保留无可靠证据时不改标签的能力。这是基于已有实现提出的集成方案，不是已经发布或跑通的端到端现成方法。新骨干不能直接加载Scaffold的anchor sidecar或旧匹配缓存。

[RoMaV2官方实现](https://github.com/Parskatt/RoMaV2)提供稠密匹配、双向设置与不确定性相关输出，但不是五类语义分类器，也不证明源标签可靠。SEM320证明本机特定实现可执行；SEM324明确反对无条件复制标签；SEM329只支持保守foundation复核，不能外推为已解决deck／stay_cable。若扩大类别或加入更强遮挡审计，必须作为新的独立变量。

专家容量不必停在B2：[Mask2Former官方模型库](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md)有Swin-L语义模型及权重，可作为后续五类渲染域监督候选，而不是已知本桥更优的替代。先用同类专家分离重建变化，再另做容量／结构比较。其[官方仓库](https://github.com/facebookresearch/Mask2Former)已归档，当前环境、预训练权重和相关许可尚未验证。

### GAGS：更直接回应跨视角语义不一致的表示候选

[GAGS论文](https://arxiv.org/html/2412.13654v2)通过深度／距离相关的SAM粒度和可学习粒度选择，减少不同视角特征监督冲突；其输出是可从新视角渲染的语义特征场，不是本比赛的现成五分类器。作者也指出复杂结构的部件级区分可能损失信息，因而不能保证桥面、塔柱和细索边界获益。

[官方仓库](https://github.com/WHU-USI3DV/GAGS)公开预训练3DGS之后冻结几何、蒸馏特征的流程；[训练代码](https://raw.githubusercontent.com/WHU-USI3DV/GAGS/main/train.py)确有特征／粒度解码器和feature_mode下不进行增密的分支。本轮未完整审计所有参数是否保持RGB逐字节不变；新骨干适配后必须另测。[Feature 3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs)可作为多通道特征渲染基础参照。[Gaussian Grouping](https://github.com/lkeab/gaussian-grouping)有身份编码、SAM掩码监督及空间一致性实现，可作分组基线；不把实例身份等同于本比赛五类标签。

本桥的拟议扩展是“新骨干＋借鉴GAGS的多粒度特征蒸馏＋官方五类标签监督的解码器／二维专家”，再研究与参考源复核的互补性。应明确标注为我们需要开发和验证的变体，而非GAGS原论文结果；不能用SAM或CLIP伪标签覆盖官方标签。它与SEM230的offset logits残差、SEM311的冻结二维特征不是同一机制，但也不因名称不同就默认不会重复语义混合失败。

### SPC-GS：最接近联合目标的参考，不是即插即用候选

[SPC-GS论文](https://arxiv.org/html/2503.12535)结合场景布局初始化、视频生成补充视角及SAM2提示约束训练视角和伪视角语义，联合处理重建与语义。其主要证据来自室内稀疏输入；对本桥采用生成影像补全几何存在不可验证结构的风险，此判断为场景适配推断。

截至本次访问，[项目页](https://gbliao.github.io/SPC-GS.github.io/)仍标示代码待发布；[官方仓库](https://github.com/gbliao/SPC-GS)可见README、assets及预处理数据／预训练模型链接，但未看到完整训练实现。不能声称已经验证可复现整套训练。[SemanticSplat](https://semanticsplat.github.io/)同样联合几何／外观／语义，但属于稀疏输入前馈方法，项目页的Code链接返回自身页面；本轮未确认可用作者训练代码，因此不把它列为当前已可落地整套替代。

### 结论与停止条件

当前最合理的具体主线是PGSR／SOF对照后的重建＋重训渲染域专家＋重建RoMaV2保守复核。GAGS是优先补充的三维语义表示研究参考；SPC-GS提供联合一致性思想，但不足以支持直接整套迁移。所有组合均尚未在本桥运行；是否有得分增益、创新性与跨项目泛化需要实测及更细的近邻方法比较。关键机制、可用实现和适配反例已有一手来源，停止泛化检索，未将资料筛查写成实验完成。

本次仅追加本节、登记追问及主入口；模型与既有运行产物不变，无提交或推送。复核仍使用仓库health与diff检查，不用文档检查替代GPU兼容或模型验证。
