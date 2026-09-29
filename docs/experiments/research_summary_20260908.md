# 阶段性研究总结：从冻结基线到第二参考源复核

> 历史快照说明：本文只总结截至2026-09-08的RGB030＋SEM161／SEM329阶段，其中“当前”“正式”等用语均指当时状态。2026-09-21的全量质量候选、255／256空间确认模型及其数据角色以仓库根目录[PROJECT_STATUS](../../PROJECT_STATUS.md)和[README](../../README.md)为准。

更新日期：2026-09-08；实验截止：SEM-329-SECOND-SOURCE-VETO-FINAL。本页是本阶段的总结入口，汇总已完成实验及其证据。日常任务看[项目状态](../../PROJECT_STATUS.md)，精确定义看[指标文档](../../METRICS.md)，配置、seed、Git来源及原始结果索引看[实验登记表](../../experiments/registry.yaml)。

可视化入口：中文离线网页报告（来源记录：`../../outputs/research_report_20260908/index.html`；未随公开包提供）、8页PDF摘要（来源记录：`../../outputs/research_report_20260908/research-summary.pdf`；未随公开包提供）、完整压缩包（来源记录：`../../outputs/research_report_20260908.zip`；未随公开包提供）。包含十组原生对比、最新RGB与SEM329语义、SEM161/326/329同轨迹视频，以及失败/阻塞/计划在内的146项登记。解压后直接打开`index.html`，不依赖网络或原目录。输出按仓库规则不入Git。

[VIS006记录](../../experiments/runs/VIS-006-SEM329-FLYTHROUGH/report.md)：复用VIS001的144个相机、24fps、6秒，补齐真实推理；三代表fresh重放、144帧RGB/保护类、18项冻结hash及离线/视频解码检查通过。新视角没有GT，不新增质量分数、不改变SEM161正式身份；报告保留真实模糊、浮点和帧间变化。

当前有两个需要区分的成果：正式可搬迁方案是 **RGB-030-FULL + SEM161**；在同74张开发图的最新三臂比较中，**SEM329取得最高整体mIoU和foundation IoU，并比SEM326减少误删**。SEM326保留为foundation precision略高的选择。新研究分支的工程探针已经通过，提交包尚未集成该分支。

## 1. 已经建立的主线

RGB骨干统一比较后选择Scaffold-GS。后续语义模块沿用冻结RGB，先得到静态五类语义，再通过渲染域二维模型和类别专用分支处理薄结构及误报。该路线已经形成可接收官方COLMAP外部相机的RGB与语义输出接口，并完成同宿主GPU上的干净Ubuntu容器重放。来源：[RGB-060](../../experiments/runs/RGB-060/report.md)、[SUBMIT-030](../../experiments/runs/SUBMIT-030-SEM161-BUNDLE/report.md)、[SUBMIT-040](../../experiments/runs/SUBMIT-040-CLEAN-UBUNTU-REPRO/report.md)。

以下是同一`sem_001_scaffold_static_v1`、74张开发图、含背景全局mIoU的历史进展。它记录整个组合的演进，各行存在训练配方及seed差异，不是贯穿全程的单变量消融：

- [SEM001](../../experiments/runs/SEM-001/report.md)：61.610304%，静态五类三维语义基底。
- [SEM111](../../experiments/runs/SEM-111-PROTECTED-FUSION-FINAL/report.md)：76.063477%，渲染域二维证据与类别保护融合。
- [SEM141](../../experiments/runs/SEM-141-CABLE-BINARY-FINAL/report.md)：77.489877%，原生分辨率`stay_cable`专用分支。
- [SEM151](../../experiments/runs/SEM-151-FOUNDATION-GATE-FINAL/report.md)：83.093383%，foundation单向审核。
- [SEM161](../../experiments/runs/SEM-161-DECK-BINARY-FINAL/report.md)：83.387017%，deck单向恢复，当前正式冻结包。
- [SEM329](../../experiments/runs/SEM-329-SECOND-SOURCE-VETO-FINAL/report.md)：83.705003%，在SEM161上增加稠密对应的foundation复核及第二参考源否决，当前优先研究候选。

正式推理顺序为：官方相机 → 冻结Scaffold RGB与SEM001三维语义 → 渲染域二维语义 → SEM141拉索分支 → SEM151基础审核 → SEM161桥面恢复。SEM329在最后增加两步：SEM326依据首个可靠参考源删除错误foundation；SEM328固定的第二源规则在有可靠foundation证据时否决部分删除。第二源不存在或无可靠像素证据时保持SEM326结果。

## 2. 研究方向得到了什么结论

**三维监督与结构清理。** 贡献筛选、冲突正则、视角残差、anchor清理和Hough结构模板都留下了实现与诊断证据。部分策略降低误报，却同时损伤真实结构，未形成保留方案。索的同源错误会在多视角中重复出现，跨视角一致本身不能证明标签正确。来源：[SEM002阶段报告](../../experiments/runs/SEM-002/report.md)、[SEM003](../../experiments/runs/SEM-003/report.md)、[SEM004](../../experiments/runs/SEM-004/report.md)、[SEM130](../../experiments/runs/SEM-130-ANCHOR-CROSSVIEW-CONSENSUS-OOF/report.md)。

**渲染域训练和类别专家。** SEM100–111证明渲染RGB仍提供可用辨识信息，也暴露真实图像与渲染图像之间的域差。专用拉索、基础审核和桥面恢复随后形成SEM141/151/161，是本阶段累计质量提升的主要组成部分。各分支的作用、局部退化和冻结参数保留在其独立报告中。

**几何、桥面表示与边界训练。** SEM200–263以及SEM270–300覆盖几何投影、XYZ/RGBD、offset责任、稀疏轨迹、点级细化和foundation边界损失。它们提供了局部正面消融或误差定位，但当前配方没有替换SEM161。尤其SEM250稀疏点覆盖不足不能外推为稠密对应无效；SEM230的旧证据不足以支持纯粒度因果解释，SEM250“未读check”的旧说法也已有勘误。来源：[ADR-053/057及后续决定](../../DECISIONS.md)、[SEM263](../../experiments/runs/SEM-263-NESTED-POINT-OOF/report.md)、[SEM300](../../experiments/runs/SEM-300-FOUNDATION-SIGNED-DISTANCE-OOF/report.md)。

**RGB侧改进。** MapAnything/DA3初始化以及后续瞬态、环境建模等尝试已有记录，当前没有候选替代RGB-030-FULL。匹配控制上的改善与超过冻结RGB的收益是两个不同问题；困难视角和原生样例用于保留局部代价。来源：[GEO-010-10K](../../experiments/runs/GEO-010-10K/report.md)、[GEO-020-3K-INT-R1](../../experiments/runs/GEO-020-3K-INT-R1/report.md)、[RGB092](../../experiments/runs/RGB-092-TRANSIENT/report.md)、[RGB101](../../experiments/runs/RGB-101-ENVIRONMENT/report.md)。

## 3. 最近SEM310–329：从特征选择到可验证纠错

各项均有独立配置与报告；下列百分比差为百分点（pp）。OOF指按空间折生成训练侧留出预测；本项目复用了历史calibration/check，解释为增量开发证据。

1. [SEM310](../../experiments/runs/SEM-310-DINOV3-FROZEN-FEATURE-OOF/report.md)：DINOv3架构与接口冒烟完成，独立HF权重申请被拒，正式预训练质量实验阻塞。不能由此判断DINOv3效果差。
2. [SEM311](../../experiments/runs/SEM-311-DINOV2-FROZEN-FEATURE-OOF/report.md)：公开DINOv2-B/14对冻结MiT控制，在复用108张calibration上mIoU/foundation IoU提高2.038/9.878pp；但相对SEM160降低2.653/12.947pp。特征来源有优势，完整0/4替换配方未保留。
3. [SEM312](../../experiments/runs/SEM-312-DINOV2-COMPLEMENTARITY-DIAG/report.md)与[SEM313](../../experiments/runs/SEM-313-DINOV2-FOUNDATION-REVIEW-OOF/report.md)：诊断支持只审核旧foundation。固定复核在111张check上相对SEM160 mIoU/foundation IoU提高0.811/3.980pp，recall下降1.003pp；同轮MiT对照没有相同收益。
4. [SEM314](../../experiments/runs/SEM-314-DINOV2-FOUNDATION-REVIEW-FINAL/report.md)：全219训练后的最终DINO复核，在74开发图上相对SEM161 mIoU只提高0.087080pp、foundation IoU提高0.437443pp，recall下降3.260272pp；169–175区域foundation IoU下降11.369pp。OOF正信号未完整迁移到最终组合。
5. [SEM320](../../experiments/runs/SEM-320-ROMAV2-DENSE-PREFLIGHT/report.md)与[SEM321](../../experiments/runs/SEM-321-ROMAV2-DENSE-GEOMETRY-DIAG/report.md)：RoMa v2完整官方权重已独立验证并实际运行。最初三组几何筛选只有一组同时通过覆盖与几何条件；源访问措辞偏差已登记。该完整权重运行不再受SEM310的独立HF申请阻塞。
6. [SEM322](../../experiments/runs/SEM-322-POSE-SOURCE-ACCESS/report.md)、[SEM323](../../experiments/runs/SEM-323-POSE-SOURCE-EXPANSION/report.md)、[SEM324](../../experiments/runs/SEM-324-DENSE-LABEL-AGREEMENT/report.md)：固定pose-top3选源扩展至105目标，91个可达。源标签一致率94.086%仍不足以直接整图替换；foundation4→background0的定向纠错更有价值。SEM324为512网格诊断，不把采样计数当原生图像mIoU。
7. [SEM325](../../experiments/runs/SEM-325-DENSE-FOUNDATION-REVIEW-OOF/report.md)与[SEM326](../../experiments/runs/SEM-326-DENSE-FOUNDATION-REVIEW-FINAL/report.md)：真实原生foundation复核形成216张OOF输出，111 check相对SEM160 mIoU/foundation IoU提高0.812045/3.969178pp；最终全219参考库+SEM161的74开发对比仍有提高，详见下节。新增真实foundation误删成为明确代价。
8. [SEM327](../../experiments/runs/SEM-327-SOURCE-BOUNDARY-DIAG/report.md)：固定2px源边界保护诊断可救回81.343%误删，却放弃41.443%正确纠错，较dense的foundation IoU下降1.451937pp。保留低误删折中，未将其叠加到下一方案。
9. [SEM328](../../experiments/runs/SEM-328-SECOND-SOURCE-VETO-OOF/report.md)：第二个可靠源明确预测foundation时否决删除。111 check相对SEM325 mIoU/foundation IoU再提高0.009221/0.047017pp，recall提高0.423304pp，precision下降0.177683pp。三折中一折IoU微降，原始正确像素净减少1039，保留此代价。
10. [SEM329](../../experiments/runs/SEM-329-SECOND-SOURCE-VETO-FINAL/report.md)：固定上述规则完成全219源库+SEM161最终三臂比较，取得小幅IoU与召回增益，成为后续研究优先起点。

## 4. 最新同口径三臂对比

来源为SEM329封存评分及独立复算：相同74张开发图、1320×989 PINHOLE语义PNG、GT为行的累计5×5混淆矩阵，mIoU为含背景五类平均。数据划分`sem_001_scaffold_static_v1`：219训练参考源、74开发、7隔离、100无标注；seed20260805。开发图已历史复用。

| 方案 | mIoU（%） | foundation IoU（%） | foundation precision（%） | foundation recall（%） |
|---|---:|---:|---:|---:|
| SEM161 正式基线 | 83.387017 | 72.770345 | 78.275843 | 91.186564 |
| SEM326 单参考源复核 | 83.682675 | 74.242876 | 81.475684 | 89.319965 |
| SEM329 第二源否决 | 83.705003 | 74.354428 | 81.244656 | 89.761802 |

相对SEM161，SEM329的mIoU/foundation IoU提高 **0.317985/1.584083pp**，precision提高2.968813pp，recall下降1.424762pp。相对SEM326，再增加 **0.022328/0.111552pp**，recall恢复0.441838pp，precision下降0.231028pp。

第二源恢复1431个旧删除像素：739个是真foundation，672个是真背景，20个是真其它类。它减少SEM326误删的23.670724%，保留原正确纠错的91.866376%，原始正确像素净增67。相对SEM161，最终仍纠正7590个像素、损伤2383个真foundation、80个仍错，净正确像素增加5207。错误类别互换不能算完全纠正。

整体增益之外：四个原有空间组相对SEM161的mIoU/foundation IoU均升；相对SEM326三组升、一组微降。seg-010的recall恢复3.578815pp后仍为83.634565%，比SEM161低10.403741pp。逐视角mIoU相对SEM326是22升、17降、35平，说明改进并不均匀。

固定140–146的**逐视角含背景mIoU均值**较SEM161提高0.953033pp，较SEM326仅再提高0.002042pp；按SEM161排名固定的最差8张没有新增收益。早期实验的“worst10%”有不含背景等不同定义，不能跨定义直接拼接。

历史内部+0.30/+1.50pp门在SEM329均满足，90%recall门仍未满足。这些原门的数值保留用于追溯；候选研究价值按实际收益和代价判断，既不因小门差否定全部增益，也不因总体提升忽略局部漏检。

## 5. 工程证据与当前边界

SEM329的74份输出独立重构一致，三臂逐图/整体/分组/固定子集指标独立复算一致。4次实际新进程外相机测试（原尺寸两次、小尺寸一次、局部新位姿一次）均通过，RGB、class1/2/3和18项冻结产物保持一致。固定六例已保存和检查。小尺寸测试是零改动接口检查，新位姿只覆盖一个局部扰动；二者不能替代有GT的质量评估。

本轮74张第二源预测利用现有匹配缓存耗时2.916秒，4次fresh外相机测试总59.753秒，峰值CUDA allocated约2508.446MiB。这些是各阶段实测，前者不含原匹配成本，后者也不是部署单相机延迟承诺。来源：[SEM329报告](../../experiments/runs/SEM-329-SECOND-SOURCE-VETO-FINAL/report.md)。

仍待处理的实际问题是：第二源未覆盖的旧误删、可靠几何下的语义冲突、seg-010召回，以及整套源RGB/源GT/匹配权重的部署成本。当前规则只能救回自己的删除，不能补回SEM161原本漏掉的foundation。尚无统计显著性、全链独立盲测或官方隐藏成绩结论。

## 6. 记录覆盖、来源与版本

整理前审计：登记表145条（含计划、失败和阻塞），135个实验目录都有报告；SEM310–314、SEM320–329这15项都有配置、报告、Git来源、seed与划分。SEM002是S1的总报告，实际子实验以SEM020/021/022等登记；三个前置失败项目明确未创建配置。早期E0/E1/E2/E3/E4B2/E4B3来自用户历史摘要，原始配置/产物标注`not_available`，保留参考身份。

最新三臂的共同来源为Git基点`49cdcdb1c9915a9544d568754af95ea20a85ec9c`加每阶段封存的工作树源码；配置[SEM329 config](../../experiments/runs/SEM-329-SECOND-SOURCE-VETO-FINAL/config.yaml)的SHA256为`f1b41fbbfd814ef611ac7cc362acb7e157c7d85396ed7988f12f27db98c43584`。数据manifest为[data/manifests/contest_dataset_v1.json](../../data/manifests/contest_dataset_v1.json)，SHA256 `0a188d4b4cfa5ffc2a7f151018f1cb012782e8f9860a3338aed084478d675b3e`；[最终划分](../../experiments/splits/sem_001_scaffold_static_v1.json)SHA256 `1764d3281fedc94e03c85f2ffc272e324bb12c0d8dba1dc1955078f71d147854`。OOF记录沿用`sem_005_oof_spatial_v1`，具体calibration/check目标清单以各ID的config为准。

历史主线的seed分别为SEM001=20260731、SEM111=20260902、SEM141/161=20260805；SEM151为无新训练的确定性冻结概率门。SEM111之后的上述记录基点为49cdcdb并以各报告源码快照补足，SEM001在登记表中明确为未提交工作树来源。每个ID的`config`、`report_path`及输出清单由登记表关联，不能只凭当前HEAD复现整个历史。

SEM329五个结果清单位于被忽略的`outputs/sem329/`：`developer-predict-20260908-r1`、`developer-audit-20260908-r1`、`external-20260908-r2`、`developer-score-20260908-r1`、`quality-audit-20260908-r1`。各自`result.json`的SHA256在登记表与实验报告中保存。本次汇总核对这些清单；其中主评分SHA256为`1fa850cce879345bd9c323be96ae840431d775cb8070418772608053c88792c1`，独立质量复算SHA256为`8a320276936abd0b61ebdc68df233f426b4e7a2f0da27724da6d123e3b69de84`。实际PNG、权重、日志继续留在各运行目录。

失败也有记录，例如SEM329外相机r1的小尺寸输出读取错误及r2修复。正式冻结配置保留`preregistered`原值，其运行完成状态从报告和登记表读取；不应为了文案统一修改已封存配置。

本次为文档整理，没有新增实验或重新评分。新文档和大量既有研究改动尚未提交Git；模型/数据/完整输出依项目规则不进入Git，需独立保管。源码/文档版本归档与输出备份应分别核对。整理前的[状态全文](../history/project_status_pre_summary_20260908.md)及[路线全文](../history/roadmap_pre_summary_20260908.md)均保留，旧“当前”措辞只代表对应历史阶段。

整理核对：两份历史存档与整理前正文一致（仅调整相对链接）；489份实验源码/测试/配置等文件的hash前后一致。本文与METRICS的两张三臂表、五项历史主线mIoU及SEM329五份结果清单均核对一致，登记项仍为145条。仓库健康检查通过136份JSON兼容YAML、273项测试（系统Python下53项可选依赖跳过）；本地文档链接及差异空白检查通过。核对不重新读取目标GT或运行模型。

## 7. 下一阶段的三个事项

1. 以SEM329作研究起点，若继续实验，另登记训练侧剩余误删诊断，先区分第二源缺少覆盖与对应标签冲突，再选择一个推理时可用的新变量。当前尚未执行该诊断。
2. 方法选定后，另登记全400 RGB/300标签拟合、依赖重建及提交包复现；原74分数届时作为开发历史保留。
3. 将本文作为阶段讨论底稿，后续按需展开论文、视频和PPT；当前先完成代码/文档版本与产物备份范围的确认。
