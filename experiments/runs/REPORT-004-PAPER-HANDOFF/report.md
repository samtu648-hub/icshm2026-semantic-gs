# REPORT-004：开发历程与论文素材交接包

## v2：以作者充分理解项目为交接目标

2026-09-29，根据用户“负责写文章的要充分理解我们的项目”的要求，增加独立[作者必读网页](https://samtu648-hub.github.io/icshm2026-semantic-gs/author_guide.html)与[可编辑理解文档](https://samtu648-hub.github.io/icshm2026-semantic-gs/author_guide.md)。当前完整ZIP（来源记录：`../../../outputs/paper_handoff_20260929.zip`；未随公开包提供）为82274477 bytes，SHA256 `85ec3d8a389a6679ca0e39c749d896c712a785aec6610c7bf426e440b3a2762b`；内含1175个内容文件（不含manifest自身）。

12章覆盖任务与外部相机输入、数据角色、冻结RGB、三条系统身份、三维到二维专家、Mask2Former、空间验证、失败假设、容量取舍、指标读图、贡献局限和写作复述。每章包含解释性正文、图／视频链接、实验来源、应掌握的结论、自测及参考回答；另有16项术语和三遍阅读路线。方法链、输入域、已完成状态和评价边界均与保存报告／配置对应，没有加入新实验结论或正式英文论文。

文稿源为`scripts/report_assets/paper_handoff_author_guide.json`，构建器生成网页与Markdown；原报告、素材、索引继续可用。首页及论文底稿优先链接作者文档。新纳入随包的接口文档解释外部相机约定。构建器重跑会重置旧验收回执，避免新版沿用旧版“通过”的声明。

验证：来源／数字／链接／配色／完整视频解码9028项通过；真实file://断网Chrome56项通过，新增作者文档目录、桌面／移动端、12章、16术语、自测展开及返回图组验证；零外网请求与浏览器错误。ZIP完整检查后在中文及空格新目录解压，7897项文件／链接／数字检查和56项浏览器检查通过。人工查看作者首页、方法章和容量章的桌面／手机截图，文字、图像与布局正常。44份native素材及8段视频与v1逐字节相同。验收记录位于`outputs/paper_handoff_20260929_author_qa/`。

包中来源哈希记录构建时状态；之后仅更新REPORT004自身的版本、验收计数与ZIP摘要，原351条科学登记不变，避免将自引用ZIP摘要当成构建前来源。未训练、推理、评分、修改模型、Git提交或上传。作者是否真正掌握需由阅读与复述反馈确认；本次交付提供相应材料和自测，不把文档完成等同于人的理解已完成。

v2收尾：`bash scripts/repo_health.sh`通过，710项测试、100项可选依赖跳过；可选本地比赛数据未提供而跳过。开始时2738份既有源文件作哈希快照，最终仅10份本任务明确修改的构建／验证／入口／登记／状态文件变化，其余保持。351条原科学登记逐对象完全相等；ZIP摘要复核一致，`git diff --check`通过。见`outputs/paper_handoff_20260929_author_qa/closeout.json`与`repo-health-final.log`。

回退：原版网页（来源记录：`../../../outputs/paper_handoff_20260929_versions/v1/paper_handoff_20260929/index.html`；未随公开包提供）和原版ZIP（来源记录：`../../../outputs/paper_handoff_20260929_versions/v1/paper_handoff_20260929.zip`；未随公开包提供）完整保留。只需改回旧入口，不涉及任何模型回退。下列初版记录的数值与旧ZIP哈希仅描述v1。

## v1 初版验收记录

2026-09-29：离线材料及最终ZIP异路径验收完成；仓库健康检查见文末收尾记录。

## 交付与范围

本任务按用户确认的计划整理截至2026-09-23的既有证据，没有训练、模型推理、重新评分、模型选型、上传或Git提交。父记录为REPORT003与QUAL026；数据manifest为`contest_dataset_v1`，材料分别标注历史74、OOF、空间v1/v2、全300/全400训练拟合、无标注100和无GT轨迹144。整理任务seed为null，各科学结果的seed、commit、配置、父实验、split和源哈希在独立索引中。

- [离线网页](https://samtu648-hub.github.io/icshm2026-semantic-gs/index.html)
- 完整ZIP（来源记录：`../../../outputs/paper_handoff_20260929_versions/v1/paper_handoff_20260929.zip`；未随公开包提供）：82,224,780 bytes，SHA256 `450fa411966edff2fdd76aae46fb60d3f448fe45293c74e27ef61754ada415cf`。
- [开发历程](https://samtu648-hub.github.io/icshm2026-semantic-gs/development_history.md)：12个阶段，按问题、尝试、结果、停止或保留理由组织。
- [论文底稿](https://samtu648-hub.github.io/icshm2026-semantic-gs/paper_notes.md)：8组可改写中文段落及写作边界。
- [结果表](https://samtu648-hub.github.io/icshm2026-semantic-gs/results_summary.md)：9组、72行，含逐类、逐seed、逐区域、训练长度及成本。
- [中英图注](https://samtu648-hub.github.io/icshm2026-semantic-gs/captions.md)：10组图、6类8段视频；44份原图/原尺寸素材、26份图表/组合图。统计图同时提供SVG/PNG。
- [完整实验索引](https://samtu648-hub.github.io/icshm2026-semantic-gs/data/experiment_index.json)：351条历史登记；REPORT004不计入历史分母。342个原实验报告目录；随包快照按登记ID展开，存在共用来源，不能将复制文件数量当独立实验数。

包内manifest记录1171个内容文件，94,837,961 bytes，不含manifest自身；没有外部字体、CDN、远程脚本或运行时服务依赖。

## 科学与材料口径

1. SEM001→SEM161的61.610304%→83.387017%是同历史74图上的系统演进，不是单变量消融；与SEM437全300拟合95.678649%分开。
2. SEM350的B2融合版80.917724%与REPORT002的raw B2 75.324650%明确区分。Mask2Former raw为86.513086%，保留057等退化例和训练预算/RGB底座差异。
3. v1/v2各两个语义seed，RGB仅seed0；跨区域差异、召回代价、v2成本与历史开发暴露均保留。逐区域表可直接查阅。
4. RGB170＋SEM437仅作为已有全量展示；ADR175推荐未来未知视角使用RGB166预算配方，其匹配最终语义尚未训练。RGB175/176才是QUAL026的配对空间模型；全量RGB166/170只作图像语境。
5. 图片使用固定五类配色，语义图仅从原class ID或原固定色表导出；原尺寸语义像素逐项复核。视频保留原配色，分别标注早期/后期图例。没有生成式修补或美化。
6. V03只把已保存的144帧30k/50k按完全相同文件名并排编码，12fps；没有重渲染。V05按301—400编号播放，不称连续时间采集或未见测试集。

## 来源与缺项

报告、配置和关键结果JSON随包保存，个人路径替换为`$PROJECT_ROOT`/`$USER_HOME`；源文件hash保留。不可随包的大型运行产物在历史报告中改为明确的来源文字，避免失效的本地超链接。

RGB132—135原登记使用`config_path`，已找到并打包配置。SEM002是独立阶段汇总报告，补入材料但不新增历史登记分母。剩余16项缺项包括E0/E1/E2/E3/E4B2/E4B3的报告及配置共12项，以及SEM006、SEM061、SEM062、SEM002的4份未建立独立配置。不存在所选图片或视频缺失；不把这些历史缺项等同于模型质量失败。见[来源说明](https://samtu648-hub.github.io/icshm2026-semantic-gs/source_notes.md)。

## 验收证据

- 最终来源、数字、链接、配色、哈希及完整视频解码：8441项通过，见`outputs/paper_handoff_20260929_qa/audit-final-r2.json`。
- 独立Chrome真实`file://`断网浏览：43项通过，覆盖1440/1366/390宽度、搜索/状态筛选、证据详情、原图展开、八视频播放和跳转、文字放大；0网络请求、0浏览器错误。
- 最终ZIP完整校验并解压到含中文和空格的新目录；7315项离线文件/数字检查和43项真实浏览器检查再通过。解压版按字节hash复核，不依赖原仓库或原路径。
- 开始时1853份既有源码、测试、配置及文档逐项hash快照，制作过程无修改；根目录状态文件和登记表只追加本任务记录。科学配置、模型和旧汇报未覆盖。

失败尝试完整保留：首次构建发现SEM002为未单列登记阶段汇总、SEM005应引用SEM052资格审计；首次Chrome因沙箱进程通信限制未启动，独立断网Chrome在沙箱外成功执行；移动端初检发现长文本/grid溢出并修复；首次最终来源核验因软链接被解析成匿名化外部目录而报告2个路径失败，改为记录逻辑仓库路径后通过。上述均为材料工程失败，不修改模型指标。

## 复现与回退

构建器为`scripts/build_paper_handoff.py`，独立审计为`scripts/audit_paper_handoff.py`，浏览器检查为`scripts/check_paper_handoff.mjs`；文稿源为`scripts/report_assets/paper_handoff_content.json`。构建命令：

```bash
python3 scripts/build_paper_handoff.py
python3 scripts/audit_paper_handoff.py --package outputs/paper_handoff_20260929 --output outputs/paper_handoff_20260929_qa/audit.json --source-root . --decode-videos
```

包内保留构建前证据快照；任务完成后registry会新增REPORT004，旧351条科学登记不变。后续重构建会重新绑定当时源哈希。SHA256应以对应构建版本为准，不能用当前HEAD替代历史源码快照。

回退只需停止引用本交接包，继续使用旧REPORT003等材料；没有模型或训练代码回退操作。交接包不包含checkpoint、完整数据集或最终比赛提交包。剩余工作为伙伴选图和改写正文、核对定稿论点与模型角色；新训练与比赛提交均另行处理。

## 最终收尾

`bash scripts/repo_health.sh`通过：710项测试，100项可选依赖跳过；上传安全、秘密模式、Markdown链接、配置和shell语法检查通过。可选本地比赛数据目录未提供而跳过，该项不影响本次保存结果的交接。最后`git diff --check`通过。351条原登记与开始时逐项相等；新增REPORT004后登记总数352。1853份任务开始前已存在的源码/测试/配置/文档哈希全部保持；新文件仅为材料构建、审计、内容和交付记录。原ZIP及内容hash在收尾后再次核对一致。
