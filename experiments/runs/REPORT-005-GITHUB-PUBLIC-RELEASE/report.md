# REPORT005：公开研究仓库与在线论文素材

2026-09-29 已完成公开发布与匿名访问验收。父记录 REPORT004；科学结果截至 2026-09-23，本任务未训练模型、推理或重新评分。

## 阅读与下载入口

- [公开仓库与 README](https://github.com/samtu648-hub/icshm2026-semantic-gs)
- [在线可视化](https://samtu648-hub.github.io/icshm2026-semantic-gs/)
- [作者必读](https://github.com/samtu648-hub/icshm2026-semantic-gs/blob/main/docs/paper/author_guide.md)
- [首个 Release 与素材下载](https://github.com/samtu648-hub/icshm2026-semantic-gs/releases/tag/paper-handoff-20260929-v1)
- [发布与在线验收记录](https://github.com/samtu648-hub/icshm2026-semantic-gs/blob/main/publication/verification.json)
- [逐文件来源与加工](https://github.com/samtu648-hub/icshm2026-semantic-gs/blob/main/publication/source_manifest.json)

Release tag：`paper-handoff-20260929-v1`；对应公开提交：`1ffbcfb2d2a825e6a7fe3c02721afeff2ee717e6`。后续 main 提交仅补齐验收与发布记录，不移动该标签。

附件 `paper-handoff-20260929-public.zip` 为 82,221,694 bytes，SHA256：`4dbb424cb9e0abf189375463e3852f553f6636fca3eec35611d983b7bae219ba`。仓库 `publication/release.json` 和 Release 同名 `.sha256` 固定这一摘要；ZIP 内部验收收据是在最后封装前写入，不能用其候选包摘要替代最终附件摘要。

## 公开内容与证据边界

包含完整可发布源码、训练／评价／诊断脚本、测试、配置、实验报告、必要 manifest／split 元数据、12 章作者指南、开发历程、结果表、失败分析、中英图注及第三方来源。10 组图、44 份原尺寸素材和 6 类 8 段视频由原交接包复用；全部选定媒体逐字节一致。Git 中只保存小型论文图；完整媒体置于 Release 和 Pages。

历史 351 条登记已封存为 `scripts/report_assets/paper_handoff_registry_20260923.json`，与新增 REPORT004／005 分开；不是 351 次已完成实验。不同评价角色不拼接提升曲线。RGB170＋SEM437 是已有全量展示，RGB166 是未来未知视角推荐 RGB 底座，其匹配最终语义尚未完成。SEM001→SEM161 的历史显著提升与容量留出退化、357 桥面漏检等反例均保留。

组员 SHM26P2 查阅版本固定为 `dceb0e67faa6b4c0fd6fba754f15c713c5cbf721`。对照说明区分实际组件、教师／渲染输入、划分和成本；未独立复现对方运行，不直接排名异协议分数，也未融合模型。

公开副本采用新 Git 历史。原研究来源 commit `49cdcdb1c9915a9544d568754af95ea20a85ec9c` 与当前未提交源码分别记录，不能把公开首次提交冒充历史实验版本。18 个脚本在副本中改为 `--artifact-root`、`ICSHM_OUTPUT_ROOT`、项目 outputs 默认的路径优先级；不更改训练算法或配方。个人路径替换记入来源清单。保留第三方许可及版本说明，没有添加覆盖全部内容的新许可。

原始照片集、LabelMe 标注、完整 COLMAP、权重、PLY、运行日志、缓存、第三方整仓和官方 Word 模板未上传。精选照片／渲染媒体属于本次明确授权的例外。原私有仓库仍为 private，原历史未推送到新仓库；旧离线材料和模型保留。

## 验收结果

- 原工作区健康检查：715 项测试，100 项可选依赖跳过，无失败；公开副本、本地干净克隆、公网匿名干净克隆：715 项测试，103 项跳过，无失败。新增 3 项跳过明确依赖未分发的历史运行产物，不把它们算作已验证的模型训练。
- 实际公开源码和附件的发布审计通过，覆盖逐文件 SHA256、Python 语法、个人路径、敏感字段、禁止资产、体积和科学指标一致性。最终清单见 `publication/source_manifest.json`，不是原私有仓库的完整历史快照。
- 公开下载附件：1,177 个内容文件及 manifest 校验成功；结构、数值、链接和完整视频解码 7,911 项通过。70 张图片无 EXIF；44 份独立素材包含 42 PNG 与 2 JPG。
- 中文及空格新目录解压后断网浏览器：56 项通过。匿名线上浏览器：62 项通过，包括 GitHub README／作者文档／两方案说明／Release、三种视口、实验查询、图片加载、8 段视频播放与跳转；无脚本错误和站内失败响应。
- GitHub Actions [运行 36527815312](https://github.com/samtu648-hub/icshm2026-semantic-gs/actions/runs/36527815312) 已成功部署。构建只下载固定 Release、验证总 SHA256 与每文件摘要，无 GPU、比赛数据或研究目录依赖；支持 Release 和手动部署。
- 本轮开始的 1,922 份源文件中，原有 1,918 份逐字节保持。4 份必要变化为冻结历史索引的构建脚本、登记表、任务和项目状态；新增发布材料单独列出。原实验算法文件没有被本次公开加工修改。

## 失败与修复记录

公开测试初检的 3 个本地产物依赖改为有理由的可选跳过；2 个 Git 来源测试在创建独立首次提交后通过。发布前修正了路径清理对引号的处理、PNG/JPG 计数及全局忽略文件对导出清单的影响。SSH 推送未响应，改用已有 GitHub CLI 凭据的 HTTPS 推送成功。

首次 Pages deploy 被新建环境默认仅允许 main 的规则拒绝；只添加精确标签 `paper-handoff-20260929-v1` 的权限后，重试成功。匿名浏览器的首轮词汇表读取和第二轮固定 200ms 播放断言早于网络就绪；改为等待 DOM、实际解码播放和 seek 完成后，62 项验收通过。保留失败日志，不将这些发布／验收问题记成模型实验结论。

## 使用与后续修订

写作者先读 README 与作者必读，再按论点核对图号、实验 ID、划分及来源。缺失的历史报告字段仍显式登记；不宣称从公开仓库已经复现完整训练。后续科学实验和正式比赛提交另立任务。

若发现公开内容问题，以新提交／新 Release 版本修复；网站可通过修订 `publication/release.json` 指向已验证附件后手动部署。保留现有 tag、附件和历史，不强推、不删除原材料。原私有研究目录继续作为内部回查与后续工作的来源。
