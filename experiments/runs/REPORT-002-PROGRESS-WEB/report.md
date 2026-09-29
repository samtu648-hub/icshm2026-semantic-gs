# REPORT-002-PROGRESS-WEB · 项目进展离线网页

2026-09-18。状态：completed_offline_presentation。本任务为结果展示，未训练、未调参、未改变模型晋级决定。

## 展示与来源

用户确认中文白底单页、六个里程碑：SEM001、SEM161、SEM331原始五类B2、SEM349原始五类Mask2Former、RGB148＋SEM380、RGB140＋SEM383。SEM381→382→383作为DINOv3研究分支从SEM378出发。SEM380称为全300训练拟合最高分，不称默认部署或泛化最佳；保留SEM386独立Dev30补充。

实验配置见[config.yaml](config.yaml)。各来源的实验ID、父实验、Git、配置、seed、manifest与split在离线来源快照中保留。主要manifest为`data/manifests/contest_dataset_v1.json`；历史`sem_001_scaffold_static_v1`74图、全量`released_all400_rgb_all300_sem_v1`及Dev30分开解释。原始实验与用户未提交修改保持。

网页位于离线入口（来源记录：`../../../outputs/project_progress_20260918/index.html`；未随公开包提供），打包文件为ZIP（来源记录：`../../../outputs/project_progress_20260918.zip`；未随公开包提供）。解压整个目录后双击index.html。

## 补做的可视化与指标

- 用户明确授权补齐冻结模型的展示渲染/推理，不允许新训练或调参；独立输出位于`outputs/progress_web_work/r1`。
- 冻结RGB110渲染144帧耗时54.623秒；已有26个轨迹视角的native/official RGB完全一致，0/71/143复渲染一致，模型artifact和RGB state签名未变。
- SEM331原始五类B2使用原3000步checkpoint，144帧推理44.315秒；保存054原生预测完全一致。SEM349使用原6000步checkpoint，144帧推理88.404秒；054预测及已有26轨迹概率完全一致。输入RGB哈希与正式18冻结产物全部保持。
- 74张保存B2原始预测逐项核验来源哈希，与封存GT按原协议独立双计数复算，五类mIoU为75.3246499139691%。这不是SEM350中`new_b2`融合分数80.917724%。未再次训练、改变输出或选择阈值。复算耗时1.188秒。
- 六段视频采用同一VIS001官方参考坐标系144帧，12fps/12秒；左右RGB/语义，2640×990 H.264，末行补齐偶数高度。视频统一官方SIMPLE_RADIAL展示网格；后期native图像采用既有RGB重畸变和最近邻类别重畸变。PNG与预测原件不改，视频是有损展示副本。
- 困难视角159/160/146/057使用SEM380原native照片/RGB/GT/预测；四图共享展示裁剪，057小基础沿用旧汇报框。96–98帧无GT，仅定性。类别2唯一正式名称为`stay_cable`。

## 工程与失败保留

[素材准备](../../../scripts/prepare_progress_media.py)、[构建打包](../../../scripts/build_progress_web.py)、[来源审计](../../../scripts/audit_progress_web.py)、[离线浏览器检查](../../../scripts/check_progress_web.mjs)。页面不依赖外网、服务器、框架、远程字体或CDN；JSON直接内嵌，所有媒体为相对路径。

第一次B2任务在读取054闭合帧时因展示脚本目录名缺少日期失败，没有完成新轨迹推理；修正路径后在b2-r2重新执行。第一次CPU复算因导入函数名称错误失败；修正为既有`independent_confusion`后在score-r2完成。原b2/score失败记录、日志和started/failure文件完整保留。首轮浏览器最后检查误把无src的空放大框判为未加载，39项已过；修正检查器后41项通过。首次打包相机校验把额外trajectory_manifest.json与两个已登记txt文件集合直接比较而失败，改为比较登记的两个键后通过；无相机/素材更改。首次repo_health因ZIP尚未生成的链接失败，生成后全量通过。没有修改冻结模块或实验配方。

## 验收

- 两轮真实断网file协议浏览器检查各41项通过：原目录与独立ZIP解压目录均零网络请求、零页面错误；六段144帧视频可解码播放、定位和全屏，键盘切换、四图同坐标裁剪、图片放大、来源往返全部通过。
- 1440×1000、1366×768、1920×1080与390×844无页面横向溢出；已检查桌面流程图、里程碑、指标表及手机困难视角截图。
- 两处静态审计通过：79个清单文件加清单自身共80文件、27个HTML本地引用、47项媒体资产；复制素材和全部视频输入哈希一致，历史B2与全量指标映射通过。全部六阶段相机元数据与VIS001封存哈希一致。
- 仓库健康检查通过576项测试（81项可选依赖跳过）；配置、上传安全、秘密扫描、Markdown链接、语法与git diff --check通过。
- ZIP约50MiB；包含独立HTML/CSS/JS、媒体、指标及21个来源快照，不含模型权重或训练数据全集。
- 验收证据：原目录浏览器（来源记录：`../../../outputs/progress_web_qa/r2/browser.json`；未随公开包提供）、搬迁浏览器（来源记录：`../../../outputs/progress_web_qa/relocated/browser.json`；未随公开包提供）、静态审计（来源记录：`../../../outputs/progress_web_qa/static-r2.json`；未随公开包提供）、搬迁静态审计（来源记录：`../../../outputs/progress_web_qa/relocated-static.json`；未随公开包提供）、仓库健康日志（来源记录：`../../../outputs/progress_repo_health_r2.log`；未随公开包提供）。

## 风险与回退

历史74图被重复开发使用；全300/400为训练拟合；单seed微小增益不证明显著性或未知视角能力。原B2与M2F训练步数不同，方法演进不是逐项单变量累加。视频无GT，不新增质量指标。困难裁剪只服务展示。

所有模型、图像、视频与QA产物留在忽略目录；只登记轻量源码、配置和报告。未上传、发布、提交或push。回退只需停用新网页，继续旧离线组会报告；无需恢复模型或重训。
