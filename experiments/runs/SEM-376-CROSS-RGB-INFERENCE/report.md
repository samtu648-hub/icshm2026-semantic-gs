# SEM-376-CROSS-RGB-INFERENCE

2026-09-15 SEM376完成：冻结双向交叉、零训练。Standard＋SEM371/375 mIoU95.265719%/94.965801%；新Scaffold＋SEM371/375为94.857052%/95.065000%。两语义均在自身训练RGB域更好；不能宣称SEM371跨域普遍更强。保持Standard＋SEM371主线，不采用交叉替换。1088新推理、18fresh概率精确、1200组独立混淆、RGB/语义/18冻结保持。下一步仅评审交叉对照与准备Standard配对的一个语义优化变量，不自动训练或融合。

实验绑定：Git `49cdcdb1c9915a9544d568754af95ea20a85ec9c`；配置`config.yaml` SHA256 `38ca661c56814d442b9b8abaf24054ec269514a9a104340250baf8ceff455f2c`；seed20260805；manifest `data/manifests/contest_dataset_v1.json`；split `released_all400_rgb_all300_sem_v1`、`outputs/quality/all_data/RGB-120-r2/split.json`。固定RGB130/RGB136和SEM371/SEM375的6000端点。新推理记录分别绑定输入RGB export和semantic checkpoint SHA256，详见`input_hashes.json`及各`inference/result.json`。对角线复用QUAL007/SEM375真实结果，未重复训练或推理对角线。

问题：同一语义checkpoint可否跨RGB骨干使用、是否优于匹配模型？本轮每个固定RGB输入只改变语义权重；完成2×2交叉，输入尺寸/原生推理/类ID/标签/相机及数值设置保持。用全300训练视角标签评价；100无标签及144轨迹无GT。不是隐藏相机泛化测试、不是多seed显著性检验。

- standard_sem371: 五类mIoU 95.265719%。
- standard_sem375: 五类mIoU 94.965801%。
- scaffold_sem371: 五类mIoU 94.857052%。
- scaffold_sem375: 五类mIoU 95.065000%。

swap_semantic_on_standard: standard_sem371 → standard_sem375，全局mIoU -0.299919pp；逐图44改善/256退化。各类IoU变化：background -0.058264pp、deck -0.210846pp、stay_cable -0.635215pp、tower -0.229902pp、foundation -0.365366pp。

swap_semantic_on_scaffold: scaffold_sem371 → scaffold_sem375，全局mIoU +0.207949pp；逐图240改善/60退化。各类IoU变化：background +0.059686pp、deck +0.146603pp、stay_cable +0.641776pp、tower +0.079602pp、foundation +0.112076pp。

swap_rgb_for_sem371: standard_sem371 → scaffold_sem371，全局mIoU -0.408667pp；逐图39改善/261退化。各类IoU变化：background -0.105592pp、deck -0.219120pp、stay_cable -1.150873pp、tower -0.224439pp、foundation -0.343312pp。

swap_rgb_for_sem375: standard_sem375 → scaffold_sem375，全局mIoU +0.099200pp；逐图187改善/113退化。各类IoU变化：background +0.012358pp、deck +0.138328pp、stay_cable +0.126119pp、tower +0.085065pp、foundation +0.134129pp。

交互差为0.507867pp：SEM371→SEM375的收益方向随RGB输入改变。SEM371在Standard上更优，SEM375在Scaffold上更优；两模型各自在自身训练RGB上五类IoU均高于其跨域输入。支持训练输入域匹配影响输出，不支持“某一个语义模型独立于RGB总是更强”或“RGB更清晰对任意固定语义模型必然更优”。四个现有组合中Standard＋SEM371全300拟合最高，继续保留该配对；不自动融合、混合域重训、调门或部署。输入域改变包含外观/纹理/伪影等，未进一步拆分具体机制。

已查看固定159/160/146困难ROI、174整图，以及同一官方轨迹71/72/73四组合PNG。前两列RGB完全相同，后两列RGB完全相同，语义差异来自更换checkpoint。

Standard RGB上，SEM375在159困难ROI修对812/损伤720，局部mIoU提高，左侧细索断口有局部改善，但deck修对104/损伤460；160困难ROI修对400/损伤1025，deck修对69/损伤559、stay_cable修对187/损伤237；146困难ROI修对832/损伤1669，其中背景损伤1291，塔邻近黄色误报更明显。159远端ROI有局部正收益（修对210/损伤131），不改变全300负结果。

新Scaffold RGB上，SEM371在159/160产生更明显的左侧拉索断续和桥面内细线缺口。换回SEM375，159真stay_cable修对1361/损伤133，但deck修对69/损伤633，背景损伤1719；160真stay_cable修对1670/损伤89，deck修对73/损伤618。匹配模型有收益但未解决原159额外deck误报。174天空ROI四组合都为正确背景；没有语义天空增益。

轨迹71–73：Scaffold＋SEM371左侧细索有较大缺口；Scaffold＋SEM375在这些帧更连续。Standard两语义在所看帧桥梁主体保持，细线仍有变化。无GT，不能断言新增预测一定真实或144帧零闪烁。完整144帧视频/544视图可浏览，但未逐帧人工检查。

核验：两新臂各544PNG，共1088；各9图独立新进程重载，18概率数组/PNG精确一致。全四组合×300=1200逐图混淆使用独立逐类逻辑掩码重新计数，与存档全部一致；全局指标精确复算。检查点/原训练源码、两套544RGB文件、原三RGB、RGB136及18冻结哈希保持。零优化器、零参数更新、零重渲染；RGB一致性本轮用文件与端点哈希验证，SEM375已有544 fresh native闭环作为先前证据，未冒称本轮重渲染。

资源：两组主推理267.819/267.389秒，两fresh10.142/10.219秒。共享GPU锁串行运行，旧权重直接只读复用，不复制大型checkpoint。未下载、上传、安装、删除、提交或push。

交付：`outputs/quality/SEM-376/r1/comparison/index.html`含四组合全544视图同框裁剪、300标签、固定九图及144帧视频；`audit.json`含全300指标、四种配对变化、固定15区域计数。JavaScript语法、5052本地图片资源及ffprobe144帧验证通过；离线PNG已检，浏览器本地file交互与视频播放未由工具验证，未绕过此前访问限制。

剩余风险：同一300已参与两个语义模型训练，交叉结果仍为历史训练视角，不是新域独立测试；小ROI类别缺失会改变mIoU平均项，必须结合修对/损伤与类别计数。轨迹无GT，不按本轮结果事后选类/区域融合。回退：原模型与默认配对未改变，继续Standard＋SEM371即可，交叉结果独立保留。

最终收尾：repo_health通过523测试（72可选跳过）；最终页面JS语法通过。新增产物约0.085GiB；训练/推理与后处理队列均结束。git diff --check通过；最终封存见completion.json。
