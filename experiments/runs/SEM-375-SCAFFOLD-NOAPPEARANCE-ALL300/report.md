# SEM-375-SCAFFOLD-NOAPPEARANCE-ALL300

2026-09-15 SEM375完成：RGB136冻结渲染、同配方全300/6000步Mask2Former。mIoU95.065000%，较旧SEM370 +0.091762pp、较Standard SEM371 −0.200719pp，五类IoU均低于Standard。1200逐图混淆独立复算、九图fresh概率精确、544 fresh RGB精确、旧三RGB/三语义及18冻结保持。保留新Scaffold候选，仍建议Standard作为语义开发主线；未替换系统。下一步仅评审新对照页及沿Standard定义一个语义优化变量，不自动启动训练。

实验绑定：Git `49cdcdb1c9915a9544d568754af95ea20a85ec9c`；配置`config.yaml`（SHA256 `6be79bf8c6048948773105098df5ff2e7d61557b5f86699dbe3d1da2a35be835`）；seed20260805；manifest `data/manifests/contest_dataset_v1.json`；split `released_all400_rgb_all300_sem_v1`，split SHA256 `d0054ccc3c490516dadc2eb4bc978360856a12a1527a28e087da2feb9ad2ee14`。父RGB136与配方父SEM370/371/372。所有值属于全300训练视角拟合、单seed；100张无标签及144轨迹无GT指标。

本次仅改变RGB输入域为RGB136无外观嵌入渲染，使用原Mask2Former Swin-L ADE预训练初始化、768裁块、作者优化器/损失及6000有效更新。完整采样与SEM370/371一致，resolved配置与SEM370一致，初始模型state与SEM371精确；没有训练RGB、联合几何或历史多专家融合，没有用SEM375分数选步数或阈值。

训练循环1567.478秒，阶段1586.825秒，AMP累计跳过6次后按父配方重试，实际6000成功更新；峰值allocated 8097.570MiB/reserved 8590.000MiB。全544推理268.279秒，单图推理中位450.689ms。保存3000/6000两个检查点，固定6000评价，未扫描端点。

全300五类mIoU：旧Scaffold/SEM370 94.973239%；Standard/SEM371 95.265719%；Mip/SEM372 95.162132%；新Scaffold/SEM375 95.065000%。新模型相对Standard 84视图改善/216退化；相对旧Scaffold161改善/139退化；相对Mip99改善/201退化。指标按全局混淆矩阵计算，逐图胜负按各图mIoU，仅描述拟合，不是独立显著性测试。

- background: IoU 99.381739%，precision 99.709620%，recall 99.670210%；较Standard IoU -0.045906pp。
- deck: IoU 97.025901%，precision 98.200746%，recall 98.781976%；较Standard IoU -0.072518pp。
- stay_cable: IoU 94.534291%，precision 97.237123%，recall 97.143646%；较Standard IoU -0.509097pp。
- tower: IoU 94.204858%，precision 96.873788%，recall 97.158559%；较Standard IoU -0.144837pp。
- foundation: IoU 90.178213%，precision 94.411146%，recall 95.263650%；较Standard IoU -0.231236pp。

固定同位置复核：159困难区域新模型出现桥面蓝色背景误报，相对Standard修对758像素、损伤2670像素，其中真背景损伤1989；该ROI mIoU低于Standard。160困难区域左侧拉索比Standard更连续，真stay_cable修对951/损伤321，但真deck修对134/损伤549；该ROI总mIoU高于Standard，不能掩盖桥面代价。146困难区域修对1299/损伤437，其中tower修对260/损伤42，但较小structure-146区域净损伤，区域口径明确分开。159远端桥梁保留，未出现Mip同帧远塔/远桥面消失。174天空固定ROI四套语义全为正确背景；RGB线条改善没有在这个区域形成额外语义收益。057结构区tower修对282/损伤113，但整体修对645/损伤746，不能称全区改善。精确指标和原框见region_audit.json。

已查看159 full/困难ROI、160困难/结构ROI、146困难ROI、174 full、057结构ROI，以及连续轨迹71/72/73四路PNG。新Scaffold在这三帧保持远端桥梁，无Mip同处雾块/主缆明显大缺口；自身桥面内黄色细线仍有断续/变化。无GT，不计算轨迹准确率，不把三帧检查解释为144帧零闪烁。页面生成全部固定九图三种面板及全544选择器，未逐张人工审查。

核验：四臂×300=1200逐图混淆由独立逐类逻辑掩码计数，全部与推理记录精确一致；全局IoU/precision/recall复算通过。九图新进程概率与训练后保存数组精确一致。训练后的新进程原生RGB544张逐像素精确，官方轨迹元数据一致；原三RGB、原三语义、18冻结产物及RGB136端点/输入哈希保持。两保存检查点全部state tensors有限。

交付：`outputs/quality/SEM-375/r1/comparison/index.html`、`trajectory.mp4`、`audit.json`、`region_audit.json`及`fresh_rgb_audit.json`。离线页面资源与JavaScript语法、ffprobe144帧/1760×712/12fps通过。此前浏览器工具对本地file页面访问受限，本轮未绕过，未声称已测试页面交互或实际视频播放；已通过本地PNG复核图像。

风险：小ROI缺失类别时mIoU对极少误报敏感，例如纯背景出现少量类2误报会让参与平均的类数增加，不能把49.99%与100%解读为半数像素改善；同时提供混淆、修对/损伤像素。RGB输入改变包含外观和几何变化，不能把语义差值单独归因外观UID或某个结构机制。当前验证支持Standard工程主线，不证明隐藏相机的绝对最佳；未来新RGB候选也应匹配语义。SEM363历史融合/SEM373三维语义与此分数不同口径。

来源许可证沿用SEM371：Mask2Former commit9b0651c6c1d5b3af2e6da0589b719c514ec0d69a（MIT），Detectron2 a2f4a8771ab77e8411c26b27f24f9489a28a2453（Apache-2.0），ADE预训练CC-BY-NC-4.0。没有安装新第三方、上传、Git提交或push。准备时调用不存在的`python`命令在注册前失败，改用`python3`完成；未产生失败训练/重训，原始shell记录保留。

回退：旧系统完全未替换，继续使用RGB130/SEM371或旧Scaffold即可；新候选独立目录留存，无需删点或恢复覆盖文件。

最终工程收尾：repo_health通过523测试（72可选跳过）、git diff --check通过；最终页面JavaScript语法通过。新增产物约2.047GiB，无剩余训练或推理进程。最终封存见`outputs/quality/SEM-375/r1/completion.json`。
