# PROJECT_STATUS 历史存档：2026-09-08 阶段整理前

本文保存阶段总结整理前的完整正文；仅将 Markdown 相对链接改为从存档目录解析。文内“当前”“下一步”“未运行”和时间目标均代表当时记录，阅读现状请回到[项目状态](../../PROJECT_STATUS.md)与[阶段总结](../experiments/research_summary_20260908.md)。历史协议、失败数值和勘误原样保留。

---

# Project Status

- 2026-09-08 / SEM329完成（ADR-078）：固定第二源否决已完成全219源库+SEM161的74最终三臂对比。SEM161/326/329的mIoU为83.387017%/83.682675%/83.705003%，基础IoU72.770345%/74.242876%/74.354428%。SEM329较SEM326 mIoU/基础IoU/recall +0.022328/+0.111552/+0.441838pp，precision−0.231028pp；较SEM161 mIoU/基础IoU+0.317985/+1.584083pp，但recall仍−1.424762pp至89.761802%。救回739误删（23.671%），保留7590正确纠错（91.866%），净正确像素较SEM326再+67。四空间组对SEM161均升，对SEM326三升一微降；seg-010 recall仍仅83.635%，worst8无新增收益。74独立原生重构/三臂质量复算/四次fresh实际外相机/六固定图/RGB与18hash通过；小尺寸输出读取器r1失败留存，r2只修读取、无改规则。SEM329作为优先研究候选保留，SEM326保留精度折中，正式SEM161及包不覆盖；非盲测、不声称显著性，未启动后续优化/all-data。

- 2026-09-08 / SEM327→328完成（ADR-076/077）：源边界2px只读诊断救回81.343%误删却损失41.443%正确纠错，较dense基础IoU−1.452pp，保留低误删折中。另立第二可靠源label4才否决删除的实际SEM328，未叠加边界：105/111均保住并微增总体IoU；check较SEM325 mIoU/基础IoU/recall +0.009221/+0.047017/+0.423304pp，precision−0.177683pp。救回4977误删（23.433%），保留95.415%原正确纠错；fold1基础IoU−0.057060pp，其余两折升，三折recall均升。216原生输出独立重构/混淆与改动计数/RGB/18hash通过，无新匹配/训练/74/外相机渲染。内部+0.30/+1.50pp验收门不作为否定这些实际增益的依据，旧记录不改。保留SEM328作为召回改善的OOF候选，下一固定全219+SEM161最终组合验证，正式SEM161及SEM326不覆盖。

- 2026-09-08 / SEM326完成（ADR-075）：全219参考库+SEM161固定最终复核已在74开发与四次实际外相机验证。74 mIoU83.387017%→83.682675%（+0.295658pp）、基础IoU+1.472531pp、precision+3.199842pp；recall−1.866600pp至89.319965%，纠正8262/误删3122/仍错100。四个既有空间组mIoU/基础IoU均升，140–146均值mIoU+0.950991pp，worst10%+0.018234pp；主增益略低于+0.30/+1.50pp，绝对recall未达90%，三门未过，不晋级但保留正面开发候选。72/74选源可达、444方向匹配88.825s，74原生独立重构/独立质量复算、四实际官方探针与class1/2/3/RGB/18hash通过；两次包装实现失败及修复留存。正式SEM161不变；下一回到训练侧定位误删，只据诊断选择一个新保护变量，不在74扫阈值、不立即换模型或重建环境。

- 2026-09-08 / SEM325完成（ADR-074）：实际1320×989基础4→0复核在105 calibration及固定111 check均通过全部质量门。check相对SEM160 OOF的mIoU/基础IoU/precision为+0.812045/+3.969178/+4.989950pp，recall−1.806420pp至96.798472%；纠正131201、误删21239、仍错530。三折基础IoU+4.410876/+5.837501/+0.259974pp，第三折收益小、第一折recall−2.892720pp仍需关注。check固定pose-top3有95/111可达，666方向匹配130.595s；216张原生PNG独立重构、质量复算、class1/2/3/RGB/18hash通过。r1归档虚拟Torch模块失败已保留，只修归档后r2的105预测与r1逐字节一致。RoMa仍512，不是原生匹配；复用增量OOF不称盲测。正式SEM161不变，下一独立登记全219参考库+SEM161的最终74开发/外相机验证，不再扫当前OOF阈值。

- 2026-09-08 / SEM323→324完成（ADR-072/073）：固定pose-top3直接扩到剩余105 calibration目标，91/105可达（rank1=86），315对中242对过门；各折43/45、39/41、9/19，第三折仍弱。630方向匹配124.292s，独立315对复算/重放及RGB/18hash通过。随后单独SEM324读105目标和91已选源标签角色（114唯一标签），在512网格上支持覆盖15.836%旧错误，源标签一致94.086%；全量照搬可纠正48069却误改164085，不能直接融合。20方向分解中foundation4→background0纠正23474/误删3707（约1.92pp采样recall代价），三折净正，优先进入独立原生分辨率基础定向复核实验。无语义预测导出/模型替换，SEM161不变；无需继续泛化扩样或重建环境。

- 2026-09-08 / SEM322完成（ADR-071）：同3目标、仅位姿排序的固定top3参考源诊断通过；rank1有2/3目标可达、top3有3/3，9对中6对过原覆盖/几何保留门。首个过门源017→013（rank3，可靠覆盖6.893%/极线保留82.950%）、001→103（rank1，6.322%/81.471%）、032→029（rank1，3.218%/94.258%）；均胜弱identity。18方向推理/新进程重放、独立位姿/9对几何复算、RGB及18hash通过；peak allocated2508.446MiB，SEM320重叠两对数组精确一致。无target轨迹解析、标签/真实target/74读取、语义转移或生产更改。保留小样本源可达性正证据，不能称mIoU/全场景泛化改善；下一固定策略扩样需另登记，不扫描K/角度/阈值。历史check编号可作同折source，访问口径已明确。

- 2026-09-08 / SEM320→321完成（ADR-069/070）：RoMa v2官方完整1.096GB权重hash、两个固定源码版本、隔离512运行与新进程重放通过；峰值CUDA allocated2497.295MiB、reserved2716MiB，未改现有环境/上游/SEM161。SEM320两个历史check编号作为同折训练source解码，与“111图像不读”的广义措辞冲突，已保留偏差而非声称全协议零偏差；任何标签/真实target/74图像均未读。另立SEM321只读几何诊断，3组可靠覆盖9.795%/0.002670%/3.218%，cycle后极线保留67.426%/63.636%/94.258%；只有第三组同时过门，整体失败。独立Torch采样/径向逆解/essential矩阵复算计数与门一致，18冻结hash/RGB不变。8新测试与repo_health（128配置/233测试，46可选跳过）通过。停止本单源筛选，不转移标签或调阈值/增分辨率；保留局部几何正证据。当前共视选源依赖已知target轨迹，新相机选源仍未解决，下一若研究需另立可达性诊断。

- 2026-09-08 / SEM-314完成未晋级（ADR-068）：唯一全219 DINOv2最终头3k训练阶段149.915s，固定0.5基础复核的74开发审计相对SEM161 mIoU/基础IoU仅+0.087/+0.437pp，precision+3.114pp但recall−3.260pp至87.926%；主增益、召回与空间门失败。169–175组基础IoU−11.369pp，固定140–146均值mIoU−0.892pp；worst10%+3.601pp不能覆盖失败。219训练隔离/权重、74独立重构、2实际头native重放、3实际官方COLMAP探针、RGB/1/2/3/18hash及74独立质量复算通过，6固定样例已存。27针对性测试和repo_health（126配置/225测试、46可选跳过）通过。停止本final配方，无调阈值/补救重训/生产替换；SEM313正面OOF证据保留但未迁移为合格final。正式仍RGB-030-FULL+SEM161，下一研究为独立稠密对应预检，非已运行。

- 2026-09-08 / SEM-312→SEM-313完成（ADR-067）：同108只读互补诊断支持DINOv2删除方向，另立固定p4<0.5只复核旧foundation的零训练配对。SEM-313在108/111均通过全部预注册门：check相对SEM160 mIoU/基础IoU/precision为+0.811/+3.980/+4.575pp，recall−1.003pp至97.602%，基础FP减少118404；三折IoU+7.113/+2.462/+0.380pp。胜过相同MiT复核mIoU+0.830pp/IoU+4.090pp。438独立重构、6实际头重放、4实际新规则官方相机探针和两组独立质量复算通过，RGB/1/2/3及18冻结项不变。保留OOF研究候选，正式仍RGB-030-FULL+SEM-161；108已用于方向选择、111历史复用，非全链盲测。未读74、未训final、无扫描/生产替换。

- 2026-09-08 / SEM-311已完成未晋级（ADR-066）：用户在DINOv3申请被拒后选择公开DINOv2；固定896/448、三折6头/18000步/438 OOF耗时13.405分钟。复用108上DINOv2相对冻结MiT mIoU/foundation IoU +2.038/+9.878pp且三折均胜；但相对SEM-160为−2.653/−12.947pp，precision−13.919pp、recall+0.758pp，foundation FP净增376974，第三折IoU−22.701pp，全部逐折安全门失败。438输出、6头native独立精确重放、正式训练头官方两尺寸四探针、108独立复算和18冻结hash通过。无111/74评分、scan/retry/final；保留特征来源正面证据，淘汰完整0/4替换配方，SEM-310旧代码/产物及正式SEM-161不变。

- 2026-09-08新开发授权（ADR-065）：论文/展示后置，先启动SEM-310冻结DINOv3/冻结MiT配对，再做稠密对应预检，方法选定后另注册全400 RGB/300标签拟合。SEM-310已实现，两臂各2步的随机DINOv3架构测试、两份独立原尺寸精确重放、两臂×官方相机native/320×240四项工程探针通过；18冻结项与219 RGB/219基线核验通过。正式预训练实验被官方HTTP401/本机未登录阻塞，未启动六模型质量训练、未评108/111/74。新单测3/3通过，repo_health通过；训练环境额外全量测试有5项既有依赖/浮点测试问题，详见报告。生产RGB-030-FULL + SEM-161不变。

> 每次研究或工程 Goal 的第一入口。

## 当前状态

- 2026-09-07 `SEM-300` completed_calibration_gate_failed：固定16px signed-distance辅助损失的六模型18k步/438 OOF运行完整结束，18个冻结artifact、RGB与class 1/2/3保护、所有重载均通过；只解码既有108 calibration标签，111 check/74 developer均为0访问。候选相对SEM-160 foundation IoU/precision +1.968/+4.377pp、mIoU +0.403pp，但recall −3.983pp至91.623%；相对同轮五类control，foundation IoU/precision/recall/mIoU再降−2.122/−0.920/−2.372/−0.430pp，且fold2 IoU较SEM-160低−7.153pp。校准门关闭，无scan/retry/final，signed-distance方向淘汰，`SEM-161`不变。

- 2026-09-07 `SEM-290` read-only raw five-class audit completed: the three sealed SEM-270 five-control checkpoints replayed all 219 inherited OOF views with 18 frozen artifacts intact; exactly the already reused 108 calibration labels were decoded and the 111 check/74 developer sets remained unread. The fixed raw argmax mIoU is 0.860263, −1.826pp against the same control's protected fusion; deck/`stay_cable`/tower/foundation IoU are −2.276/−3.517/−1.041/−1.928pp. Raw cable/foundation recall gains are only +0.048/+0.587pp while background→cable rises 65.09%. Therefore raw native960 five-class replacement is not evidence for replacing specialists, and no learned expert-fusion task is opened. The separately registered signed-distance follow-up also failed in SEM-300; `SEM-161` remains unchanged.

- 2026-09-07 SEM-280已完成并未晋级：固定2倍真实foundation内侧16px边界损失相对同轮五类控制找回recall +0.917pp，却损失foundation IoU −2.984pp、precision −3.769pp和mIoU −0.607pp；相对SEM-160只有IoU +0.862pp/precision +1.400pp，recall −0.923pp至94.683%。fold1/2 IoU为−1.671/−5.737pp，逐折安全门失败。六模型18k步/438概率及PNG、当前checkpoint逐位重放、官方相机native+小尺寸、RGB/1/2/3保护及18冻结hash通过；跨独立训练运行的旧control有0.0330%融合像素漂移，登记警告。未读111/74质量指标、未训final；停止倍率/半径/阈值扫描，正式SEM-161不变。

- 2026-09-07 SEM-270已完成，二分类未晋级：同native960三折六模型/18k步/438份预测耗时46.25分钟。二分类对SEM-160基础IoU/precision/recall为-4.060/-4.067/-0.544pp、mIoU-0.825pp，第三折基础IoU仅48.929%且大量背景外扩。五类匹配控制反而IoU+4.779pp、precision+6.146pp、mIoU+0.973pp，但recall-1.761pp，也未满足原门。保留五类正面开发证据，不偷换候选晋级；不评111/74、不训final。独立六模型重载/438概率及PNG/同采样/权重/官方相机native+小尺寸、RGB/1/2/3保护、18冻结hash均通过。calibration事后诊断：五类误删23,658真基础，其中81.81%在16px真值边界内；下一值得检验的是保留五类上下文的基础召回/较宽边界训练监督。

- 2026-09-07 SEM-263严格嵌套OOF已完成，未晋级：18个从原始预训练初始化的内层教师、6个匹配点头、438张预测，总耗时2.185小时。候选对匹配控制deck recall +0.981pp，但IoU -0.076pp；对SEM-160 OOF IoU +0.242pp、mIoU +0.037pp、precision 96.258%，recall仍-0.403pp。召回门及胜过控制IoU门失败，不读本轮111/74质量指标、不训final。独立审计核对419对视角/6,864,896同采样点，219零残差/保护/RGB和实际训练头官方相机native/小尺寸通过，18/18冻结哈希一致。保留正面召回消融证据，正式SEM-161不变；下一重点是独立foundation/background专用分类研究。

- 2026-09-07 SEM-260/261/262开发迭代完成：95.46%的calibration deck FN在4px真值边界；细特征点头对粗对照IoU+0.956pp，训练类别平衡再+0.089pp。最佳SEM-262对SEM-160 OOF的deck IoU+0.275pp、mIoU+0.035pp、precision97.47%，但recall−1.606pp，因此未晋级，check/developer本轮不评分、无final。两轮均允许真实bg/deck双向修正；新头重载、219零残差/保护/RGB及实际新分支外相机/小尺寸门通过。保留正面消融证据，不继续扫描权重；随后严格嵌套OOF的独立匹配检验见SEM-263。
- SUBMIT-040干净Ubuntu容器复现完成：新依赖/新编译扩展、编号140官方相机两次native和320×240 RGB/语义精确重放，2161包文件不变；不挂载旧Python。补齐torch_scatter/colorama并保留r1失败。范围是同GPU/宿主内核的新用户空间，非第二机器。论文证据提纲已整理，正文/视频/PPT和官方提交尚未完成。

- 2026-09-07新开发授权：启动SEM-260五类误差预算，随后按证据推进SEM-261冻结骨干点级语义细化；允许background/deck双向修正，不把全部旧deck保护。SUBMIT-020的18/18冻结项已重新验证存在且哈希一致。当前正式回退仍为RGB-030-FULL + SEM-161。
- 评估勘误优先于以下历史摘要：SEM-250代码实际评估全部219个target（含111 check），只保证同折source/target拟合隔离；原“check未读”不实。直接稀疏点修正失败不能否定稠密方法。SEM-230的纯粒度因果和工程门表述按ADR-053收窄；桥面不是全局IoU最低类别。见ADR-057。

- 2026-09-07长任务已收束：SUBMIT-030完整SEM-161可搬迁接口通过18/18冻结哈希、2,161文件内容清单、两次独立重载、140官方相机RGB/语义逐位重放、320×240相机和非空输出拒绝；未官方提交。SEM-240真实贡献/最大贡献责任匹配三折淘汰（deck IoU/recall −4.303/−6.728pp，check/developer未评）。RGB-100/101的匹配双30k完成；候选相对控制改善，但相对冻结RGB的140–146为−0.188dB，低于−0.100dB门，且固定视觉门失败，drop且未打开新语义指标。
- 2026-09-07 `SEM-250-COLMAP-TRACK-DECK-DIAG`已完成独立稀疏对应必要性预检：官方`images.txt`轨迹重三角化的61,170点在严格三折source-only OOF下只覆盖1/198,169个SEM-160 deck漏检（0.0005046%）；零误报oracle deck IoU上界仅+0.00000635pp，固定多数证据precision 50%。未写候选/预测、未读check或74-view，禁止以splat/阈值挽救；这条稀疏轨迹来源不能支持桥面修正。
- SEM-230解释修正：旧失败数值不变，但实际是从零二分类offset logits拟合，缺少anchor匹配控制，贡献日志只是opacity，外相机仅测旧sidecar。不能将这些记录当作纯粒度因果结论或新分支工程门；以下旧摘要的相关强表述以ADR-053及SEM-240报告为准。
- 2026-09-07 `SEM-230-OFFSET-SEMANTIC-OOF`已完成并在108-view calibration淘汰：唯一变量是将anchor共享语义替换为每个generated offset的background/deck零初始化残差，RGB/几何/opacity/appearance/densification/相机全部冻结。三折10k、219 OOF、贡献权重和checkpoint重载完成；零残差语义与RGB严格一致、外部COLMAP相机兼容、`stay_cable`/tower/foundation及foundation拒绝区逐位不变。但相对SEM-160 OOF，deck IoU/precision/recall为−3.259/−2.262/−1.200pp，mIoU −0.699pp，precision仅93.366%。111-view check、74-view developer和final训练均未打开；`SUBMIT-020`的18/18项哈希仍一致，`RGB-030-FULL + SEM-161`不变。
- 2026-09-07 已完成并关闭独立桥面几何/早期融合研究线：`SUBMIT-020`已用实际SHA-256冻结RGB-030-FULL + SEM-161及全部父产物。SEM-200/201离散几何、SEM-210连续XYZ、SEM-211外观×XYZ晚期联合、SEM-220 RGB+log-depth+normal三折9k早期融合、SEM-221双向仲裁均未通过108-view calibration，111-view check与74-view始终未打开。最接近的SEM-220仅使deck IoU/recall +0.0219/+0.0668pp，低于+0.1/+0.2pp门；SEM-221反而使deck IoU/recall −0.7658/−1.0519pp。当前提交候选不变。
- 2026-09-06 RGB-090/091/092首轮全部完成：219训练图诊断不支持统一RGB高误差解释deck漏检；匹配控制及DINO降权候选均跑满唯一30k。RGB-092相对控制PSNR −0.34963 dB、LPIPS +0.001509、worst10/140–146 PSNR −0.96932/−0.44268 dB；054/140天空条带减轻但159/160水岸伪影加重，数值和视觉门均失败，drop本配方。训练/重载/保护/数据隔离通过，144帧同轨迹视频和原生局部图已保存。SEM-180按父门关闭，新预测/语义标签读取均0，不声称桥面改善；旧RGB-030-FULL + SEM-161全部回退与74张旧语义/RGB逐份哈希一致，协议ADR-048。

- 当前日期：2026-09-08
- 最近更新：2026-09-08
- 当前阶段：正式冻结与可搬迁包仍为`SEM-161-DECK-BINARY-FINAL`；新研究候选`SEM-329-SECOND-SOURCE-VETO-FINAL`在同74的三臂mIoU/基础IoU最高，较SEM326减少误删但略损precision，尚未替换正式包。`SEM-326`与`SEM-151`、`SEM-141`、`SEM-111`和`RGB-030-FULL Scaffold-GS + SEM-001`全部保留
- 当前实验：`SEM-141`已完成全219张final-3k与冻结0.70的74-view开发审计；相对SEM-111的mIoU/索IoU/recall为+1.43/+6.88/+10.29pp，worst-10%与140–146为+1.64/+1.58pp，deck/tower/foundation逐像素完全不变。背景→索增加28.91%、precision下降1.81pp，故冻结现有checkpoint和阈值，不再据74-view调参
- 当前实验：`SEM-150-FOUNDATION-GATE-OOF`已通过并冻结0.10；`SEM-151`的74-view final审计相对SEM-141使foundation IoU/precision +27.90/+31.54pp、recall保持91.19%、mIoU +5.60pp、background→foundation −76.12%，70视角提高、4持平、0下降，class 1/2/3逐位一致，RGB/重载/外部相机全门通过。冻结规则，不再据74-view调参
- 当前实验：`SEM-160-DECK-BINARY-OOF`已通过并冻结0.99；111-view check使deck IoU/recall +0.437/+0.841pp、mIoU +0.093pp、precision保持95.25%，保护像素逐位不变。允许唯一SEM-161全219 final与一次开发审计
- 当前实验：`SEM-161-DECK-BINARY-FINAL`已完成唯一全219 final-3k与冻结0.99的一次性74-view审计；相对SEM-151的mIoU +0.294pp，deck IoU/recall +1.367/+2.190pp，precision 95.04%，158–161均值+2.690pp且四帧均升。63视角升、3平、8降，background→deck为1.24866倍；保护像素、RGB、重载和官方外部相机全门通过。冻结当前checkpoint/阈值，不再据74-view调参
- 当前实验：`SEM-170-DECK-ARBITRATION-OOF`已在calibration止损并drop；12个阈值全失败，最佳0.70相对SEM-160使deck IoU/recall −0.933/−1.157pp，删除242,977个既有deck且其中124,445个为真deck。111 check与74-view均未打开，说明冻结二分类概率可补漏但不能安全否决既有3D桥面；SEM-161保持不变
- 当前实验：`SEM-171-DECK-CONTINUITY-OOF`已在calibration止损并drop；28个半径/阈值组合全失败，最佳半径3/阈值0.98相对SEM-160使deck IoU −0.0102pp、recall仅+0.0065pp，1,847个新增中只有486个真deck却有935个`stay_cable`。111 check与74-view未读，说明剩余闭合缺口主要是索穿越而非可安全补的桥面洞
- 当前实验：`SEM-172-DECK-INTERIOR-OOF`已完成三折final-3k并在calibration止损；0.41044 GPU小时、三checkpoint严格重载。12个阈值全失败，最佳0.99相对SEM-160使deck IoU/recall −0.018/−0.106pp、precision +0.081pp，check和74-view未读。内部加权独立替换drop，但其precision/recall偏移可作为一次冻结概率集成的互补证据
- 当前实验：`SEM-173-DECK-ENSEMBLE-OOF`已在calibration止损并drop；36个组合全失败，最佳alpha=0.75/阈值0.99仅使deck IoU +0.0032pp，却使recall −0.0496pp、mIoU略降，check和74-view未读。双向删除、闭运算补洞、内部加权训练和冻结集成都未超过SEM-160，当前桥面后处理/同架构loss系列结束，SEM-161继续冻结为最强候选
- 当前实验：`RGB-080-LATE-PRUNE`已完成并drop。15k后150轮原生opacity-only pruning从候选自身清掉13,742 anchors；mean PSNR/SSIM/worst-10%分别改善+0.0389 dB/+0.00101/+0.0679 dB，但LPIPS退化+0.00050，且固定样例主要横条、幽灵索和近岸拉丝仍在，未过主门与视觉门
- RGB诊断：`RGB-070-APPEARANCE-PF`已完成并drop。最近训练相机appearance相对`RGB-030-FULL`使mean PSNR/SSIM/LPIPS分别变化−0.217 dB/−0.00226/+0.00424，最差10%和140–146 PSNR下降0.387/0.278 dB；空中长条、重影索线和岸边拉伸的位置不变。当前RGB最佳仍为`RGB-030-FULL`，appearance假设已隔离为非主因，后续只允许另立ID的几何/高斯支持度诊断
- 耦合诊断：`RGB-071-SEMANTIC-COUPLING-DIAG`已完成。SEM-001的13,166,331个background→`stay_cable`误报像素RGB MAE比正确背景低10.62%，说明大面积静态语义误报并非重建差的区域；SEM-111的749,936个索→背景漏检像素RGB MAE/真实边缘强度为索TP的2.21/2.54倍，说明渲染精度是二维渲染域细索召回的瓶颈之一。结论仅为开发集关联，不作因果/泛化主张
- 几何清理前置：`RGB-072`已完成303训练视角审计。低支持群整体与正常结构混合，不能全局清除；累计主导≥256像素、支持≤4视角、mean/high-error RGB MAE双门的极端交集仅29/189,073 anchors、18,078训练主导像素。已另立`RGB-073`锁死该极小mask作一次可逆74-view预检，不扫描阈值
- 清理预检：`RGB-073`因逐像素精确重放门失败而止损；`RGB-074`保持同一29-anchor mask并完成数值成对审计。mean PSNR/SSIM/LPIPS虽微幅改善+0.00108 dB/+0.000065/−0.000116，主要天空横条、幽灵索和岸边拉丝在054/140/151/159/160/183仍无可感知改善，故不提升、不做语义重跑，RGB-030-FULL仍是当前最佳。低支持事后清点路线到此停止
- 训练时清理：`RGB-080-LATE-PRUNE`完整30k已验证原生pruning可稳定删除6.88%的候选15k anchors，但它只带来小幅PSNR/SSIM收益，LPIPS略差，主要浮点/拉丝未被清除；不执行语义/外部相机升级门，不扫描opacity阈值。RGB-030-FULL继续是当前RGB最佳
- 提交截止：2026-09-30
- 团队：3人，负责人尚未映射到任务
- Ubuntu数据：已在既有路径完成检查；数据 manifest 与仓库版本一致

## 导师讨论摘要：SEM-001 至 SEM-006

这里的“当前最佳”指通过预注册质量门、工程门和冻结边界后仍可保留的候选，
不是从所有诊断和二维审计中事后挑选单项最高数字。`SEM-161`是当前最强**开发候选**；
`SEM-151`、`SEM-141`与`SEM-111`保留分级回退，`RGB-030-FULL Scaffold-GS + SEM-001`是可靠独立回退。

| ID | 方法与唯一改动 | 主要想验证什么 | 已完成的真实结果 | 当前结论 |
|---|---|---|---|---|
| `SEM-001` / S0 | 在冻结 Scaffold RGB anchor 上训练静态五类 logits；不改变 geometry、RGB、opacity | 普通三维语义 sidecar 能达到怎样的可复现基线 | 74-view mIoU 0.616103；`stay_cable` IoU 0.068759；background→stay_cable 13,166,331 px；RGB差0 | **保留，当前唯一合格三维语义基线** |
| `SEM-002` / S1 | 按 Gaussian 对像素的实际贡献筛选监督；测试累计贡献0.95和相对最大贡献0.05两种门 | 低贡献 anchor 是否在细拉索像素上收到错误监督 | 最强候选 `SEM-022`：mIoU +1.51 pp、误报−760,439 px、`stay_cable` IoU +0.87 pp，但 tower IoU −0.89 pp | **部分支持但 drop**：破坏保护类，不能作为可靠升级 |
| `SEM-003` / S2 | 只在7,681个训练侧 background/`stay_cable` 冲突 anchor 上加入 Dirichlet concentration regularizer | 显式压制跨视角类别冲突能否减少背景误报 | 相对匹配CTRL，mIoU −0.1011 pp、`stay_cable` IoU −0.1925 pp、误报 +345,469 px | **drop**：目标指标反向恶化 |
| `SEM-004` / S3 | 给7,069个冲突 anchor 增加与官方观察方向相关的一阶残差，只调 background/`stay_cable` 相对 logit | 静态 anchor 是否需要轻量视角条件语义 | mIoU仅 +0.0123 pp，`stay_cable` IoU +0.0197 pp，误报−0.2222%；RGB差0 | **drop / unproven**：方向正确但量级近乎等价 |
| `SEM-005` / U1-A | SegFormer-B2 三折空间 OOF 二维分割与伪标签教师资格审计；不是3D模型 | 图像中是否有可识别拉索信息，以及二维教师能否安全给100张无标注图生成伪标签 | 二维74-view mIoU 0.804696、`stay_cable` IoU 0.750613；但冻结阈值下验证选择性precision 0.911232<0.95，折一致性也失败 | **not eligible**：证明二维信息存在，但未生成伪标签、未训练半监督3D |
| `SEM-006` / Cable3D | 计划用训练侧多视图GT中心线三角化 semantic-only微高斯；先执行 `SEM-060-CABLE-GEO` 几何门 | 细拉索瓶颈是否主要来自错误的三维表示/anchor绑定 | 40-view训练侧独立check：2 px precision 0.10423、误差中位/P90 16.55/65.44 px、>8 px远误报69.11%；尺度门失败 | **drop_geometry_not_reliable**：只完成几何预检；微高斯、`SEM-061/062`及10k均未实现或训练 |
| `SEM-070` / POST-CLEAN | 冻结 SEM-001，只对训练侧 dominant-anchor 做保守 `stay_cable→background` 概率转移 | 在不伤害真实拉索的前提下清除跨视角稳定误报 | 五阈值 check 误报下降63.01%–90.61%，但 recall 仅保留30.33%–61.47%，均低于98%；74-view raw/cleaned完全一致；RGB差0、checkpoint/官方相机门通过 | **show-only / fallback**：没有安全可清除的 anchor，仍使用 SEM-001 |
| `SEM-080` / STRUCTURAL-DIAG | 冻结 SEM-001 后的逐视角 tower-to-deck 连续线谓词；不是 anchor 删除 | 是否能只保留桥梁结构索线、清除水面/天空黄色包络 | 219训练视角检出12,351条结构线；check background→`stay_cable` −94.45%，但 recall仅保留12.49%、mIoU −4.03pp；样例还显示正常主缆/吊索被删；RGB差0，74-view未读 | **drop**：错误的tower→deck拓扑假设，不作展示或正式候选 |
| `SEM-090` / CABLE-NETWORK-DIAG | 在SEM-080失败后加入主缆、吊索、斜拉索三类Hough模板 | 多类线网能否既保留正常索又去掉江水包络 | 8训练视角预检找到三类候选，但样例仍保留江水大块黄色；RGB差0，219-view/74-view未读 | **preflight drop**：raw语义Hough会自证伪线，不能作为独立物理证据 |
| `SEM-100` / RENDER-SEG-PF | 将冻结三折二维OOF教师的输入从真实RGB换为同相机冻结Scaffold渲染RGB；不训练 | 渲染RGB是否仍含可用的索线证据 | 8训练OOF视角：`stay_cable` recall 95.90%（真实95.18%），但IoU −11.42pp、background→`stay_cable` +115,339 px；RGB差0，验证未读 | **preflight supports new preregistration**：有索线证据但域误差显著，不能直接作为外部相机模型 |
| `SEM-101` / RENDERED-SEG-OOF | 在同相机冻结Scaffold渲染RGB上复用SEM-051的三折SegFormer-B2训练配方 | 渲染域训练能否修复SEM-100的二维误报漂移 | 219训练图严格OOF：mIoU 84.97%、`stay_cable` IoU 83.61%、P/R 86.09%/96.67%、背景误报率1.41%；五门全过，RGB差0、74-view未读 | **qualified for separate final-model registration**：仍不是三维候选或SEM-001替代 |
| `SEM-102` / RENDERED-SEG-FINAL | 将SEM-101合格三折配方固定为全219图单一final-3k checkpoint，再作一次性审计 | 渲染域二维模型能否在未见的74视角和官方相机上超过SEM-001保留门 | 74-view mIoU 0.750301，`stay_cable` IoU/recall 0.685329/0.810826，background→`stay_cable` −96.05%；但deck/tower为−2.42/−9.32pp，最差10%为−7.17pp；RGB=0、重载与外部相机均通过 | **drop / show-only**：三个保护门失败；不重训、不重选，SEM-001保持回退 |
| `SEM-110` / PROTECTED-FUSION-OOF | 只用SEM-101空间OOF二维概率替换SEM-001的background/`stay_cable`决策；其deck/tower/foundation预测逐像素冻结 | 互补两路语义能否清理背景黄色而不伤桥梁结构 | 108-view calibration 选0.80；111-view check mIoU 0.807461、索IoU/recall 0.871190/0.945650、background→索−93.28%，三类保护预测完全一致 | **qualified for final**：训练侧通过；74-view未读 |
| `SEM-111` / PROTECTED-FUSION-FINAL | 以SEM-110固定0.80规则融合冻结SEM-001类图和SEM-102概率；不训练、不调参 | 是否同时保住桥面/塔柱/基础并改善拉索误报 | 74-view mIoU 0.760635、索IoU/recall 0.638137/0.711445、background→索−97.36%；deck/tower/foundation逐像素与SEM-001一致；最差10% +0.52pp；RGB字节相同与外部相机通过 | **历史正式低误报候选**：SEM-001保留为回退 |
| `SEM-112` / EDGE-RECOVERY-OOF | 从SEM-110高置信0.80结果出发，只补回同时有冻结RGB边缘、SEM-001索类和高置信索种子邻接的低置信索像素 | 是否能提升细索recall/IoU而不重新引入背景黄区 | 108-view calibration 三个分位均失败；最佳0.90仅mIoU +0.04pp、索IoU +0.18pp、recall +0.64pp，未达+0.20pp质量门；check/74-view未读 | **drop**：SEM-111候选和SEM-001回退均不变 |
| `SEM-120` / DEV-THRESHOLD-SWEEP | 只扫描SEM-111保护融合的 `stay_cable` 阈值；deck/tower/foundation逐像素不变 | 在开发集上找到更好的索线完整性—背景误报折中 | 开发分数选0.20：mIoU 0.770685、索IoU/recall 0.687827/0.858168、背景→索743,906；0.30取得最高mIoU/索IoU 0.771317/0.690498，背景→索629,688 | **completed development-only**：两者供视觉比较，不构成新盲测或正式替代声明 |
| `VIS-004` / DEV-THRESHOLD-FLYTHROUGH | 在144个同pose新相机位姿上并排渲染SEM-111 0.80、SEM-120的0.20与0.30；无标签/参考RGB | 数字折中在连续视角上是否保住桥索、避免大片黄色扩张 | 144帧/6秒视频完成；0.20比0.30多347,371索像素，frame0091显示其扩张低置信黄色区；0.30保留主要恢复线且三类结构逐像素相同 | **completed developer visualization**：0.30作为下一开发机制视觉基线，非正式替代 |
| `SEM-121` / CONTINUOUS-FUSION-OOF | 一次拟合连续二路校准替换0.30硬阈值；仅可变background/`stay_cable`通道，三类结构冻结 | SEM-001索先验能否抑制低置信背景扩张，同时保住0.30主缆恢复 | 111 check：mIoU/索IoU/recall/背景→索从0.790574/0.798087/0.970561/2,812,695变为0.778500/0.746121/0.975556/3,948,987；三类结构逐像素一致，74-view未读 | **drop**：二元先验不能分离犹豫带；禁止在本实验重调公式 |
| `SEM-122` / UNCERTAINTY-BAND-DIAG | 对固定0.20−0.30索像素带只读测量，不写新预测 | 哪些不变信号真正区分新增真拉索与背景黄色误报 | 186,758 px中真索37.63%、背景61.16%；96.48%贴0.30种子但真索仅37.96%，1–2 px环为44.95%；无RGB/预测变化 | **completed diagnostic**：连通性单独不足，不产生候选 |
| `SEM-123` / BAND-RECOVERY-DEV-SWEEP | 从0.30保护融合起，只选0.20−0.30带的固定联合子集补回 | 三项弱证据的交集能否比0.30更好地恢复真索而不重染背景 | 分数leader `p20_d02_any`：mIoU/索IoU/recall +0.029/+0.161/+1.908pp、背景→索+73,598；保守Pareto `p20_d04_sem001_stay_cable`：表中最高mIoU/索IoU且背景→索仅+28,552；结构类完全一致 | **completed developer sweep**：两候选待同pose视觉复核 |
| `VIS-005` / SELECTIVE-RECOVERY-FLYTHROUGH | 在VIS-001的144个新pose并排渲染hard 0.30、SEM-123分数leader和保守Pareto；无标签/参考RGB | SEM-123的定量补回是否会在连续新视角中重新扩张成背景黄区 | 144帧/6秒完成；leader/保守相对0.30新增242,108/98,369索像素，最高新增帧中增量集中在既有索/塔邻域，未见VIS-004硬0.20式大片新黄区；结构类逐帧一致且不删除0.30已有索 | **completed developer visualization**：外观复核通过，但无GT视频不构成泛化或替代结论 |
| `SEM-130` / ANCHOR-CROSSVIEW-CONSENSUS-OOF | 只在SEM-123 `p20_d02_any`新增像素上要求同一dominant frozen anchor 获得其他同空间OOF折相机支持；从不删hard-0.30索或保护结构 | 跨视角重复二维证据能否区分真实桥索与黄色背景 | 219训练相机得到345,375条支持观测；108-view calibration中K=1/2/3/4/6/8均低于hard-0.30的mIoU/索IoU。最佳K=8为0.795336/0.817223，仍低于0.796303/0.821512；不读check/74-view，RGB差0 | **drop**：背景伪索也在其他视角获得同anchor OOF支持；不在本实验调阈值或来源规则 |
| `SEM-140` / CABLE-BINARY-OOF | 在原生960分辨率上把五类教师改为background/`stay_cable`专用二分类，仍按三折空间OOF训练；结构类只由SEM-001提供 | 专用二分类是否能恢复细索并降低五类竞争造成的漏检 | 111-view独立check相对SEM-110：mIoU +0.64pp、索IoU +2.83pp、recall +1.37pp、背景误报−23.66%；三类结构逐像素不变 | **qualified for final**：0.70与完整配方冻结后只允许SEM-141一次最终转换 |
| `SEM-141` / CABLE-BINARY-FINAL | 用SEM-140冻结配方在219图训练唯一final-3k，再以0.70融合SEM-001保护结构 | 专用高分辨率拉索分支能否直接提高开发语义分数，同时不改结构类 | 74-view开发集相对SEM-111：mIoU +1.43pp、索IoU +6.88pp、recall +10.29pp；worst-10%/140–146 +1.64/+1.58pp；结构逐位一致，RGB/重载/外部相机通过；背景误报+28.91% | **保留为SEM-151直接回退**：冻结checkpoint/0.70；SEM-111/SEM-001继续保留 |
| `SEM-150` / FOUNDATION-GATE-OOF | 仅用冻结SEM-101 OOF class-4概率把SEM-140类比基线已有foundation单向拒绝为background；1/2/3不变 | 现有二维教师能否安全审核高召回、低precision的foundation | 111-view check：foundation IoU/precision +25.81/+26.29pp、recall 98.60%、总mIoU +5.35pp、background→foundation −66.84%，1/2/3逐位一致 | **qualified for final**：0.10只由calibration选择并冻结；74-view未读 |
| `SEM-151` / FOUNDATION-GATE-FINAL | 对SEM-141已有foundation应用SEM-150冻结0.10与SEM-102 class-4概率；只允许4→0 | 能否解决foundation低precision，同时不动桥面、塔柱和拉索 | 74-view相对SEM-141：总mIoU +5.60pp，foundation IoU/precision +27.90/+31.54pp、recall 91.19%，background→foundation −76.12%；70升4平0降；1/2/3逐位一致，RGB/重载/外部相机通过 | **保留为SEM-161直接回退**：冻结阈值、父模型和规则，SEM-141/111/001继续保留 |
| `SEM-160` / DECK-BINARY-OOF | 原生960分辨率non-deck/deck三折空间OOF；只允许foundation门前后均为background的位置补为deck | 独立桥面分支能否修补SEM-151在斜视角的桥面碎裂 | 111-view check相对SEM-150：deck IoU/recall +0.437/+0.841pp、mIoU +0.093pp、precision 95.25%，保护像素逐位一致 | **qualified for final**：0.99与配方冻结，只允许SEM-161一次final转换 |
| `SEM-161` / DECK-BINARY-FINAL | 全219图训练唯一final-3k，以冻结0.99在SEM-151原background中单向0→deck | 是否提升桥面完整性，尤其158–161斜视角，同时保留其他语义 | 74-view相对SEM-151：mIoU +0.294pp，deck IoU/recall +1.367/+2.190pp、precision 95.04%；158–161均值+2.690pp且四帧均升；63升3平8降，保护/RGB/重载/外部相机全过 | **当前最强开发候选**：推荐隐藏评测；079–082与1.24866倍背景误补风险由SEM-151直接回退覆盖 |

因此目前可以对外表述为：原始 Scaffold-GS 仍是冻结 RGB 骨干优胜者；当前最强开发语义组合
是 `SEM-161 = SEM-141拉索二分类保护融合 + SEM-151冻结foundation审核 + SEM-161冻结deck补洞`。
它先单向拒绝低置信foundation，再只把foundation门前后均为background的位置补为deck；既有
`stay_cable`、tower、foundation和deck从不被删除。SEM-151/141/111作为逐级回退，SEM-001仍是
可立即回退的独立静态三维语义方案；SEM-161的最终泛化结论必须由官方隐藏评测给出。

`SEM-070-POST-CLEAN` 已完成上述唯一一次注册评测：只读取219张训练侧标签计算 dominant-anchor 统计，按固定 hash parity 在 calibration/check 两半比较 `0.50/0.60/0.70/0.80/0.90`，由于所有阈值都不能保留98%的真实 `stay_cable` recall，冻结全零 `[anchor_count]` boolean mask 后读取74张验证图。结果 raw/cleaned 完全一致，background→`stay_cable` 没有下降，因此正式替代门失败；该结果不支持继续在 SEM-070 内调参。

`SEM-130-ANCHOR-CROSSVIEW-CONSENSUS-OOF` 则测试了不同于SEM-070的局部正向证据：先固定SEM-123 `p20_d02_any`的候选补回，再仅保留那些dominant frozen anchor 在其他相机也获得OOF `stay_cable`支持的新增像素。来源被严格限制到目标所在空间OOF折的其他相机，且该折模型未见目标相机；219张训练相机的无标签阶段产生345,375条支持观测。可是在108张独立calibration中，K=8虽然将raw恢复的243,164个像素压至94,503个，仍保留79,724个background而仅12,932个真实`stay_cable`补回。其mIoU/索IoU为0.795336/0.817223，低于hard-0.30的0.796303/0.821512；K=1/2/3/4/6也未达门。这个结果表明多视角的同源二维教师会一致地确认黄色背景伪索，不能作为独立物理判据。按协议不读111张check或74张开发标签，不在SEM-130内重调；训练相机RGB前后隔离最大差为0。

`SEM-080-STRUCTURAL-DIAG` 是另立的、训练侧限定的逐视角机制：raw `stay_cable` 只有落在连续、近竖直且端点接触预测 tower/deck 的 Hough 线附近才保留，否则概率质量移至background。它在219张训练图中检出12,351条结构线，证明实现不是全零；check half清除了94.45%的background→`stay_cable`，但使`stay_cable` recall仅保留12.49%、mIoU下降4.03pp。样例复核还显示红色删区包含正常主缆、吊索和部分斜拉索：tower→deck直连不覆盖主缆的tower↔tower/锚点和吊索的deck↔主缆连接。访问日志、checkpoint重载和RGB=0都通过，74-view、无标注和外部相机均未读。故该规则视觉与度量均为drop，不以验证集调参或作为展示。

`SEM-090-CABLE-NETWORK-DIAG` 是SEM-080之后独立注册的三类线网诊断：主缆tower-top↔tower-top/边界、吊索deck↔主缆、斜拉索tower↔deck。8张训练视角GPU预检工程门通过且能检测三类候选，但样例仍把左侧大块江水黄色保留为主缆。原因是Hough线从同一张错误语义图提取，tower-top→边界模板会让江面伪线自证为桥索。故视觉门即刻失败，不运行219-view或74-view，也不在此实验调参。

`SEM-100-RENDER-SEG-PF` 转而使用独立的冻结渲染RGB作为二维教师输入，在8个训练侧OOF视角上直接量化域差。它没有训练新参数，也没有读取74-view、无标注或隐藏输入：真实/渲染预测均先完成，再读取训练标签。结果中，渲染输入的 `stay_cable` recall 为95.90%，说明桥梁索线证据没有消失；但 `stay_cable` IoU低11.42pp，background→`stay_cable`多115,339像素，样例031出现水面/远景黄色误报。因此下一步不能直接部署真实RGB教师，而是应在新的ID内预注册训练侧冻结渲染RGB上的二维训练与空间OOF；在模型冻结前不读74-view。该预检的RGB artifact、重复渲染PNG hash均一致，最大RGB差为0。

`SEM-101-RENDERED-SEG-OOF` 随后只在这219张同相机冻结渲染RGB上，按SEM-051不变的三折空间OOF配方训练。三折产生95/84/40张不重叠OOF预测，合并mIoU为84.97%，`stay_cable` IoU/precision/recall为83.61%/86.09%/96.67%，background→`stay_cable`占真实背景1.41%，预注册五门全通过。三折checkpoint均重载成功，RGB artifact/三张重复PNG hash相同且最大差0。样例误差主要为标签边界及细线偏移，不再出现SEM-080那种删除正常索线的机制性失败。这证明训练侧的渲染域漂移可以被相同二维配方吸收；但没有读取74-view、无标注、隐藏输入或外部相机，因此不构成最终质量或SEM-001替代证据。

`SEM-102-RENDERED-SEG-FINAL` 将唯一合格的SEM-101配方固定为全219图final-3k后，严格按“checkpoint重载→无标签RGB渲染→写完74份预测→一次性读标签”的顺序完成审计。它在全局和索类上大幅改善：mIoU从0.616103升至0.750301，`stay_cable` IoU/recall从0.068759/0.368036升至0.685329/0.810826，background→`stay_cable`从13,166,331降至520,271（−96.05%）。但deck/tower IoU分别下降2.42/9.32pp，最差10%不含背景mIoU下降7.17pp，均越过预注册保护边界；代表性四联图还显示细索、桥面边界与局部塔体残差。因此候选正式drop/show-only，不因总体数字漂亮而绕过保护门。RGB artifact/像素=0、checkpoint重载及官方外部相机（RGB/五logits/0–4 PNG）均通过；但它们只证明工程接口，不改变质量失败结论。

`SEM-110-PROTECTED-FUSION-OOF` 随后只在219训练图检验一个新的推理规则：冻结
SEM-001预测为deck、tower、foundation的像素永远不变，其他像素仅由空间OOF二维教师
在background/`stay_cable`之间决定。0.80阈值仅由108张calibration最大化误报下降而选，
并在111张独立check上使background→索下降93.28%、索IoU/recall为87.12%/94.56%、
mIoU升至80.75%；所有保护像素严格一致。该训练侧结果只允许新ID最终审计。

`SEM-111-PROTECTED-FUSION-FINAL` 固定使用上述0.80，先写出全部74张融合图才读取标签。
它通过每项正式保留门：mIoU为0.760635，`stay_cable` IoU/recall为0.638137/0.711445，
background→索从13,166,331降至347,412（−97.36%）；deck/tower/foundation的预测及IoU
与SEM-001逐像素一致，最差10%从0.423440升至0.428629，140–146升至0.758237。RGB字节
相同、checkpoint重载及官方`CAMERA_FROM_WORLD + SIMPLE_RADIAL`输出也通过。因此它成为
当前合格候选，但SEM-001仍随部署包保留为独立回退。

## 已冻结的决定

1. 独立 Proxy 不再作为比赛主线；保留为问题分析和负面实验。
2. RGB 骨干必须经过统一数据、划分、预算和评测的公平比较后冻结。
3. 冻结后，语义训练不得改变 RGB 几何、颜色和 opacity。
4. 100 张无标注图像优先用于 RGB 重建；语义伪标签必须单独证明增益。
5. 最终渲染入口必须接受官方 COLMAP 参考坐标系中的外部相机参数。
6. 原始数据、checkpoint、PLY、日志和训练输出不得进入 Git。

完整依据和可推翻条件见 [DECISIONS.md](../../DECISIONS.md)。

## 早期 E1/E2 历史参考（不同协议）

以下数字来自早期实验报告或用户提供的实验总结，原始实验目录当前不在本工作树，
且数据/划分/实现协议与上面的当前 R1/R2 主线不同。它们只用于追溯，不能与
`SEM-001` 至 `SEM-006` 的74-view结果直接排名，也不覆盖上面的当前保留结论；
registry 中无法确认的字段标为 `not_available`。

| 项目 | 当前结果 | 证据状态 |
|---|---:|---|
| RGB留出视角PSNR | 19.013 dB | 用户提供的E1总结 |
| RGB留出视角SSIM | 0.7640 | 用户提供的E1总结 |
| RGB留出视角LPIPS-Squeeze | 0.3849 | 用户提供的E1总结 |
| 普通语义3DGS mIoU | 0.579755 | 用户提供的E2总结 |
| `stay_cable` IoU | 0.138621 | 用户提供的E2总结 |
| 三种子共同背景→拉索误报 | 约517万像素 | 用户提供的E2总结 |

R1 冻结的同条件400-RGB基线如下（74个空间隔离验证视角；LPIPS为AlexNet）：

| 骨干 | PSNR | SSIM | LPIPS | 冻结结论 |
|---|---:|---:|---:|---|
| Standard 3DGS | 15.04685 | 0.80318 | 0.45093 | 保留工程回退 |
| Mip-Splatting | 14.22588 | 0.78709 | 0.50359 | 淘汰 |
| Scaffold-GS | 20.59640 | 0.85018 | 0.32237 | 冻结用于R2 |

自动过滤曾将背景→拉索误报减少 93.43%，但 mIoU 从 0.579755 降至 0.557671，`stay_cable` IoU 从 0.138621 降至 0.062299；真实细拉索保留率只有 28.73%。这说明当前硬过滤不能替代普通语义基线。

完整30,000-iteration运行已完成 Standard-300/400、Mip-400 和 Scaffold-400；均具备冻结 AlexNet LPIPS、checkpoint重载及外部官方相机门。关闭CloudCompare/Chrome并设置`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`后，Standard-400在其余冻结协议完全不变的条件下完成。Scaffold-400 在同条件400-RGB比较中明显优于Standard和Mip（PSNR 20.59640 vs 15.04685/14.22588，SSIM 0.85018 vs 0.80318/0.78709，LPIPS 0.32237 vs 0.45093/0.50359）；最终前二名Standard与Scaffold均通过五类语义sidecar烟雾测试。RGB-060已冻结Scaffold-GS。

`VIS-002-STANDARD-FLYTHROUGH` 已使用冻结 `RGB-011-R2-ALLOC` Standard-400 和与Scaffold视频完全相同的144个仅相机元数据插值pose生成6秒RGB H.264视频；不训练、不读取参考RGB/标签。逐帧1320×989 RGB、视频144帧/6秒、冻结PLY和无参考数据边界均已审计通过。该视频如实保留Standard在中段新视角的泛白/floaters，只作可视化对照，不改变Scaffold冻结决定。见 `experiments/runs/VIS-002-STANDARD-FLYTHROUGH/report.md`。

SEM-001 是已完成的冻结 Scaffold RGB 静态五类基线：219 张带标注空间训练侧训练、74 张空间隔离标注相机验证、10,000 iteration。其本地 mIoU（含/不含 background）为 0.61610 / 0.56334；deck/tower IoU 为 0.90100 / 0.83487，但 `stay_cable` IoU 仅 0.06876、precision 0.07796，background→stay_cable 误报为 13,166,331 像素（真实背景 15.0076%）。冻结 RGB artifact hash、三相机 RGB PNG hash 与逐像素最大差均为 0；checkpoint 重载、独立外部 COLMAP `CAMERA_FROM_WORLD` 相机 RGB+五 logits+类别 PNG 均通过。完整结果和限制见 `experiments/runs/SEM-001/report.md`。

SEM-002/S1 已完成：disabled-gate 与 SEM-001 logits/类别图严格一致、RGB差为0；贡献 adapter 的全保留数值差在 `2e-5` 预注册容差内。仅用219个语义训练相机冻结累计0.95与相对最大0.05门控，两种8-iteration CUDA、RGB隔离、重载和官方外部相机烟雾均通过。两个74视角10k主折均改善背景→stay_cable：SEM-021为−37,017 px（mIoU +0.80pp），但不足+1pp且tower −0.54pp；SEM-022为−760,439 px、mIoU +1.51pp、stay_cable IoU +0.87pp，但tower −0.89pp超过−0.5pp保护线。故S1未证实，SEM-001保留；不启动SEM-023、S2/S3或三折三seed。见`experiments/runs/SEM-002/report.md`。

SEM-003/S2 已完成独立预注册验证：219张训练图的exact all-contributor诊断从189,073个anchor冻结7,681个bg/stay_cable冲突且保护类证据低的eligible anchors，全程未使用验证、无标注或隐藏测试信息。Dirichlet表示CTRL几乎复现SEM-001（mIoU 0.61621 vs 0.61610），候选则为0.61520；相对CTRL，stay_cable IoU从0.06887降至0.06694，background→stay_cable从13,106,856增加到13,452,325（+345,469 px），困难140–146略降，只有最差10%略升。两组RGB最大差均为0，checkpoint重载和官方外部相机门均通过。故限定冲突正则按主折门drop，不调lambda/阈值、不做三折三seed，也不启动SEM-004；见`experiments/runs/SEM-003/report.md`。

SEM-004-DIAG 是后续独立机制的只读可行性诊断，绝非SEM-003重试：固定SEM-001在219训练/74验证只读视角重渲染并精确解码冻结RGB贡献。验证完全复现SEM-001（background→stay_cable 13,166,331）；训练误报93.54%、验证误报97.77%距离任意GT cable超过8 px。788个训练stay_cable多边形的72.21%满足预注册细线阈值；误报区主导anchor落入既有训练侧冲突mask的概率为6.401%，正确背景为0.0935%，训练半部固定方向桶有7,069/14,863（47.56%）稳定anchor复现。该四项关联使其`hypothesis_ready_for_new_preregistration`，但SEM-001仍是唯一冻结语义基线；S3必须另立Goal、只改一个变量并取得运行授权。全部诊断保持RGB=0、checkpoint重载和外部官方相机接口。见`experiments/runs/SEM-004-DIAG/report.md`。

SEM-004/S3 已在独立 Goal 中完成：`SEM-040-EQ` 的零残差概率差为0、类别图一致且RGB差0；`SEM-041-PF` 证明只有7,069×3方向系数有梯度，峰值792 MiB，重载和外部相机门通过。随后单次10k主折耗时3,294.01 s、峰值822 MiB，mIoU为0.616226（相对SEM-001仅+0.000123），stay_cable IoU为0.068957（+0.000197），background→stay_cable为13,137,079（−29,252 px、−0.2222%）。保护类、最差10%、140–146、RGB=0、重载和外部相机均通过，但未达到+1pp mIoU和至少1%误报下降门，故`drop_unproven_main_fold`；不改cap/mask/L2/LR/seed，不启动S4、伪标签或三折三seed。完整证据见`experiments/runs/SEM-004/report.md`。

SEM-005-PF/U1-A 已完成独立SegFormer-B2三折空间OOF审计：219张SEM-001训练侧图恰好各有一次OOF预测，固定74-view在阈值提交前未打开。三折训练合计0.2468 GPU h、峰值约3,226 MiB；训练侧OOF冻结stay_cable阈值0.911得到precision/recall 0.950004/0.903823和99.54%视角覆盖。一次性74-view审计的二维mIoU为0.804696，但同一阈值的stay_cable选择性precision仅0.911232，低于0.95；fold 0/2的独立precision也低于0.95一致性门。故结论为`not_eligible_for_pseudo_labels`，不对100张无标注图生成训练用soft labels，不启动3D半监督，SEM-001保持冻结。完整证据见`experiments/runs/SEM-052-PL-AUDIT/report.md`。

SEM-060-CABLE-GEO 已完成不训练模型的训练侧拉索三维几何预检：由219张SEM-001训练侧标签确定性划分177 fit、40 check、2 local guard，固定74-view、无标注图和隐藏输入均未加载。fit得到24,825个至少三独立相机支持的候选和79条分段直线，但check的2px precision仅0.10423（门槛0.90），中位/P90投影误差16.55/65.44px，69.11%投影像素距真值中心线超过8px，且候选尺度审计失败；只有2px recall 0.41090和40/40覆盖通过。因此登记`drop_geometry_not_reliable`，不改阈值/split/seed，不实现SEM-061/062或训练SEM-006，SEM-001继续为冻结baseline。完整证据见`experiments/runs/SEM-060-CABLE-GEO/report.md`。

2026-08-03 主机审计已确认冻结 Scaffold-GS 上游 commit `59c833b56bbbf510f3f64d40f81721995caced66`、checkpoint、PLY、config、74视角评测和数据 manifest 均存在；当前工作树仍有用户既有未提交改动，未提交或推送。新增的 `GEO-000` 使用同一 `rgb_r1_spatial_v1`、同一 400-RGB准备场景和同一评测入口运行 3,000 iteration：PSNR 20.06078、SSIM 0.83855、LPIPS-Alex 0.37803，checkpoint 重载通过。它只是后续几何候选的公平3k控制，不能替代30k RGB-030-FULL。

官方相机解析、四元数/位姿闭环、投影/反投影、Sim(3) Umeyama测试已通过。由400个官方相机生成的 camera graph 有400节点、36,928条共视边；航段切分输出24个最小4图航段，保留33个候选跳变、接受23个断点，明确不把400张图当连续视频。MapAnything Apache 已在隔离环境完成真实 16图 smoke、303训练图/32窗口离线几何、P0融合和同预算10k；DA3-Large 已完成真实8图 smoke、303训练图/70窗口离线几何、融合与3k Scaffold 接入，但质量门淘汰；VGGT-Ω、LingBot 仍无真实模型执行。四模型共同的连续0→10k、Top2→30k、静态语义和跨空间折规则已由 `docs/experiments/geometry_prior_evaluation_v2.md` 冻结。

新鲜 `GEO-000-10K` 控制在74个空间验证视图取得 PSNR/SSIM/LPIPS-Alex 为 20.48798/0.85027/0.33167；MapAnything P0 候选为 20.48281/0.84441/0.35755。候选 PSNR几乎持平，但 LPIPS恶化0.02587、SSIM下降0.00586、最差10% PSNR下降0.75640 dB，140–146同样退化；两者均通过 checkpoint 重载与官方外部 `CAMERA_FROM_WORLD` 相机 1320×989 RGB 门。因此该 P0 xyz+原始RGB融合配方淘汰且不进入30k，冻结 Scaffold 原初始化保持。完整证据见 `GEO-010-PF-R1`、`GEO-010-P0`、`GEO-000-10K` 和 `GEO-010-10K` 报告。

`GEO-011-3K` 是唯一允许的 MapAnything 添加式止损验证：完整保留 72,292 个原始初始化点，
只新增 1,719 个经过三视角、置信度和低覆盖过滤的普通表面点。连续 0→3k、74视角固定评测、
checkpoint 重载和外部官方相机门均完成，但相对相同 `GEO-000-10K` 3k 控制仅 PSNR +0.01864 dB、
LPIPS-Alex +0.00437、最差10% PSNR -0.16258 dB、困难140–146 PSNR -0.12083 dB，未达预注册
PSNR +0.2 或 LPIPS -0.01 门。因此该配方也淘汰，不启动其10k、30k或重复 seed；完整证据见
`experiments/runs/GEO-011-3K/report.md`。

## 当前主线

```text
Standard / Mip-Splatting / Scaffold-GS公平比较
                         ↓
                    冻结RGB骨干
                         ↓
                  五类Semantic GS
                         ↓
             Visibility-aware supervision
                         ↓
             Gaussian跨视角冲突统计
                         ↓
     bg/stay_cable冲突点的view-conditioned residual
                         ↓
      SEM-070 conservative anchor post-clean candidate
                         ↓
      SEM-080 tower-to-deck structural-clean diagnostic (drop)
                         ↓
      SEM-090 three-family raw-semantic line network (preflight drop)
                         ↓
      SEM-100 frozen rendered-RGB domain preflight (supports new preregistration)
                         ↓
      SEM-101 rendered-domain SegFormer spatial OOF (five gates passed)
                         ↓
      SEM-102 final rendered-domain checkpoint → once-only 74-view audit
                         ↓
      deck/tower/worst-10% retention failed → protected fusion hypothesis
                         ↓
      SEM-110 training-only OOF (passes; threshold 0.80 frozen)
                         ↓
      SEM-111 final fusion audit (historical formal gates passed)
                         ↓
      SEM-120 completed 74-view developer threshold sweep (0.20 leader; 0.30 Pareto alternative)
                         ↓
      SEM-121 continuous two-stream calibration (111-view check drop; 74-view unread)
                         ↓
      SEM-122 read-only 0.20--0.30 uncertainty-band diagnosis (background-majority)
                         ↓
      SEM-123 joint selective-recovery developer sweep (two developer visual candidates)
                         ↓
      VIS-005 same-pose no-GT visual review (no broad hard-0.20-style expansion)
                         ↓
      SEM-130 strict same-fold cross-view anchor consensus (calibration drop; check/74 unread)
                         ↓
      SEM-140 native-resolution cable-binary spatial OOF (training/check gates passed)
                         ↓
      SEM-141 all-training final + frozen 0.70 developer audit (all targets passed)
                         ↓
      SEM-150 foundation one-way OOF gate (0.10 frozen on training only)
                         ↓
      SEM-151 frozen final audit (all targets passed; direct rollback)
                         ↓
      SEM-160 native-resolution deck-binary spatial OOF (0.99 frozen)
                         ↓
      SEM-161 all-training final + frozen 0.99 audit (all targets passed; current best)
                         ↓
      SEM-170 bidirectional deck arbitration (calibration drop; true-deck deletion)
                         ↓
      SEM-171 protected morphological gap recovery (calibration drop; cable-majority additions)
                         ↓
      SEM-172 fixed deck-interior loss OOF (calibration drop; precision/recall trade)
                         ↓
      SEM-173 frozen probability ensemble (calibration drop; current series closed)
                         ↓
      SUBMIT-020 content-addressed freeze of RGB-030-FULL + SEM-161
                         ↓
      SEM-200/201 discrete frozen-depth deck projection (precision/coverage failure)
                         ↓
      SEM-210 continuous XYZ field → SEM-211 appearance/XYZ conjunction (both drop)
                         ↓
      SEM-220 matched RGB+depth+normal early fusion, 3-fold 9k (tiny gain; gate fail)
                         ↓
      SEM-221 bidirectional RGBD arbitration (true-edge deletion; research line closed)
                         ↓
      SEM-230 per-offset semantic residual OOF (all RGB/protection gates pass; deck IoU −3.259pp; calibration drop)

RGB-030-FULL direct render (no post-training geometry cleanup)
                         ↓
      RGB-070 nearest-camera appearance fallback (all global metrics regress; drop)
                         ↓
      RGB-071 RGB/semantic coupling audit (reconstruction limits thin-cable recall only)
                         ↓
      RGB-072 training support/residual audit (low support alone is not separable)
                         ↓
      RGB-073 exact replay gate stop → RGB-074 fixed 29-anchor numerical audit
                         ↓
      tiny numeric gain but no perceptible floater removal → do not promote; keep RGB-030-FULL
```

## 下一步

1. SEM329的固定最终三臂验证已完成，有小幅正向IoU与召回收益，作为优先研究起点保留。若继续优化，另立训练侧OOF剩余误删诊断，先区分第二源无覆盖与匹配标签冲突，再选择一个推理可得的新变量；该后续尚未执行，不在74扫阈值/源/区域，不因未满90%而否定当前增益。
2. 方法选定后另注册all400 RGB/all300标签最终拟合并复现；RGB域变化需重建对应语义头，原开发分数不得当成独立评估。当前SEM161及回退链保持只读。
3. 论文/视频/PPT继续后置于方法选择，期间持续记录正负结果、来源和回退；无官方上传或Git push。

三线协议与 SEM-230 代码审计解释修正见 `docs/experiments/long_task_20260907.md`。SEM-230 的失败配方仍淘汰，但旧日志不足以证明像素责任权重或训练后外相机门已通过。

## 当前阻塞与待决定

- 2026-09-08 / SEM-310更新：用户截图已确认本机登录成功，随后报告DINOv3申请被拒；原SEM-310仍保留该独立HF权重授权阻塞。替代DINOv2研究及RoMa v2官方完整权重的独立来源/工程/实际GPU匹配与SEM325–329质量实验现已完成，不能再将该HF申请视为这些已完成实验的阻塞；无需重建环境。

- 2026-08-05 GPU根因已辨明并绕过：驱动内核/用户态均为580.173.02且主机RTX 5070正常；受限命令命名空间以空tmpfs覆盖`/dev`，隐藏了真实`/dev/nvidia*`。主机执行上下文的`nvidia-smi`和冻结Python CUDA探针均通过，SEM-003全部GPU任务随后完成。无需重装驱动，也与Docker数据迁移或显存不足无关。
- 隔离 CUDA 12.8 / PyTorch 2.11.0+cu128 环境与三个固定上游均已在 Git 忽略位置建立，并完成真实 CUDA 编译和 500-iteration预检；受限 shell 的 CUDA gate 曾为 false，但主机上下文 RTX 5070 gate 已恢复并完成 SEM-070 full，不应重装依赖或改用 CPU 冒充正式结果。
- R1没有冻结阻塞；12GB显存运行Standard时必须关闭大型桌面GPU应用并使用记录在案的allocator设置。RGB-041-PF确认关闭Handy后可稳定完成1k控制；LPIPS AlexNet已冻结并完成全部完整结果的补算。
- R1 的执行记录、停止条件和补丁边界见[环境预检与授权边界](../../docs/reproduction/r1_preflight.md)。完整训练已获授权，但 OOM、磁盘低于100GB或共同协议变更仍会立即停止并登记。
- `SEM-001-PF` 的两项启动前环境缺口已修复并保留失败记录；SEM-001 随后完成真实 10k 训练（3,076.65 s、824 MiB PyTorch allocated）及完整空间验证。它暴露的静态 stay_cable 背景误报是 S1 的研究风险与比较基线，不是当前提交方案已解决的问题。
- `SEM-070-POST-CLEAN` 已完成正式 full（Git HEAD `49cdcdb1c9915a9544d568754af95ea20a85ec9c` 加工作树实现）：五个阈值均未满足训练侧 recall 门，正式验证保持 SEM-001 原样；VIS-003 已生成 144 帧 raw/cleaned 并排视频，逐帧两侧一致。该候选不替代冻结基线。
- `SEM-100-RENDER-SEG-PF` 已完成训练侧八视角域差预检：冻结渲染RGB保住`stay_cable` recall，却使其IoU降低11.42pp并新增115,339个background→`stay_cable`误报；工程隔离/RGB=0通过，74-view保持关闭。该结果只允许另立渲染域训练预注册，不允许直接部署真实RGB二维教师。
- `SEM-101-RENDERED-SEG-OOF` 已完成219张训练侧冻结渲染RGB的三折3k严格OOF：mIoU 84.97%、`stay_cable` IoU 83.61%、P/R 86.09%/96.67%、背景→索误报率1.41%，五门均过；三折重载、RGB=0与访问隔离通过，74-view仍关闭。它只允许新ID注册最终二维模型，提交回退保持不变。
- `SEM-102-RENDERED-SEG-FINAL` 已完成该唯一最终模型与一次性审计：全局mIoU +13.42pp、索IoU +61.66pp、background→`stay_cable` −96.05%，RGB=0、重载和官方外部相机均通过；但deck −2.42pp、tower −9.32pp、最差10% −7.17pp，三个保护门失败。按ADR-022登记drop/show-only，不重训、不重选或再审计，SEM-001保持回退。
- `SEM-110` 已完成219训练图保护融合OOF：0.80由calibration固定，独立check使background→`stay_cable` −93.28%、索IoU/recall 87.12%/94.56%，且deck/tower/foundation预测逐像素不变；74-view未读。
- `SEM-111` 已完成固定规则74-view和官方外部相机审计：mIoU 0.760635、索IoU/recall 0.638137/0.711445、background→`stay_cable` −97.36%，三类保护结构完全不变、最差10% +0.52pp；全门通过。它保留为历史正式低误报候选，SEM-001保留回退。
- `SEM-141` 已完成全219图final-3k和冻结0.70的74-view开发审计：mIoU 0.774899、索IoU/precision/recall 0.706983/0.842841/0.814334，三类保护结构完全不变，worst-10%与140–146均改善；全部目标与工程门通过，推荐隐藏评测。其背景→索较SEM-111增加28.91%，不允许再按复用开发集调参。
- `SEM-150`与`SEM-151`已完成foundation单向验证：训练OOF只在calibration选出0.10并通过独立check；74-view final使mIoU 0.830934、foundation IoU/precision/recall 0.727703/0.782758/0.911866，70视角升、4平、0降，且class 1/2/3逐位不变、RGB/重载/外部相机通过。74-view已复用，下一证据只能是官方隐藏评测，失败时回退SEM-141。
- `SEM-160`与`SEM-161`已完成deck单向恢复：训练OOF只在calibration选出0.99并通过独立check；唯一final checkpoint冻结后，74-view使mIoU 0.833870、deck IoU/precision/recall 0.914667/0.950392/0.960525，158–161四帧均升，保护像素与RGB/重载/外部相机通过。background→deck达到1.24866倍且079–082下降，下一证据只能是官方隐藏评测，失败时直接回退SEM-151。
- `SEM-130` 已完成严格同空间OOF折跨视角anchor支持的108-view calibration：K=1/2/3/4/6/8均未恢复hard-0.30的mIoU与索IoU，K=8仍以79,724背景补回换得12,932真索。它未读111-view check或74-view，禁止在该网格或来源规则内重调；RGB前后最大差0。
- GeometryPrior 当前风险：MapAnything 两种初始化均已淘汰；DA3 P0 也已在3k匹配筛选中 drop。`GEO-020-PF-R2` 的 DA3-Large P0 smoke、训练侧几何和融合已通过，峰值 Torch 显存 4671 MiB；`GEO-020-3K-INT-R1` 工程门通过但质量明显落后控制，且使用了运行时依赖 shim。最终回退冻结 Scaffold 原始初始化。
- GitHub private 仓库尚未创建，remote 尚未配置。
- 尚未选择许可证。
- 主办方尚未发布最终 LPIPS、mIoU背景处理、文件命名和提交接口协议。
- 三名组员姓名与责任映射尚未登记。
