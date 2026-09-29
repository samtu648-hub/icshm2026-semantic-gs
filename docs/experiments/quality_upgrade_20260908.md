# 质量优先升级执行记录

用户已授权完整计划；ADR-079仅允许独立研究副本。正式RGB030+SEM161和18冻结文件不变。总协议为`experiments/runs/QUAL-001-FOUNDATION/config.yaml`。

## 2026-09-09位姿方向实际完成（ADR-082，覆盖下方旧状态）

- 用户要求先只沿官方位姿可能有误的方向推进。RGB117/118均完成训练侧相机一致性诊断、独立重放/公式复算，未修改相机/内参或训练新模型，未解码74开发RGB/观测/标签。
- RGB117近邻468对/420可评分，64200留出描述子匹配官方F/图像F中位距离0.104524/0.104942px；RGB118较大基线587对/137可评分，8818匹配为0.162070/0.176293px。两组未触发事前明显不一致候选门，不认证真实位姿或全303/400覆盖。
- 57616官方轨迹留一预测中，1461条>5px，其中978条来源最大视线角<1°，是弱三角化与长尾的关联证据，不是坏位姿数量或已改善重建。较大基线450对匹配不足，初次6组均匀重放也均弃权，已另补6个可评分对fresh和全部8818检查匹配的独立算式验收。
- RGB115从头30k/RGB已验收PSNR12.02759；SOF两臂静态10k及独立评分已验收mIoU26.30469%/33.60657%，Scaffold为60.68385%。所有已启动批次结束，没有后台继续新语义训练；旧方案不动。
- 当前不依据本轮结果大范围重估相机；只保留覆盖不足、足够视差下的局部残差与亚像素校准问题作为后续相机诊断。其余骨干/专家/GAGS/跨视角研究未取消，但本轮不扩展。

## 2026-09-09最新续推（ADR-081）

- SOF112/113完整30k、参数有限、74双fresh RGB及独立原生探针均通过，PSNR13.33622/14.00706，明显弱于Scaffold20.65405。RGB116只读7k/15k/30k显示早期已有训练样本/开发差距、后半程开发进一步退化；作者与新接口至多1量化级差，不因接口小数差异解释当前低分，也不把总loss当单项正则因果。
- SEM330新Scaffold静态10k已完成，两个新进程精确重放74概率/PNG后独立双计数评分，mIoU60.68385%、deck IoU90.1900%、stay_cable5.6974%。历史静态索类也很弱；不同seed/旧RGB只作额外参照，不与完整SEM161/329成绩直接对比。目标标签只在固定评分阶段打开，没有阈值搜索。
- RGB115修复版runner已接入，以已通过的RGB114诊断result/patch/helper作精确前置门，独立共同初始化300步及探针通过；`pgsr-safe-serial-20260909-r1`已启动从头30k，之后数值/重载/RGB/接口验收，完整结果尚无。等待中的`sof-static-after-pgsr-20260909-r1`在GPU释放后各自训练两个SOF静态分支并评分，PGSR失败不直接拒绝SOF。
- quality62测试、repo_health335测试/63可选跳过通过。所有原数据、失败模型、冻结18文件和未提交工作树保留；无commit/push/上传/费用。下面早期运行状态为过程记录，最新状态以本节及各run终态为准。

## 固定推进顺序

1. QUAL-001：冻结/数据/相机契约、完整三折访问清单、单GPU锁与不可覆盖运行记录。
2. RGB110 Scaffold、RGB111 PGSR、RGB112 SOF-NVS、RGB113 SOF-GEO，各先300步工程、再30k完整训练、独立重载和同口径RGB/语义评测。四组303训练/74开发/原生1320×989/同训练DLT；新方法不能只凭RGB平均分淘汰。
3. 每骨干静态语义10k、B2五类768/512、索/桥面960/480；3k匹配后独立比较6k/12k。仅训练OOF校准阈值，含关闭项。保留Scaffold及用户证据评审通过的一条新骨干。
4. 两路线比较Mask2Former-Swin-L；GAGS分别none/fixed-medium/learned，16维、16→128→5、30k、半分辨率蒸馏/原生评测、融合0/.25/.5/.75/1。必须冻结RGB并沿真实骨干投影/排序/遮挡合成。
5. 重建RoMa参考库先复现SEM329，再独立测试两真实训练来源一致且桥面专家确认的0→1补回；保护2/3/4，无证据弃权，不自动复制索类/把深度当GT。特征与新复核做2×2消融。
6. 两系统各三seed和完整空间三折，共12验证单元，内部校准训练另计；RGB/初始化/语义/特征/参考库全外折隔离。内层语义交叉拟合允许共享外层训练RGB，不冒称独立重建验证。历史复用数据不称新盲测。
7. 方法确定后才all400/all300拟合及重建语义、独立包复现、论文/视频/PPT；9月24日就绪检查可使用成熟包兜底，研究不强制9月20日冻结。

## 当前能力和运行入口

- `quality_preflight.py`只读相机元数据与hash，生成完整空间三折访问清单，不解码GT。
- `run_quality_rgb.py`固定配置/source/input hash、seed、实际相机列表、日志、失败分类、启动时磁盘门和本研究GPU互斥；`quality_upstream_worker.py`仅适配划分/共同初始化/随机种子/GUI。显式CPU图像缓存是无损内存优化，不改分辨率或容量。
- 上游可能写PLY，所以每run复制稀疏元数据，仅共享只读图像；PGSR flat-sparse是相同元数据副本，不增加观测或重新SfM。
- 2026-09-09实测发现PGSR TXT reader隐式丢弃误差>2px的1761点（项目共同DLT阈值3px）。worker现明确读取各run的共同`sparse/0/points3D.ply`72,292点，保留训练/原生加载器点数和PLY哈希。早期PGSR smoke-r1使用70,531点，工程证据保留但排除公平质量对照；新attempt重新检查。Scaffold原生锚点体素化是表示机制，与输入点云被额外过滤分开记录。
- 原生训练完成不等于重载、外相机、多通道特征或质量通过。后续语义/GAGS/复核重训和12单元验证尚未执行，不能把配置登记当完成。
- `prepare_quality_sources.py`固定四个官方仓库commit及本机兼容补丁，扩展编译和dacite/trimesh置于隔离目录，不修改历史环境；构建失败日志保留。
- `audit_quality_rgb.py`复制检查点到两个独立重载目录，运行各自作者render.py，核对74视角输出，再用原评测PSNR/SSIM/LPIPS。PGSR按原图名映射，另两者按序号严格映射。
- `probe_quality_adapter.py`实际加载PGSR/SOF检查点，测试16通道有符号特征合成/梯度、RGB张量及像素不变性、外相机输入；SOF滤波器只由训练相机元数据重建。未通过该测试不进入特征训练。
- `diagnose_quality_stages.py`根据带hash的输入manifest原样保存每级输出/概率/原图/重建图，导出修改掩膜、修对/改错计数；无同视角照片不自动判定重建缺失，插值视角拒绝绑定真实照片或标签真值。
- `quality_review.py`只实现两真实训练来源共同支持的桥面0→1决策核；RoMa匹配准入/参考库重建尚未接入，不能将此代码视为SEM329已迁移。
- `quality_scaffold.py`提供新协议Scaffold接口，复用原生生成高斯及其锚点字段继承，外相机固定appearance index0；历史接口不改。`quality_semantic_fields.py`实现静态五类字段及16→128→5分类头和严格模型绑定，CPU梯度/存取测试通过，未做真实语义训练或GAGS蒸馏。
- `export_quality_rgb_inputs.py`给新专家生成与部署同渲染路径的PINHOLE输入，训练219与开发74分开选择，要求完整RGB模型和实际探针绑定，不读取真实图或标签；入口已编译，尚未GPU执行。
- `run_quality_stage_a.py`是一次性本地串行任务，复用已验收Scaffold，对PGSR/SOF逐臂30k→双fresh重载/固定RGB评分→实际原生接口探针。按已有输入和源码哈希验证前提，任一步失败停队列，不自动选型、缩小实验、重试或购买资源。

## 本轮实际进度（2026-09-09）

本节原始启动记录由以下续推结果更新：RGB111完整30k已结束但包含6080个NaN高斯，原`serial-20260909-r1`在适配器门停止，不能再称PGSR仍在该队列训练。RGB114逐损失/算子诊断定位多视角pixel_noise范数反传污染，独立safe-cycle适配正在进行带异常追踪的3k窗口验证；PGSR方法不因此淘汰。新run在评分前校验保存PLY全部字段有限性。详见[RGB114报告](../../experiments/runs/RGB-114-PGSR-NUMERIC-DIAG/report.md)。

[QUAL002六视角诊断](../../experiments/runs/QUAL-002-PAIRED-GAP-DIAG/report.md)已归档真实PINHOLE照片/RGB/7级语义/3专家概率：33169个最终deck→background漏检无前级正确deck后删，但26375个raw五类专家已识别deck，现有融合不使用该分支做deck补回。6683个不满足现有SEM161补回准入条件；不在这些开发图调阈值或推断全图precision。静态完整概率和逐源追踪尚未全部补齐。

续推通过显式`--arms`启动独立SOF两臂，不把省略的PGSR标为通过。`sof-serial-20260909-r2`已实际开始RGB112完整30k、RGB113随后。RGB114 safe-cycle窗口最终3000步通过（615.78秒，约5967MiB allocated），RGB115新完整30k未运行且不属于当前SOF队列。原失败队列和模型全部保留；新队列只做选定RGB臂，非完整计划完成。

- 完整Scaffold新对照已完成；74张两fresh重载字节一致。PSNR20.654049810932、SSIM0.850192733256、LPIPS0.319108837561，约29.4分钟含原生导出，峰值allocated6578.979MiB。不是新骨干收益。
- 原启动记录：PGSR共同初始化/CPU图像缓存smoke及真实特征/RGB探针通过，之后30k发生上述数值失败。300步缓存优化峰值allocated520.533MiB，完整30k为3103.737MiB，短程数字不能代表质量或后期峰值。
- SOF-NVS和SOF-GEO分别完成300步原生训练，约58.7/58.5秒，峰值allocated6275.719MiB；NVS真实接口探针已过，GEO独立探针排队。300步尚不足以衡量两种几何配方。
- 原队列`outputs/quality/queues/serial-20260909-r1`已停止，终态result保留。新attempt的started/event/result才代表其状态；队列完成仍只代表选定RGB部分，不自动开始语义、全链验证或发送通知。
- 隔离运行环境37项新增测试全通过；仓库健康310测试/56可选依赖跳过；18冻结hash再次通过。对应原始报告、源快照、失败及哈希均保留。

PGSR完整多视角阶段当前明显慢于300步段；SOF后期点数也会变化。因此目前只给实测耗时，不把短程线性外推当完整排期承诺。最终资源/吞吐由每个完整run记录，阶段B以后尚未实测。

路径由参数传入，GPU需在实际有设备访问的环境运行：

```bash
python3 scripts/quality_preflight.py --artifact-root "$ICSHM_OUTPUT_ROOT" --output outputs/quality/preflight-new.json
python3 scripts/run_quality_rgb.py --experiment RGB-110-SCAFFOLD-CTRL \
  --artifact-root "$ICSHM_OUTPUT_ROOT" --source "$R1_THIRD_PARTY_ROOT/scaffold-gs" \
  --python "$R1_ENV_ROOT/bin/python" --cuda "$R1_CUDA_HOME" --mode smoke --attempt r1
```

工程验证后再`full`；失败attempt不覆盖。锁只覆盖本研究runner，启动前仍检查其他GPU训练。原生依赖/源码/公开权重分别审查许可证，研究运行不代表商业使用或外部分发获准。

## 交付和判定

2026-09-09阶段B接入：[SEM330](../../experiments/runs/SEM-330-QUALITY-STATIC/report.md)实现统一原生静态训练和CPU标签访问审计。219及三折114/128/177训练标签已核验，OOF目标未用于计数或拟合；RGB仍共享，非全链隔离。GPU执行先刷新因数值门变更过期的Scaffold探针，再8步独立smoke、从零10k；后续一次性队列等待SOF两臂验收成功，不修改原队列。预检不是语义训练完成，训练器输出预测后仍需独立评测/外相机验收；B2/二分类专家/GAGS/RoMa不包含在该队列。

报告PSNR/SSIM/LPIPS、五类IoU/P/R、前景/全类mIoU、固定困难区域、连续视角、纠错/误改计数和资源。无几何真值不报告Chamfer/F-score；插值72帧没有同视角GT。质量折中由用户评审；失败/不确定结果保留。各实验report与registry以实际运行状态为准。

## 2026-09-09双线实施（ADR083，覆盖上述旧运行时态）

本轮用户再次明确要求实施完整语义与针对性重建双线，故恢复原完整研究优先级，不再把ADR082“仅相机/旧批次结束”当当前执行状态。新增QUAL003、SEM331、SEM332均在运行前登记配置，未覆盖历史模型或配置。

- [QUAL003](../../experiments/runs/QUAL-003-COVERAGE-INIT-DIAG/report.md)首轮完成：四臂74原生覆盖及独立计数无真实前景alpha<0.5；实际初始化1455异常交叉预测轨迹留存（976弱角），独立集合复算通过；PGSR训练12样本PSNR略升而74开发随7k/15k/30k下降。没有相机/初始化/概率修改，不把关联当因果。
- [SEM330](../../experiments/runs/SEM-330-QUALITY-STATIC/report.md)新增PGSR静态10k和两fresh74审计，mIoU22.940321%。四臂development静态均已完成；12静态OOF由新的串行批次补齐，各折重新训练/计权重。
- [SEM331](../../experiments/runs/SEM-331-QUALITY-EXPERTS/report.md)四臂219+74原生专家RGB导出完成。首个Scaffold五类OOF0重新训练3000步，两个新进程95视角概率/PNG精确，RGB/18冻结不变；仅1/48专家，没有新完整系统分数。实际训练4.8分钟，含导出和重放7.34分钟，峰值allocated3223MiB。当前环境差异及排期外推限制单列。
- `queue_quality_dualline.py`已实际启动`outputs/quality/queues/dualline-20260909-r1`，一次只运行一个重型GPU步骤，100GiB阶段启动保留线，活动依赖hash封存。覆盖/导出已过，当前逐折静态和三专家拟合及独立审计。严格3000/10000步，资源或工程失败停止留证，不擅自淘汰骨干、截断预算或改模型容量。
- `queue_quality_fusion.py`已实际启动CPU依赖任务`outputs/quality/queues/fusion-dualline-20260909-r1`。拟合完成才开始四组融合校准/评分，继而[SEM332](../../experiments/runs/SEM-332-DECK-EVIDENCE-OOF/report.md)单变量桥面证据；当前只有等待事件，没有假造融合结果。它不是定时通知或持续自主研究，完成止于人工评审。
- `evaluate_quality_fusion.py`继承四步保护职责，使用108训练OOF封存阈值后才做本轮111/74评分；旧内部加权校准改为新统一mIoU字典序，影响单列。SEM332独立deck优先、只用min双专家证据/旧背景准入、含关闭项。保存逐阶段预测和真实修改、五类/逐图/OOF组计数，无自动晋级或新跨视角规则。

本轮新增16项专家/覆盖/初始化/保护/融合测试，全92质量测试通过；仓库健康365测试/63可选跳过通过。真实GPU重放和外置标签检查不能由可选依赖跳过替代，已经完成的真实审计在各报告记录。所有源、失败与18冻结文件保留，无commit/push/上传或购买算力。

仍未完成：四臂全部专家/校准、完整官方外相机专家链、局部边界与连续轨迹评审包、用户新骨干选择；以及6k/12k、高容量专家、GAGS、SEM329新参考库/双源deck及四开关消融、三seed/全链空间三折和最终打包。当前实现是可运行的首轮研究链，不是原计划整体完成。具体实时进展以该批次最新event或终态result为准；静态文档仅记录本轮核对时点。
