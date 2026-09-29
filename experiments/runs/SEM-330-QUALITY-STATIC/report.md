# SEM-330 新协议原生静态语义训练

更新2026-09-10 / ADR083。四骨干development及12个训练侧OOF共16个静态模型均完成10000步和两fresh独立审计；`dualline-20260909-r1`在07:07正常结束。二维专家及融合结果归SEM331，不能把静态结果当完整系统。下方“运行中/待完成/暂停”等段落保留过程历史，由本节覆盖时态。

9月10日只读核对队列所有16条静态审计完成事件，终态result SHA `0aaf734a041552afb45ed460bd40a97bf8bbdebac932beb32ebad083e5dbb64a`；其204条各类子阶段result链接均匹配，18历史冻结产物再次通过。没有本批后续静态训练。所有分区单独拟合/计权重并绑定各自父RGB，模型预测两fresh原生重放完成；本轮共享RGB的语义OOF仍非完整重建空间审计。完整官方外相机专家链和三seed/三外折验收另计，未自动晋级。

## 新增完成：PGSR静态10k

沿用本ID原config SHA `9693ab0632dfcf9d565120d97ac964fc5cf77e79118c2f488172e469e392f425`、seed20260805、219/74与1320×989；父RGB115/seed0、共同72292点及官方坐标不变，基底commit仍为`49cdcdb1c9915a9544d568754af95ea20a85ec9c`加每次封存源码。先独立8步smoke，再10000步完整拟合；训练611.995s，含导出665.594s，峰值allocated1142.078MiB/reserved2556MiB。

`outputs/quality/semantic/SEM-330-PGSR-development-full-dualline-20260909-r1/result.json` SHA `ac2716fd5bb393de1b46e79cec44eabc3a830c7b4af2490d154f3715d41476dc`；独立`outputs/quality/audits/SEM-330-PGSR-development-full-dualline-20260909-r1/result.json` SHA `974cb5f29c92268449c2f5091d2536997c239625dbc513fcbcba785276d9b808`，25.484s。两fresh74概率/PNG精确，独立混淆计数通过，18冻结/固定RGB不变。

- 全类mIoU22.940321%、前景6.681555%；各类IoU为background87.975385%、deck7.776218%、stay_cable3.845980%、tower7.893160%、foundation7.210861%。
- deck precision84.018333%/recall7.892968%；foundation precision25.132535%/recall9.183527%。静态漏检明显，不能将其当作未完成的B2完整链结果，或以此提前淘汰新骨干。
- QUAL003该检查点原生覆盖诊断及独立复算也已通过：真实前景无累计覆盖<0.5，不支持低覆盖硬上限解释这批静态漏检，但不证明几何正确。

四臂development模型复用；新批次从Scaffold OOF0开始，补12个10k OOF，每折重新计权重和训练语义，只共享303训练RGB，仍是语义筛选而非全链空间审计。活动代码/配置锁定，不改旧结果或阈值。

## 两SOF批次实际终态（12:10完成）

`sof-static-after-pgsr-20260909-r1`终态为`completed_two_sof_static_comparisons_2d_experts_pending`。两组与Scaffold使用同config SHA9693ab0632dfcf9d565120d97ac964fc5cf77e79118c2f488172e469e392f425、seed20260805、219/74和1320×989；各10000步、18冻结及001/054/140 RGB不变，74概率/PNG两次fresh重放和独立计数通过。

- NVS训练1894.2336s，allocated990.8838MiB/reserved1262MiB；全类mIoU26.304695%、前景11.842706%；deck IoU18.662630%/recall18.851456%，stay_cable IoU2.446482%，tower IoU17.137536%，foundation IoU9.124176%。
- GEO训练1650.1818s，allocated964.4375MiB/reserved1250MiB；全类mIoU33.606569%、前景19.956412%；deck IoU34.773236%/recall35.229236%，stay_cable IoU3.364705%，tower IoU21.723851%，foundation IoU19.963858%。
- 结果是广泛构件漏检，GEO优于NVS但均落后Scaffold；不是显存不足而缩分辨率或截断训练。不能以静态代替完整专家链结论，也不把差值全部归因于几何。
- 独立证据在`outputs/quality/audits/SEM-330-RGB-112-SOF-NVS-full-sof-static-after-pgsr-20260909-r1/result.json`及同名RGB113目录；训练结果在`outputs/quality/semantic/`对应目录，保留代码/配置/模型/输入完整绑定。
- 新方向是RGB117相机一致性诊断，不修改本实验任何配置、旧输出或已评测预测。

## 最新完成：Scaffold静态10k与独立审计

`static-after-sof-20260909-r1`已完成，不再等待。新原生探针、独立8步smoke、219训练/74预测10k均通过；10k训练913.1572s、峰值allocated836.6548MiB，三固定相机RGB/模型张量及18冻结一致。训练result SHA `2222a7c78e53b4a7dc6233dbcc5da5121a500a0b9aff1419e9bbf1fcaf38046e`。149个保存产物哈希再次核对。

本轮新增独立`audit_config.yaml`与`audit_quality_static.py`，在两个新进程（PID2340455/2341001）逐值重放全部74 float32概率和类别PNG，一致后才解码目标标签；双重混淆计数精确，18冻结前后通过。25.8356s，result `outputs/quality/audits/SEM-330-Scaffold-development-r1/result.json` SHA `da9b97ca979cd27f3a541a8f0757f3f46c40f121f61cb41449e54be83ac7bcb3`。

- 全五类mIoU60.6838501%，前景mIoU54.9563116%。
- IoU依次为background83.5940%、deck90.1900%、stay_cable5.6974%、tower85.7319%、foundation38.2059%。
- deck precision95.5333%/recall94.1607%；stay_cable precision6.6397%/recall28.6462%；foundation precision39.4142%/recall92.5725%。全部逐视角及原始混淆已保存，无阈值搜索。
- 历史SEM001同74静态结果mIoU61.6103042%、stay_cable IoU6.8759380%；新静态mIoU低0.92645pp，索类误报是历史静态也存在的瓶颈。历史RGB模型、训练seed20260731和旧相机路径不同，因此只作额外参照，不能把差值归因于单独骨干、适配或seed，更不能与SEM161/329完整融合分数直接比较。

下一为两SOF完整30k模型分别匹配8步/10k静态及独立审计；同一SEM330配方，不因RGB落后先淘汰、不改字段容量/分辨率。一次性`outputs/quality/queues/sof-static-after-pgsr-20260909-r1`已实际启动，event000为`waiting_for_predecessor_resources`；只等待PGSR当前GPU批次退出，其科学前提仍为各自SOF RGB验收。PGSR数值失败不等于SOF不能继续；没有将等待算训练完成。二维专家、OOF、GAGS与跨视角复核不包含在此静态批次。

本轮新增语义重放/评分、PGSR前置门测试均通过，quality合计62测试；`outputs/quality/repo-health-audit-r1.log`为335测试/63可选跳过通过，默认数据目录缺失按预期跳过，真实外置标签已单独核验。报告以下早期“尚未训练/等待SOF”保留为历史，不覆盖本节完成状态。

## 实验定义

- config SHA256：`9693ab0632dfcf9d565120d97ac964fc5cf77e79118c2f488172e469e392f425`；Git base `49cdcdb1c9915a9544d568754af95ea20a85ec9c`加保留的工作树，每attempt封存实际代码/配置/输入hash与源码副本。
- 父实验：QUAL001、RGB110/112/113/115、SEM001；数据manifest为`contest_dataset_v1`，语义划分为`sem_001_scaffold_static_v1`与`sem_005_oof_spatial_v1`。当前实际预检只针对已完成RGB110，未将其他臂提前验收。
- 同款静态配方：原生1320×989 PINHOLE，五类，初始logits=0；AdamW lr0.05、weight_decay0、加权NLL clamp1e-8，10000步；独立8步smoke不计质量。语义seed20260805、父RGB seed0。
- 类别权重从本run实际允许训练PNG重新计数，median/count裁至[0.05,20]；不复用全219的权重到内折。静态表示为PGSR/SOF原生高斯或Scaffold锚点，后者继承到其生成偏移；该参数化差异不能当成纯几何因果。
- OOF分别训练114/128/177张，预测95/84/40张；共用原303训练图RGB，只是筛选用语义交叉拟合，**不是完整空间泛化审计**。最终完整链隔离实验另做。

## 已实现并验证的边界

- `run_quality_static.py`检查完整RGB记录、共同初始化、独立74重载、准确原生探针、源/扩展及有序辅助依赖清单。语义附件绑定骨干、检查点、拓扑、配置、输入、seed和运行时。
- 训练只优化语义logits；检查原生模型直接张量和嵌套MLP均冻结，每步检查loss/梯度/更新后参数有限性；坏梯度拒绝进入优化器，不清零救场。
- 完成训练后存取侧文件并核对logits/固定相机语义重放，检查001/054/140三相机的RGB数组及全部RGB张量不变。这里实现的是同进程重载门，独立进程语义验收仍待实现/执行，不冒称已完成。
- 导出原生float32五类概率和uint8类别PNG，不在本训练器解码目标标签、不搜索阈值；完整run的成功状态仍是`pending_independent_evaluation`。官方畸变输出包装器沿用已有新接口，训练后的实际外相机验收仍待执行。
- 全量训练必须匹配单独成功smoke的父模型、划分、配置、代码及检查点hash；无任意减少迭代/缩分辨率参数。全局GPU非阻塞锁及100GiB剩余空间门保持。

## 真实预检及保留的失败

- development-preflight-r1因旧Scaffold探针引用的`src/quality_backbone.py` hash不再对应新增有限性门而停止；0训练步，尚未解码标签。原结果/日志保留。不是旧RGB质量无效，也不是新语义方法失败。
- CPU-only预检可独立审计训练标签，同时明确留下`pending_native_probe_refresh`；GPU smoke/full仍严格禁止过期探针。r2完成四划分；r3增加辅助源/运行时绑定；r4增加实际源码副本归档，全部保留，无新预测。
- r4完整219及三折114/128/177训练标签的逐图SHA与逐类计数全部通过；每组18冻结产物前后通过。没有解码对应74或OOF目标标签；读取过公开的标签manifest元数据，故不声称目标元数据完全未访问。
- 全219计数为background239361935、deck15037817、stay_cable23862905、tower5450542、foundation2186921，与历史训练计数独立吻合；各折单独重算，不读取manifest的全局计数作loss权重。
- 真实GPU锁准入测试在SOF活动期间返回`BlockingIOError`，未创建smoke输出目录、没有抢占GPU，日志`outputs/quality/static-lock-admission.log`保留。这是预期准入拒绝，不登记成已启动GPU实验。
- 首轮CPU优化器测试暴露选定Torch的非连续NLL输入反传限制；显式contiguous不改数值/布局/预算，修复后10个新增测试通过（含真实CPU优化器降loss、坏梯度拒绝、嵌套RGB冻结、标签隔离和队列前置门），全quality套件56测试通过。合成测试不是本桥语义质量证据。

最终CPU证据目录前缀`outputs/quality/semantic/SEM-330-Scaffold-`，后缀及result SHA256：

- `development-preflight-r4`：`e652af530854ae080ac1cf94c9b1d34149a84bfcbe4ea7ce16eed4033d074999`。
- `oof-0-preflight-r4`：`656f8cfddb2a786b480170fd9accf97db6cb9f9ac4999c34e77566f89fc86f67`。
- `oof-1-preflight-r4`：`baa8ac4a9959d493cbcb831c8176b27932f15d1161bfa683a96382af05ac7297`。
- `oof-2-preflight-r4`：`350ba254a8d0e98c08f48f7f165cb4579ad815e90c19930031ca799f72d4b43f`。

## 一次性后续队列与剩余工作

`queue_quality_static.py`以本机batch锁等待已有SOF112/113两臂成功终态和各自验收事件，随后仅执行Scaffold新探针→本ID development分区8步smoke→从零10000步。等待不持有GPU锁；任何子步骤失败、代码/协议变化、空间不足即停并保留记录，不自动重试或降配。不会启动B2、GAGS、RoMa、三折或其他骨干，也不会通知/上传/付费。

运行记录：`outputs/quality/queues/static-after-sof-20260909-r1`已实际启动，`event-000.json`记录`waiting_for_existing_sof_queue`，unix1788896139.725638；核对时没有终态result和GPU子步骤。它是已经运行的等待批处理，不是已经开始语义训练。等待中的14个执行依赖及配置同样锁定，若要更改RGB115共享runner依赖，应在本批完成后另开attempt，不边跑边改。没有定时轮询或通知自动化。

最终仓库健康检查通过：`outputs/quality/repo-health-static-r1.log`，329测试/63可选依赖跳过；隔离环境`static-all-quality-tests-final.log`为56项quality测试全过。默认仓库数据目录跳过不影响上述外置真实标签审计。SOF队列15个代码指纹与本后续队列14个代码指纹均核对未变；`git diff --check`通过。

GPU拟合与独立评测、各折静态OOF、分骨干B2/二分类专家及校准均仍未完成。完整系统选择和最终空间/seed验证不由本ID代替。所有输出Git忽略，旧18冻结产物、失败模型、提交包及用户改动保留；无commit/push/上传。回退仍是RGB030+SEM161。
