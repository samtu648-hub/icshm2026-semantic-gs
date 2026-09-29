# SEM348：官方soft接口的固定相机工程审计

2026-09-12完成。24个固定相机×三折共72组CPU检查及新进程精确重放通过；独立四邻点插值的全量概率最大差为0。原始专家soft工程文件可复现，但它与现有hard接口不等价，未改现有输出或正式包。

## 协议及语义边界

先登记config及代码/父结果哈希，再复用SEM344固定frame0060–0083三折保存的五类概率。唯一变量是官方像素网格的重采样定义：现有PINHOLE判类后最近邻hard，与原专家概率双线性重畸变/归一化后argmax。无新推理、训练、标签访问、阈值选择或相机调整。

原始概率经现有冻结adapter映射为float32五通道归一化score，log(max(p,float32.tiny))作为有限log-probability；不是未归一化模型logits，也不是已校准正确率。soft只属于原专家，不代表0.05或校准融合后的最终概率。完整定义见[soft接口契约](../../../docs/submission/semantic_soft_contract.md)。

## 实际差异（没有GT，不是准确率）

72组共93994560模型/相机像素，soft argmax相对existing raw改变39624（0.042156%）。同一24相机由三个同seed OOF模型分别检查，不能称72个独立视角或三seed。

- oof-0：变化12168，其中原background→stay_cable 1310、原stay_cable→background 1525。
- oof-1：变化11510，其中原background→stay_cable 1119、原stay_cable→background 1337。
- oof-2：变化15946，其中原background→stay_cable 3083、原stay_cable→background 3483。

总计原索标签减少8897、增加7790；这是预测集合变化，不是GT索损失/修复。背景→索5512与索→背景6345并存，不支持以插值“清理黄色”。

- 对已有selected硬标签，原专家soft argmax分歧39645像素。
- 对已有fused硬标签，原专家soft argmax分歧1035711像素。

后两项包含基础门/融合职责本身的差异，不能全归因于相机插值。逐图及每类5×5转换完整保存在result；没有把原专家分数伪造为最终融合置信度，也没有生成部署替代标签。

## 验收证据

实验Git `49cdcdb1c9915a9544d568754af95ea20a85ec9c`，seed20260805，manifest `data/manifests/contest_dataset_v1.json`；精确相机名单、父SEM344相机metadata/hash及源码记录在config。工作区实际代码另按SHA256封存。配置SHA256 `aa05adb901f438da131daffc6165da74c867a8fd49335a00e32fd521b5c3813f`。

- 首次结果（来源记录：`../../../outputs/quality/external_camera/SEM-348-r1/result.json`；未随公开包提供）：`89ed8bb871f075c5bfdd5cea99087250c6d30c8aa7f60a62508a810214d95542`。
- 新进程重放与独立插值（来源记录：`../../../outputs/quality/external_camera/SEM-348-replay-r1/result.json`；未随公开包提供）：`a326bb4250c88b6416265104e576e318d215f302d4ee13ae862361ba1ebbbb37`。
- 所有72概率与log数组digest逐项相同；独立NumPy四邻点/1⁄32权重插值与原adapter全量差为0、argmax精确相同。独立性限于插值累加实现；相机反畸变map使用同一已验收canonical map，不把它称为另一个相机模型独立证明。
- 概率有限/归一化、exp(log)数值误差及argmax通过；nearest概率gather的argmax精确等于原hard raw，三臂hard全部精确复现。
- 219源RGB与18正式冻结前后哈希一致；24轨迹RGB/official RGB/float digest在三个语义附着折间精确一致。原SEM344结果文件及适配器未修改。
- 按事前选择仅frame0060每折保存一个soft/log工程NPZ，全部72保存数组摘要，独立重放不复制大概率缓存。样本内包含existing三个hard对照。
- 恒等相机、实际径向插值独立比较、非法概率拒绝三项测试通过。完整仓库健康检查通过：474项测试（69可选跳过）、173配置，安全/秘密/链接/语法通过；日志（来源记录：`../../../outputs/quality/external_camera/repo-health-sem348-20260912.log`；未随公开包提供）。旧文件无缺失，原有源码/实验结果未改变，既有修改仅七份状态/计划/登记文档；Git diff格式通过。本轮两个CPU阶段均正常结束，无新GPU作业。

## 决定、剩余及回退

保持已有hard输出规则；soft仅作为明确命名的原始专家附加工程证据，不接入正式包、不强制其argmax覆盖hard。若后续消费者要求PNG必须等于soft argmax，需另登记固定质量评估并请用户确认；本轮无GT轨迹不能判定该取舍。

当前仅覆盖原生1320×989 SIMPLE_RADIAL和固定24相机；其它规格、完整融合概率语义、全219最终模型及干净安装仍未验收。SEM347背景门保持off，黄色背景没有因本轮接口检查得到安全修复。

[全219最终拟合资源草案](../../../docs/experiments/mask2former_final_fit_resource_plan.md)已准备：不重复三折，只补缺失模型；已有74开发RGB逐项hash通过、未读GT。建议4GiB新增上限、100GiB保留线和45–60分钟运行窗口；运行器/精确角色/预算控制须先落实，具体训练再请求确认，当前未启动。

回退无需恢复文件：继续使用既有raw、封存0.05与RGB030+SEM161正式包即可。本轮均为新诊断文件；无上传、购买、Git提交/推送。
