# SEM349：全219最终模型及74双重载验收完成

2026-09-13全部结束。用户对具体方案批准后，仅运行一次全219最终拟合，没有重复SEM341三折训练。当前已补齐最终Mask2Former检查点，尚未进行新的74质量评分或正式包晋级；下方等待确认/准备措辞保留为历史。

## 最终执行与验收

- 6000有效更新全部完成，训练1584.316秒（26.41分钟）；整批1749.648秒（29.16分钟）。没有缩图、减模型容量、截断预算或自动重启。
- AMP共有5次无效梯度跳过，恢复后才计有效更新；峰值allocated 8084.516MiB（7.895GiB）。全6000日志有限性、学习率poly0.9、采样轨迹与CPU封存计划独立复算通过，全部219训练视角被采样。
- 全219标签计数与CPU预检一致，作者set/mask/dice/no-object损失不变。3k/6k各650个状态张量全为有限值，检查点绑定/角色/step与保存时RNG不变性通过；3k只溯源，唯一最终端点6k。
- 74张float32五类原生概率和uint8 PNG完整导出，耗时50.669秒；两次新进程严格载入同6k模型，74/74概率与PNG逐项精确一致，未复制fresh概率缓存。独立CPU核对每图形状、值域、simplex和argmax。
- 293源RGB及18正式冻结产物前后哈希通过；约1.80GiB实际batch体积低于4GiB上限，宿主收尾磁盘104.322GiB，符合100GiB保留线。无本项目训练/推理进程，GPU仅804/12227MiB、5%利用率且无列出的计算进程，不称完全空闲。
- 仓库健康通过：478测试（69可选跳过）、174配置；日志（来源记录：`../../../outputs/quality/mask2former/SEM-349-r1/repo-health-20260913.log`；未随公开包提供）。新审计脚本已经实际通过；收尾记录（来源记录：`../../../outputs/quality/mask2former/SEM-349-r1/closeout.json`；未随公开包提供）确认旧文件无缺失，既有修改仅九份状态/路线/报告/登记文档，原源码/配置/旧结果保持；最终链接、配置与Git差异格式通过。

## 封存证据

配置仍为`6eaaadd86d407b5f2cd27295591c20808605a16e86de52183652827112f2520b`，commit `49cdcdb1c9915a9544d568754af95ea20a85ec9c`、seed20260805、semantic_train219/validation74名单不变；实际源码额外hash绑定。

- [批准](approval.json)：`110415053bbf2c278e13db600b1c73bf030338a4bc56b59343a85d669e7cfd83`。
- 训练结果（来源记录：`../../../outputs/quality/mask2former/SEM-349-r1/fit/result.json`；未随公开包提供）：`42ffcbc870190d52f45f06b6b672a132e637ec3d67108647794905a0fb279f68`。
- 完整队列（来源记录：`../../../outputs/quality/mask2former/SEM-349-r1/queue/result.json`；未随公开包提供）：`583cbf7ad2c5c4c5e49a39d7ec418dad2d9e81051d26b7bdec5d1aeeed40cc38`。
- 独立CPU审计（来源记录：`../../../outputs/quality/mask2former/SEM-349-r1/final-audit/result.json`；未随公开包提供）：`974dc00731ebf66a888c4fbb6c416ff2ddca8d4df27ddf8ab34c3a4a28d68732`。
- 宿主核验（来源记录：`../../../outputs/quality/mask2former/SEM-349-r1/host-final.json`；未随公开包提供）：`40be59a7d7bde490a12bcec6f09d850d9a4e39c6655a08ea01ce860c5663d466`。
- 最终model.pt SHA256：`a159373db84590fe73509aab60c5f928d0a9beac5b20bead53092e33217bf1a7`；3k：`d392d3bec9b0c0e9aa03ac26f769668ee80aee43ae09813058f1ae25059cf1ae`。

## 下一步与边界

当前证明最终模型及预测可复现，不证明其质量已经胜过旧正式链。训练期间只读核对了旧SEM161/新Scaffold+B2的同74预测文件、名称、尺寸和新基线引用的标签manifest哈希，没有GT解码或新评分。[固定评价草案](../../../docs/experiments/sem349_fixed_evaluation_draft.md)已准备；下一核对实际相机/网格语义并另登记固定评价，保持raw/原校准融合/0.05，不重选阈值。历史74不能称独立盲测。

SEM347背景门保持off，SEM348 soft不覆盖原hard，黄色背景问题未因本次拟合而被宣称解决。正式RGB030+SEM161及旧模型不变，可直接继续使用；无上传、购买、Git提交或推送。

## CPU准备与批准过程（历史）

2026-09-13更新：用户针对具体训练请求回复“好的继续推进”，已封存[批准记录](approval.json)并复核绑定、磁盘及宿主GPU。授权执行一次全219×6000、74预测及两fresh；下方“等待确认/未获批准”为原CPU准备记录。当前尚无完成训练结果。

2026-09-12完成运行器与CPU准备。GPU训练尚未启动，未创建批准记录；不能把CPU就绪写成最终模型已训练或GPU工程已通过。

## 已完成且可复用的准备

- 新增独立FinalContext、训练/重载入口及一次性队列，未修改SEM341或冻结模块。使用原模型构造、作者优化器、AMP更新、768裁块/512滑窗、6000有效更新和原生输出实现。
- 全219训练、74预测与RGB父划分精确绑定，二者不交叉。TrainingLabels仅准许219；74标签未解码、没有新准确率分数。293现有RGB及18正式冻结文件前后SHA256通过。
- 219训练图标签逐像素类别计数与各自manifest一致，6000步预定采样覆盖全部219。实际训练时每步轨迹必须与此CPU计划一致；独立新进程用choice/randrange重新复算全部6000步、角色与计数通过。
- 损失保持作者分类/mask/dice/no-object权重。**Mask2Former不使用逆频率CE类别权重**；训练类别统计仅作数据审计。此前资源草案的“类别权重”措辞已澄清，不引入B2的加权CE。
- 整个batch（包括fit、fresh与日志）合并计4GiB上限；每个检查点预留900MiB、每图概率/PNG预留27MiB，写入前仍须保留100GiB。不会复制fresh概率、存优化器大快照或导出219训练预测。
- 队列只允许批准batch中的一个fit目录和两个fresh目录。必须有绑定config、CPU结果、batch与用户确认原文的批准记录；不存在批准记录时不能启动GPU执行。任何阶段失败立即停止，保留文件，不自动重试或换batch继续。

训练类别计数（background/deck/stay_cable/tower/foundation）为：239361935, 15037817, 23862905, 5450542, 2186921。这些是219训练标签像素统计，不是模型质量指标。

## 待用户确认的具体执行范围

一次全219、6000有效更新的Mask2Former-Swin-L最终拟合，seed20260805，从原官方ADE预训练开始，不从任何OOF或稳定性拟合模型继续。3k仅留溯源，6k为唯一最终端点；保持RGB、模型容量、损失、裁块和原推理规则。

训练完成后仅导出固定74 PINHOLE原生概率和raw PNG；再由两个新进程严格重载同一个6k检查点，全部74概率/PNG精确比较。预计45–60分钟，训练阶段硬上限60分钟、每次推理/重放15分钟、整批硬上限90分钟。所有上限是失败停止线，不保证实际耗时或显存峰值。

当前只请求“补缺训练＋74预测/两fresh”的范围。不会在本队列读取74 GT或评分，不将74用于选阈值/checkpoint，不改0.05、不开SEM347背景门，不更新正式包，不上传/购买/Git提交推送。

CPU核验已完成，GPU构造/更新与新最终角色的真实运行仍须在获批后验收；旧SEM341可运行证据不能把尚未执行的新GPU路径标为通过。任一失败留存并报告，不以隐式重跑掩盖。

## 证据与复现

实验commit `49cdcdb1c9915a9544d568754af95ea20a85ec9c`，seed20260805；manifest `data/manifests/contest_dataset_v1.json`；精确219/74名单和训练配置在config。实际新旧代码额外按SHA256绑定，保留脏工作区。config SHA256 `6eaaadd86d407b5f2cd27295591c20808605a16e86de52183652827112f2520b`。

- CPU预检（来源记录：`../../../outputs/quality/mask2former/SEM-349-r1/preflight/result.json`；未随公开包提供）：`c7dd8e26693f04f53ce95e6991ab3353cb8910a5bb80c4666f94a98bfc754f81`。
- 独立就绪审计（来源记录：`../../../outputs/quality/mask2former/SEM-349-r1/readiness-audit.json`；未随公开包提供）：`9a5bbfe4e02602b5f7bd1f520c234aceed36e599f0fe1512ee7c56dd188a1211`。
- [运行器](../../../scripts/run_quality_mask2former_final.py)、[一次性队列](../../../scripts/queue_quality_mask2former_final.py)与[角色/预算控制](../../../src/quality_mask2former_final.py)已完成语法与CLI检查。四项针对实际裁块RNG一致、开发角色泄漏、跨stage/pending写预算及无批准拒绝的测试通过；仓库健康通过：478测试（69可选跳过）、174配置，安全/秘密/链接/语法及Git差异格式通过；日志（来源记录：`../../../outputs/quality/mask2former/repo-health-sem349-20260912.log`；未随公开包提供）。旧文件无缺失，旧源/旧结果不变，既有文件修改仅七份计划/状态/登记文档。

已授权CPU就绪命令（已执行）：

```bash
python3 scripts/run_quality_mask2former_final.py   --artifact-root "$ICSHM_ARTIFACT_ROOT"   --batch outputs/quality/mask2former/SEM-349-r1   --output outputs/quality/mask2former/SEM-349-r1/preflight   --mode preflight
```

以上目录已经存在，不要覆盖或重复执行。获明确确认后创建独立批准记录，再用已验收GPU环境调用下面命令；当前未执行：

```bash
"$ICSHM_ARTIFACT_ROOT/r1/envs/r1-cuda128/bin/python"   scripts/queue_quality_mask2former_final.py   --artifact-root "$ICSHM_ARTIFACT_ROOT"   --batch outputs/quality/mask2former/SEM-349-r1   --preflight outputs/quality/mask2former/SEM-349-r1/preflight/result.json   --approval experiments/runs/SEM-349-MASK2FORMER-FINAL/approval.json
```

批准记录字段固定为decision、config_sha256、preflight_sha256、batch、user_confirmation；只有实际收到用户针对本次范围的确认后才填写，不预填“已批准”。

## 后续评价与回退

本实验结束仍不等于新整链击败正式系统。下一固定评价协议需明确相同74相机/网格/GT manifest、旧正式链和新Scaffold+B2对照、raw/原校准融合/0.05三个固定候选；已历史复用74只能称开发结果。当前未编写或运行新的质量评分，不把CPU准备范围偷换为完整最终验收完成。

困难轨迹、其它相机规格、多seed/全链空间验证仍后续独立登记，GAGS不混入。无需恢复旧文件即可回退：现有raw/0.05及RGB030+SEM161正式包始终保留。新增训练失败亦保存，不删除旧模型腾空间。
