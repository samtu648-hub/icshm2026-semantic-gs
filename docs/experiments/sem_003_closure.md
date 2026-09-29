# SEM-003 结案记录

> 状态：`completed_drop_main_fold`。本记录冻结 S0/S1/S2 的当前结论；不授权 SEM-004/S3 训练、伪标签、半监督、lambda/阈值重试或三折三 seed。后续独立的 `SEM-004-DIAG` 只读诊断不构成S3启动。

## 当前语义回退

`SEM-001` 是当前唯一冻结的 Scaffold-GS 五类静态语义基线。它使用 219 张空间训练侧标注图、74 张空间隔离验证图、seed 20260731 和 10,000 iteration；RGB 几何、颜色、opacity、anchor feature、RGB MLP、appearance embedding 及其 optimizer state 均冻结。

| 实验 | 含背景 mIoU | stay_cable IoU | background→stay_cable | 结论 |
|---|---:|---:|---:|---|
| `SEM-001` | 0.616103 | 0.068756 | 13,166,331 px | 冻结 S0 基线 |
| `SEM-021` | 相对 S0 +0.80 pp | 相对 S0 +0.52 pp | −37,017 px | drop：mIoU不足+1pp，tower −0.54pp |
| `SEM-022` | 相对 S0 +1.51 pp | 相对 S0 +0.87 pp | −760,439 px | drop：tower −0.89pp，超过保护线 |
| `SEM-003-CTRL` | 0.616209 | 0.068865 | 13,106,856 px | Dirichlet 表示匹配对照 |
| `SEM-003` | 0.615198 | 0.066940 | 13,452,325 px | drop：相对 CTRL 的 mIoU −0.10pp、stay_cable −0.19pp、误报 +345,469 px |

S1表明低贡献监督与目标误报有关，但类别无关的全局门控伤害 tower。S2 的训练侧 bg/stay_cable 冲突诊断确实从189,073个 anchor 冻结出7,681个不承载主要 deck/tower/foundation 证据的 eligible anchors；然而限定 Dirichlet 浓度正则仍使目标误报相对匹配 CTRL 增加。因此结果否定的是这一预注册的正则假设，而不是接口、冻结或坐标实现失败。

每个完成的 S0/S1/S2 full run 都保持 RGB 最大绝对像素差为0、semantic checkpoint重载通过，并能接收官方 `CAMERA_FROM_WORLD` 外部相机。原始逐视角结果、checkpoint、日志和渲染产物均为 Git 忽略数据；完整来源见 `experiments/runs/SEM-001/report.md`、`experiments/runs/SEM-002/report.md`、`experiments/runs/SEM-003/report.md` 与 `experiments/registry.yaml`。

## 2026-08-05 独立提交接口复验

使用冻结 `SEM-001` RGB/语义 checkpoint，独立进程重新加载后只读取外部 `cameras.txt`、`images.txt` 和请求的 `140.png` pose；不读取该视角参考 RGB 或标签。输入为官方 `SIMPLE_RADIAL` 与 `CAMERA_FROM_WORLD`，identity Sim(3) 明确写入证据。

| 输出 | 结果 | SHA-256 |
|---|---|---|
| RGB | 1320×989、8-bit RGB；与历史 full/reload artifact 字节一致 | `563e041da15e713f2acc3139f2b922d129b7db7910b18d18b8ac6489e3de10d3` |
| semantic PNG | 1320×989、单通道 uint8、值域0–4；与历史 artifact 字节一致 | `e516db5b8eed1dc3eef952e170ecba03cfe95a9359ceb45655f967cc2c240085` |
| semantic logits | `[5,989,1320]`；与历史 artifact 字节一致 | `42631a3ed6381bdb8205f7f982820dc20aa0cd2922b51ca1d34721369f9a03ff` |
| 审计 evidence | 包含相机、Sim(3)、输出形状和值域、无参考输入声明 | `f18da6b492ad6d0686c16be1ca3ba9ceb948e1d2fc830f416956e9a20a53da8b` |

此复验证明当前 checkpoint/接口可审查，但不假定主办方尚未发布的最终批量相机文件、命名或指标格式。最终包装器仍须以官方提交协议为准。

## 2026-08-05 SEM-004-DIAG follow-up

冻结 SEM-001 的独立只读诊断已完成训练/验证误报、LabelMe细线、精确冻结贡献和训练半部方向统计；其为一个新的独立预注册假设形成了训练侧关联证据，但没有训练模型、改变本结案的S1/S2 drop、或授权S3。详见 `experiments/runs/SEM-004-DIAG/report.md`。

## 2026-08-05 SEM-004/S3 independent outcome

后续独立Goal已授权并完成受限一阶观察方向残差。零残差路径与SEM-001完全等价，16步工程门与10k主折均保持RGB=0、静态logits冻结、checkpoint重载和官方外部相机接口。10k相对SEM-001的mIoU仅+0.0123pp，background→stay_cable仅−29,252像素（−0.2222%），未达到+1pp和至少1%误报下降门，因此S3按门drop。SEM-001仍是回退，不启动S4、伪标签、三折三seed或事后超参数重试；详见`experiments/runs/SEM-004/report.md`。
