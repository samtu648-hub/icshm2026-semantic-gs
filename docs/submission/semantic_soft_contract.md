# 研究专家soft输出契约：SEM348

2026-09-13更新：[SEM351](../../experiments/runs/SEM-351-FINAL-CAMERA-ACCEPTANCE/report.md)补齐最终模型30固定原生规格相机、全量fresh及raw-soft独立工程审计。原hard不改、融合概率仍未定义；无GT/质量晋级，其它相机规格和更广验证仍缺。

本文件定义SEM348的工程诊断文件，尚未接入正式包，也不是主办方新发布的强制接口。原正式RGB030+SEM161及SEM344硬标签规则保持不变。

## 文件含义

输入为同一SEM341检查点在PINHOLE网格产生的五类归一化score。相机使用SEM344已验证的官方COLMAP坐标与SIMPLE_RADIAL内参。只从已保存数组映射，不重新渲染或推理。

`expert_probability`为float32、形状[5,989,1320]，按已有相机适配器将概率双线性映射到官方像素网格，再逐像素归一化。通道顺序为background、deck、stay_cable、tower、foundation。数值表示模型的归一化score，不是已校准的正确率。

`expert_log_probability`为log(max(expert_probability, float32.tiny))。它是有限的log-probability文件，不是模型未归一化的原始logits。零概率用浮点最小正常正数表示；指数恢复误差、归一化误差和argmax一致性须验收。若相机映射导致无有效概率支持，停止并登记，不悄悄填成背景。

`expert_soft_argmax`为上述专家概率的argmax，uint8，类别0–4。它属于待评审的接口反事实，不能覆盖现有正式标签。

`existing_raw`、`existing_selected`、`existing_fused`是SEM344保存的三种硬标签，在PINHOLE上执行各自规则后，最近邻映射至官方像素网格。它们是对照，与专家soft argmax不保证相同。

## 不可混用的输出

原始专家概率不是0.05基础门或旧校准融合的最终概率。那些规则会采用其它分支或覆盖某类，现有代码并未定义统一的最终概率分布。不得把原始soft文件改名为final confidence，也不得通过强制提高目标类分数、交换通道或写one-hot伪造融合置信度。

若调用方只要求硬标签，现有nearest hard契约继续有效。若调用方要求PNG必须等于soft argmax，就需要作新的输出规则决定：插值后再判类会改变像素，需要固定协议下的质量证据及用户确认。无GT轨迹上的差异数量只能证明两种定义不等价，不能判断哪一种更准确，也不能声称黄色背景已修复。

本轮只覆盖固定24帧×三模型的当前原生SIMPLE_RADIAL内参。它不证明其它相机型号/尺寸、全219最终模型、完整融合soft接口或干净安装均已通过；这些验收需分别补齐。

## 复核入口

- [SEM348配置](../../experiments/runs/SEM-348-SOFT-CAMERA-AUDIT/config.yaml)事前固定相机、容差、代码与父结果哈希。
- [SEM348报告](../../experiments/runs/SEM-348-SOFT-CAMERA-AUDIT/report.md)记录真实执行状态与证据。
- [CPU审计脚本](../../scripts/audit_quality_soft_camera.py)保留原数组、计算三组5×5类别转换，并提供新进程精确重放与独立四邻点插值复算。

运行使用显式`--artifact-root`和新的`--output`目录；独立验收增加`--replay`指向第一次完成结果。输出目录必须不存在，避免覆盖旧成果。实验样本只保存预登记frame0060的三份soft数组，其余帧记录完整数组摘要及类别转换。
