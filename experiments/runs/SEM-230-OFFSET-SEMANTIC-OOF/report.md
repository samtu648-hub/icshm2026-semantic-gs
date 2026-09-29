# SEM-230 — per-generated-offset Scaffold semantic residual OOF

- 状态：`completed_calibration_gate_failed`；不晋级，保持 `SUBMIT-020 = RGB-030-FULL + SEM-161`
- 父实验：`SUBMIT-020-SEM161-FREEZE`、`SEM-001`、`SEM-150`、`SEM-160`、`SEM-161`
- 配置：`experiments/runs/SEM-230-OFFSET-SEMANTIC-OOF/config.yaml`（seed `20260907`，每折 10,000 steps）
- 输出：`external_gitignored:ICSHM_OUTPUT_ROOT/r2/sem_230/offset_semantic_oof/full-20260907-r2`
- 数据隔离：219 training views 的固定空间三折 OOF；候选与贡献清单先封存，才读取 108-view calibration。111-view check、74-view developer 均为 `false` / 未读。

## 假设与唯一变量

每个 Scaffold anchor 生成 10 个 offsets。SEM-230 保留冻结的五类 anchor logits，仅为每个 generated offset 加入独立、零初始化的 background/deck 残差 `[anchor, offset, 2]`。RGB、几何、opacity、appearance、densification、相机、网络、阈值和 RGB 域均未改变；最终 background/deck 用 argmax 双向竞争，`stay_cable`、tower、foundation 及 SEM-150 foundation rejection 区逐像素保护。

## 工程与隔离证据

- `SUBMIT-020` 的 18/18 项产物在本次开始时均存在且 SHA-256 一致；没有覆盖冻结输出。
- 零残差相对冻结 SEM-001 的五类 tensor 最大差为 `0.0`、逐 tensor bit-identical；001/017/093 的 RGB PNG SHA-256 完全一致，所有冻结 RGB 产物 hash 未变。
- 官方外部相机合同通过：`140.png` 的 `CAMERA_FROM_WORLD + SIMPLE_RADIAL` 产生 `989×1320` RGB、`[5,989,1320]` logits 与 `uint8[0,4]` 语义图，未读目标 RGB 或标签。
- 三个 offset-residual checkpoint 均严格重载一致；219 个 OOF prediction 与 219 条 contribution record 已封存。活跃 generated offsets 总计 45,358,180，opacity-weighted residual L1 总计 1,688.2197。
- 一次技术恢复被如实保留：首个 r2 进程在候选封存后、calibration 汇总写入前停止，因为单图报告错误要求每张图都出现五类；未输出任何 aggregate 质量数值。修复仅删除了该错误的单图五类指标调用，重放了同一份逐项 hash 验证的候选，未改变模型、预测、门、阈值，未读 check/developer。

## OOF 资源与重载

| fold | train / OOF views | training seconds | inference seconds | GPU hours | peak MiB | reload |
|---|---:|---:|---:|---:|---:|---|
| 0 | 114 / 95 | 2816.8283 | 28.0930 | 0.78245 | 547 | bit-identical |
| 1 | 128 / 84 | 2865.8632 | 25.1073 | 0.79607 | 547 | bit-identical |
| 2 | 177 / 40 | 2833.9425 | 11.7983 | 0.78721 | 493 | bit-identical |
| total | — | 8516.6340 | 64.9985 | 2.36573 | 547 | all passed |

## 封存 calibration（108 views）

对比是相同 OOF 视角上的 `SEM-160`（冻结 `SEM-161` 的 OOF 前身），不是已经关闭的 74-view developer 审计。`SEM-230` 在允许的 lane 内发生 610,402 个 background→deck 与 339,365 个 deck→background 变化；250,188,212 个 background/deck 像素可变，2,303,775 个 foundation rejection 像素保持保护。

| 类别 | SEM-160 IoU / precision / recall | SEM-230 IoU / precision / recall |
|---|---|---|
| background | 98.304 / 99.557 / 98.736% | 98.069 / 99.476 / 98.578% |
| deck | 94.530 / 95.628 / 98.800% | 91.271 / 93.366 / 97.600% |
| stay_cable | 89.636 / 94.142 / 94.931% | 89.636 / 94.142 / 94.931% |
| tower | 86.328 / 88.866 / 96.798% | 86.328 / 88.866 / 96.798% |
| foundation | 65.596 / 67.636 / 95.606% | 65.596 / 67.636 / 95.606% |
| mIoU (with background) | 86.879% | 86.180% |

门的结果：deck IoU `−3.2591pp`（要求 `+0.2pp`）、recall `−1.1995pp`（要求 `+0.5pp`）、precision `93.3661%`（要求 `≥94%`）、mIoU `−0.6988pp`（要求不降），均失败。`stay_cable`/tower/foundation 与 foundation rejection 均完全一致，RGB 也完全一致，但保护工程门不能抵消桥面质量退化。因而没有 final 训练、check 或 developer 审计。

## 可视化与哈希

同视角 GT / SEM-160 / SEM-230 / 改正或伤害面板及桥面局部放大图已输出到 `calibration_panels/`：001、017、093 各一张面板与一张 deck zoom。

- `result.json`: `404bbff9527598db39856bd12ba7616e5e27ee74e319f9e00d0e8422e8ef0816`
- `calibration_result.json`: `eab7a320570986854a0d71419b355762e1c579020cfc974850b0b968a22cd462`
- `prediction_manifest.json`: `790aa681cbb3ed65defbf56a699b17589aa5448eca5b74b416e2822bc23a5e47`
- `contribution_manifest.json`: `d68b8f139949f8d68dfca253c0757f7078e3ca589c77c26390661dfa9619819b`
- `zero_residual_equivalence.json`: `5287abff90653632642bbb3c056708a7dfed273a62365dfdb71948549010c135`

## 结论与回退

SEM-230 未超过 SEM-161，且在其更早、更严格隔离的 OOF calibration 已显著落后 SEM-160；立即淘汰，不以阈值、训练步数、残差权重或后处理挽救。结果否定了“只把共享 logits 拆成独立 offset background/deck 残差”足以分离桥面边缘与邻近背景的假设。可立即回退且唯一保持的正式方案仍是 `RGB-030-FULL + SEM-161`。
