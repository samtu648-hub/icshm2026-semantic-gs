# SEM-180-RGB-INPUT-AUDIT — Frozen deck new RGB input audit

- 状态：not_run_parent_RGB_gate_failed；按协议关闭
- 父实验：RGB-092-TRANSIENT,SEM-161-DECK-BINARY-FINAL
- 配置：本目录 config.yaml；seed 0；split rgb_r1_spatial_v1。
- 2026-09-06：父RGB-092完整30k的数值及视觉门失败，按ADR-048不启动语义新输入审计。新语义预测0、打开语义评估标签0；未重训，也未把旧sidecar挂到新高斯上。
- 已实现冻结0.99/960/480输入替换、严格保护、旧输入重放、五类评估和官方外部格式检查的条件代码；仅执行并测试了RGB失败的关闭路径，真实通过路径的端到端推理与外部语义验证未运行。
- 不产生候选五类指标，不声称桥面收益或否定桥面输入假设。当前SEM-161历史deck IoU/precision/recall仍为91.4667%/95.0392%/96.0525%，mIoU83.3870%；完整五类历史指标列于RGB-092报告。
- 旧SEM-161 checkpoint、74份预测和对应旧RGB全部哈希验证通过。Git HEAD `49cdcdb1c9915a9544d568754af95ea20a85ec9c`加保留工作树；条件结果保存在`ICSHM_OUTPUT_ROOT/r2/sem_180/input_audit/full-20260906-r1/result.json`，SHA256 `6a1c6e9dad020b268167fb5e6666d779ab9bd16de125730d085170cf3f6143ba`。
- 回退方式：直接继续使用未修改的RGB-030-FULL + SEM-161及其原sidecar/逐级父方案。提交配置未切换。
