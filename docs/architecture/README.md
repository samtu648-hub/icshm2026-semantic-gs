# Architecture

- [Scaffold-GS + GeometryPrior + VC-Semantic 总体方案](scaffold_geometry_prior_semantic_plan.md)：冻结 RGB、训练期几何先验、语义路线和提交边界；具体执行状态以项目状态文件和实验记录为准。
- [GeometryPrior 统一评估协议 v2](../experiments/geometry_prior_evaluation_v2.md)：四个前馈候选与 GEO-000 控制的预注册比较规则。
- [A6000 GeometryPrior 交互准备矩阵](../experiments/a6000_geometryprior_interaction_readiness.md)：仅记录硬件迁移后的控制、资源和唯一 ID 规则；不授权重跑已淘汰配方。
- [SEM-003 结案记录](../experiments/sem_003_closure.md)：S0/S1/S2 的指标、负面结论和独立外部相机接口复验。
- [VC-SemGS计划](vc_semgs_plan.md)：当前推荐主线。
- [RGB Backbone公平比较与冻结协议](rgb_backbone_bakeoff.md)：Standard、Mip-Splatting、Scaffold-GS和条件候选3DGS-MCMC。
- RGB 骨干已由 RGB-060 冻结为 Scaffold-GS；后续 GeometryPrior 挑战不得改写该已登记事实。
- 所有架构图和实现描述必须标明“已实现”“已验证”或“计划”，避免把计划写成结果。
