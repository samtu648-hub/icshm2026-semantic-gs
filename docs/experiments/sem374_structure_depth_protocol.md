# SEM374 — 人工标注筛选多视角结构点的深度约束

用户2026-09-15批准先Standard、Scaffold后置。配置及每阶段源码哈希为执行依据，独立输出outputs/quality/SEM-374。

## 单一机制与预算

原RGB130全400/30k＋SEM373冻结6000步语义端点（两套Adam完整复用），各2000步同400相机5轮；1500标签访问、500仅RGB访问。RGB损失0.8L1+0.2DSSIM更新XYZ和SH，0.1语义NLL只更新logits，两臂均相同。联合额外0.05结构损失到XYZ。尺度、旋转、opacity、数量、曝光冻结，不增密剪枝、无平面损失。不重复冻结语义训练，不改Mask2Former/融合。

结构点选择：从原初始化COLMAP points3D及官方track恢复真实观测。仅全部可见人工标签一致的deck/tower/foundation，至少3个合格标注视图，内距标签边界>2px，各观测重投影≤2px，最大观测夹角≥2°。100无标注图不制造结构标签。按point_id modulo5==0留作新增深度损失外检查，其余用于fit；两类点都曾参与原RGB初始化，因此不是独立几何真值。真实stay_cable因索面包络与实体线语义差异，不作深度目标。

渲染：native inverse depth为∑Tα/z，同次白色override得到∑Tα，两者相除（分母下限1e-6）形成期望逆Z，按官方COLMAP edge-origin xy减0.5后双线性采样。e=(pred_inverse_z−reference_inverse_z)/reference_inverse_z；对当次视角全部fit点平均SmoothL1(e,0,beta=0.05)。不按当前误差/alpha动态筛点或按类调权。控制臂也同样forward并记录损失，结构geometry view全部detach；仅联合开放XYZ，其余几何和语义参数不从depth收到梯度。

这测试“标注筛选＋稀疏深度”整体机制；没有无标注筛选深度臂，不能单独归因语义筛选。该监督约束结构内部稀疏深度，不是直接轮廓损失，也不能新增缺失实体。

资源预估30–50分钟GPU阶段、显存4–6GiB、新增5–8GiB，使用共享gpu.lock且检查非协作GPU计算进程；不按质量停止或搜索权重。来源：[原始3DGS](https://github.com/graphdeco-inria/gaussian-splatting)，原生深度功能支持工程路径；本实验是自定义稀疏约束，非声称复现该库的单目深度配方。上游commit/许可证沿SEM373记录，未复制整仓或改上游。

## 固定核验与评价

1. 结构观测角色互斥、标签一致性/投影/坐标、实际梯度阻断、有限差分、inverse-depth override oracle、原九RGB精确；失败现场保留。
2. 两臂起点模型及两Adam、2000完整采样、冻结参数/652298点、端点fresh重载；训练中检查有限值。
3. 全400 RGB、全300五类3D语义投影为训练视角拟合；旧九图相同ROI、边界RGB及各类RGB误差；官方144帧同轨迹无GT仅视觉。原RGB130已有544导出与评价复用，新两臂完整评价。
4. 三套按fit/check与三结构类别报告相对逆深度绝对误差和alpha，check仅不参与新loss；不能把优化自身代理目标下降等同于真实几何改善。
5. 浏览页面沿SEM373统一裁剪/视频布局，另展示结构点和深度残差；报告拉索、桥面/塔/基础以及背景的实际得失。不自动部署/扩Scaffold/扫参。
