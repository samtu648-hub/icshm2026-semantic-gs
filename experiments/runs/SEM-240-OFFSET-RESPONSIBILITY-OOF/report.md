# SEM-240-OFFSET-RESPONSIBILITY-OOF

状态：`completed_calibration_gate_failed`，两臂均 drop，不拟合 final。

## 协议与因果范围

父实验 SEM-160 OOF（冻结 SEM-161 的 OOF 前身），seed 0，HEAD `49cdcdb1c9915a9544d568754af95ea20a85ec9c` 加本次工作树。
配置 `config.yaml`，数据 `contest_dataset_v1`，固定 `sem_005_oof_spatial_v1`。
两臂相同 offset 参数表、Dirichlet(1,1) 先验、二类概率渲染与保护融合；唯一变量为标签证据的责任分配。
控制累计所有 offset 的真实像素 `T*alpha`；候选仅把证据分给该像素最大贡献 offset。
使用闭式分类后验拟合，不是 10k 梯度训练，也没有网络、阈值、权重或 checkpoint 搜索。

三折 train/OOF/guard 分别为114/95/10、128/84/7、177/40/2，均通过集合隔离。
每个 fold 只读取自身 train 标签来拟合；全部219预测封存后读取108 calibration指标。
111 check 与74 developer指标未读、final未拟合。这里的“未读check”指未用作评估；一些check视角作为其他折的训练输入是既有OOF协议的一部分。
继承的 SEM-001 保护基底曾在全部219视角训练，故这是**增量 OOF**，不是从底到顶完全独立的 OOF。calibration也已多次用于历史研究，不称盲测。

## 结果（108-view calibration）

SEM-160 → 全贡献控制 → 最大贡献候选：

- mIoU：86.8790% → 86.6663% → 85.9478%。
- deck IoU：94.5301% → 93.5866% → 90.2270%。
- deck precision：95.6281% → 97.6380% → 97.8276%。
- deck recall：98.8000% → 95.7545% → 92.0717%。

候选五类 IoU / precision / recall（百分数）：

- background：97.9512 / 99.0870 / 98.8433。
- deck：90.2270 / 97.8276 / 92.0717。
- stay_cable：89.6361 / 94.1423 / 94.9307。
- tower：86.3283 / 88.8658 / 96.7983。
- foundation：65.5965 / 67.6357 / 95.6056。

固定门：IoU +0.2pp、recall +0.5pp 和mIoU不下降均失败；precision≥94%及所有保护/RGB门通过。
候选相对匹配控制 deck IoU再降3.3596pp、recall再降3.6827pp。
相对SEM-160，deck真阳性净少505,174，假阳性净少185,633；这不是健康的边缘修正，而是过度删除。
017同视角局部图可见桥面下缘产生锯齿与缺口；三个预先固定样例均已输出，未用于修改配方。

## 工程与资源

- 真实渲染颜色 adjoint 贡献恒等式相对误差 `7.82955e-6`（门 `1e-4`）。
- 独立诊断与原生 foreground 最大差 `1.04308e-5`（门 `2e-5`），winner权重和误差 `1.84960e-7`。
- 6个参数表 checkpoint逐值重载通过；每折新候选二分类分支经140官方相机重投影输出合法5通道，未调用旧sidecar替代。
- RGB拟合前后浮点 tensor逐位相同，六项冻结RGB哈希不变。
- class 2/3/4 和foundation拒绝区域完全一致；background/deck没有额外边界锁。
- 三折拟合时间28.804 / 35.336 / 46.019秒；端到端171.479秒；峰值Torch allocated618.005MiB。

## 产物与哈希

外部根：`$ICSHM_OUTPUT_ROOT/r2/sem_240/full-20260907-r1/`。
`prediction_manifest.json`：`0c73db354ad09fc577b0c33996757aee384ddca05b5afeb3ec4a292820e43444`。
`calibration_result.json`：`9cf64fa86aa43482f966e5274fcb239a033c4f92930f6b929fda037e4ede0832`。
各折checkpoint、贡献及支持视角记录、工程记录、全部预测、`control_panels`/`candidate_panels`和真正四列局部放大图均保留。
完整五类混淆、P/R/IoU与每视角变化在calibration JSON；checkpoint哈希在fold_results及result JSON。

## 决定

SEM-240没有提供超过SEM-161的资格证据，不晋级、不读取check/developer、不再改责任阈值/支持视角数来挽救。
本结果否定的是**最大贡献硬责任配方**，不能推广为所有学习式对应无效。控制优于候选还表明，丢弃次贡献者可能丢失真实轮廓证据。
短期保留SEM-161，先完成隐藏评测与失败类型审计。若未来再开语义研究，应有新的、独立的跨视角对应或边界可见性证据，并验证它在漏检区有效，不能继续基于同一冻结高斯的简单投票/硬归属。
