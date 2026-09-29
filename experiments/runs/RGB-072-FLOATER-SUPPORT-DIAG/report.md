# RGB-072-FLOATER-SUPPORT-DIAG — 冻结Scaffold浮点支持度诊断

- 状态：completed_training_only_diagnostic
- 父实验：`RGB-030-FULL`, `RGB-070`, `RGB-071`
- 数据：仅303个训练相机及其RGB；不读取74-view参考RGB或任何语义数据

## 目的

判断空中/水面的半透明长条是否对应“只被极少训练视角支持、却主导大量像素且训练残差高”的
anchor。该条件若不成立，基于孤立点/低支持度的全局清洗就没有安全依据，尤其可能同时删除
真实细索。

## 固定统计

- 使用冻结rasterizer的逐像素最大边际贡献generated offset，并精确映射回189,073个anchor；
- 单视角至少主导16像素才计一次支持；累计记录主导像素、训练RGB MAE、贡献、active视角数；
- 同时记录view-dependent generated opacity、scale与各向异性；至少累计256个主导像素的anchor
  才进入高残差排名；
- 本ID不生成prune mask、不渲染候选。若训练侧确有可分离群体，必须另立ID并在读取74-view
  参考RGB前冻结唯一清理规则。

## 结果

303个训练视角在89.69秒内完成；冻结artifact哈希前后完全一致。训练渲染聚合PSNR为
23.3819 dB，逐视角mean RGB MAE均值为0.018183。189,073个anchor中，182,722个曾
生成正opacity offset，135,898个至少主导一个像素；59,614个累计主导至少256像素。

累计≥256像素且有效主导支持≤4视角的anchor有10,946个，其RGB MAE中位/q95/q99为
0.022681/0.057426/0.111310，说明“低支持”本身并不等价于坏点。进一步固定训练侧异常
条件：mean RGB MAE≥0.15且高误差像素比例≥0.50，只剩29个anchor、18,078个训练主导
像素。它们是可做一次保守可逆屏蔽的极小群体，但本诊断不生成mask。

结果与逐anchor数组位于`ICSHM_OUTPUT_ROOT/r2/rgb_072/floater_support/full-20260903-r1`；
`result.json`/`anchor_statistics.npz` SHA256为
`abfb0ee5e7b51cd87a8acb050ff036a2c422e00d55882519b40995b230aba2e0` /
`563a90d9b87e4696eaf09dd4b5a1500db0a5490951e9fdf65fd0048da061fecd`。

## 决定

低支持anchor整体不可直接清除；其分布与大量正常结构混合。只允许另立
`RGB-073-RESIDUAL-SUPPORT-PRUNE-PF`验证上述29-anchor极保守群，不扫描阈值；若不能
改善固定RGB指标和预选浮点样例则drop。

