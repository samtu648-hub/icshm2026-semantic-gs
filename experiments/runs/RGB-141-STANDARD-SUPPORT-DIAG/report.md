# RGB-141-STANDARD-SUPPORT-DIAG

2026-09-16 completed_readonly。父RGB140，HEAD 49cdcdb1c9915a9544d568754af95ea20a85ec9c；seed0、manifest contest_dataset_v1、split released_all400_rgb_all300_sem_v1，正式配置见config.yaml。执行源码SHA见输出result.json；源码与旧工作区清单在outputs/quality/rgb141保存。此次没有优化器、增密、训练、模型更新。

复用RGB123/124与RGB134/135结论，另在RGB140的159岸线、远端、水面、真实stay_cable和结构边界各取绝对ROI opacity导数前二，冻结后共8个去重贡献者，进行8×400有限去除反事实和全400边际导数。有限去除只在内存临时设置opacity logit=-100，每次立即恢复；父PLY哈希及内存opacity均保持。RGB区域残差采用float RGB，略不同于历史8bit量化MAE，不能把量化差解释为新模型变化。

八个贡献者有81–106个可测贡献视角，均同时出现改善/损伤的有限去除结果；这直接排除“少视角=伪影”作为当前证据的解释。例如水面/边界344333的91个贡献视角中，有限去除51改善/39恶化；另一个水面214694为106视角、40改善/66恶化。远端117356为89视角，去除34改善/54恶化。真实stay_cable区域贡献者442838也得到低于1的支撑权重0.88948，去除52改善/43恶化；该点不因此被定义为假索。岸线120562权重1.09236，去除28改善/61恶化。少数零变化/浮点抵消解释改善与恶化数不必等于可测贡献视角数。

信息来源可靠于“原生当前模型的光度边际导数、实际去除贡献和照片误差”，不可靠于“真实三维几何、语义正确性、跨视角独立对应”。它尚不能保证伪影与真索分离，但足以实现一个连续、缺证中性的训练分配假设。按用户要求不设逐类质量硬门，进入一次事先固定的匹配预算全30k对照，不扫描权重或改为opacity修复。训练只读取原有RGB损失梯度，不输入诊断标签/ID。

完整selection、3200行逐视角反事实、权重数组和父哈希在outputs/quality/rgb141/diagnostic-r1；GPU过程约85.58秒。新权重公式与预算参见docs/experiments/rgb143_support_densification_protocol.md。默认模型未切换，回退无需修改任何模型。

## 完成后续对照与交付

RGB142新预算控制与RGB143加权排序均完成全400/30k匹配实验，结论不采用本次固定配方，见[完整对照报告](../RGB-143-STANDARD-SUPPORT-DENSIFY/report.md)。本诊断未进行训练或修改父模型。159照片/父渲染/残差图和8贡献者跨视角证据图已加入outputs/quality/rgb141/review-r1/index.html；前者使用保存PNG，原始诊断仍为浮点残差，两者口径分列。最终源码封存于evidence-code-r2，保留初始快照和失败现场。
