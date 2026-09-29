# QUAL-019：配对照片域监督审计

2026-09-21完成。独立CPU重读30张GT及两臂预测，两个混淆矩阵与保存结果逐项一致；初始state、6000采样、render/photo哈希、checkpoint前后哈希和冻结RGB九项审计全部通过。另逐行核验两份6000步journal，sample、学习率及每步开始时CPU/CUDA RNG哈希4/4精确；不同AMP回退步不会改变后续成功更新的随机起点。主结果为SEM433相对SEM432五类/前景mIoU−0.327939/−0.392040pp、索边界F1−0.022513pp；结构类IoU为deck +0.407246、`stay_cable` −1.966317、tower +0.156236、foundation −0.165325pp；三区前景为−1.337625/−0.129959/+0.140094pp。事前五门全部失败，状态`completed_not_promoted`。

封存后又只用RGB140 render对两个模型各推理301--400共100张，无标签、无目标照片访问；与当前SEM378、真实照片人工对照和变化图组成交互审阅页（来源记录：`../../../outputs/quality/QUAL-019/review/index.html`；未随公开包提供）。页面支持100无标签/v1确认切换、下拉、上一张/下一张和方向键，已在127.0.0.1浏览器实际渲染检查。无标签视觉只用于发现错误形态，不改变失败结论。

`summary.json`、pair audit与public100结果SHA256分别为`434bab87a846ddaa126e30e95ae5e4be59e0688a3b66f2469910a933bbc9e3e1`、`08c061a1868c957c9e9e5ae566c4b2409c8bb191ff6c97a3b55c6d22ee1a2df1`和`3080d63d2f1d8082d4dc48a0c977e8f0f4b765943c82148f12d11f4eec647eaf`。回退为继续使用原二维锚/生产候选，不部署SEM432/433。
