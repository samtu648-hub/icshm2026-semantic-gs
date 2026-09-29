# RGB-071-SEMANTIC-COUPLING-DIAG — RGB重建误差与语义错误耦合审计

- 状态：completed_read_only_diagnostic
- 阶段：R2 read-only cross-task diagnostic
- 父实验：`RGB-030-FULL`, `SEM-001`, `SEM-111-PROTECTED-FUSION-FINAL`
- 配置：`experiments/runs/RGB-071-SEMANTIC-COUPLING-DIAG/config.yaml`

## 问题与边界

检查语义问题是否随RGB重建精度恶化，以及不同语义路线是否具有不同耦合机制。只读取已
复用的74-view开发集、冻结RGB/reference对、现有GT与SEM-001/SEM-111预测；不改模型、
预测、阈值或提交候选。相关性与像素分层仅作机制诊断，不作因果证明或新鲜泛化证据。

## 预注册统计

1. 逐视角计算PSNR/SSIM/LPIPS-Alex与mIoU（含/不含背景）、`stay_cable` IoU的Pearson和
   Spearman相关；
2. 逐像素RGB MAE按语义correct/wrong、background正确/background→`stay_cable`、
   `stay_cable` TP/→background/→other分层；
3. 同时记录真实RGB的固定前向亮度边缘强度，辨别“重建颜色误差”和“真实细线/边界难度”。

## 结果

74个视角的整体PSNR与语义质量没有正相关：SEM-001的PSNR–mIoU Pearson/Spearman为
`−0.141/−0.236`，SEM-111为`−0.081/−0.161`。视角内容、类别占比和难度混杂了全图
PSNR，因此不能用“某视角PSNR低”直接解释该视角mIoU。感知误差LPIPS与SEM-111
mIoU的Pearson/Spearman为`−0.305/−0.377`（LPIPS越高越差），与SEM-111
`stay_cable` IoU为`−0.363/−0.488`，存在中等方向一致关联。

像素分层给出了更明确的机制差异：

| 语义路线与像素层 | 像素数 | mean RGB MAE | 真实边缘强度 |
|---|---:|---:|---:|
| SEM-001 正确背景 | 74,226,997 | 0.056194 | 0.008418 |
| SEM-001 background→stay_cable | 13,166,331 | 0.050229 | 0.012783 |
| SEM-111 stay_cable TP | 2,167,514 | 0.048863 | 0.016744 |
| SEM-111 stay_cable→background | 749,936 | 0.108138 | 0.042492 |

SEM-001的大面积background→`stay_cable`误报处RGB MAE比正确背景低10.62%，并不集中
在RGB重建较差像素。相反，SEM-111漏掉真实索线时RGB MAE为命中索线的2.21倍、真实
边缘强度为2.54倍；其所有错误像素RGB MAE也为正确像素的1.68倍。这与SEM-111使用
冻结渲染RGB作为二维输入、SEM-001使用跨视角静态anchor logits的结构差异一致。

结果位于`ICSHM_OUTPUT_ROOT/r2/rgb_071/semantic_coupling/full-20260903-r1/result.json`，
SHA256为`fb12191767b55ab1b53db478fa0a210e9c3a5a08f4ad0843a05ceaa5978431d9`。

## 结论

RGB重建精度是SEM-111细索召回的真实瓶颈之一，但不是SEM-001大面积背景误报的主因。
因此几何改进应以恢复细线和减少浮点为目标；不能期待它单独修复静态3D语义的跨视角
类别混合。该开发集统计是关联证据而非因果证明，不直接触发清点、重训或语义规则修改。
