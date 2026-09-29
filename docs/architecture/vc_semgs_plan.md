# VC-SemGS 计划

> 状态：计划，尚未在本仓库实现。  
> 全称：Visibility- and Conflict-aware View-conditioned Semantic Gaussian Splatting。

## 问题

普通语义3DGS把固定语义属性附着到Gaussian，隐含语义跨视角一致。已有数据审计和E2结果表明，部分`stay_cable`二维多边形包含拉索之间的天空、水面或远景包络，使固定语义产生稳定的背景→拉索错误。

本方法不判断某个标签是否“物理真实”，而是检测一个可见Gaussian在不同观察方向下是否收到稳定或冲突的官方监督。

## 数据流

```text
冻结RGB场
   ├── 静态5类logits
   ├── 基于边际渲染贡献的可见性门控
   ├── 多视角类别分布与冲突分数
   └── 仅对bg/cable高冲突点启用低阶视角残差
```

## 约束

- 不新增独立Proxy几何；
- 不使用相机ID作为语义条件；
- 视角条件只使用连续观察方向，先一阶SH、必要时二阶；
- 残差只能在`background`与`stay_cable`间转移logit；
- 不覆盖`deck`、`tower`、`foundation`；
- 语义分支不得改变RGB；
- 只通过空间隔离的未见视角决定是否保留。

## 实施顺序

1. S0：冻结RGB上的五类静态logits。
2. S1：可见性监督。
3. S2：Gaussian多视角冲突统计。
4. S3：冲突点视角条件残差。
5. U1：仅在主方案稳定后测试软伪标签。

定量门槛见 [ROADMAP.md](../../ROADMAP.md)，指标定义见 [METRICS.md](../../METRICS.md)。

