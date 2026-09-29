# RGB-114：PGSR数值异常诊断

日期2026-09-09；父RGB111完整30k，seed0、原303/74 split及manifest，Git base49cdcdb1c9915a9544d568754af95ea20a85ec9c加每attempt代码/config/input哈希。诊断只用训练RGB，不读开发RGB或语义标签；原模型、旧运行及18冻结产物不变。

## 已确认事实

- PLY检查：7k共399222点，无非有限字段；15k共573936点，651点含NaN；30k同总数，6080点含NaN。xyz/opacity/scaling/rotation都有污染，SH字段未检测到非有限数。
- 7k PLY是在当步裁剪/优化前保存，训练checkpoint在当步之后保存；诊断恢复训练checkpoint点数356504，不把两者点数差异误作输入被改。
- r3在7k checkpoint上完成20步、未观察到非有限值，只能说明短窗口正常。原checkpoint没有RNG/采样器状态，诊断不声称逐步复现原训练轨迹。
- r4延长观察上限至3000步，在7642步、更新前发现xyz非有限梯度；完成641次优化后停止。
- r6保留当前计算图，在7121步分别重算各损失梯度：geo_loss产生xyz240/opacity80/scaling240/rotation320个NaN元素；image_loss、normal_loss、ncc_loss、min_scale_loss均有限。这定位到多视角几何梯度，尚不是完整修复验收。
- 非确定性CUDA归约及重启采样状态差异使各次首个异常步不同，不拿一次20步成功宣称数值稳定。

## 失败和诊断工具修正

- r1：checkpoint受新PyTorch weights_only默认限制；只完成PLY清点，没有执行反传。其顶层旧状态字符串写为完成诊断，必须结合子记录setup错误理解，不能当梯度实验完成。
- r2：受限NumPy scalar导入路径不兼容，未执行反传。
- r3：只对校验过的自有checkpoint允许NumPy scalar/dtype，不关闭weights_only安全加载；20步通过。
- r5：再次捕获坏梯度，但逐分量mean在原生no_grad块中构造，拆解失败；r6用明确enable_grad修正诊断，不修改训练损失。
- r7：补上作者入口的`--detect_anomaly`。此前在入口外启用的异常检测会被train.py默认参数关闭，因此r3–r6有效证据是显式梯度/参数检查，不声称开启了原生异常追踪。

各次目录为`outputs/quality/diagnostics/RGB-114-numeric-r1`至`r7`，全部保留。r7终态及算子级定位以其result/backward.log为准。

r7最终在7769步、训练相机092触发`LinalgVectorNormBackward0`，前向来源为作者train.py第234行`pixel_noise = torch.norm(pts_projections - pixels, dim=-1)`。该步各损失标量均有限，总loss0.267640293；因此只检查loss值不足以排除梯度污染。

## 独立safe-cycle候选

`safe_cycle_config.yaml`登记单变量修复诊断，独立source副本只改变train.py多视角循环计算；原source不变。先按原可见性与pixel_noise阈值在无梯度路径选择对应，再对被接受的同一组对应重新建立可求导循环，保留原权重。被排除的无效投影不进入反传；不清零模型/梯度、不关闭几何损失、不增添新的几何阈值。

三项CPU自动微分测试通过：有限inlier的损失/梯度与原算法精确相同；无效零深度不会污染有效梯度；无有效对应时正常弃权。`RGB-114-safe-cycle-r1`从原7k checkpoint执行最多3000步带异常追踪的诊断，状态以其终态result为准，不能用单元测试或诊断替代新的完整30k。

本次修复没有声称原始RGB低分的全部原因已解除，也没有从诊断checkpoint导出新参赛模型。安全候选若通过仍须另ID全预算训练/重载/字段接口/语义筛选。

最终窗口结果：`RGB-114-safe-cycle-r1`完成3000次优化（7001–10000），带原生异常追踪、每步梯度与参数有限性检查，全窗口无非有限值；615.782757秒，peak allocated5967.247559MiB，18冻结/原检查点hash一致。终态`diagnostic_limit_reached_no_nonfinite_observed`是达到预注册上限，不是失败或完整训练收敛。result SHA256 `efa216e4d034d3202b266713193d04d8cee8fdaf1e3993216a7a126ee40b89b3`。RGB115独立完整训练尚未执行；本诊断任务完成，但数值稳定性不外推到30k或其它种子。

## 已实现保护与边界

新重建runner在评分前清点所有保存的模型PLY字段；有非有限值会记录numeric_saved_model_integrity并停止，坏模型及清单保留。原生adapter错误现在包含字段和计数，不放宽检查。该保护不改变训练配方，也不能替代早期训练梯度守卫。

潜在除零分支需先用算子轨迹确认；PyTorch官方说明，先除零再用mask排除仍可能污染梯度，修复应发生在计算前，不能靠清零NaN梯度：[Autograd机制](https://docs.pytorch.org/docs/2.14/notes/autograd.html#division-by-zero-in-autograd)。本机实际版本2.11.0+cu128，不将文档版本写成运行版本。

回退：历史RGB030+SEM161保持原状。修复后的PGSR仍须独立完整30k、RGB重载、原生接口和匹配语义测试；本报告不宣布PGSR方法淘汰或修复成功。
