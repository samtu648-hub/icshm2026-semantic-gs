# SEM-403 原生射线责任重放预检协议

## 目的

SEM398--402关闭的是训练后父Gaussian二分家族；下一主线需要观察像素射线内部的真实alpha-transmittance贡献结构。SEM403只验证现有AbsGS原生前向缓冲区能否被无损解码并逐像素重放，不读取任何语义标签，不训练，也不提出质量结论。

## 固定输入

使用冻结RGB152端点和`semantic_direction_refinement_v1`中三张训练相机：`001.png`、`120.png`、`300.png`。每张图固定取16×12均匀网格，共576个像素。禁止访问所有语义标签、30张确认相机、31张buffer相机和无标签外部相机。

直接调用现有编译扩展的原生forward，使用由Gaussian ID确定的三通道诊断颜色。只读解码其`GeometryState`、`BinningState`与`ImageState`缓冲区中的深度、二维均值、conic-opacity、深度排序point list、tile ranges、像素最后贡献者与最终透射率。CPU重放必须严格复用原生规则：`alpha=min(0.99, opacity*exp(power))`、`alpha<1/255`跳过、下一透射率低于`1e-4`时停止、前向`alpha*T`合成。

## 事前门

三张图合并后必须全部满足：诊断颜色、最终透射率和逆深度最大绝对重放误差分别不超过`1e-5`；非空贡献trace比例至少90%；接受贡献者的深度顺序违反数为0。RGB152正常RGB在预检前后仍须对`001.png`逐像素一致。

通过仅授权另立标签诊断，届时再事前冻结边界/内部采样、surface-set公式与训练授权门。失败则先修复可观测性工具或改用独立trace renderer，不得使用不可信贡献数据训练。
