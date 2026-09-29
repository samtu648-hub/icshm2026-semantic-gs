# SEM-432：双分支 Render/Render 匹配对照

2026-09-21完成。该实验不引入真实照片监督，只把同一render裁块通过两个独立的Mask2Former损失图，等权平均后做一次更新，用于控制SEM433额外损失采样与计算量。

256训练/30确认/17缓冲、seed20260805、6000条view/crop/flip、ADE初始化、优化器和冻结RGB152均与SEM396绑定。训练6000步用时2744.087秒，5次标准AMP回退，峰值reserved 9370MiB；6000行采样精确、冻结RGB哈希精确，确认标签在checkpoint封存前访问为0。最终checkpoint SHA256为`e3eeca1c5a6df11b11344fc035468cf4c560ce7641f3d231afa9b58ab30f3a3a`。

空间确认五类/前景mIoU为89.436596%/86.957823%，索2px边界F1为77.024876%。相对单损失图SEM396分别+0.231990/+0.290446/−2.494449pp；deck/foundation IoU提高1.418722/2.023909pp，但`stay_cable` IoU下降2.120622pp。这证明仅增加第二个随机loss图已会改变大构件/细边界取舍，不能把SEM433直接和SEM396比较来归因照片域。

本ID仅作匹配对照，不晋级、不替换SEM396或生产RGB140+SEM378。训练/推理结果SHA256分别为`a52e23065600fcccf6be26478d59940c513883f79c185ebd3daa0bba60725ec3`和`e78c2096e5040cf992cf905a506a5b2d070933c5b9b87e7878e74107dce8b850`。
