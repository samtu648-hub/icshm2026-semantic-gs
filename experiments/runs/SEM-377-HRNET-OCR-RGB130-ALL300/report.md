# SEM-377-HRNET-OCR-RGB130-ALL300

2026-09-15 已注册，准备执行一次固定配方的 HRNetV2-W48 + OCR 五类训练。输入为冻结 RGB130 的全部 400 张原生 1320×989 渲染，其中有人工标签的 300 张用于训练与训练视角拟合评价；其余 100 张只作无 GT 合理性检查。父对照是同一 RGB130 输入上的 SEM371 Mask2Former，绝不重训、覆盖或替换。

配方依据：官方 HRNet OCR Cityscapes 配方使用 SGD、momentum 0.9、weight decay 5e-4、OCR/non-backbone 10× 学习率和 poly(0.9) 调度。本机单卡采用 batch 1、累积 4，因此按官方 global batch 12 的线性缩放采用 base LR 0.003333；使用原生 768×768 随机裁块和翻转，不缩小全图、不引入多尺度或额外模型。12,000 个更新相当于 160 个有效 300 图 epoch，长于 SEM371 的 6,000 步，这是针对 HRNet+OCR 的单次、预先固定收敛预算，并非结果驱动的搜索。

上游：HRNet Semantic Segmentation commit `0bbb2880446ddff2d78f8dd7e8c4c610151d5a51`（MIT）与 HRNet Image Classification commit `f760c988482cdb8a1f69b10b219d669721144582`（MIT）。权重使用官方发布的 ImageNet `HRNet_W48_C_ssld_pretrained.pth`；初次 cosine/cutmix 下载损坏已保留在独立权重目录且不加载，ssld 权重已用 `weights_only=True` 成功读取。类别固定为 `background, deck, stay_cable, tower, foundation`，其中 class 2 正式名称唯一为 `stay_cable`。

数值预检通过后，`train-r1` 在第一个累积边界检测到初始 AMP scale 过高而停止；此前无优化器更新，记录已保留。`train-r2` 使用标准 GradScaler 初值 1024 完成了 12,000 个裁块前反传，但事后核对确认其配合累积4只进行了 3,000 次 SGD 参数更新，与原配置把“12,000 更新”误写为裁块次数不符。该批以及 `infer-r1` 的 74.931333% 训练视角 mIoU 均保留为协议不符合的失败尝试，不用于架构结论或模型选择。唯一合规的正式批次将固定为 24,000 个同分布原生裁块、累积4、恰好 6,000 次 SGD 更新；优化器、LR、损失、模型、数据、seed、RGB130、推理窗口均不变，不构成参数搜索。

正式训练结束后将在独立输出目录登记实际资源、模型大小、300 图混淆矩阵、固定九图/ROI、100 无标注预测、同一 144 帧轨迹视频与可浏览对照页。所有 300 图数字将明确标为训练视角拟合，不称为未知视角或盲测。
