# RGB-080-LATE-PRUNE — 训练后半程原生 opacity pruning

- 状态：completed_primary_and_visual_gate_failed
- 父实验：`RGB-030-FULL`
- seed：0
- split：`rgb_r1_spatial_v1`（303 train / 74 development validation）
- 唯一研究变量：anchor growth仍在15,000停止，但原生opacity/访问统计与pruning延长到29,900

## 动机与实现边界

RGB-074证明按训练低支持/高残差事后屏蔽29个anchor只带来不可感知的微小变化，不能清除
天空横条、幽灵索和岸边拉丝。RGB-080改在训练过程中让后半程已经失去稳定opacity支持的
anchor被Scaffold-GS原生规则持续清理，而不是再按验证图或人工点选删点。

上游`training_statis`同时收集opacity访问量和anchor增长梯度，直接把`update_until`改成30k
会同时延长增点，破坏单变量原则。因此隔离补丁将两类统计拆开：0–15k行为与RGB-030-FULL
一致；15k后不保留屏幕空间增长梯度、不新增anchor，只继续原生`min_opacity=0.005`、
`success_threshold=0.8`、100步周期的pruning，最后一次发生在29,900。其余训练超参数、数据、
seed、相机、30k预算和评测全部冻结。

## 预注册判定

本实验只跑一个完整30k，不扫描`min_opacity`、访问门、周期或结束步数。候选必须在固定
054/140/151/159/160/183原分辨率对比中明确减少半透明横条/浮点/拉丝，且不出现可见的
真实索线、塔体或桥面侵蚀；同时满足以下量化门：

- mean PSNR相对RGB-030-FULL至少`+0.05 dB`，或LPIPS-Alex至少改善`0.003`；
- mean SSIM最多下降`0.001`；
- worst-10%与140–146 mean PSNR最多各下降`0.10 dB`。

这是有实际作用幅度的训练时清理，不用近零阈值保护基线。若视觉目标未达到或量化保留门
失败，完整保留RGB-030-FULL并登记负结果；不得在同一ID事后改阈值或重跑seed。只有RGB门
通过，才执行冻结语义兼容与官方外部相机门。

## 结果

唯一一次30k训练、74-view渲染与冻结AlexNet评测均完成。运行耗时30分54.84秒（包含首次
下载约528 MB上游VGG权重、上游附带评测和统一评测），主机最大RSS 4,329,580 KiB、无swap。

15,000到29,900共执行150轮opacity-only pruning。候选自身日志中anchor由199,837降至
186,095，共删除13,742（6.8766%）；第一轮删除680，最后10轮合计删除243，清理平滑收敛。
最终PLY为186,095 anchors，RGB-030-FULL最终PLY为189,073；由于两者是分别训练的CUDA
运行，最终数量差`−2,978`不能替代候选自身逐轮日志作因果计数。

| 指标 | RGB-030-FULL | RGB-080 | 差值 | 门结果 |
|---|---:|---:|---:|---|
| mean PSNR (dB) | 20.596398 | 20.635305 | +0.038907 | 未达+0.05主门 |
| mean SSIM | 0.850177 | 0.851184 | +0.001007 | 保留门通过 |
| mean LPIPS-Alex | 0.322373 | 0.322874 | +0.000501 | 未达−0.003主门且略退化 |
| worst-10% PSNR (dB) | 17.734000 | 17.801923 | +0.067924 | 保留门通过 |
| 140–146 PSNR (dB) | 21.338724 | 21.355694 | +0.016970 | 保留门通过 |

固定视觉样例中，054/140/159/160的PSNR分别改善0.179/0.228/0.223/0.270 dB，159/160的
LPIPS也改善0.0146/0.0122；但151 PSNR下降0.244 dB，183下降0.059 dB。原分辨率左右对比
确认候选有局部锐度变化，但天空水平条带、江面长斜线/幽灵索和近岸拉丝仍明显存在，部分
只改变位置或形态，没有达到“明确移除且不伤结构”的视觉门。六张左右对比保存在外部输出
`visual_audit_rgb030_vs_rgb080/`，左侧为RGB-030、右侧为RGB-080。

## 决定与产物

`keep_or_drop = drop_primary_and_visual_gate_failed`。SSIM、worst-10%和困难区间改善，说明
训练后半程原生pruning并非完全无效；但PSNR主门差0.0111 dB，LPIPS反向，核心浮点/拉丝
没有被稳定清除。因此不替换RGB-030-FULL，不执行只在RGB门通过后才允许的语义兼容和
官方外部相机门，也不在本ID改opacity阈值、周期、结束步或追加seed。

源码基线diff、隔离源码diff和实验补丁SHA256分别为
`0030e06dd814742a1c1537c6ff6b4ec6067492861bf9ca56304db0de162141ac`、
`d5af1ccfa6d8b0d82d6705052965edcaa5cb68ab84b68752ec1e7343220dc7d9`、
`a4a1a4b2bb9ef4d294a4099608ff5557a0e12ef55c304c3b22129483da631d77`。
metrics summary/per-view、checkpoint和PLY SHA256分别为
`84cb0db5fc3a62fc496c4f53e8be7eba2ec9a82365ce2ebfc24a0289a6a0ce4b`、
`8cf84d153d7b599578ae0e5657c7e413b4bb2597ac171f472c314e740ebba91f`、
`b869d9bc00bf02370bb27fa157b15e77c8c4e3fdc457b9fc42b74a380d3822ab`、
`e81fcb8a43933f041abed68a220cd264cb09fdad7abeeb7a2fec217738979c87`。
