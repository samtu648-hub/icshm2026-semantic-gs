# RGB-149-STANDARD-GT-REGION-GRADIENT

状态：completed_research_evidence。C 只在匹配控制 A 上增加 GT 结构区域原始尺度与 4 倍粗尺度真实照片梯度一致性，不叠加 B；完整图先求 RGB 前向差分，再按区域加权。

## 配方

- 300 张现有标签使用 `GT != background` 并集加 4px 原生膨胀；100 张无标签视角严格只用原全图 RGB loss。
- 固定 12 视角 `calibration-r2` 系数为 `0.3103426341938125`，加权辅助图像梯度中位数/基础梯度中位数为 `0.2`；artifact SHA256 `ec275093a88f43d6a14876bcc26aba265bae5f5d12394b3ac7af50378b7d992a`。
- 常量图合成反例中正确“先求梯度再加权”为 `0.0`，错误“先遮罩再求梯度”为 `0.0032258066348731518`。废止的 `calibration-r1` 未用于训练。

## 结果

- 全 400 训练视角拟合：PSNR `23.291085001` dB，SSIM `0.896749491`，LPIPS `0.244210317`；相对 A 为 `+0.010789820` dB、`+0.000560419`、`-0.001497136`。
- PSNR/SSIM/LPIPS 分别在 `256/298/334` 张视角优于 A，在 `144/102/66` 张退化。
- 公共结构区域 RGB L1 宏平均 `0.028133433`，相对 A 下降 `3.377%`；多尺度梯度 L1 `0.017084465`，下降 `5.169%`，为三臂最低。
- 类别 RGB MAE（0-255）：background `4.843783`、deck `10.013036`、`stay_cable` `5.216487`、tower `4.642976`、foundation `4.691218`；五类均优于 A，但结构颜色改善通常弱于 B。
- 固定 ROI：159 shore `7.004616 -> 6.505499`、159 far occlusion `5.682813 -> 5.030024`、160 shore `6.513156 -> 6.414343`、174 sky `4.595524 -> 4.424631`；159/160 near 分别退化为 `8.593525/9.783492`。159 water ghost `2.869583`，仍略差于 A。
- 最终 `906,989` 高斯，较 A 增加 `13.931%`；模型 `214.515 MiB`，训练 `2814.540 s`，峰值 allocated `2203.141 MiB`，固定九图 CUDA 渲染中位 `15.624 ms`。

## 验证与判断

- 30,000 次辅助调用中 22,500 次有标签、7,500 次无标签；无标签调用的辅助项为零。参数梯度探针有限，XYZ、DC、scale、rotation、opacity 均有非零辅助梯度。
- config SHA256 `5e3b17893e5ffcfb8efcbe91f3ae7507e10eeefaccbcbe14476937ab3b9cac5e`；run result SHA256 `489e745f154c8feb8ba74e6d855d417f91b1efe38d96e710f7d42841fb8375f4`；evaluation SHA256 `b440a750eb1f5005626f1d6c52da34f2732b47ce6a3ad580c97b4b959711d2a1`。
- 与 A/B 相机序列严格一致；完成 400 图、固定九图/ROI 和 144 无 GT 轨迹评价。comparison SHA256 `236e07a4bca408fa94b5c0797adab5ca36db0d0297fd6d40960611aeef8b8193`。

C 证明区域真实照片梯度监督能降低目标梯度误差，并以低于 B 的容量成本改善全局 SSIM/LPIPS；但结构颜色改善较 B 弱，159/160 近景明显退化，水面幽灵仍未解决。全部结果为单 seed 训练拟合，没有 held-out 全链或匹配语义证据。因此保留 C 为机制研究证据，不替换现有系统候选/默认，不扫描尺度或系数，不自动追加训练。

浏览：`outputs/quality/rgb147/review-r1/index.html`；同步轨迹：`outputs/quality/rgb147/review-r1/trajectory-a-b-c.mp4`。

最终验证与 A 共用同一代码和交付检查：实际 Torch artifact 环境目标测试 4/4 通过；`repo_health` 通过 547 项测试（76 项可选跳过），234 份配置、`py_compile`、`git diff --check`、所有登记哈希及 144 帧视频元数据均通过。
