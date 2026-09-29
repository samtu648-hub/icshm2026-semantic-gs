# RGB-148-STANDARD-GT-REGION-RGB

状态：completed_research_candidate。B 只在匹配控制 A 上增加 GT 结构区域真实照片 RGB L1；标签只定位附加照片监督区域，不提供目标颜色或轮廓。

## 配方

- 300 张现有标签使用 `GT != background` 并集加 4px 原生膨胀；100 张无标签视角严格只用原全图 RGB loss，不生成伪标签。
- 固定 12 视角 `calibration-r2` 系数为 `0.26004419260021433`，加权辅助图像梯度中位数/基础梯度中位数为 `0.2`；artifact SHA256 `ec275093a88f43d6a14876bcc26aba265bae5f5d12394b3ac7af50378b7d992a`。
- `calibration-r1` 使用中位逐视角比值而非冻结公式要求的中位梯度范数比，已保留但废止，未用于训练。

## 结果

- 全 400 训练视角拟合：PSNR `23.305737475` dB，SSIM `0.896585986`，LPIPS `0.244487219`；相对 A 为 `+0.025442295` dB、`+0.000396914`、`-0.001220233`。
- PSNR/SSIM/LPIPS 分别在 `275/283/291` 张视角优于 A，在 `125/117/109` 张退化。
- 公共结构区域 RGB L1 宏平均 `0.026990828`，相对 A 下降 `7.301%`；多尺度梯度 L1 `0.017260002`，下降 `4.195%`。
- 类别 RGB MAE（0-255）：background `4.856563`、deck `9.695273`、`stay_cable` `5.008529`、tower `4.005266`、foundation `4.331204`。四个结构类均优于 A，background 略差。
- 固定 ROI：159 shore `7.004616 -> 6.591633`、159 far occlusion `5.682813 -> 5.111743`、160 shore `6.513156 -> 6.386037`、160 near `8.861171 -> 8.555357`、174 sky `4.595524 -> 4.092626`；159 near 仅 `7.739203 -> 7.733764`。159 water ghost `2.828997 -> 2.898719`，未解决水面幽灵且略退。
- 最终 `991,322` 高斯，较 A 增加 `24.524%`；模型 `234.460 MiB`，训练 `2909.908 s`，峰值 allocated `2368.134 MiB`，固定九图 CUDA 渲染中位 `17.360 ms`。

## 验证与判断

- 30,000 次辅助调用中 22,500 次有标签、7,500 次无标签，精确匹配 300/100 相机比例；无标签调用的辅助项为零。参数梯度探针有限，XYZ、DC、scale、rotation、opacity 均有非零辅助梯度。
- config SHA256 `36d24842a517fb0371365afc48dd11d70ed5961b9ddf68a13fdc73d9ff11c423`；run result SHA256 `2f3429542e1508cf57855ac15a4a0fcbb1eea0bc608e3d026d71416bd9d660a0`；evaluation SHA256 `b180776786418812865c2fa0db61f35c7e14e552fab28d223b469e6bf9dd8766`。
- 与 A/C 相机序列严格一致；完成 400 图、固定九图/ROI 和 144 无 GT 轨迹评价。comparison SHA256 `236e07a4bca408fa94b5c0797adab5ca36db0d0297fd6d40960611aeef8b8193`。

B 是三臂中更均衡的结构区域颜色拟合候选，但收益伴随最大容量、训练和渲染成本，且固定 ROI 仍有混合结果、水面幽灵未改善。全部结果为单 seed 训练拟合，未训练匹配语义模型，也没有 held-out 全链证据。因此保留 B 为研究候选，不替换 RGB140+SEM378 系统候选或 RGB130+SEM371 默认，不扫描系数或自动追加训练。

浏览：`outputs/quality/rgb147/review-r1/index.html`；同步轨迹：`outputs/quality/rgb147/review-r1/trajectory-a-b-c.mp4`。

最终验证与 A 共用同一代码和交付检查：实际 Torch artifact 环境目标测试 4/4 通过；`repo_health` 通过 547 项测试（76 项可选跳过），234 份配置、`py_compile`、`git diff --check`、所有登记哈希及 144 帧视频元数据均通过。
