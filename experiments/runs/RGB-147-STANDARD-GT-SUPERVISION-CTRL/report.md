# RGB-147-STANDARD-GT-SUPERVISION-CTRL

状态：completed_matched_control。A 为本轮从原 72,292 点初始化重新训练的匹配控制，沿用 RGB140 全图 RGB 损失、AbsGS 增密、官方 400 相机、1320x989、seed 0 和 30,000 iteration。

## 结果

- 全 400 训练视角拟合：PSNR `23.280295180` dB，SSIM `0.896189072`，LPIPS `0.245707452`。
- 公共 4px 膨胀结构区域：RGB L1 宏平均 `0.029116654`，原始尺度与 4 倍粗尺度梯度 L1 宏平均 `0.018015776`。
- 300 标签图逐图等权类别 RGB MAE（0-255）：background `4.852316`、deck `10.307871`、`stay_cable` `5.467610`、tower `5.031724`、foundation `5.056701`。这些是照片拟合误差，不是语义 IoU。
- 最终 `796,089` 高斯，模型 `188.285 MiB`，训练 `2618.659 s`，峰值 allocated `1988.181 MiB`，固定九图 CUDA 渲染中位 `15.077 ms`。

## 身份与验证

- Git commit `49cdcdb1c9915a9544d568754af95ea20a85ec9c`；config SHA256 `641dd068168257d9e18d28542da30a34fe7422f5b3da3e917b263e13d0d1c0cc`。
- 初始化 PLY SHA256 `10a88fb018c27530b3bbfa0f7000408693ba75228067f5311e7775e4755e68f2`；最终 run result SHA256 `8e584368d64bd765b60b7b435abdb317b051154e1ad1a74a048a0e49721f7f9b`；evaluation SHA256 `4086e66cbf8f6ea118f426bb8c76aa3ba735789564667f1e32f45f639f46d4ce`。
- 最终绑定 `preflight-r4` SHA256 为 `b51b4b374563001d0ad0769eda18bad498ca068485678ae2a23c375d3586e6d6`。严格逐位 checkpoint 判据的 `preflight-r1/r2` 失败均保留；r4 使用已测 CUDA 复跑包络，不把工程预检当质量门。
- A/B/C 的 30,000 步相机序列严格一致，SHA256 `26366aa1eca45de584f02f7ea51777ff4aca83f4998e45b2616d6b4da79af28f`。A 控制路径辅助调用为 0，原始 loss 对象级 no-op 已在预检证明。
- 完成 400 张训练视角和同一 144 帧官方参考轨迹渲染，固定九图 fresh 重载一致。轨迹无 GT，只供定性浏览。

## 判断

A 是本次新训练的匹配控制，不是历史 RGB140 结果的重命名。所有全局分数均为训练拟合，单 seed 和 CUDA 原子非确定性限制微小差异解释。A 仅用于隔离 B/C 的新增监督，不改变 RGB140+SEM378 系统候选或 RGB130+SEM371 默认。

完整比较：`outputs/quality/rgb147/comparison-r1/result.json`；浏览：`outputs/quality/rgb147/review-r1/index.html`；同步轨迹：`outputs/quality/rgb147/review-r1/trajectory-a-b-c.mp4`。

## 最终验证

实际 Torch artifact 环境 `python -m unittest tests.test_standard_rgb_supervision -v` 为 4/4 通过；轻量系统环境相同四项因无 Torch 按设计跳过。`bash scripts/repo_health.sh` 通过 547 项测试（76 项可选依赖跳过）及全部上传安全、secret、Markdown 链接和 Shell 检查；`python3 scripts/validate_configs.py` 通过 234 份 JSON-compatible YAML；`python3 -m py_compile` 与 `git diff --check` 通过。comparison SHA256 为 `236e07a4bca408fa94b5c0797adab5ca36db0d0297fd6d40960611aeef8b8193`；同步视频 SHA256 为 `fdafcf9471a38c5749ad8b9df93644ab455acbbfcd69572e01f0373c643843b5`，`ffprobe` 确认 3960x1024、12 fps、12秒、144帧。
