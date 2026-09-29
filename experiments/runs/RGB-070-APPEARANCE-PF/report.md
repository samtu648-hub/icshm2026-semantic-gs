# RGB-070-APPEARANCE-PF — nearest-training-camera appearance fallback preflight

- 状态：completed_gate_failed
- 阶段：R2 frozen RGB inference diagnostic
- 父实验：`RGB-030-FULL`
- 配置：`experiments/runs/RGB-070-APPEARANCE-PF/config.yaml`

## 假设

冻结 Scaffold-GS 对未见相机使用的 appearance embedding 与目标相机没有物理对应：
上游验证渲染用 test-list ordinal 索引训练专属 embedding，外部提交适配器则固定 index 0。
仅依据官方相机中心选择最近训练相机的 embedding，可能改善整体色调、雾感和对比度，
但不会改变或修复几何拖影。

## 唯一变量与边界

候选仅将 appearance ID 改为303个训练相机中官方坐标中心距离最近者；并列时取较小
training embedding index。RGB-030-FULL 的 anchor、offset、geometry、opacity、covariance/
color/opacity MLP、appearance表本身和相机均冻结。不读取目标RGB或语义标签选择ID，不拟合
参数，也不扫描距离公式。

74-view 已在历史实验中使用，因此这里只能作 developer-only paired evaluation。原始
RGB-030-FULL checkpoint、原渲染和外部 index-0 回退全部保留。

## 预注册保留门

- mean PSNR 至少提高0.10 dB；mean SSIM不得下降；mean LPIPS-Alex至少下降0.005；
- PSNR最差10%与140–146均不得下降超过0.10 dB；
- 冻结RGB六个artifact哈希前后相同；
- 未同时通过则drop，不在本ID修改距离、姿态权重、邻居数或做RGB拟合。

## 结果

已按注册规则完成74-view冻结推理。最近训练相机 appearance 的映射范围为
`0..302`；例：`054→325`（训练embedding index 325对应的 cameras.json 训练块位置）、
`140→137`、`151→045`、`159→325`、`183→185`。这里箭头右侧记录的是训练图像名；
渲染脚本另存了实际0-based appearance index与相机中心距离。

| 指标 | RGB-030-FULL | RGB-070 | 差值（候选−基线） |
|---|---:|---:|---:|
| mean PSNR (dB) | 20.596398 | 20.379328 | -0.217070 |
| mean SSIM | 0.850177 | 0.847921 | -0.002255 |
| mean LPIPS-Alex | 0.322373 | 0.326608 | +0.004236（更差） |
| worst-10% PSNR (dB) | 17.734000 | 17.346695 | -0.387305 |
| 140–146 PSNR (dB) | 21.338724 | 21.060697 | -0.278027 |

逐视角仅25/74的PSNR、34/74的SSIM和28/74的LPIPS改善。预选样例中，151的
PSNR虽提高2.848 dB，但159下降0.520 dB；视觉复核确认空中/水面半透明长条、重影索线
和近岸拉伸的位置没有改变，符合“appearance不改变geometry”的预期。

冻结artifact的渲染前后哈希完全一致。产物：
`ICSHM_OUTPUT_ROOT/r2/rgb_070/nearest_appearance/full-20260903-r1`；
`render_result.json` SHA256为
`e090189a68859c3e50a5b2c3082306a76ce8c3d65dafa7db3c044c631c978926`，
`metrics_summary.json`为
`ffefe48af40e6fd8cdf80948022dbb5fd84a5c9e489f735ff4d736f409e0086b`。

## 保留决定

- keep_or_drop：drop
- reason：mean PSNR/SSIM/LPIPS、最差10%与140–146全部未过预注册门；不在本ID扫描距离、姿态、邻居数或根据目标RGB选embedding。继续保留`RGB-030-FULL`及其原始验证渲染。
