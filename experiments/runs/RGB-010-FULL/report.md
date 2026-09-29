# RGB-010-FULL — Standard 3DGS 300 RGB R1 full run

- 状态：completed
- 阶段：R1
- 负责人：unassigned
- Git commit：working tree uncommitted
- 父实验：RGB-010
- 配置：`experiments/runs/RGB-010-FULL/config.yaml`

## 假设

在 RGB-010 500-iteration screening 已验证的执行路径上，300-RGB Standard 3DGS 完整训练将给出100张无标注RGB价值对照的一侧证据。

## 与父实验的唯一主要差异

唯一变化是训练迭代从500增加至30,000；数据、官方相机、空间划分、初始化、seed、分辨率、评测实现保持固定。

## 数据与划分

- Dataset manifest：`data/manifests/contest_dataset_v1.json`
- Split ID：`rgb_r1_spatial_v1`（219 train / 74 validation）
- 是否使用100张无标注RGB：否
- 是否使用伪标签：否

## 命令与环境

```bash
scripts/run_r1_rgb_candidate.sh RGB-010-FULL standard-3dgs \
  "$R1_OUTPUT_ROOT/rgb_r1_spatial_v1/scene_300_eval" 30000
```

- GPU：RTX 5070（12,227 MiB）
- Python/PyTorch/CUDA：isolated Python 3.12 / PyTorch 2.11.0+cu128 / CUDA compiler 12.8
- 上游commit：`54c035f7834b564019656c3e3fcc3646292f727d`
- 实际训练 seed：0（上游 `safe_state()` 固定）；空间 split seed：20260730。

## 指标

| 指标 | 结果 |
|---|---:|
| PSNR | 14.41954 dB |
| SSIM | 0.79165 |
| LPIPS Alex | 0.48452 |
| mIoU（含背景） | not_available |
| mIoU（不含背景） | not_available |
| `stay_cable` IoU | not_available |
| 背景→拉索误报 | not_available |

## 逐视角与困难视角

- PSNR 中位数：14.39314 dB；最差10%均值：10.90163 dB（`081,080,082,079,078,077,054,076`）。
- SSIM 中位数：0.80001；最差10%均值：0.69717（`159,160,161,158,157,147,081,080`）。
- 困难视角 140–146：PSNR均值 14.68521 dB、中位数 14.63453 dB；SSIM均值 0.77585、中位数 0.76925；LPIPS均值 0.47796。
- Git 忽略的逐视角CSV：`$R1_OUTPUT_ROOT/r1/runs/RGB-010-FULL/r1_metrics_per_view.csv`；映射和摘要同目录保存。

## 定性可视化

原生74视角渲染和GT仅存于 Git 忽略目录。外部官方 `SIMPLE_RADIAL` 相机 `140.png` 通过提交适配层渲染为 1320×989 RGB；其元数据确认使用 `CAMERA_FROM_WORLD`、identity Sim(3)，且未读取该视角RGB或标签。

## 失败和异常

- 30,000 iteration、checkpoint、PLY和原生74相机渲染均成功；checkpoint 的独立重载验证通过（853,670 Gaussians）。
- 原始包装进程在原生渲染之后以退出码2结束，日志指向当时的`run_r1_rgb_candidate.sh`第75行 shell 语法错误。当前脚本已通过`bash -n`；保留退出码2和日志，不把它隐藏。对已经完成的原生渲染手工运行同一固定重命名与`--skip-lpips`评测，成功产生74视角映射、PSNR和SSIM。无需重训，且该恢复步骤不使用任何测试数据。
- LPIPS 使用冻结的 `lpips==0.1.4` AlexNet 在同一74张保存渲染上补算完成；未重训或重渲染。

## 结论

资源摘要（Git 忽略的`r1_resource_summary.json`）：端到端wall time 1,988.11 s（0.552 GPU h）、host peak RSS 4,546,256 KiB、采样GPU显存峰值11,705 MiB、checkpoint 614,651,606 B、PLY 211,711,691 B、整个输出981,693,876 B。原生74视角渲染约21.1 s，即约0.285 s/视角；外部相机的冷启动接口门总耗时约6.2 s（含模型加载与回映射）。

## 保留决定

- keep_or_drop：undecided
- reason：300-RGB完整对照具备完整指标与接口证据；但 Standard-400 在本机显存不足，300-vs-400 的因果对照尚不能完成。

## 下一步

在至少16GB、推荐24GB显存的机器上按相同协议补跑 RGB-011-FULL；不得将本机 OOM 解释为400张无标注RGB的效果。
