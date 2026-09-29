# RGB-040-PF — 3DGS-MCMC 400 RGB mechanism preflight

- 状态：completed
- 阶段：R1 extension；父实验：RGB-060（已冻结，不被本预检改写）
- 配置：`experiments/runs/RGB-040-PF/config.yaml`

## 目的与预注册门槛

这是用户授权的限时鲁棒性探针，用于检验 3DGS-MCMC 是否可能改变 RGB-060 的 Scaffold-GS 冻结结论。它不是完整候选比较，也不启动语义主线。

上游在 `iteration > densify_from_iter` 时才执行核心 Gaussian relocation/add；默认 `densify_from_iter=500`，因此 500 iteration 不能真正检验 MCMC。预检固定为 1,000 iteration，并用同数据、同划分、同分辨率、同 seed 的 `RGB-041-PF` Standard 3DGS 作为对照。

只有同时满足以下条件才会建议另行授权完整训练：构建和训练成功、checkpoint 重载、显式 74 视角渲染、外部官方 COLMAP 相机入口；并且相对 `RGB-041-PF` 不退化，且达到 PSNR +0.3 dB、LPIPS -0.01 或有记录的困难视角修复之一。该门槛不是官方评分规则。

## 固定协议

- 数据与划分：`contest_dataset_v1.json`，`rgb_r1_spatial_v1`，303 train / 74 validation，允许全部400张 RGB。
- 相机：官方 `CAMERA_FROM_WORLD`；全帧 `SIMPLE_RADIAL` → `PINHOLE`，1320×989，无裁剪。
- 初始化：同一 prepared-scene 的训练专用 DLT 三角化点云（72,293点），`--init_type sfm`。
- 上游：`https://github.com/ubc-vision/3dgs-mcmc.git` @ `7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13`，Gaussian-Splatting License。
- MCMC 参数：`cap_max=685296`（与 Standard-400 最终 Gaussian 数一致）、`noise_lr=500000`、`scale_reg=0.01`、`opacity_reg=0.01`。

## 已验证结果

MCMC 的隔离 CUDA 12.8 环境成功编译。最初构建发现缺失嵌套 GLM 子模块以及新版 CUDA 对 `<cstdint>` / `<cfloat>` 的显式头文件要求；补齐受固定 commit 约束的 GLM，并应用最小头文件兼容补丁后成功。显式 split 读取也修复为将 `test.txt` 的 `.png` 名称与上游无扩展名 `image_name` 规范化比较；第一次失败日志保留在 Git 忽略输出目录，不被覆盖。

| 指标（74空间隔离验证视角，1000 iter） | 结果 |
|---|---:|
| PSNR | 14.66473 dB |
| SSIM | 0.77793 |
| LPIPS Alex | 0.57130 |
| 最差10% PSNR / SSIM / LPIPS | 10.04963 / 0.68573 / 0.69788 |
| 训练 wall time | 26.96 s |
| checkpoint reload | passed（92,263 Gaussians） |
| 外部官方相机 | passed；`CAMERA_FROM_WORLD`、identity Sim(3)、官方 `SIMPLE_RADIAL` 输出1320×989、不读RGB/标签 |

固定评测文件、训练/渲染日志、第一次失败日志、checkpoint、PLY 和外部相机证据均在 Git 忽略的 `$R1_OUTPUT_ROOT/r1/runs/RGB-040-PF`。该结果只表示 MCMC 的1k机制预检完成，绝不是30k候选排名，也不能与 RGB-060 的 Scaffold 数值直接比较。

## 匹配对照与判定

用户关闭 Handy 后，`RGB-041-PF` 以完全相同数据、split、1,000 iteration、seed、输入分辨率和固定评测入口成功完成。首个 OOM 尝试的输出仍保留在 Git 忽略的 `$R1_OUTPUT_ROOT/r1/runs/RGB-041-PF-attempt1-handy-vram-blocked`。

| 指标（74视角，1000 iter） | 3DGS-MCMC | Standard 3DGS | MCMC − Standard |
|---|---:|---:|---:|
| PSNR | 14.66473 | 18.13317 | -3.46844 dB |
| SSIM | 0.77793 | 0.81465 | -0.03672 |
| LPIPS Alex | 0.57130 | 0.46033 | +0.11098 |
| 140–146 PSNR / SSIM / LPIPS | 11.61698 / 0.74232 / 0.58114 | 19.04362 / 0.80829 / 0.44175 | 全部明显更差 |

MCMC 未满足任何预注册晋级条件，反而在全局、最差10%和140–146困难视角均显著落后于相同预算的 Standard。因此不授权30k MCMC完整训练，不重新打开 RGB-060，也不改变 Scaffold-GS 冻结决定。

## 保留决定

- keep_or_drop：drop
- reason：在真正执行 MCMC relocation/add 的公平1k筛选中，MCMC 三项RGB指标和140–146困难视角均显著低于 Standard-1k；工程通过不抵消质量负证据。
