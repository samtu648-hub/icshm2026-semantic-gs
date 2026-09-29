# RGB-020-PF — Mip-Splatting CUDA 12.8 build preflight

- 状态：completed
- 阶段：R1；父实验：RGB-001；Git commit：working tree uncommitted
- 配置：`experiments/runs/RGB-020-PF/config.yaml`

## 已验证执行

固定上游 `dda02ab5ecf45d6edb8c540d9bb65c7e451345a9` 已在 Git 忽略的 R1 Python 3.12 / PyTorch 2.11.0+cu128 / CUDA 12.8 (`sm_120`) 环境中完成 CUDA 扩展编译。最小补丁包括 CUDA 12.8 头文件兼容和显式 `sparse/0/test.txt` 验证划分读取；没有调整共同输入、相机或分辨率。

`train.py --help` 与真实 RGB-020 500-iteration 重试均成功。首次包装脚本调用包含该版本不支持的 `--disable_viewer`，在训练前以退出码2失败并已登记；移除该无效参数后独立重试完成 checkpoint 加载、74相机渲染及固定 PSNR/SSIM。真实 screening 指标属于 RGB-020。

## 结论

Mip-Splatting 已通过构建与启动门，可进入授权的 RGB-020-FULL。LPIPS 权重未缓存完整时，任何 partial metrics 均不可用于排名。
