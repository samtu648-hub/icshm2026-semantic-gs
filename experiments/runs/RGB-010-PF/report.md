# RGB-010-PF — Standard 3DGS CUDA 12.8 build preflight

- 状态：completed
- 阶段：R1；父实验：RGB-001；Git commit：working tree uncommitted
- 配置：`experiments/runs/RGB-010-PF/config.yaml`

## 已验证执行

固定上游 `54c035f7834b564019656c3e3fcc3646292f727d` 已在 Git 忽略的 R1 Python 3.12 / PyTorch 2.11.0+cu128 / CUDA 12.8 (`sm_120`) 环境中完成 CUDA 扩展编译。最小补丁包括 CUDA 12.8 头文件兼容和显式 `sparse/0/test.txt` 验证划分读取；没有变更数据、输入分辨率或训练预算。

`train.py --help` 与真实 RGB-010 500-iteration 训练均成功，后者还证明 checkpoint 可加载、从74个显式验证相机渲染、以及固定 PSNR/SSIM 入口可运行。真实 screening 指标和资源证据属于 RGB-010，不重复记在本预检中。

## 结论

Standard 3DGS 通过构建与启动门，可进入授权的完整 RGB-010-FULL / RGB-011-FULL 比较。LPIPS AlexNet 的权重缓存不属于编译通过条件，缺失时任何指标条目仍禁止排名。
