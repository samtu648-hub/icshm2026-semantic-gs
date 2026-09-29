# GEO-020-3K-INT-R1 — DA3 P0 + Scaffold engineering gate retry

状态：`completed_drop_quality_gate`，从 iteration 0 连续训练至 3,000，不续训旧目录。

旧的 `GEO-020-3K-INT` 目录为空且保留不动；本 retry 使用相同 DA3 P0 初始化 PLY、Scaffold
commit、split、seed、分辨率和 3k 预算。R1 Python 3.11 的 CUDA 扩展已重编并通过导入检查；
`R1_IMPORT_SHIM=/tmp/icshm_wandb_stub` 仅提供冻结训练脚本在未启用 wandb 时所需的导入，
不安装或修改冻结环境。

训练已完成，随后使用独立 render 进程重新加载 1k/3k checkpoint，并以官方
`CAMERA_FROM_WORLD` 相机在空间隔离验证折渲染 1320×989 RGB；两次评测均完成。

结果（本地协议，非 10k 排名）：

| iteration | PSNR mean | SSIM mean | LPIPS-Alex mean | worst-10 PSNR |
|---:|---:|---:|---:|---:|
| 1000 | 17.2517 | 0.7891 | 0.5520 | 14.4404 |
| 3000 | 18.9824 | 0.8110 | 0.4746 | 16.4954 |

训练耗时约 243 s；checkpoint SHA-256、逐视角结果、外部路径和运行元数据保存在
Git 忽略的输出根目录，并已写入 config/registry。该结果只证明 DA3 P0 PLY 能被
Scaffold 稳定加载、训练、保存、重载和接受官方外部相机；不改变最终盲测只运行
Scaffold 的约束。与 GEO-000-10K 的同预算 3k 控制（PSNR 20.2008、SSIM 0.8391、
LPIPS 0.3734）相比，变化为 PSNR -1.2184 dB、SSIM -0.0280、LPIPS +0.1012，
最差10% PSNR -1.0661 dB，未满足预注册保留门。因此结论为 `drop`，不启动
GEO-020-10K、30k、重复 seed 或无边界调参，回退冻结 Scaffold 原始初始化。
