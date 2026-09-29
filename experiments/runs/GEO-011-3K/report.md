# GEO-011-3K — MapAnything P0 additive hybrid-initialization stop-test

状态：`completed_drop`。这是一次独立的、3,000 iteration 止损筛选，不是对
GeometryPrior v2 的 10k 排名或 30k 胜者声明。

代码冻结 commit：`848e5d8b3eea22c75ca8995fdfef7443d2297af7`。其余用户既有未提交
R1/R2 修改保留且未纳入本实验代码提交。

## 唯一变量

完整保留 `GEO-000-10K` 的原始 DLT 初始化点和 RGB 顺序，仅从已审计的
`GEO-010-P0` P0 融合产物补充少量 MapAnything 点。新增点必须同时满足：普通表面、
至少 3 个支持视角、calibrated confidence ≥ 0.75、通过原融合的多视图深度一致性，且
到任一原始初始化点的最近距离至少为 `1.5*r0`。Map 的细线 2-view 例外不适用；最多新增
3,000 点。所有筛选、体素去重、hash 和点数都由 config 与 manifest 记录。

`GEO-000-10K` 的连续 0→3k checkpoint、逐视角 CSV、输入 PLY、seed、split 和固定
评测证据已确认存在，因此它是唯一允许的匹配 3k 控制；历史 `GEO-000` pilot 不参与本比较。

## 已完成的 CPU 预检

数据 manifest 于本阶段再次检查并与仓库 `contest_dataset_v1.json` 一致。已审计成功的
`GEO-010-P0` NPZ、fusion manifest 与 `GEO-000-10K` input PLY hash 均匹配，随后用冻结
代码生成外部 Git 忽略的 hybrid PLY：

- 原始 DLT 点：72,292；全部坐标和 RGB 作为输出 PLY 前缀逐字节保留；
- Map 输入点：100,000；质量门后：9,075；低覆盖和 `r0` 体素去重后新增：1,719；
- hybrid 总点：74,011；`r0=0.05992645844916069`；
- hybrid PLY SHA-256：`5f49481cf693dc63ba4254abddab3167a70371b894e9850a796460d762d732e3`；
- hybrid manifest SHA-256：`b2587c96c6fcce5b6bd0df608d7ae31ab6f6a0c2471b0f1924305183d7b7c814`。

新外部 Scaffold 场景只以只读链接复用原 `scene_400_eval` 的图像和相机文件，并替换自己的
`sparse/0/points3D.ply`；其 PLY hash 与上述 hybrid PLY 一致。该检查证明 PLY 结构、来源
hash、官方坐标和基线不变性；随后实际训练也证明该 PLY 可由冻结 Scaffold loader 加载。

## 实际运行与工程门

早期 driver/library mismatch 在受控重启后已恢复；本实验随后在隔离的冻结 Scaffold CUDA
环境中从零以单一进程连续训练 0→3,000 iteration，并在 1k/3k 保存 checkpoint 和固定的
74-view 评测。运行记录的 wall time 为 270 s（约 0.075 GPU h）；采样到的最高显存占用为
7,430 MiB，不能当作未采样瞬间的严格峰值。

- 1k：PSNR 19.37729、SSIM 0.82450、LPIPS-Alex 0.43417；
- 3k：PSNR 20.21943、SSIM 0.83874、LPIPS-Alex 0.37775；中位数为
  20.56786 / 0.85100 / 0.36513；
- 3k 最差 10%：PSNR 17.39895、SSIM 0.74352、LPIPS-Alex 0.51450；
- 3k checkpoint：78,516,021 bytes，SHA-256
  `b40339644c86c50bb7195ead3c2eaaa98ce87c212fcdabb35c1b1b1fda956d7b`；最终 PLY 有
  86,813 个 Gaussian；
- 固定 74-view native render 的输出 mtime 区间为 41.09 s，即约 0.555 s/view（这是完整
  写盘区间，不是孤立 GPU kernel 基准）；
- checkpoint 已在全新 Scaffold 实例中重载；外部官方 `140.png` `CAMERA_FROM_WORLD` 相机
  在 identity Sim(3)、不读取参考 RGB/标签的情况下输出 1320×989、官方 SIMPLE_RADIAL
  像素网格 RGB。其 PNG 和 metadata 的 SHA-256 见 Git 忽略的运行证据目录。

原始运行、配置、数据 manifest、split、seed、command、训练日志、逐视角 CSV、资源采样和
artifact hashes 均保留在 `ICSHM_OUTPUT_ROOT/geometry_prior/hybrid_screen/runs/GEO-011-3K`。

## 与匹配 3k 控制的比较

唯一允许的控制是 `GEO-000-10K` 的连续 0→3k artifact。74 个同一空间验证视角的 paired
比较结果（candidate − control）为：

| 指标 | 控制 | GEO-011-3K | 变化 |
|---|---:|---:|---:|
| PSNR mean (dB) | 20.20079 | 20.21943 | +0.01864 |
| SSIM mean | 0.83907 | 0.83874 | -0.00032 |
| LPIPS-Alex mean | 0.37339 | 0.37775 | +0.00437 |
| PSNR 最差10% (dB) | 17.56152 | 17.39895 | -0.16258 |
| 困难 140–146 PSNR mean (dB) | 20.75725 | 20.63642 | -0.12083 |
| 困难 140–146 LPIPS-Alex mean | 0.37547 | 0.37733 | +0.00186 |

PSNR mean 的 paired bootstrap 95% percentile interval 为 [-0.06106, +0.10283] dB；
LPIPS-Alex 为 [+0.00318, +0.00546]，说明后者的退化在该固定折上方向一致。困难视角
140–146 中只有 140–142 的 PSNR 小幅提高，143–146 均下降，其中 146 为 -0.65133 dB。

## 预注册决策

只有当相对 `GEO-000-10K` 的 3k checkpoint 达到 PSNR +0.2 dB 或 LPIPS-Alex -0.01，
最差 10% 和困难 140–146 不出现实质退化，并通过 checkpoint reload 与外部官方
`CAMERA_FROM_WORLD` 1320×989 RGB 门，才新建 10k 候选。否则本 MapAnything 混合初始化
路线登记为 drop，不运行其 10k 或 30k。

未达到 PSNR +0.2 dB，也未达到 LPIPS-Alex -0.01，并且最差 10% 和困难视角出现退化。
虽然工程门均通过，预注册规则要求本配方 `drop`：不得为它运行 10k、30k、重复 seed 或
无边界调参。该结论只否定此“保留原始点 + 1,719 高置信 Map 补点”的初始化，不等同于
MapAnything 全部几何价值的结论。
