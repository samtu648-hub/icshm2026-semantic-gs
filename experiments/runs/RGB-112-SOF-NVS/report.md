# RGB-112 SOF-NVS

状态：completed_30k_rgb_and_independent_reload_audit，2026-09-09。RGB工程验收完成，但当前RGB画质明显落后Scaffold；完整语义系统收益未验证。

完整运行`outputs/quality/runs/RGB-112-SOF-NVS-full-sof-serial-20260909-r2`用时2965.2391s；7k/15k/30k保存PLY均有限，74双fresh作者渲染精确、最终原生字段/RGB不变探针通过。74开发PSNR13.33621657、SSIM0.781145398、LPIPS0.520761675；同seed0新Scaffold对照为20.65404981/0.850192733/0.319108838，没有RGB收益，不将工程通过当质量晋级。训练result SHA `c6fec359d5ed6347acf8a31cd4f2243cb9712d4b9f872dda44302e0fd934fb2d`；audit/probe位于同名目录。

新增[RGB116诊断](../RGB-116-SOF-CHECKPOINT-DIAG/report.md)确认作者与新接口RGB在74图最多1量化级差；7k已经存在训练样本/开发差距，15k以后开发进一步下降。没有隔离单项损失因果；两臂仍按计划做同款语义，不提前淘汰。

父实验RGB110/QUAL-001；seed0、303/74、1320×989、共同训练DLT72292点；作者commit b9eb4170c843014f5f96d54924976161bd675469，NVS配置不启用每图外观拟合。basecommit49cdcdb1c9915a9544d568754af95ea20a85ec9c及实际源码/配置/输入哈希由run保存。

- smoke-r1：原生300步、检查点与PLY保存通过，58.668813s；allocated6275.719238MiB/reserved6922MiB。model/input.ply与共同初始化字节一致。
- 真实模型接口探针通过：16通道原生特征合成最大线性误差7.450580597e-7，特征梯度有限非零，RGB张量和固定视角RGB不变。滤波器按作者流程仅由303训练相机元数据重算，不利用目标相机的RGB。
- 外相机检查是已知官方姿态加合成非零畸变/偏心内参，不是几何真值验证。静态语义、B2专家与GAGS蒸馏尚未训练。
- 原`serial-20260909-r1`在PGSR数值门停止，未运行本臂。后来独立SOF队列完成上述30k及验收；两臂都已结束，不再是活动RGB任务。语义训练另外登记SEM330，不改本臂配方/旧证据。

工程失败及Python3.12、SM120、缺trimesh、appearance关闭时checkpoint保存等修复全部保留，见QUAL-001/third_party。没有降分辨率/表示容量。18冻结文件不变。

证据：`outputs/quality/runs/RGB-112-SOF-NVS-smoke-r1/result.json` SHA256 `f3acd767cec3885c676644048e12360db66e763d4297d66bd31aa06ab4757336`；`outputs/quality/probes/SOF-NVS-smoke-r1/result.json` SHA256 `11f82cecc975e42e01996dea82ca1def9df94ab20726ea2a58f0d1d5cc87bb2d`。研究副本独立，旧冻结方案可回退。
