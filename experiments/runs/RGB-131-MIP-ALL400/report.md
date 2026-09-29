# RGB-131-MIP-ALL400

2026-09-15 已授权并注册，准备中，未训练/无新指标。父实验、commit、seed和数据角色见config.yaml。

共同原生1320×989、官方COLMAP全400相机、RGB120原303来源初始化、RGB seed0/30k。保留上游原生设置；Standard opacity_lr0.025、Mip0.05且kernel_size0.1，记录源码差异。M2F严格复用SEM370 Swin-L预训练/768采样/6k/seed20260805，独立初始化，只输入对应渲染。

RGB400和语义300均为训练视角拟合；100无标注无GT指标；九视角与固定统一轨迹仅诊断。不增加专家/融合，不替换旧系统、不删、不上传、不push。预计3–5小时/15–30GiB，资源实测监测，无任意质量门。

来源 https://github.com/autonomousvision/mip-splatting.git，commit dda02ab5ecf45d6edb8c540d9bb65c7e451345a9；Gaussian-Splatting research/evaluation license，许可证哈希见outputs独立provenance；完整已安装源码hash见started.json。

实际批次 outputs/quality/backbone_comparison/mip-smoke-r1：completed_smoke，20.08秒，smoke=True。详见该目录result.json/train.log；无质量结论。

资源干扰：另一任务RGB124在Mip启动后占GPU；发现后对本轮Mip原进程SIGSTOP，待外部GPU核验完成后SIGCONT，同一进程/状态恢复，暂停224.251秒。见mip-full-r1/interruption-r1.json。未重启/减步/调配方。Mip wall与扣暂停wall均非严格独占GPU基准，暂停前存在共享；仅进程自身allocated峰值归属Mip。

实际批次 outputs/quality/backbone_comparison/mip-full-r1：completed_all400，2234.16秒，smoke=False。详见该目录result.json/train.log；无质量结论。

完整实际结果已核验：completed_all400，2234.163s。逐项指标、局部退化、资源和不自动替换的判断见[QUAL007](../QUAL-007-BACKBONE-COMPARISON/report.md)。
