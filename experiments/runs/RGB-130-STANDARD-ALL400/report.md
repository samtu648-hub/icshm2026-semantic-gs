# RGB-130-STANDARD-ALL400

2026-09-15 已授权并注册，准备中，未训练/无新指标。父实验、commit、seed和数据角色见config.yaml。

共同原生1320×989、官方COLMAP全400相机、RGB120原303来源初始化、RGB seed0/30k。保留上游原生设置；Standard opacity_lr0.025、Mip0.05且kernel_size0.1，记录源码差异。M2F严格复用SEM370 Swin-L预训练/768采样/6k/seed20260805，独立初始化，只输入对应渲染。

RGB400和语义300均为训练视角拟合；100无标注无GT指标；九视角与固定统一轨迹仅诊断。不增加专家/融合，不替换旧系统、不删、不上传、不push。预计3–5小时/15–30GiB，资源实测监测，无任意质量门。

实际批次 outputs/quality/backbone_comparison/standard-smoke-r1：failed_preserved，10.03秒，smoke=True。详见该目录result.json/train.log；无质量结论。

实际批次 outputs/quality/backbone_comparison/standard-smoke-r2：completed_smoke，20.08秒，smoke=True。详见该目录result.json/train.log；无质量结论。

来源 https://github.com/graphdeco-inria/gaussian-splatting.git，commit 54c035f7834b564019656c3e3fcc3646292f727d；Gaussian-Splatting research/evaluation license，许可证哈希见outputs独立provenance；完整已安装源码hash见started.json。

实际批次 outputs/quality/backbone_comparison/standard-full-r1：completed_all400，1913.57秒，smoke=False。详见该目录result.json/train.log；无质量结论。

完整实际结果已核验：completed_all400，1913.568s。逐项指标、局部退化、资源和不自动替换的判断见[QUAL007](../QUAL-007-BACKBONE-COMPARISON/report.md)。
