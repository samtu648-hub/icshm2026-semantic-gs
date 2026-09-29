# SEM-371-STANDARD-ALL300

2026-09-15 已授权并注册，准备中，未训练/无新指标。父实验、commit、seed和数据角色见config.yaml。

共同原生1320×989、官方COLMAP全400相机、RGB120原303来源初始化、RGB seed0/30k。保留上游原生设置；Standard opacity_lr0.025、Mip0.05且kernel_size0.1，记录源码差异。M2F严格复用SEM370 Swin-L预训练/768采样/6k/seed20260805，独立初始化，只输入对应渲染。

RGB400和语义300均为训练视角拟合；100无标注无GT指标；九视角与固定统一轨迹仅诊断。不增加专家/融合，不替换旧系统、不删、不上传、不push。预计3–5小时/15–30GiB，资源实测监测，无任意质量门。

沿用SEM340/370来源：Mask2Former commit 9b0651c6c1d5b3af2e6da0589b719c514ec0d69a（MIT）；Detectron2 commit a2f4a8771ab77e8411c26b27f24f9489a28a2453（Apache-2.0）；ADE语义预训练权重CC-BY-NC-4.0，SHA256 f0c081e4fb76d3387141341cd799f1a615aa8178d6873157e1d11e0c8773ef38。仅本地研究训练，无对外发布。

实际完成：standard/mask2former/result.json记录6000有效更新，训练1724.967秒、阶段1743.178秒；完整采样序列和解析配方与SEM370相同，RGB哈希保持，峰值allocated8085.789MiB/reserved8560MiB。全300同口径评价待队列完成，不将九图指标当作全300结果。

完整实际结果已核验：completed，1743.178s。逐项指标、局部退化、资源和不自动替换的判断见[QUAL007](../QUAL-007-BACKBONE-COMPARISON/report.md)。
