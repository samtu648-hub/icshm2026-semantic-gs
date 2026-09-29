# GEO-010-PF — MapAnything preflight

状态：blocked_dependency，不是模型质量失败。

确定性的两个窗口 dry-run 已生成 manifest。真实预检在模型执行前退出：主机
没有 mapanything 包或获批 checkpoint。没有下载、没有 GPU 推理，也没有修改冻结
Scaffold 环境。

因此本实验只证明 v1 的窗口/依赖 fail-closed 行为；它不构成 MapAnything
几何、显存、坐标或 Scaffold 初始化结果。真实执行前仍需登记来源、固定 commit、
代码许可证、checkpoint 许可证和 checkpoint hash。
