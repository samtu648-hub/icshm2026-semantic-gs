# GEO-020-PF — Depth Anything 3 preflight

状态：`blocked_dependency`，不是模型质量失败。

The deterministic two-window dry-run passed. The real preflight generated its
manifest and exited before model execution because the isolated host does not
contain `depth_anything_3` or a DA3 checkpoint. No download was attempted and
the frozen Scaffold environment was not modified.

2026-08-04 来源/许可证复核：官方代码仓库为 Apache-2.0，且公开说明 DA3 支持已知位姿
条件的多视图深度；但官方 model-zoo 将 `DA3-LARGE` 与 `DA3-LARGE-1.1` checkpoint 列为
CC BY-NC 4.0。主机 GPU 可用性门已恢复，但本机仍没有 DA3 source package 或 checkpoint。
未下载、未安装、未执行模型；在比赛用途是否符合该 checkpoint 条款得到明确确认前，不得绕过
或改用其他权重。`DA3-BASE` 的 Apache-2.0 权重不是本预注册的 DA3-Large，不能静默替换。
