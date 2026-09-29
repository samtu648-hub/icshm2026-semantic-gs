# RGB-145-STANDARD-GEOMETRY-CTRL

完成本轮实际全400/30k新预算控制，不能冒充历史RGB140。父RGB140、seed0、官方相机、原72292点初始化与候选完全匹配；两臂读取同一照片几何缓存，控制只计算关联、实际XYZ替换0次。

Git 49cdcdb1c9915a9544d568754af95ea20a85ec9c加run/code源码快照；manifest contest_dataset_v1；split released_all400_rgb_all300_sem_v1；全部400为训练拟合，0留出。

结果：{"PSNR": 23.25012676885586, "SSIM": 0.8940959862647526, "LPIPS": 0.2512525022402406}；最终710516高斯；训练40.547分钟。

完整对照、局部损伤、资源、审计与保留判断见RGB146报告。旧默认与系统候选保持；本任务结束，无额外语义训练或参数扫描。


## 最终交付核验（2026-09-16）

仓库健康检查通过：543项测试中72项可选跳过，231份配置语法、上传安全、secret扫描、Markdown链接及Shell语法通过；可选本地原始数据目录缺省跳过，实际400训练输入另由配对审计核验。`git diff --check`通过。浏览页同步裁剪、400逐图跳转、144帧切换、12秒视频播放与6张事后放大图均检查通过。

本组输出当前占用27.438GiB，包含两臂全恢复状态、模型、匹配缓存、失败现场与渲染；磁盘可用86.763GiB（收尾快照时点，非额外空间承诺）。最终交付源码及文档快照：`outputs/quality/rgb144/delivery-source-r1/manifest.json`；实际运行版本仍以每臂code和matching/diagnostic快照为准。完整核验入口：`outputs/quality/rgb144/delivery-audit.json`。旧模型哈希保持，未提交或推送。
