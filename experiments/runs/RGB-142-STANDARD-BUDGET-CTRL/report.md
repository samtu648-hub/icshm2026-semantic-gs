# RGB-142-STANDARD-BUDGET-CTRL

完整全400/30k新预算控制及配对评价已完成。它与RGB143共享预算、原生配方和支撑统计计算，唯一臂间差别是大高斯分裂排序是否乘连续权重。不能冒充历史RGB140。

父RGB140；HEAD 49cdcdb1c9915a9544d568754af95ea20a85ec9c加封存源码；seed0；manifest data/manifests/contest_dataset_v1.json；split released_all400_rgb_all300_sem_v1。400指标均为训练拟合，144轨迹无GT。

控制结果：{"PSNR": 23.250105546307932, "SSIM": 0.8940896702009621, "LPIPS": 0.2515451814979315}。高斯708574。

完整方法、配对指标、固定区域、资源、视觉收益与损伤、审计和回退见[RGB143报告](../RGB-143-STANDARD-SUPPORT-DENSIFY/report.md)。控制完整训练记录、PLY与所有预检保持在outputs/quality/rgb141/control-r1。
