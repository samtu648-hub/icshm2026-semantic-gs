# RGB-167-SCAFFOLD-GROWTH20K-ALL400

Status: completed; retain as a modest full-data bridge-fit research candidate, without replacing RGB140+SEM378 or claiming unknown-view generalization.

2026-09-22 用户授权一次 Scaffold-GS anchor growing/pruning 截止步数实验。父实验为 `RGB-136-SCAFFOLD-NOAPPEARANCE-ALL400`；唯一主要变量是 `update_until: 15000 -> 20000`。正式运行必须从相同 72,292 点、SHA256 `10a88fb018c27530b3bbfa0f7000408693ba75228067f5311e7775e4755e68f2` 和 seed 0 从零连续训练到 30k，不从父 checkpoint 续训。

其余配方逐项冻结：全400 RGB、官方 COLMAP 相机、`appearance_dim=0`、无逐相机 appearance UID、`update_from=1500`、`update_interval=100`、`success_threshold=0.8`、`densify_grad_threshold=0.0002`、`min_opacity=0.005`、30k 总步数及全部 30k LR schedule、`voxel_size=0.001`、10 offsets、MLP、损失和原始分辨率。保存 15k/20k/30k 模型及可恢复 checkpoint。

工程预检仅验证训练、保存、重载、UID 无关性和显存，不读取其质量选择参数。正式评价对三个端点使用同一流程：全400训练照片 PSNR/SSIM/LPIPS、300标签图四个桥梁类的 RGB 区域指标、159--160/174/235--245/385--392 固定困难视角、同一144帧外部相机轨迹，以及 anchor 数、累计增长/裁剪、模型大小、训练时间、峰值显存和渲染速度。30k 与 RGB136 30k 严格并列；全400只表述为训练视角拟合。

选择以桥梁结构连续性和困难区域为主，重点检查 `stay_cable`、远塔、桥面边缘、基础、岸线、水面重影、天空伪影、轨迹闪烁/拉丝/漂浮物。若只增加 anchor 和训练拟合，或困难区域/轨迹变差，则保留负结果并继续采用15k截止。不得自动运行25k/50k、其他阈值、其他seed、Scaffold 50k或匹配语义训练。

启动前证据：commit `49cdcdb1c9915a9544d568754af95ea20a85ec9c`；manifest `data/manifests/contest_dataset_v1.json`；split `released_all400_rgb_all300_sem_v1` / `outputs/quality/all_data/RGB-120-r2/split.json`。RGB166训练与评价完成后，`outputs/quality/gpu.lock` 已确认释放，主机无其它 GPU 计算进程。RGB136、RGB166及既有冻结模型只读。

`smoke-r1` 失败保留：32步从零训练、模型/PLY重载、checkpoint有限值和三相机×四UID精确一致均通过（70,530 anchors，checkpoint 61,890,339 bytes），但上游续训入口在 PyTorch 2.6+ 默认以 `weights_only=True` 读取其自生成、含 NumPy scalar 的checkpoint，抛出 `UnpicklingError`，未执行第33步。旧RGB/语义哈希未变，GPU锁已释放。后续只在可信的本实验自生成checkpoint工程重载分支显式传递 `weights_only=False`；正式30k仍从零且不启用该兼容分支。新目录 `smoke-r2` 重跑，不覆盖失败现场。

`smoke-r2` 完成全部工程门：32步从零训练、模型和checkpoint保存、新进程模型/PLY重载、checkpoint有限值、三相机×四UID逐像素精确一致，以及可信自生成checkpoint从32步恢复到33步均通过；32/33步均为70,530 anchors。训练峰值 allocated/reserved 为665.895/946.000 MiB。RGB136、RGB166、冻结语义18项及Scaffold源码前后哈希一致。预检只作工程验收，未使用分数选参；正式轨迹仍须从零启动且命令不得含`start_checkpoint`。

## 正式训练与完整性

`full-r1` 从相同72,292点初始化、seed 0、全400官方COLMAP相机连续训练到30k；命令没有`start_checkpoint`。训练日志逐项确认`appearance_dim=0`、`update_from=1500`、`update_interval=100`、`update_until=20000`、`success_threshold=0.8`、`densify_grad_threshold=0.0002`、`min_opacity=0.005`、`voxel_size=0.001`、10 offsets，以及全部30k学习率与`lr_max_steps`。初始化SHA256仍为`10a88fb018c27530b3bbfa0f7000408693ba75228067f5311e7775e4755e68f2`。

15k/20k/30k三份模型与可恢复checkpoint均保存并由独立进程重载。anchor数分别为187,044、201,736、201,736；20k后拓扑冻结。累计184次调整事件共增长238,041、裁剪106,835个anchor，满足70,530 + 238,041 - 106,835 = 201,736；15k时累计增长199,864、裁剪83,350。三个端点均通过有限值、PLY/模型重载、`appearance_dim`缺席，以及159/160/174三相机×UID 0/158/399/999逐像素精确一致检查。

正式训练总墙钟1792.125秒；原生模型保存时刻为15k 887.742秒、20k 1188.644秒、30k 1778.162秒。峰值allocated/reserved为996.834/4464 MiB。正式结果SHA256为`a6cf5508115081adae16ebb17ab999d84c6b7291bc67f5586fbec06fe372ed32`。

## 同口径评价结果

全400仅代表参与训练照片的拟合，不是未知视角泛化：

| 模型端点 | PSNR dB | SSIM | LPIPS |
|---|---:|---:|---:|
| RGB136 30k，增长截止15k | 23.203860 | 0.892380 | 0.250840 |
| RGB167 15k | 23.074501 | 0.886342 | 0.261531 |
| RGB167 20k | 23.128344 | 0.889970 | 0.255244 |
| RGB167 30k，增长截止20k | 23.211430 | 0.893567 | 0.249458 |

RGB167-30k相对RGB136-30k为PSNR +0.007570 dB、SSIM +0.001187、LPIPS -0.001382；全图收益很小，不能单独作为选择依据。

300张标签照片的固定四类RGB区域结果如下（PSNR越高越好；索额外列出对比保留和可见支持）：

| 类别 | RGB136 30k PSNR | RGB167 30k PSNR | 差值 | 结构证据 |
|---|---:|---:|---:|---|
| deck | 22.704067 | 22.761017 | +0.056949 | MAE 10.6316→10.5276；边界梯度误差0.023984→0.023797 |
| stay_cable | 29.916095 | 29.919773 | +0.003678 | 对比保留0.737080→0.746793；可见支持0.638836→0.647050；边界梯度误差0.024216→0.023744 |
| tower | 31.288129 | 31.058189 | -0.229940 | MAE 5.3635→5.6106，区域拟合有损；边界梯度比0.867497→0.867630近似持平 |
| foundation | 30.564873 | 30.812193 | +0.247320 | MAE 5.8377→5.6693；边界梯度误差0.020095→0.019637 |

固定困难组的RGB167-30k相对RGB136-30k：159--160为PSNR +0.165116 dB、SSIM +0.005083、LPIPS -0.003863；174为-0.011359 dB、-0.000453、+0.000689，近似持平略退；235--245为+0.112599 dB、+0.002608、-0.004248；385--392为+0.076915 dB、+0.005858、-0.013028。385--392桥梁手工裁剪为24.390872/0.859477→24.495455/0.863270。159岸ROI改善0.759282 dB，但159近景ROI下降0.583367 dB；160近景提高0.914557 dB，160岸ROI下降0.093499 dB。174天空ROI PSNR提高0.029223 dB但MAE略升，未见新增大块天空伪影。

困难视角并排检查显示RGB167-30k在桥面边缘、主缆/吊索和部分基础轮廓略清晰，远塔没有一致的量化改善；水面与岸线仍有明显平滑、横向拉丝和重影，延长增长没有解决这些既有缺陷。固定144帧外部轨迹的接触图及71--73连续帧未见新增大块漂浮物或跳变；无GT时序描述量的二阶差分均值由0.033709升至0.034253，p95由0.069972升至0.070876，属于小幅上升，且受相机运动混杂，不能解释为精确闪烁分数或未知视角准确率。

完整评价结果SHA256为`375f854b8f5e4954a562a04c8deff653de0502e511a8e96b308c01e1768e41f6`；统一轨迹视频SHA256为`6a94b7ee57f413b6f6e65c9cc30ef0ce23ff27bc43272634cf07362c3720986c`。

## Anchor与成本

| 端点 | anchor | checkpoint | 保存模型 | 原生耗时 | 峰值allocated/reserved | 中位渲染时间 / FPS |
|---|---:|---:|---:|---:|---:|---:|
| RGB167 15k | 187,044 | 156.361 MiB | 56.313 MiB | 887.742 s | 959.318 / 2432 MiB | 7.956 ms / 125.70 |
| RGB167 20k | 201,736 | 168.636 MiB | 60.851 MiB | 1188.644 s | 996.834 / 2760 MiB | 8.236 ms / 121.41 |
| RGB167 30k | 201,736 | 168.636 MiB | 60.851 MiB | 1778.162 s | 996.834 / 4464 MiB | 8.686 ms / 115.13 |
| RGB136 30k | 189,887 | 158.736 MiB | 57.280 MiB | 1731.838 s | 937.488 / 3522 MiB | 8.137 ms / 122.90 |

相对RGB136，RGB167-30k增加11,849 anchors（+6.240%）、保存模型+3.571 MiB（+6.235%）、checkpoint +9.899 MiB（+6.236%）、可比原生训练时间+46.323秒（+2.675%）、峰值allocated +59.346 MiB（+6.330%）、reserved +942 MiB（+26.746%）；中位渲染耗时+6.751%，FPS下降6.324%。RGB136历史运行没有生命周期仪器，无法诚实重建其累计增长/裁剪总数；这里只比较其最终anchor。

## 结论与选择

将增长/裁剪截止从15k延长到20k，在30k终点带来小而多项同向的桥梁训练视角收益：deck、foundation、困难组和`stay_cable`可见支持改善，轨迹未见明显新增大范围不稳定。代价是约6.24%更多anchor/模型容量和约6.75%更慢的中位渲染，且tower区域PSNR下降0.229940 dB，水面/岸线拉丝没有解决。按预注册规则，RGB167可保留为“全数据桥梁拟合研究候选”，但证据不够强到替换当前RGB140+SEM378，也不能宣称未知视角泛化改善。

不建议现在直接运行Scaffold 50k：本实验的增量已经很小，主要残留缺陷更像几何支撑/水面表示问题；先做独立空间留出或桥梁支撑诊断，比继续增加训练步数更有信息价值。未运行25k、50k、其他阈值、其他seed或任何匹配语义训练。

评价后再次逐项核对26个受保护条目：RGB136全部已登记模型、RGB166五个端点及评价、冻结语义manifest均与训练前哈希精确一致；Scaffold源码manifest也一致。全部失败预检、中间模型、checkpoints、渲染、视频和评价证据均保留。

最终仓库验收：`python3 scripts/validate_configs.py`通过328份配置；RGB167五个脚本`py_compile`通过；`PYTHONPATH=. pytest -q tests/test_quality_scaffold.py tests/test_quality_backbone.py`为9 passed；`git diff --check`通过；`bash scripts/repo_health.sh`为696 passed、97个可选依赖测试跳过，全部七阶段通过。首次未设置`PYTHONPATH`的定向pytest在collection阶段因找不到仓库模块失败，随后只修正测试入口环境并通过，未触发训练、评价或产物修改。
