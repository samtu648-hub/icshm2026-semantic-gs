# RGB-092-TRANSIENT — Protected DINO uncertainty weighting

- 状态：completed_candidate_not_promoted；drop，保留研究产物
- 父实验：RGB-030-FULL
- 配置：本目录 config.yaml；seed 0；split rgb_r1_spatial_v1。
- 2026-09-06 完成唯一匹配控制30k和候选30k、最终74-view审计、固定视角对比及144帧视频。不覆盖冻结 RGB-030-FULL 或 SEM-161。
- Git HEAD：`49cdcdb1c9915a9544d568754af95ea20a85ec9c` 加保留工作树；候选配置SHA256 `a987bb2f65623ebdd7cb2bf7d17c500cb4246122a8bec93351f471d1d2c7e41a`；控制配置SHA256 `cdceece7e98de0d43bea5b45ed1a151cc1e1c1e9bb6f54fa7e40cb66336867c8`。
- 数据：manifest `0a188d4b4cfa5ffc2a7f151018f1cb012782e8f9860a3338aed084478d675b3e`；split `141944ca14288536a73a55586763b3104493112069a536bc0dfcc2d9f0081847`。303训练RGB、219训练标签保护、84无标注全监督；1320×989；seed 0。74-view是已复用开发审计，不是盲测。

## 结论与数值门

这次实现和完整训练均成功，但固定配方没有达到替代标准。仅部分天空条带减轻，不能称为桥面问题已解决。

- 匹配控制RGB-091：PSNR **20.674209** dB，SSIM **0.851700781**，LPIPS Alex **0.319847499**；worst10 PSNR **18.203331**，140–146 **21.219144** dB。
- 候选RGB-092：PSNR **20.324577** dB，SSIM **0.851364495**，LPIPS Alex **0.321356529**；worst10 PSNR **17.234006**，140–146 **20.776462** dB。
- 相对匹配控制：PSNR **−0.349632 dB**，LPIPS **+0.001509**（变差），SSIM **−0.000336**；worst10 **−0.969325 dB**、140–146 **−0.442682 dB**。只有SSIM保留门通过，主收益门和两项困难视角保留门失败。
- 相对历史RGB-030：PSNR −0.271821 dB，SSIM +0.001188、LPIPS −0.001016；worst10 −0.499994 dB、140–146 −0.562262 dB。历史困难视角保留门同样失败，不能用局部SSIM/LPIPS改善绕过主门。

## 视觉与mask审计

六个固定完整视角逐一检查，并查看原生像素分块和054/159/160桥面局部图；没有滤镜、锐化、补洞或后删点。逐图判断与面板哈希见本目录`visual_review.json`。

- 054、140：天空的长横条、竖向痕迹确有减轻；水面并未恢复到参考图状态。
- 151：部分亮线变弱，但暗幽灵线和边缘块状伪影仍在，属于混合变化。
- 159、160：左侧江面的成束拉丝/幽灵结构更明显，近岸几何依然严重拉伸；桥面局部仍有纵向拖影。不能因为其他天空视角改善而接受这两处退化。
- 183：整体接近，水面纹理缺失和岸线模糊仍在，未建立明确主要伪影清除证据。

因此达到“两处局部改善”，但未达到“其他视角无明显恶化”，视觉门失败。未见明确新增塔体/桥面大洞不等于证明几何逐像素或逐点不变。

冻结final不确定性头在219训练图上降权了 **62.4329% 的可降权背景**；没有视角超过95%可降权背景全被降权。训练期间头更新21,320次，head-start后累计标注图像像素降权比例59.5274%；两个比例的分母/时间范围不同。样例002/274显示天空及部分水面被大片降权，而四结构类及8px邻域保持权重1。没有真实瞬态mask，不能把覆盖率当作检测准确率；也不能把失败归结为一个未经隔离验证的唯一原因。

## 工程验证与资源

- 初始点云、相机列表与控制/历史完全同哈希；全1损失/梯度等价、SSIM邻域、结构保护、未标注全监督、梯度隔离、RNG隔离测试通过。
- 训练前8项CPU合同测试通过；训练和视频结束后补充原生1320×989 CUDA等价测试，实际Torch环境共 **9/9通过**。审计流程4项单元测试通过。
- 完成后的`bash scripts/repo_health.sh`通过：99份配置有效，默认环境152项测试中19项依赖相关跳过；训练专用9项已在实际CPU/CUDA环境单独运行并全部通过。上传安全、密钥模式、Markdown链接、shell语法和`git diff --check`通过；未提交或推送Git，也未改动原有不相关工作树内容。
- RGB checkpoint严格重载、不确定性头重载、final mask检查均通过。训练及final mask审计的保护违规均0，开发集mask读取0。
- 控制/候选完整包装耗时分别 **1751.724/1925.804秒**（0.4866/0.5349小时，含保存、重载和评估；不是纯GPU算子计时），候选开销约+9.94%。30秒采样的设备显存峰值分别7716/7726 MiB。
- 最终anchors为195,876/170,453，候选少12.98%；这是独立训练最终数量差，不是已确认删除了25,423个错误结构。
- 144帧、24fps、6秒、3960×1026三联视频生成通过。所有模型使用同一冻结VIS-001轨迹、官方COLMAP参考坐标系、CAMERA_FROM_WORLD、SIMPLE_RADIAL网格及固定appearance index 0；不读取参考RGB或标签。抽查0/48/96/143帧：画面方向和排列一致，96帧候选水面仍有明显暗色幽灵线。未声称逐帧人工审片。
- 原RGB-030保存产物全哈希不变；另逐份验证SEM-161 checkpoint、74个语义PNG及对应旧RGB PNG完全不变。

## 语义结果与回退

RGB数值及视觉门失败，`SEM-180-RGB-INPUT-AUDIT`按预注册条件关闭，生成新预测0、打开新语义评估标签0。未验证新RGB是否单独有桥面收益，也没有新外部语义输出；不能把条件关闭写成桥面改善或桥面假设被否定。

当前保留的 **RGB-030-FULL + SEM-161** 五类历史开发指标如下（IoU / precision / recall；不是本次候选的新评估）：

- background：98.513% / 99.283% / 99.219%。
- deck：91.467% / 95.039% / 96.053%。
- stay_cable（Class ID 2）：70.698% / 84.284% / 81.433%。
- tower：83.487% / 88.382% / 93.779%。
- foundation：72.770% / 78.276% / 91.187%。

mIoU为83.387%。旧anchor sidecar没有挂接到新高斯上，提交配置未切换。SEM-180的条件推理代码已实现，但通过分支未做真实端到端语义评估，后续不能免除该验证。

## 产物与复现依据

路径均相对`ICSHM_OUTPUT_ROOT`，模型、图像、视频和日志不进入Git：

- `r1/runs/RGB-091-CTRL`、`r1/runs/RGB-092-TRANSIENT`：完整训练、native保存、execution/reload、逐视角RGB CSV和汇总。
- `r2/rgb_092/final_audit/full-20260906-r1`：六视角四联图、原生分块、三张桥面局部、数值与Codex视觉检查记录；最终result SHA256 `bc5515c669cd976d8a6bd33875b7ae867f4cb97b333720b098674f954a1e9589`。
- `r2/rgb_092/mask_audit/full-20260906-r1`：219-view训练侧mask；result SHA256 `fc4fd0c9f4599a2fc5b161fad82061c08cec7317d138e804a23a409fad3c6068`。
- `r2/rgb_092/flythrough/full-20260906-r1/rgb030_rgb091_rgb092_same_pose.mp4`：视频SHA256 `660b12b15842e1054b9b8f3d61c6b0d44dcc6677c9f036c63a2bb84caa6d9e1c`。
- `r2/rgb_092/fallback_verification/full-20260906-r1`：旧方案逐份核验；result SHA256 `486380a05cf0568dcd25494177dd742419728be1f59d21e1f3754f6874c6acd3`。
- 候选checkpoint SHA256 `c2dd3d0652dc4a8daab7dc2a0271c0a8a8c070a1089ecb181fea99dd19f34543`；PLY `5dddda4e0d0408c0b6731d7ffecfa3c37ecada90a51d7b6f40c5f40084758082`；不确定性checkpoint `255a7c07191e68cdec709b5c363e2fc7ae9528cfcbdb93254e1b915e9fe8be6d`。
- 上游Scaffold commit `59c833b56bbbf510f3f64d40f81721995caced66`；隔离补丁`scaffold_transient.patch` SHA256 `e9f4f7c37f77bd96787cf2f4c928482540e30493909d777d71649c16688d735b`。两组使用同一训练module SHA（下文），实际启动命令、源码与DINO/219标签哈希见execution.json。控制与候选的runner哈希有工程后处理差异（补加不确定性重载探针），训练模块和patched train.py相同。

本轮到此关闭：保留全部成功/失败记录，不继续扫描权重、阈值、appearance或opacity；这只淘汰当前配方，不否定所有训练时异常建模方法。

## 工程记录

- 8项冻结Torch环境CPU测试已通过：全1损失/梯度等价、结构8px保护、SSIM邻域梯度、sigma阈值、无标注全监督、非法输入、特征梯度隔离及Python/NumPy/Torch随机流隔离。
- 首次隔离补丁因上下文/缩进未正确匹配而未应用；修正后通过check，原始源码diff哈希保持登记值。
- 首次2600步冒烟在2560步附近收到exec会话退出143，无CUDA traceback，未产出final RGB checkpoint；所有日志及transient_2500保留，登记为interrupted而非通过。
- 用户级长任务首个队列因CLI负号后缀解析错误退出2，未启动训练；修正为显式等号后，queue-r2执行同配方smoke retry及唯一控制/候选full队列。以上均为工程修复，未据质量选择超参或额外full seed。

- Smoke retry完成2600步、原生保存与RGB严格重载：79,987 anchors，执行包装耗时234.92秒；checkpoint SHA256 `9751f3cdf1362e0970f5c3bbc3449cef2dc974ea2d5843279800404d75557c99`，PLY SHA256 `723d452c6aeb156390768906cd98f4bdd2bc5a752c13de9f354cd5612501cd76`。原始RGB父产物哈希完全不变。未评估或据smoke质量选参；上游保存后的原生渲染仍生成了74张未评分图，任何训练梯度与mask均仅来自303训练相机。
- 正式控制/候选使用模块SHA256 `9c9a021a04bd36df4a43fcb91849ea2a24094e39d2c588f0069425ba8e906614`，补充保护feature初始化期间Python/NumPy RNG；该工程修复经独立测试，不改变注册损失/权重/时序。
