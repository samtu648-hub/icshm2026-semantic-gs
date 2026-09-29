# SUBMIT-030-SEM161-BUNDLE

状态：`completed_local_interface_audit`。最终可搬迁包已构建并审计；待团队的对外提交决定。父实验SUBMIT-020，正式候选不变。

## 本地可复核结果

- 输入严格为COLMAP `cameras.txt`、`images.txt`；`CAMERA_FROM_WORLD`、`SIMPLE_RADIAL`，不接收参考RGB或标签。
- 推理链是冻结 RGB-030-FULL → SEM-001 → SEM-102 → SEM-141 (0.70) → SEM-151 (0.10) → SEM-161 (0.99)，并可输出SEM-001/111/141/151回退。
- SUBMIT-020的18/18项源产物于2026-09-07再次验证为存在且SHA-256一致；新包从独立工作目录重载后，140相机最终RGB与SEM-161 map均逐像素重现冻结外相机输出。
- 审计以两次新进程重放140、一个320×240相机及非空输出目录拒绝测试完成；所有推理均不读取GT。最终审计根：`$ICSHM_OUTPUT_ROOT/submission/sem161-audit-20260907-r4/`。
- 最终可复核审计为2,161文件、两次重载均精确，环境Torch 2.11.0+cu128、Transformers 5.5.3、NumPy 1.26.4、Pillow 10.2.0、OpenCV 4.11.0。bundle manifest SHA256 `1b14407d1f374673ebba1c6e626871abe25f47ff7b4351115bf4177463d99506`；审计结果 `a481460749c2ef5d348058bb7b664ed82edb3def1a065b18e3cd3ded970d9b6f`。

## 外部状态和边界

官方说明要求从测试视角同时产出RGB和官方类别ID语义图；隐藏集按视觉保真和语义mIoU各半计分。官方材料包在2026-09-30前邮件提交，且还包括英文README/代码、可复现数据链接、10分钟视频、PPT和10–15页论文。当前没有获得隐藏相机或上传/命名接口，因此没有官方分数、没有对外上传，也不把本地重放称为隐藏评测。

SegFormer B2保留NVIDIA Source Code License for SegFormer原文，限制非商业研究/评估用途；Scaffold快照亦保留其研究许可。发布前仍须团队确认项目代码许可证、数据分享权限、成员信息与所有材料。

协议见 `docs/experiments/long_task_20260907.md`。
