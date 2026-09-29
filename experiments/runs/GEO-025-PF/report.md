# GEO-025-PF — VGGT-Ω joint co-visibility preflight

状态：`completed_passed_for_full_geometry`。固定来源、许可证、checkpoint、隔离环境和CPU strict-load均通过；
首个25图bf16 smoke通过全部预注册功能、坐标、重复性和12GB资源门。后续全量几何及其坐标门失败由
`GEO-025-P0`单独记录，不改写本预检的通过结论，也不产生RGB质量结论。

已实现的是依赖无关 dry-run：它从官方 camera graph 生成 16 或 25 图的
anchor-plus-neighbor 联合共视集合，并写出 GeometryPrior v2 兼容的 manifest。
dry-run 不导入 VGGT-Ω、不访问网络、不下载 checkpoint、不调用 GPU。

## 2026-09-18 官方来源与访问核验

- 官方源码：`https://github.com/facebookresearch/vggt-omega`；核验时 `main` HEAD 为
  `a3ab0141f96838724423541044ff5ba301cfd36a`。因 checkpoint 访问门阻塞，未 checkout 或安装源码。
- 固定 commit 的 `LICENSE` 为 `FAIR Noncommercial Research License v1`，SHA256 为
  `27696e51930daf5d24222e2d9d65339f529153e853de0e2eb457133a1a6f5239`。许可仅允许
  noncommercial research use，分发需沿用条款，发表研究结果需致谢；本记录不构成法律意见。
- 固定 commit 的官方 README SHA256 为
  `8526d8578d3246c9042115dd22c211b53842639262833d41eaf717f56567264e`，其中明确要求先向
  `https://huggingface.co/facebook/VGGT-Omega` 申请 checkpoint 访问并等待审批。
- Hugging Face 官方 API 返回仓库 revision
  `1041e80fc0e911235d3426b0a3d9a81075111a53`、`gated="manual"`，并列出目标
  `vggt_omega_1b_512.pt`。使用环境中已有凭据对官方 resolve URL 做无下载 HEAD 探测，返回
  HTTP 403；本机也没有该模型的 Hugging Face cache。
- 官方源码 LICENSE 对 `Research Materials` 的定义包含 trained model weights；当时 HF checkpoint
  仓库的 `LICENSE.txt` 同样因访问控制不可读取，模型仓库元数据只标记 `license="other"`，因此该次
  HF 探测无法独立核验 checkpoint 许可证和 SHA256。

该 HF 403 是历史事实，没有改写为 HF 下载成功，也没有提交访问申请或接受新条款。

## 2026-09-18 Meta 官方魔搭分发渠道下载与核验

- 页面：`https://modelscope.cn/models/facebook/VGGT-Omega`；元数据 API：
  `https://modelscope.cn/api/v1/models/facebook/VGGT-Omega`。API 返回组织名 `facebook`、组织显示名
  `AI at Meta`，许可证名为 `fair-noncommercial-research-license`，许可证链接指向官方
  `facebook/VGGT-Omega` 页面。
- 分支固定为 `master`，`git ls-remote` 与 API 文件列表共同确认仓库 revision 为
  `3722a28e35bae3a12b22a78ba3c62d32c6ca883b`。文件列表使用该不可变 revision 查询；四个文件的
  直接下载 URL 均为
  `https://modelscope.cn/api/v1/models/facebook/VGGT-Omega/repo?FilePath=<name>&Revision=3722a28e35bae3a12b22a78ba3c62d32c6ca883b`。
- 本地目录为 `checkpoints/vggt-omega/modelscope-facebook-VGGT-Omega/`，由 `.gitignore` 的
  `checkpoints/` 规则忽略。下载先写 `.part`，成功后原子改名；核验时不存在残留 `.part` 文件。
- `vggt_omega_1b_512.pt`：文件 revision
  `f4fe2040e400351980060e5391e12f00a6c037b2`，4,576,706,117 bytes，SHA256
  `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`。本地字节数和独立复算 SHA256
  均与不可变 revision 的 API 元数据一致；`file` 识别为 ZIP archive，前16字节以 PyTorch ZIP
  容器常见的 `PK\x03\x04` 开头，不是 HTML 或登录页。
- `LICENSE.txt`：11,720 bytes，SHA256
  `27696e51930daf5d24222e2d9d65339f529153e853de0e2eb457133a1a6f5239`，内容为
  `FAIR Noncommercial Research License v1`；它与另行核验的官方源码 LICENSE 哈希一致。
- `README.md`：2,344 bytes，SHA256
  `bc036fd9449837e58ff553fe8ee0a7a309241691c3cc7bd709fd097f4c7582e1`；模型卡声明
  `fair-noncommercial-research-license` 并链接官方 GitHub 项目。
- `configuration.json`：64 bytes，SHA256
  `625c81a3527ce4f7a739854b431f380a987dc4af5e07e099b568e50e2e3b248c`；JSON 解析通过，声明
  PyTorch framework。
- 上述结果只证明下载内容与 Meta 官方魔搭 revision 一致。由于 HF 权重仍不可访问，本记录不声称
  魔搭 checkpoint 与 HF checkpoint 逐字节一致。

下载和校验全程未加载 checkpoint、未导入或安装 VGGT-Ω、未调用 GPU，也未修改训练环境。
以上是下载阶段的历史边界。

## 2026-09-18 执行授权与依赖解除

- `SEM-387-RGB150-FEATURE-SPLATTING-DINOV3` 的 config、report、registry、TASKS、PROJECT_STATUS、
  ADR-129 和 RISKS 已交叉核验：双臂预检及6000步已完成，固定Dev30结果不晋级，Feature Splatting任务关闭。
- 用户明确授权继续固定源码checkout、隔离环境、真实adapter、checkpoint加载、GPU独占下两个区域的
  25→16图smoke；只有全部预注册门通过后，才继续`GEO-025-10K`和精确匹配控制。
- 状态从`blocked_waiting_feature_splatting`更新为`running_source_environment_preparation`。该状态只表示
  可以执行，不表示源码、环境、加载、坐标、数值、12GB资源或质量门已经通过。
- 失败政策不变：来源、加载、坐标、数值或资源任一门失败即保留现场并停止，不临时降分辨率、换权重、
  改图像/窗口/相机/输入尺寸/评价门槛或修改协议。

## 2026-09-18 source、load 与 GPU smoke

- 官方 commit 已以 detached HEAD checkout，工作树干净；source LICENSE/README SHA256 与登记值一致。
- ignored 隔离 venv 使用 Python 3.12.3、Torch 2.11.0+cu128、torchvision 0.26.0+cu128、
  NumPy 1.26.4、OpenCV 4.11.0.86，`pip check` 通过。CPU strict load 匹配1411个key，
  参数量1,143,814,185；结果 SHA256 为
  `3b1520113ccf8d4fbe01529f5587c22ece977d91f2ebd719ecf68541c53d8cf9`。
- 两个区域继承真实 forward 前已存在的`115.png`和`126.png`共视区域。25图集合由 anchor 加
  最高 shared-POINT3D 的24个训练邻居构成；16图 fallback 是25图列表的精确前缀。
- 首次`25/bf16`在共享GPU独占锁内成功，未运行`fp16`或16图 fallback。官方 balanced/512/patch16
  将1320x989无裁剪映射为592x448。两个区域各重复前向2次，depth、confidence、predicted pose
  均逐元素完全一致。
- 共50个NPZ、每个265216像素；depth/confidence/world points全图有限，valid fraction均为1.0，
  每视图稀疏尺度对应数均远高于64，官方P0投影/反投影闭环全部通过。推理10.7451秒，整个模型阶段
  26.8184秒；peak allocated/reserved为6,680,261,632/8,279,556,096 bytes。
- run summary SHA256：
  `918162559fcf34c3fb9e7e58dc21c3240f97a1c4c7f10ece82d9a2914b09f172`；window manifest SHA256：
  `5fbe5cb4bed63a80aa2d46d59aa1862aa7587107d2a936b6757f91a19017edd6`。

Smoke只证明功能、坐标和资源门，不是质量结果。完整303训练图几何、融合、Scaffold初始化与
`GEO-025-10K`仍分别由`GEO-025-P0`和`GEO-025-10K`记录。

## 记录验证

- 下载文件大小、SHA256、checkpoint 文件头、JSON 解析、无 `.part` 残留及 Git ignore：通过；
- `jq empty experiments/runs/GEO-025-PF/config.yaml experiments/registry.yaml`：通过；
- focused VGGT tests：15项通过；`py_compile`和官方单图preprocess shape检查通过；
- `python3 scripts/validate_configs.py`：通过，247份JSON-compatible YAML配置；
- `git diff --check`（本次六个记录文件）：通过；
- `bash scripts/repo_health.sh`：最终通过，599 tests，85 skipped；可选本地数据集不存在，按clean clone规则跳过。
