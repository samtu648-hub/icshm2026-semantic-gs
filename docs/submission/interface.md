# 盲测渲染接口契约

> 状态：协议适配设计；精确命令、目录和命名待主办方最终发布。

最终程序必须：

1. 加载checkpoint、配置和必要的Sim(3)；
2. 接收外部相机模型、内参、外参、宽度和高度；
3. 输入位姿按COLMAP `CAMERA_FROM_WORLD`解释；
4. 输出对应视角RGB；
5. 输出同尺寸单通道语义PNG；
6. 语义像素值只能为0、1、2、3、4；
7. 不读取隐藏测试RGB或标签；
8. 对输出路径、文件名和评测细节使用配置，不写死未发布协议。

建议最终命令形式：

```bash
python render_submission.py \
  --checkpoint <path> \
  --config <path> \
  --cameras <official_test_camera_file> \
  --output <output_directory>
```

这只是接口形状，不是主办方已经确认的最终命令。

## R1 RGB 外部相机验证入口

`scripts/render_external_colmap_camera.py` 是已实现的 R1 RGB 接口门，而不是最终提交脚本。它只接收外部 `cameras.txt`、`images.txt` 和指定 pose 名称，严格按 `CAMERA_FROM_WORLD` 解释外参；它不读取该视角的 RGB 或标签。内部以同 K 的全帧 PINHOLE 渲染，默认再映射为外部 `SIMPLE_RADIAL` 像素网格，输出宽高直接来自相机记录。

```bash
python scripts/render_external_colmap_camera.py \
  --candidate standard-3dgs \
  --source-root "$R1_THIRD_PARTY_ROOT/standard-3dgs" \
  --model-path "$R1_OUTPUT_ROOT/r1/runs/RGB-011-FULL" \
  --iteration 30000 \
  --cameras "$ICSHM_DATA_ROOT/camera_parameters/cameras.txt" \
  --images "$ICSHM_DATA_ROOT/camera_parameters/images.txt" \
  --image-name 140.png \
  --output "$R1_OUTPUT_ROOT/r1/interface_gates/RGB-011-FULL-140.png" \
  --metadata "$R1_OUTPUT_ROOT/r1/interface_gates/RGB-011-FULL-140.json"
```

每个候选都必须在完整 checkpoint 上通过该门。Scaffold-GS 的 RGB 颜色分支包含训练相机 appearance embedding；R1 gate 因而显式记录一个确定性、已知训练 embedding index，绝不读取测试 RGB。它能证明相机接口和尺寸契约，但不能替代对未见 appearance 适配的最终提交设计；该风险会影响冻结决定。

自动格式检查应覆盖：

- RGB和语义数量一致；
- 分辨率来自相机输入；
- RGB值域和格式符合最终协议；
- 语义为单通道整数PNG；
- 所有语义值都在0–4；
- 没有缺失或额外文件。

## SEM-001 Scaffold 语义接口（完整 checkpoint 已验证）

`scripts/run_sem001.py` 是冻结 `RGB-030-FULL` 的 R2/S0 训练与接口验证入口，不读取任何测试 RGB 或标签。`scripts/render_sem001_external_camera.py` 是独立的 checkpoint 渲染入口：它接收任意外部 `cameras.txt`/`images.txt` 和指定 pose，而非假定发布数据路径。两者都将官方 `CAMERA_FROM_WORLD` 和 identity Sim(3) 转成同 K 的内部 PINHOLE 渲染，再将 RGB 和五类概率映射回输入 `SIMPLE_RADIAL` 网格。预检在外部 `140.png` 相机上验证了 RGB 1320×989、五个 `[5,989,1320]` log-probability channels 和同尺寸单通道 uint8 0–4 PNG。

```bash
python scripts/render_sem001_external_camera.py \
  --config experiments/runs/SEM-001/config.yaml \
  --checkpoint <semantic_sidecar.pth> \
  --rgb-run-dir "$R1_OUTPUT_ROOT/r1/runs/RGB-030-FULL" \
  --source-root "$R1_THIRD_PARTY_ROOT/scaffold-gs" \
  --cameras <external_cameras.txt> --images <external_images.txt> \
  --image-name <requested_pose.png> --output-dir <output_directory>
```

完整 `SEM-001` checkpoint 已在独立进程中用外部 140 相机复验：它与完整训练 runner 的 RGB、单通道语义 PNG 和 logits 文件逐字节一致，且不读取参考 RGB/标签。该脚本是当前可审查的 checkpoint/外部相机证明，不假定主办方尚未发布的最终命令、批量相机文件格式或命名规则；提交包装器仍须按主办方最终接口重新固定。

## 当前批量回退包

`SUBMIT-010-BUNDLE` 已把冻结 `RGB-030-FULL + SEM-001`、固定 Scaffold-GS 源码/许可证和批量入口封装到 Git 忽略目录。`scripts/render_submission.py` 一次加载模型后可渲染外部 `images.txt` 的全部或指定 pose，输出 `rgb/`、`semantic/`、可选 `logits/` 和逐文件 hash manifest；`scripts/audit_submission_bundle.py` 离线检查模型及输出格式。搬运副本的三个相机真实复验通过，且 `140.png` 三种输出与既有独立复验逐字节一致。

构建、运行和限制见[当前可复现提交回退](reproducible_baseline.md)。该包仍不假定尚未发布的官方命名/目录协议，也不取代 `SUBMIT-002` 的全新 Ubuntu 重建。

## SUBMIT-030：完整 SEM-161 本地提交包

`scripts/package_submission_sem161.py` 在新的Git忽略目录创建完整
`RGB-030-FULL + SEM-161` 包。它先逐项验证 `SUBMIT-020` 的18个冻结源
artifact，再只复制运行时模型、源码、许可证、运行时版本记录及冻结清单；不复制
比赛图像、标签、预测、日志或密钥。批量入口是包内
`code/scripts/render_submission_sem161.py`：同一外部相机可选输出最终
`rgb/`、`semantic/`以及`SEM-001`、`SEM-111`、`SEM-141`、`SEM-151`回退图。

本地审计以两个独立进程重放140相机，并与已冻结的外相机输出逐像素比较RGB与最终
SEM-161；另检查320×240尺寸与非空输出目录拒绝。它证明当前代码链可用，并不读取
GT，也不等于主办方隐藏测试。官方公告现已明确项目2需要从测试相机输出两种图，
但最终文件命名、目录和执行限制仍应配置化处理。
