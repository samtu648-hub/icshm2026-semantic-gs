# SUBMIT-010-BUNDLE — frozen fallback submission bundle

- 状态：completed local packaging and relocation smoke
- 父实验：`RGB-030-FULL`、`RGB-060`、`SEM-001`、`SUBMIT-001`
- 运行仓库 HEAD：`49cdcdb1c9915a9544d568754af95ea20a85ec9c`
- 运行时工作树：保留既有用户修改；打包前 tracked diff SHA-256 为 `29c38629fbd509dcd0e6963a4d9f15de9d018c7d6ae7b8ddb1dd3ba7e6f1f6c9`
- 配置：`experiments/runs/SUBMIT-010-BUNDLE/config.yaml`

## 目的与边界

把当前可复现回退 `RGB-030-FULL Scaffold-GS + SEM-001` 从原训练目录封装为可搬运目录，并从该副本加载模型、批量接收外部 COLMAP 相机、输出 RGB 与五类语义结果。它不是新的训练或精度实验，不读取参考 RGB/标签，也不把主办方尚未发布的最终命名和命令假定为正式协议。

## 实际打包

打包入口为 `scripts/package_submission_baseline.py`，冻结锁为 `configs/submission/sem001_baseline_v1.json`。脚本在复制前验证七个模型文件的预登记 SHA-256、Scaffold 上游 commit、最小 patch hash，并验证该 patch 能从实际源码树反向干净应用。

外部 Git 忽略包包含：

- RGB iteration-30000 checkpoint、PLY、opacity/cov/color MLP 和 appearance embedding；
- SEM-001 sidecar checkpoint；
- 批量渲染与审计代码的轻量快照；
- 已应用最小兼容补丁的 Scaffold-GS 源码快照、子模块源码和许可证；
- SEM-001 配置、来源记录和逐文件内容清单。

包中不包含比赛 RGB、LabelMe JSON、标签 PNG、相机文件、日志、验证预测或密钥。包大小为 296,035,835 bytes，共 2,145 个被清单锁定的文件。`artifact_manifest.json` SHA-256 为 `b6b86a8962d17c87b686e1eb93f6fc2851edb8f4b30644a9d95d809b866b1d94`；`provenance.json` SHA-256 为 `80e524b4023b5abb093abac7472912e223ea4cde47c3be538224bad5292f6200`。仓库内不保存该包本体。

## 搬运后真实渲染

使用包内 `code/scripts/render_submission.py`、包内模型副本和包内 Scaffold 源码快照，在冻结 Python 3.12.3 / PyTorch 2.11.0+cu128 环境中一次加载模型，渲染外部 `001.png`、`054.png`、`140.png` 三个官方相机。主机为 RTX 5070 12,227 MiB；三个视角实际渲染耗时 2.06546 s，PyTorch peak allocated 362 MiB。

全部视角均通过：

- 输入为 `SIMPLE_RADIAL` 和 `CAMERA_FROM_WORLD`，identity Sim(3)；
- 输出尺寸来自相机，均为 1320×989；
- RGB 为三通道 uint8 PNG；
- logits 为 `[5,989,1320]` float32；
- semantic PNG 为单通道 uint8，像素值范围 0–4；
- manifest 明确 `uses_reference_rgb_or_labels=false`。

输出 manifest SHA-256 为 `928d9b6143802f3ebe1fe75e657f12025356781f906aa70555e79d12c3efe412`。包内审计器同时验证 2,145 个包文件、七个冻结模型 artifact 和三个视角的 RGB/semantic/logits hash 与格式，结果为 passed。

`140.png` 相对先前独立 SEM-001 checkpoint 复验逐字节一致：

| 输出 | SHA-256 | 结果 |
|---|---|---|
| RGB PNG | `563e041da15e713f2acc3139f2b922d129b7db7910b18d18b8ac6489e3de10d3` | identical |
| semantic PNG | `e516db5b8eed1dc3eef952e170ecba03cfe95a9359ceb45655f967cc2c240085` | identical |
| five-channel logits | `42631a3ed6381bdb8205f7f982820dc20aa0cd2922b51ca1d34721369f9a03ff` | identical |

## 结论与未完成项

当前冻结回退已从独立包真实加载和批量渲染，证明它不依赖原训练输出目录或训练进程内存。`SUBMIT-010-BUNDLE` 登记为 `keep_as_reproducible_fallback_bundle`。

仍不能把它称为最终官方提交包：主办方尚未冻结批量相机文件、目录命名和最终评测脚本；Scaffold 外部相机仍使用固定训练 appearance embedding index 0；全新 Ubuntu/CUDA 环境从源码重建属于 `SUBMIT-002`，本实验只证明当前冻结环境和搬运副本可运行。
