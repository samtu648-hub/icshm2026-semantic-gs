# Ubuntu GPU复现说明

## 当前状态

仓库尚未冻结RGB骨干，因此不能可靠指定Python、PyTorch、CUDA、编译器或自定义CUDA扩展版本。此阶段只提供检查接口，不提供表面完整但无法保证可用的环境文件。

用户已经确认Ubuntu主机上存在比赛数据。服务器部署不需要通过GitHub复制数据，只需要把环境变量指向既有目录并验证其内容。

## 克隆后检查

```bash
cp .env.example .env
bash scripts/bootstrap_ubuntu.sh --check-only
bash scripts/check_gpu.sh
bash scripts/repo_health.sh
```

数据可以放在仓库外并通过环境变量指定：

```bash
export ICSHM_DATA_ROOT=/mnt/datasets/icshm2026/Contest_Dataset
python3 scripts/check_data.py \
  --root "$ICSHM_DATA_ROOT" \
  --write-manifest /tmp/ubuntu_contest_dataset_manifest.json \
  --fingerprint

python3 scripts/compare_manifests.py \
  data/manifests/contest_dataset_v1.json \
  /tmp/ubuntu_contest_dataset_manifest.json
```

只有输出`Dataset manifests match.`才完成`DATA-003`。实际服务器路径不得写入Git。

## R2冻结后必须补充

- Ubuntu版本；
- NVIDIA驱动；
- CUDA toolkit和运行时；
- Python版本；
- PyTorch和torchvision版本；
- 每个3DGS上游仓库URL和commit；
- 编译器和自定义扩展构建命令；
- Conda或Docker复现方式；
- 一次从checkpoint渲染官方已发布相机的命令；
- 已验证GPU型号和峰值显存。

脚本不得自动执行`sudo`、修改驱动或改写系统CUDA。系统级安装由服务器管理员完成。
