# R1 环境预检、执行记录与授权边界

> 状态：授权已于 2026-07-30 获得。以下记录区分已验证的执行事实、正在运行的完整实验和仍然受限的事项；不把未完成指标写成结果。

## 已验证事实

- GPU：一张 NVIDIA GeForce RTX 5070，12,227 MiB；主机 `nvidia-smi`确认驱动为 580.159.03。
- GPU 访问：普通仓库沙箱会遮蔽`/dev/nvidia*`；GPU命令必须在有主机设备访问的终端运行。这与 Docker data-root 迁移至`/home`无关。
- 存储：`/home`尚有约483 GB可用空间。
- 软件：系统 Python 没有`nvcc`或 COLMAP。R1 已在 Git 忽略目录建立隔离 Python 3.12 / PyTorch 2.11.0+cu128 环境及 CUDA 12.8 编译器；不修改其他项目环境。COLMAP 不属于共同场景预处理的必需依赖。
- 数据与协议：`RGB-001`的 manifest、空间 split、PINHOLE 预处理和本地评测入口已经验证。已生成 `scene_300_eval`（219 train / 74 validation）及 `scene_400_eval`（303 train / 74 validation）；`sparse/0/test.txt` 是唯一的显式验证相机列表，初始化点只由训练视角重三角化。
- 随机性：空间 split 的 k-means seed 为 20260730；三个候选上游实际训练 seed 都是其 `safe_state()` 固定的 0。二者已分别记录，所有 R1 screening 和 full runs 使用 training seed=0。

## 已执行的候选预检

- Git 忽略的固定源码根：`$R1_OUTPUT_ROOT/r1/third_party/`。实际 commits：Standard 3DGS `54c035f7834b564019656c3e3fcc3646292f727d`、Mip-Splatting `dda02ab5ecf45d6edb8c540d9bb65c7e451345a9`、Scaffold-GS `59c833b56bbbf510f3f64d40f81721995caced66`。
- 三个项目的 CUDA 扩展都已经以 CUDA 12.8、`sm_120` 编译成功。为兼容当前工具链及冻结的显式 split，应用了最小本地补丁；完整 diff 与 hash 会在 R1 结束时保存到 Git 忽略的输出日志，仓库只保留补丁说明。
- 500-iteration screening：Standard-300（RGB-010）、Standard-400（RGB-011）、Mip-400（RGB-020）及 Scaffold-400 checkpoint-fix retry（RGB-030-R1）均完成训练、checkpoint 加载、74 张显式验证相机渲染与固定 PSNR/SSIM。原始 Scaffold-400（RGB-030）在迭代500的旧 checkpoint 序列化字段错误处失败，作为失败实验保留；最小 capture/restore 修复后重试成功。
- 所有 screening 结果都标为 `completed_partial_metrics`：冻结的 LPIPS AlexNet 权重尚未完整缓存，不能用于候选排序或 RGB-060。该权重只从官方 PyTorch URL 下载到 Git 忽略的 R1 cache，完成后对既有渲染补算，无需重训。

## 已授权操作与仍适用的停止条件

1. 已执行：在 Git 忽略工作目录克隆三个固定 commit 及必要 submodule，并建立可复现的 CUDA 12.8 / PyTorch 2.11.0+cu128 环境。
2. 已执行：为 RTX 5070 的`sm_120`编译 CUDA 扩展；不进行 Docker data-root 或系统驱动改动。
3. 已执行：安装候选和固定本地评测的最小依赖。运行时、编译和补丁证据将随实验报告归档。
4. 进行中：下载冻结的 LPIPS AlexNet 权重至 Git 忽略缓存；完成后记录官方 URL、文件名和 SHA-256。

COLMAP不属于当前共同PINHOLE场景的必需项：R1预处理从发布的`images.txt`重三角化初始化点，不重新运行SfM。只有候选代码或诊断确实需要时才追加这一依赖。

## GPU与磁盘停止条件

四个受控预检（Standard-300、Standard-400、Mip-400、Scaffold-400）已经运行。没有候选因 OOM 而改分辨率或共同输入；Scaffold 的原始 checkpoint 序列化失败已登记并以独立实验 ID 的最小修复重试。完整实验按单 GPU 串行运行，以避免相互干扰。

预检合计GPU时间上限为3小时。只有通过预检的候选才进入各30,000 iteration的完整运行；完整R1四项合计先申请48 GPU小时上限。每项必须实时记录实际时间、峰值显存和产物体积；若累计达到上限、单项OOM、磁盘低于100 GB可用，或需要改变共同协议，立即停止并报告，而不是继续消耗资源。

上游源码、环境、编译缓存、checkpoint、PLY、日志与渲染输出全部放在 Git 忽略目录或仓库外。全量训练已获授权，但任何 OOM、磁盘低于100 GB、或需要变更共同协议时仍须停止并登记，而不是静默调整实验。
