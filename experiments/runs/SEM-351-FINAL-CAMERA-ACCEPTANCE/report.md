# SEM351：最终模型固定外相机与轨迹验收

2026-09-13完成：30相机最终模型推理、全30渲染及专家独立新进程重载、CPU独立重构与hard/raw-soft审计全部通过。没有新训练、标签评分、阈值选择或正式晋级。本实验关闭最终模型固定相机工程缺项，不产生新的mIoU或背景修复结论。

## 事前协议与来源

- 实验 `SEM-351-FINAL-CAMERA-ACCEPTANCE`；Git `49cdcdb1c9915a9544d568754af95ea20a85ec9c` 加绑定未提交源码；seed `20260805`；父SEM349/350/344/348。
- [配置](config.yaml) SHA256 `53dae6aa2e2a625e94936864e6d04132cedf8d9ac0f0d6efd129ee7e0ad90350`，登记时间 `2026-09-13T06:22:44.141607+00:00`；[原准备记录](preregistration_notes.md)保留首次临时脚本路径错误及启动状态。
- 数据manifest `data/manifests/contest_dataset_v1.json`，沿用SEM001角色划分；三个语义分支都是全219最终训练角色，RGB110共享303。仅读取相机元数据、冻结渲染和模型，不读取目标GT、真实照片，不选择checkpoint。
- 054为已有开发输出闭环；057/146为已见退化、158为斜视组起点，明确属于历史诊断，不是新盲样本。轨迹沿用SEM344：frame0000/0071/0143及frame0060–0083，去重26个，加4已知相机共30。
- 唯一主要变量是最终模型的外相机工程验收；原生PINHOLE下raw、SEM343封存0.05、SEM341校准融合0.35/关闭索二分类/0.1/0.98均不改。SEM347保持off，GAGS不混入。
- 官方COLMAP参考坐标系，1320×989 SIMPLE_RADIAL，先变换至模型PINHOLE推理，再按既有最近邻映射class-ID；RGB仍双线性映射。raw专家soft/log只作CPU契约诊断，既有hard不变，不输出伪造的完整融合概率。

## 实际工程验收

- CPU启动前验证全219角色、三个checkpoint与结果绑定、30相机、原293RGB及18冻结成果。最终SEM349原fit全部166项、静态分支149项、桥面专家149项产物在结束后再次逐项哈希验证通过。
- 30个相机完成RGB/最终静态语义渲染；4已知相机R/t/K与原生存档一致，RGB/static及五类/桥面概率和原保存结果逐位一致。26轨迹相机的float RGB摘要、原生uint8 RGB和官方RGB均与SEM344旧缓存完全相等。
- 每个相机在语义渲染前后复查RGB完全相等，模型RGB签名不变；官方/模型独立投影验证通过。语义更换未改变冻结重建。
- 新进程完整重载30次渲染、30次专家推理，逐张比较全部保存数组（RGB、static、five、deck、raw/selected/fused及官方hard）完全一致。fresh只保存核验记录，不复制大概率缓存。
- CPU另写表达式构造融合/0.05及独立整数坐标gather，30组均精确；所有原生概率为有限非负float32并归一化，raw→0.05只允许tower→foundation，背景/deck/stay_cable在原生及官方网格均被保护。
- 四阶段均重新验证原293RGB及18冻结成果；代码/配置/元数据哈希不变。单GPU串行，无优化器、训练入口、标签解码或自动恢复。

## 描述性结果及边界

- `all30`：0.05原生/官方各修改18/18像素；raw hard与soft argmax分歧14019；hard background→soft stay_cable 1661，反向1943；独立四邻点插值最大绝对误差0.0。
- `trajectory24`：0.05原生/官方各修改14/14像素；raw hard与soft argmax分歧11814；hard background→soft stay_cable 1238，反向1495；独立四邻点插值最大绝对误差0.0。

固定24帧仅14个tower→foundation提议，9帧非零、单帧最多3像素；不能将这种微量改动解释为黄色背景修复。全30含4已知相机的共18提议同样不能称修对18像素，因为本轮未读GT。

raw-soft插值与既有hard有14,019处不同，不构成优劣结论，也不自动更换hard。raw概率并非融合置信度；这次补充的是最终模型的原生规格raw-soft工程证据，其它分辨率/相机模型或强制hard=soft argmax仍需独立协议。

三张预定轨迹概览frame0060/0071/0083已目视核对：raw与0.05在概览尺度几乎一致；完整融合在近端桥面区域有更多孔洞/离散蓝色预测，索区和塔轮廓也存在差异。没有GT，不能把所有离散黄区计为误报，或把外观连续当作准确率提高。相机运动/遮挡会改变类别面积，本实验没有运动对齐的闪烁指标。

只有26既有轨迹相机和固定24连续帧，不是更长轨迹、所有困难视角、全新空间盲测、多seed或全链空间隔离。SEM350原开发集数值保持不变；本轮不新增质量得分。

## 时间、资源与可视材料

整队列（含四独立GPU进程及CPU审计）176.617秒，约2.944分钟。阶段耗时如下，包含自身哈希与模型加载等开销，不是纯模型单帧延迟：

- `render`：22.708秒，30相机，CUDA峰值allocated 396.708MiB。
- `infer`：62.302秒，30相机，CUDA峰值allocated 1937.705MiB。
- `fresh-render`：20.109秒，30相机，CUDA峰值allocated 396.708MiB。
- `fresh-infer`：47.917秒，30相机，CUDA峰值allocated 1937.705MiB。

整批加图/视频约0.3541GiB，低于2GiB上限；当前磁盘余量103.946GiB，高于100GiB保留线。已有模型、概率和失败记录未删除。

- 固定三帧对照图（来源记录：`../../../outputs/quality/external_camera/SEM-351-visuals-r1/trajectory_overview.png`；未随公开包提供）。
- 类别面积及0.05提议计数（来源记录：`../../../outputs/quality/external_camera/SEM-351-visuals-r1/trajectory_counts.png`；未随公开包提供），只作描述。
- 原速24帧短片（来源记录：`../../../outputs/quality/external_camera/SEM-351-visuals-r1/trajectory_native_rate.mp4`；未随公开包提供）与四倍慢放（来源记录：`../../../outputs/quality/external_camera/SEM-351-visuals-r1/trajectory_slow4x.mp4`；未随公开包提供）；两者均24帧，ffprobe确认，视频半分辨率只用于展示，推理未缩图。

## 封存结果

- `outputs/quality/external_camera/SEM-351-r1/render/result.json` SHA256 `9f98e8fa0e8d6ecbca40670f587734cc705c79b580dd2dc7424141049f1680fa`。
- `outputs/quality/external_camera/SEM-351-r1/infer/result.json` SHA256 `93478239d10f48d163e695a95054b47e7148eb4e5aa3d0f7534392e8716f4c8b`。
- `outputs/quality/external_camera/SEM-351-r1/fresh-render/result.json` SHA256 `b02e88e0afe2db8fa5e8b6cb2430a9820a88f50bd794733335d14283a4c3cbe3`。
- `outputs/quality/external_camera/SEM-351-r1/fresh-infer/result.json` SHA256 `71d247e1ddca448a53d9885a409032b68e1c00a3dfaae7cabb9a29df4fa1e449`。
- `outputs/quality/external_camera/SEM-351-r1/audit/result.json` SHA256 `fb581847153244af8196e8b9c17346bc965a13b78f5c3b44f6d71271459a4a43`。
- `outputs/quality/external_camera/SEM-351-r1/queue/result.json` SHA256 `b8da739b98749e508438bde96783f4cd80c10e6695c14f233590a50181d36387`。
- `outputs/quality/external_camera/SEM-351-visuals-r1/result.json` SHA256 `e874b131eadeed89ca21f05b8239afc0dd7cdd85fe369729deafd8e3e632f2d0`。
- `outputs/quality/external_camera/SEM-351-preflight.json` SHA256 `2850aadaf92149c0a162aa0d6099b4f3f1b7edf6bdc960bf03aeb0d892c5e9ae`。

## 决定与后续

保留当前raw主研究对照、0.05和校准融合，正式RGB030+SEM161与提交包不动。不能因为本次工程复现通过就接受SEM350中的召回代价或自动晋级完整融合。下一优先复用SEM350已存逐视角/类别转换证据，把退化与背景黄区的错误类型分清，限定诊断而不选参；若提出新背景机制，先预登记训练侧选择/校准和独立验收，不能把旧111/74换名当盲测。新增训练、多seed/全链空间重建或正式质量取舍先提交具体预算和证据确认。

回退方式：研究侧直接读未改原始概率/PNG，正式侧继续原冻结包；停止引用SEM351新增目录即可，无需重训或恢复旧文件。全部产物保留。

2026-09-13收尾：仓库健康通过478测试（69可选依赖跳过）、176配置，上传安全/秘密扫描/链接/语法及git diff --check通过。已有文件无缺失，只有8份状态/协议与登记记录更新，绑定源码与输入哈希仍精确。宿主GPU为826/12227MiB、4%，计算进程表为空、未发现所查项目作业；余量103.944GiB。全部SEM351批次结束，没有训练或推理排队。 收尾证据：`outputs/quality/external_camera/SEM-351-closeout.json`。
