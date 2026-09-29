# SEM-340：Mask2Former-Swin-L 原尺寸工程预检

2026-09-11 01:00后完成。状态：`completed_native_preflight_not_quality_validation`。

父实验为SEM336、SEM339和QUAL001。以RGB110冻结Scaffold的PINHOLE渲染为输入，仅fold0训练侧固定样本；禁止用OOF目标、缓冲或74开发标签。配置、seed、代码版本及数据哈希见同目录config.yaml。原生768裁块/512滑窗不缩小，二分类分支和旧正式包不变。

本机RTX5070已完成完整Swin-L的两次有效优化更新、原生1320×989滑窗预测、新进程逐值重放与独立CPU验收。没有缩小768裁块、减少模型容量、启用激活检查点、读取OOF目标/74开发标签或新增质量成绩。所有本轮计算已结束，尚未启动完整专家训练。

## 已完成和证据边界

- 官方完整ADE150语义预训练权重866,036,678字节；MD5 `6b4a3a5d988b780a40cf8256cca9ffa2`与作者文件名摘要一致，SHA256 `f0c081e4fb76d3387141341cd799f1a615aa8178d6873157e1d11e0c8773ef38`。不是backbone-only或panoptic权重。
- 模型215,451,514参数全部可训练。650份官方state张量中647份逐项精确加载；仅重新初始化5+no-object分类权重/偏置及类数相关的非学习loss权重缓冲。作者旧`static_query`原值按v2规则改名为`query_feat`，不是丢掉query后随机补齐。
- 原生CUDA注意力相对作者PyTorch参考前向最大误差1.1921e-7，三个输入梯度最大误差分别5.9605e-8/4.7684e-7/2.3842e-7，均过事前固定容差。它是算子一致性，不是真实桥梁精度。
- 仅解码fold0训练图`001.png`的两个随机原生768裁块，seed20260805。其在其它研究中可能是校准目标，不声称新盲数据；本次明确是fold0训练角色。独立重放不解码标签，OOF/guard/74目标标签零访问。
- 实际训练（来源记录：`../../../outputs/quality/mask2former/SEM-340/probe-eager-r5/result.json`；未随公开包提供）两次有效更新；第二步一次scaled-gradient溢出，标准GradScaler跳过无效更新、1024→512后对同裁块/同CPU及CUDA RNG重新执行，通过有限性检查后才更新。该工程预检最多12次回退，不无限重试。没有改变loss、lr、模型、精度或分辨率；完整训练稳定性仍未验证。
- 独立重放（来源记录：`../../../outputs/quality/mask2former/SEM-340/fresh-eager-r1/result.json`；未随公开包提供）与CPU验收（来源记录：`../../../outputs/quality/mask2former/SEM-340/audit-r1/result.json`；未随公开包提供）通过：6,527,400个概率值完全相同，最大差0；float32五类有限、非负且逐像素和为1。作者query合成分数明确按五类归一化，不把no-object当背景，也不把mask分数当softmax logits。此是新的专家概率协议，后续须独立OOF校准。
- 219份已冻结Scaffold输入渲染及18正式冻结产物哈希不变。模型绑定RGB检查点/拓扑身份、像素空间、源码、配置、官方权重和来源版本；加载/重放发生错配时拒绝。此次未新跑RGB渲染、官方外相机畸变映射或连续轨迹，不能把旧接口验收冒充新完整链验收。

## 实测资源

- 单卡、batch1、AMP fp16、作者像素解码器原有FP32路径，原生768/512，不使用CPU推理回退。
- 训练峰值allocated **8084.52MiB（7.895GiB）**，reserved **8564MiB（8.363GiB）**；这是Torch统计，不包含桌面等其它进程，不是整卡全部占用。
- 两次有效更新耗时0.841/0.507秒，后者含一次AMP回退；只有两步，不据此承诺长程稳定吞吐/训练总时长。
- 单张1320×989、六窗预测0.526秒，新进程0.716秒；推理allocated峰值1488.45MiB。仅固定训练图，不能当复杂全链或困难视角平均耗时。
- 新文件约1.8GiB，仍满足100GiB磁盘保留下限。无需从本次预检推断必须升级GPU，后续完整训练/概率缓存需重新核算磁盘与长程资源。

## 兼容实现与失败留存

共享只读Torch2.11.0+cu128/torchvision0.26.0+cu128，CUDA12.8/sm120；新目录安装fvcore0.1.5.post20221221、iopath0.1.9、yacs0.1.8、OmegaConf2.3.0、Hydra1.3.2、antlr4 runtime4.9.3、portalocker3.2.0、submitit1.5.3、cloudpickle3.1.2、tabulate0.9.0。复用timm1.0.26及已有只读Pillow12覆盖层。不称全依赖干净独立环境；没有降级Torch、改驱动或修改旧运行时。

源码锁定Mask2Former `9b0651c6c1d5b3af2e6da0589b719c514ec0d69a`、Detectron2 `a2f4a8771ab77e8411c26b27f24f9489a28a2453`。新补丁仅THC头文件→ATen、scalar_type API及反传连续内存适配，不改CUDA算法。训练适配层保存作者v2元数据，避免旧load hook给现有pixel_decoder键重复加前缀；所有非分类预训练张量精确核验。官方配置链中的两个resize-list eval标签经人工阅读并锁定来源；项目实际裁块不使用其随机resize。

失败全部保留，原状态不改为成功：

1. 初次两项build因头文件搜索路径缺`cuda_runtime_api.h`/`cusparse.h`失败；日志为`logs/msda-build-r1.log`和`logs/detectron2-build-r1.log`。补充现有工具链/运行时include与library路径后build-r2分别15.44/41.04秒通过，未改系统路径或驱动。
2. probe-eager-r1：算子合成sum梯度为非连续视图，原包装器未转连续内存，0训练步；修复后保持原数值容差通过独立梯度对照。
3. probe-eager-r2：严格审计拒绝旧query键名，0步；按作者已存在迁移规则保留原张量改名。
4. probe-eager-r3：官方pkl缺版本元数据使新版head重复迁移，0步；验证完整当前键集合后附当前schema元数据，647张量精确保留。
5. probe-eager-r4：完成1步后测试包装器直接拒绝第二步AMP overflow，没有非有限参数更新；另次r5启用标准的有限次缩放回退。旧失败、配置/代码快照、日志均保留，不能把r4说成两步通过或方法质量失败。

可复用入口为`build_quality_mask2former.py`、`run_quality_mask2former_preflight.py`、`audit_quality_mask2former_preflight.py`及`src/quality_mask2former.py`。原始源码/权重/编译产物/模型/概率仅保存在Git忽略目录，不整仓复制到Git。

## 来源、验收与下一步

[作者模型库](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md)将权重列为CC-BY-NC-4.0，[代码许可证](https://github.com/facebookresearch/Mask2Former/blob/main/LICENSE)为MIT，Detectron2为Apache-2.0。三者不混用；当前技术预检不代表已经澄清参赛分发或未来商业复用许可，没有对外发布/上传。

模型运行环境167项质量测试通过；新增10项覆盖score归一化、空质量拒绝、五类mask分区、旧键迁移和元数据保留。仓库健康检查（来源记录：`../../../outputs/quality/mask2former/SEM-340/logs/repo-health-r1.log`；未随公开包提供）440项测试通过（69项可选依赖跳过），配置/Markdown链接/差异格式检查通过；不把轻量环境可选跳过说成GPU执行。旧数据未挂到默认路径时健康脚本的数据扫描按设计跳过，本次实际数据角色/哈希验收由独立CPU审计完成。

下一另登记Scaffold上完整Mask2Former五类OOF训练：明确有效batch、长程预算/调度、训练增强与作者loss配方，先有限长程吞吐和数值检查，再与已有充分训练B2在同划分比较。桥面/索分支固定，训练侧概率重新校准；不把整套架构/预训练/loss包的收益单独归因为模型变大。当前没有新的mIoU、模型晋级、74开发评分、GAGS或全链三seed/三折完成事实。

配置SHA256 `d0bee61014a21368a80f0f89d5c07f831965a9dbe3a9184579d0ccd35ea2b755`；r5 result `ddd18698f5d5f04cc4c247333ea504dbc2b8b859de55b602d0f4cae66af014f8`；fresh result `2ad87df99d5c4dfb13b9fedc96a1bc0cd8a2358e77e2b4b5bec5b86270b6a0e1`；CPU audit `b941bdd32f8478f018d6ce5a486d9746e71984d33b719e36ea74e26c1da0d54c`。更多逐文件摘要在对应result内。

回退：不启用SEM340新专家即可使用原RGB030+SEM161正式包及已有研究链；没有覆盖、删除、Git提交或推送。
