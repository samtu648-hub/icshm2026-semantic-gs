# Metrics

2026-09-21 [QUAL014双扩展汇总](experiments/runs/QUAL-014-COMPONENT-DROP-EXTENSIONS/report.md)：小构件配额两seed前景差+0.261825/−0.329470pp（均值−0.033823pp），第二seedfoundation−1.127419pp，不晋级；小索/小基础召回两次提高仅作为尺度取舍。DropGaussian新配对PSNR/SSIM/LPIPS差−0.107260/+0.003063/−0.001686，冻结双语义前景−2.814603/−1.697606pp。另RGB152/156/158同记录配方重建确认PSNR19.383911/18.391991/18.536542dB，必须区分端点重放与重训稳定，不选最好run覆盖其它记录。全部指标绑定对应ID/config/seed/split和封存源码，均为历史开发证据。

2026-09-21 [QUAL013](experiments/runs/QUAL-013-SPATIAL-V2-SEED-REPLICATION/report.md)完成v2第二语义seed：SEM414/415确认前景mIoU为87.404296%/87.157925%，索IoU81.774993%/80.834737%，2px边界F1为74.173220%/72.706601%。两seed平均AbsGS−Standard为前景+0.420671pp、索+1.928975pp、边界+3.436304pp，六个逐seed方向门全过。此处是各seed pooled指标的均值，不是集成预测成绩；保留区域与tower代价、PINHOLE网格、历史暴露及RGB seed0限制。完整数据身份、配置、源码与checkpoint哈希见对应报告。

官方尚未发布的归一化、LPIPS backbone、背景处理和裁剪规则不得自行视为最终规则。当前指标用于本地公平比较。

更新2026-09-08：本页已补齐SEM310–329评估口径与最新三臂结果；整体研究解释见[阶段总结](docs/experiments/research_summary_20260908.md)。各历史ID段落的“当前/下一步/正式门”仅代表其执行阶段，最新正式包及研究候选以[项目状态](PROJECT_STATUS.md)为准。

## RGB

- **PSNR**：基于相同值域和尺寸的RGB图计算，越高越好。
- **SSIM**：固定实现、窗口和通道策略，越高越好。
- **LPIPS**：必须记录实现、backbone、权重和输入值域，越低越好。早期E系列摘要为LPIPS-Squeeze；当前R1冻结比较为LPIPS-Alex，不跨backbone混排。

每次报告同时包含：

- 全部视角均值和中位数；
- 最差10%视角的均值；
- 140–146困难视角；
- 插值视角与空间困难视角；
- 拉索局部裁剪的定性对比。

### RGB147–149 真实照片区域监督对照

三臂使用同一官方400相机、72,292点初始化、30k、seed0和严格相同相机序列。A为匹配重训控制，B增加4px膨胀结构区域真实照片RGB L1，C增加完整图先求差分后的原始/4倍粗尺度区域梯度L1且不叠加B。全400均为训练拟合；300标签只用于监督定位和区域诊断；固定144轨迹无GT。

| 臂 | PSNR dB | SSIM | LPIPS | 结构RGB L1 | 结构梯度L1 | 高斯数 |
|---|---:|---:|---:|---:|---:|---:|
| A control | 23.280295 | 0.896189 | 0.245707 | 0.029117 | 0.018016 | 796089 |
| B region RGB | 23.305737 | 0.896586 | 0.244487 | 0.026991 | 0.017260 | 991322 |
| C region gradient | 23.291085 | 0.896749 | 0.244210 | 0.028133 | 0.017084 | 906989 |

B相对A的结构RGB/梯度误差下降7.301%/4.195%，C下降3.377%/5.169%；B/C高斯增加24.524%/13.931%。B四个结构类RGB MAE均最低但background略退；C五类均较A改善但159/160近景明显退化。B/C的159水面幽灵ROI均略差于A。故这些指标支持“区域监督改变照片拟合和原生增密”，不支持隐藏视角、几何真实性或系统晋级结论。完整数值与逐图配对见`outputs/quality/rgb147/comparison-r1/result.json`。

### R1固定本地实现

R1 的可比较RGB结果只能由`python3 scripts/evaluate_rgb.py`产生。输入是一对同名PNG：Pillow解码为RGB、转换为float32的[0,1]，不允许缩放、裁剪、mask或额外色彩变换；尺寸不一致即失败。该脚本逐视角计算并输出CSV，再以每个指标分别给出均值、中位数和最差10%均值。

- PSNR：三通道全像素 MSE，`-10 log10(MSE)`；完全相同图像记为`+inf`。
- SSIM：`opencv-python-headless==4.11.0.86`的每通道11×11高斯窗口，σ=1.5，`C1=(0.01)^2`、`C2=(0.03)^2`，`BORDER_REFLECT_101`边界，所有像素和通道平均。
- LPIPS：`lpips==0.1.4`、`net=alex`、`version=0.1`、eval模式、batch size 1；输入从[0,1]线性映射至[-1,1]。真实实验开始前必须缓存并记录AlexNet权重的来源和校验值。

`--skip-lpips`只可用于环境/渲染链路验证或明确标注为`completed_partial_metrics`的非排名运行；它不能用于候选比较、排名或RGB-060。缓存完成后必须对同一份已有渲染补算LPIPS，不能以 partial 结果替代完整指标。此实现是固定的内部公平比较工具，非官方评测规则。

## 语义

类别固定：

| ID | 名称 |
|---:|---|
| 0 | `background` |
| 1 | `deck` |
| 2 | `stay_cable` |
| 3 | `tower` |
| 4 | `foundation` |

对每类 \(c\)：

\[
IoU_c=\frac{TP_c}{TP_c+FP_c+FN_c}
\]

同时报告：

- 五类IoU；
- 含背景mIoU：五类IoU平均；
- 不含背景mIoU：ID 1–4的IoU平均；
- `stay_cable` precision、recall和IoU；
- `background → stay_cable`误报像素数及占真实背景比例；
- 每视角和每空间折统计。

### SEM-001 固定本地实现

`scripts/run_sem001.py` 对`sem_001_scaffold_static_v1`的全部验证相机读取单通道 ID PNG，按“真实类别为行、预测类别为列”累计 5×5 像素混淆矩阵；没有 ignore/void 像素。每类 IoU、全局含/不含 background mIoU、`stay_cable` precision/recall 及 background→stay_cable 的数量/真实背景比例都由这个累计矩阵计算。最差10%按每视角不含 background mIoU 升序选择；140–146报告其逐视角 mIoU 的均值。该策略是内部可复现协议，非主办方最终规则。

2/4/8 像素容差边界指标保留给有单独预注册的后续诊断；它们不是 SEM-001 的完成定义，不能用来替代本地原始像素 IoU。

官方是否包含背景、忽略区域和聚合方式仍待最终协议确认。

### SEM-070 raw/cleaned 对比

`SEM-070-POST-CLEAN` 先在219张语义训练视角上按固定 hash parity 生成 calibration/check 两半。anchor 的误报率只由 calibration dominant 像素统计，五个固定阈值只在 check half 上比较实际渲染后的五类混淆矩阵；74张空间验证视角必须在 `postprocess_mask.npz` 写入后才打开。raw 与 cleaned 使用同一冻结 SEM-001 checkpoint、RGB renderer、相机和标签，逐视角同时报告 mIoU、四类 IoU、`stay_cable` precision/recall/IoU、background→stay_cable、最差10%和140–146。

后处理候选的正式保留门是相对于同一运行中的 raw SEM-001：整体 mIoU、`stay_cable` IoU/recall 不下降，deck/tower/foundation 每类 IoU 最多下降0.5 pp，background→stay_cable 至少下降1%，最差10%和140–146 mIoU 最多下降0.5 pp，并且 RGB 最大像素差为0、artifact hash/checkpoint reload/官方外部相机契约全部通过。任何失败结果只保留为 show-only，最终回退仍为 `RGB-030-FULL Scaffold-GS + SEM-001`。

正式 SEM-070 结果：五个阈值的 check 误报下降为 63.01%–90.61%，但 `stay_cable` recall 保留仅 30.33%–61.47%，故选择 `null`、抑制 0 个 anchor；74-view raw/cleaned 含背景 mIoU 均为 `0.616103`，`stay_cable` IoU 均为 `0.068759`，background→`stay_cable` 均为 `13,166,331` 像素。RGB 最大差为0，artifact/checkpoint/官方相机门通过，唯一失败门为误报至少下降1%。

### SEM-080 训练侧结构清理诊断

`SEM-080-STRUCTURAL-DIAG` 只在219张SEM-001训练视角运行。后处理谓词读取冻结的五通道概率，保留连续、近竖直且两端分别接触预测tower/deck邻域的 raw `stay_cable` Hough线；被拒绝的像素按 `p_background += p_stay_cable; p_stay_cable = 0` 转移。标签仅在谓词完成后用于训练 calibration/check half的度量，74-view、无标注和隐藏输入不得读取。

该诊断检出12,351条 tower-to-deck 支持线并保留4,597,766 / 52,588,100（8.743%）raw `stay_cable` 像素。独立check half的background→`stay_cable`从18,363,796降至1,019,147（−94.45%），precision从38.48%升至57.90%，但recall从84.90%降至10.60%（保留12.49%），`stay_cable` IoU从0.36019降至0.09842，含背景mIoU从0.67833降至0.63807（−4.03pp）。样例可视复核还确认被删区包含正常主缆、吊索和部分斜拉索；tower→deck直连不是完整桥梁线网。因此它在视觉和训练侧官方指标上均为drop，不打开74-view。RGB artifact/三张PNG hash相等、最大像素差为0，checkpoint重载逐位一致。

### SEM-090 三类线网预检

`SEM-090-CABLE-NETWORK-DIAG` 以主缆、吊索、斜拉索三种模板修正SEM-080，但仍只从冻结raw语义 `stay_cable` 图提取Hough线。8-view训练预检检出358条主缆、168条吊索和736条斜拉索候选，RGB=0、checkpoint重载和无验证/无标注访问边界通过；但样例仍将左侧大块江水黄色接纳为tower-top→边界主缆。该视觉失败足以止损，219-view/74-view指标均不运行也不得推断；它证明同源语义Hough不能充当独立的真假索线证据。

### SEM-100 冻结渲染 RGB 域预检

`SEM-100-RENDER-SEG-PF` 在八张固定训练侧 OOF 图上，保持三个
`SEM-051-OOF` SegFormer-B2 checkpoint 与 `RGB-030-FULL` 均冻结，仅将
教师输入从真实训练 RGB 替换为同官方相机、同尺寸的 Scaffold 渲染 RGB。
真实/渲染两套预测在标签读取前全部生成；74-view、无标注、隐藏输入和外部相机
均未读取。渲染输入仍有 `stay_cable` recall 0.959029（真实输入为0.951849），
但其 mIoU 为0.789742（−4.37pp）、`stay_cable` IoU为0.772650（−11.42pp），且
background→`stay_cable` 从47,875升至163,214（+115,339）。这证明冻结渲染 RGB
仍含可识别的索线证据，同时证明真实 RGB 教师不能原样用于外部渲染域。所有冻结
RGB artifact/三张重复渲染PNG hash相等、最大像素差0、OOF checkpoint hash和五类
simplex均通过。此结果仅为训练侧域差诊断，不能与74-view候选排名混用。

### SEM-101 渲染 RGB 二维 OOF 止损门

`SEM-101-RENDERED-SEG-OOF` 在结果前固定五项训练侧空间OOF门：含背景mIoU至少
0.70、`stay_cable` IoU至少0.70、precision至少0.80、recall至少0.90，以及
background→`stay_cable`不超过真实背景的2%。模型、seed、三折、3,000 iteration、
损失和推理窗口与SEM-051一致，只把输入换为冻结Scaffold渲染RGB。每一项都必须通过；
否则不训练最终模型，也不读取74-view或外部相机。门仅判断是否值得另立最终模型，
不与SEM-001的74-view三维指标比较或选择超参数。

正式结果：219张训练图恰好各有一次空间OOF预测（95/84/40张）。含背景mIoU为
0.849734，`stay_cable` IoU/precision/recall为0.836124/0.860914/0.966708，
background→`stay_cable`为3,383,866像素、占真实背景0.014137。五项固定门均通过；
三折checkpoint重载均通过，训练总907.39 s、OOF推理总171.57 s、峰值分配显存
3,203 MiB。219张冻结渲染RGB的manifest及三张重复渲染均证明RGB artifact/PNG hash
一致、最大像素差0；74-view、无标注、隐藏输入和外部相机均未读。这仅允许新ID注册
最终二维渲染域模型，不能作为SEM-001替代或74-view质量结论。

### SEM-102 最终渲染域模型与一次性验证审计

`SEM-102-RENDERED-SEG-FINAL` 已在结果前冻结为：复用SEM-101哈希地址的219张
冻结渲染RGB，以相同SegFormer-B2、seed、3,000 iteration、AdamW、加权CE、768 crop/
flip及768/512滑窗训练一个全训练集checkpoint。checkpoint写入且重载通过前，74-view、
无标注、真实RGB和外部相机均不可打开；随后先渲染74张冻结RGB、写完74份预测清单，
才读取标签作一次性混淆矩阵审计。

正式保留门相对于记录的SEM-001：含背景mIoU、`stay_cable` IoU和recall不可下降；
deck/tower/foundation最多各降0.5pp；background→`stay_cable`至少下降1%；最差10%和
140–146不含背景mIoU最多各降0.5pp。还必须满足RGB artifact/重复像素差0、checkpoint
重载和官方`CAMERA_FROM_WORLD + SIMPLE_RADIAL`的RGB/五logits/0–4 PNG门。

该一次性审计已完成，且**没有通过正式保留门**。全219图final-3k checkpoint重载后，先在
无标签条件下渲染并写完74份预测，才打开固定标签。最终74-view含背景mIoU为0.750301
（SEM-001为0.616103），`stay_cable` IoU/precision/recall为
0.685329/0.815764/0.810826（SEM-001 IoU/recall为0.068759/0.368036），
background→`stay_cable`为520,271像素（SEM-001为13,166,331，下降96.05%）。

然而deck IoU从0.900999降为0.876788（−2.42pp）、tower从0.834870降为0.741701
（−9.32pp），最差10%不含背景mIoU从0.423440降为0.351778（−7.17pp）；三项均超过
−0.5pp保护界限。foundation从0.448718升为0.463487，140–146从0.601213升为0.750786，
RGB artifact/像素差0、checkpoint重载和官方外部相机接口均通过。故SEM-102仅保留为
show-only失败证据，不替代`RGB-030-FULL Scaffold-GS + SEM-001`，且不得用本次验证结果
重训、调参、重选或再次审计。

### SEM-110 训练侧 OOF 保护融合

`SEM-110-PROTECTED-FUSION-OOF` 不训练也不改写任一模型：冻结 SEM-001 的类图提供
deck/tower/foundation（class 1/3/4）逐像素保护区；其余 background/`stay_cable`（class
0/2）位置仅接受 SEM-101 三折空间 OOF 的 `stay_cable` 概率决定。108 张 calibration
训练视角在固定阈值网格 0.20/0.35/0.50/0.65/0.80 中预选择，0.80 因 background→
`stay_cable` 降幅最大而冻结；111 张 hash check 半部从未参与选阈值。

独立 check 上，含背景 mIoU 从 0.680865 升至 0.807461，`stay_cable` IoU/recall 为
0.871190/0.945650，background→`stay_cable` 从 16,573,599 降至 1,113,641（−93.28%）；
保护类预测像素逐位不变。219 张 SEM-001 训练侧渲染均在不读取标签时先写入，且 RGB
冻结 artifact/PNG hash 一致。该结果只授权单独、固定阈值的最终 74-view 登记，不能从
check 或验证结果再调规则。

### SEM-111 固定保护融合最终审计

`SEM-111-PROTECTED-FUSION-FINAL` 将 SEM-110 冻结的 0.80 规则直接作用于 74 张已先
写出的 SEM-001 类图和 SEM-102 五通道概率；74 张预测 manifest 完成后才读取标签。正式
74-view 含背景 mIoU 为 0.760635（SEM-001 为 0.616103），不含背景 mIoU 为 0.705681；
`stay_cable` IoU/precision/recall 为 0.638137/0.860977/0.711445，background→
`stay_cable` 降为 347,412 像素（−97.36%）。deck/tower/foundation IoU 与 SEM-001
逐位相同（0.900999/0.834870/0.448718），最差 10% 不含背景 mIoU 0.428629，高于
SEM-001 的 0.423440，140–146 为 0.758237。

全部正式质量门和工程门通过：RGB 最大差 0、artifact/PNG hash 一致、SEM-001 checkpoint
重载通过，官方 `CAMERA_FROM_WORLD + SIMPLE_RADIAL` 外部相机输出 RGB、五 logits 和
0–4 PNG 均正确。SEM-111 因此是当前本地正式合格候选；`RGB-030-FULL Scaffold-GS +
SEM-001` 仍须完整保留为可即时回退版本。

### SEM-112 边缘支持低置信索线补回

`SEM-112-EDGE-RECOVERY-OOF` 从 SEM-110 的冻结 0.80 高置信融合出发，只允许恢复同时
满足 SEM-001 class-2、SEM-101 OOF 概率 `[0.55, 0.80)`、冻结渲染 RGB 强亮度边缘及两像素
高置信索线种子邻接的像素。0.80/0.55 阈值与邻接半径固定，只有 0.80/0.85/0.90 RGB 边缘
分位在108张 calibration 中比较；111张 check 与74-view不可用于选择。

三项均未通过 calibration 优化门。最强 0.90 候选使 mIoU 0.807648→0.808044
（+0.0396pp）、`stay_cable` IoU 0.871964→0.873791（+0.1827pp）和 recall
0.927498→0.933866（+0.6368pp）；但前两项均未达到预注册的 +0.20pp，故不得选择。
background→`stay_cable` 为 674,104→725,132（+7.57%）仍在10%增长上限内，三类保护预测
逐位不变；0.80/0.85分位还分别增加11.94%/10.15%误报。结果在 calibration 即止损，check
标签、74-view、无标注、真实 RGB 和外部相机均未读，SEM-111不被改变。

### SEM-120 显式开发阈值扫掠（非盲测）

在用户授权将已打开的74张标签转为可重复使用开发集后，`SEM-120-DEV-THRESHOLD-SWEEP`
固定所有模型、RGB与保护类，只扫保护融合的`stay_cable`阈值0.10--0.90。其预注册开发分数
`0.55*mIoU + 0.30*stay_cable IoU + 0.15*stay_cable recall`选中0.20：mIoU为0.770685，
`stay_cable` IoU/precision/recall为0.687827/0.776047/0.858168，最差10%与140--146分别为
0.460220/0.770756；相对SEM-111的0.80，mIoU/索IoU/recall增加1.005/4.969/14.672pp，
但background→`stay_cable`由347,412升至743,906。

0.30是相邻的Pareto折中：它取得本扫掠最高mIoU 0.771317和索IoU 0.690498，背景→索为
629,688，代价是recall降至0.835102。两项候选的deck/tower/foundation预测均与SEM-001逐像素
一致；0.80亦已由冻结输入精确复现。此结果仅用于开发与可视化选择，不能覆盖SEM-111的历史
正式审计，更不能表述为隐藏视角泛化或最终替代证据。

`VIS-004-DEV-THRESHOLD-FLYTHROUGH` 随后以144个不读取标签或参考RGB的新相机pose直接
比较0.80、0.20、0.30。三路deck/tower/foundation逐像素一致；轨迹上0.30相对0.80多
1,307,344个`stay_cable`像素，0.20又相对0.30多347,371个。高差异帧0091表明后者的一部分
为塔右侧的大块低置信黄色扩张，0.30仍保留主缆主体。此为无GT视频的设计复核，支持把0.30
作为下一开发机制的视觉基线，不能单独支持质量或泛化结论。

### SEM-121 连续二流融合（训练 check 失败，开发集未读）

`SEM-121-CONTINUOUS-FUSION-OOF` 不再扫阈值，而是在108张训练 calibration 以等量
`stay_cable`/非索可变像素一次拟合固定形式的二维索/背景赔率与SEM-001索先验。拟合得到
截距`1.018158`、二维赔率权重`0.600878`与SEM-001先验权重`0.544541`，九步收敛。111张
独立check的硬0.30基线为mIoU `0.790574`、索IoU/recall `0.798087/0.970561`及2,812,695
background→`stay_cable`像素；连续候选为`0.778500`、`0.746121/0.975556`及3,948,987。

故连续函数虽然使recall增加0.50pp，却令mIoU下降1.21pp、索IoU下降5.20pp，并增加
1,136,292个背景误报；deck/tower/foundation逐像素保持不变。它未通过五项支配比较中的四项，
74-view、真实RGB、无标注、隐藏输入和外部相机都未读取。该公式登记drop，不能在同一实验中
调L2、采样、特征、阈值或公式；后续只诊断0.20–0.30犹豫带的误差机制。

### SEM-122 0.20–0.30 犹豫带只读诊断（开发集）

`SEM-122-UNCERTAINTY-BAND-DIAG` 验证冻结输入后，逐像素证明其测量集合恰为hard 0.20
减hard 0.30的`stay_cable`像素。74-view开发集共有186,758个该类像素，其中70,276
（37.63%）是GT `stay_cable`、114,218（61.16%）是GT background、2,264（1.21%）是其他类。
这直接量化了0.20的增量：确实补回大量真实索线，但整体是背景多数。

概率越高，真索比例由`[0.20,0.22)`的34.61%升至`[0.28,0.30]`的41.20%，仍不够可靠。
与hard-0.30种子8连通的部分占96.48%，但真索仅37.96%，所以“连接到已有主缆”不能单独解释或
清理错误区域。距离有排名能力：距种子1–2px为44.95%真索、2–4px为29.04%、4–8px为20.29%，
但最接近环也仍是背景多数。没有预测、RGB、checkpoint或外部输入被改动；结果只允许下一新ID测试
一个预注册的联合谓词，不是候选质量或泛化证据。

### SEM-123 联合犹豫带恢复开发扫掠

`SEM-123-BAND-RECOVERY-DEV-SWEEP` 从hard 0.30开始，只补回固定0.20–0.30带中符合
概率下限×种子距离×SEM-001索先验的18个交集，并把hard 0.30保留在同一表中。所有候选
deck/tower/foundation预测逐像素一致，RGB、checkpoint和模型权重未改。

既有开发分数选择`p20_d02_any`：mIoU `0.771605`、索IoU/recall `0.692107/0.854184`，
相对hard 0.30的`0.771317/0.690498/0.835102`增加0.029/0.161/1.908pp；背景→索由629,688
增至703,286（+73,598）。它补回58,136个真索，同时有73,598个背景像素。

`p20_d04_sem001_stay_cable` 是更保守的Pareto候选：仅补回SEM-001原本已是索的4px带内
像素，取得全表最高mIoU `0.771991`和索IoU `0.693816`，recall `0.845885`，背景→索为658,240
（仅+28,552）；它补回32,854真索与28,552背景像素。两个结果都只为开发视觉候选，下一步应在
同pose无标签视频中与0.30比较是否避免了0.20式黄色扩张。

### VIS-005 固定选择性恢复的同pose视频复核

`VIS-005-SELECTIVE-RECOVERY-FLYTHROUGH` 完成144个`CAMERA_FROM_WORLD + SIMPLE_RADIAL`
新pose的hard 0.30、`p20_d02_any`与`p20_d04_sem001_stay_cable`三路并排渲染。视频为6.0秒、
24fps、3960×1032 H.264；不读取标签或参考RGB，不加入展示清理。所有帧的deck/tower/foundation
与hard 0.30完全一致，两个恢复候选也都不删除hard 0.30的任何已有`stay_cable`像素。

在此无GT轨迹上，召回候选相对hard 0.30新增242,108个`stay_cable`像素，保守候选新增98,369个。
复核各自新增最多的frame0091/frame0031，增量沿既有索/塔邻域出现，未出现VIS-004硬0.20所示的
大片新增黄色背景区。这只说明已冻结规则在连续视角的外观没有明显失控，不能替代带标签指标、独立泛化
或正式候选门。

### SEM-140/141 原生分辨率拉索二分类

`SEM-140-CABLE-BINARY-OOF`只把SEM-101五类学习改为`non_stay_cable`/`stay_cable`二分类，
使用不缩放的960×960 crop和960/480滑窗；SEM-001 class 1/3/4仍逐像素保护。三折3k训练耗时
0.4093 GPU小时、峰值5570 MiB。108-view calibration从固定0.05–0.95网格选择0.70后，
111-view独立check相对SEM-110使mIoU +0.64pp、索IoU +2.83pp、recall +1.37pp，背景误报
减少23.66%，全部训练门通过。

`SEM-141-CABLE-BINARY-FINAL`随后按完全冻结的配方在219图训练唯一final-3k，checkpoint先重载并
冻结hash，74份开发预测全部写完后才读取标签。相对SEM-111，含背景mIoU为0.774899（+1.43pp），
不含背景mIoU为0.722893（+1.72pp），`stay_cable` IoU/precision/recall为
0.706983/0.842841/0.814334，其中IoU/recall分别+6.88/+10.29pp；worst-10%与140–146
不含背景mIoU分别+1.64/+1.58pp。deck/tower/foundation预测逐像素完全一致，RGB字节隔离、
checkpoint重载和官方外部相机门通过。

该候选通过全部预注册开发目标，推荐官方隐藏评测；但74-view已是复用开发集，不能称为新的盲测。
它相对SEM-111的background→`stay_cable`从347,412增至447,859（+28.91%），precision下降
1.81pp，且174单视角索IoU下降3.59pp。因此checkpoint与0.70阈值保持冻结，SEM-111与SEM-001
继续作为低误报候选和独立回退。

### SEM-150/151 foundation单向验证门

`SEM-150-FOUNDATION-GATE-OOF`只允许冻结五类二维教师的class-4概率把已有foundation
预测拒绝为background，不得新增foundation，也不得改动deck、`stay_cable`和tower。0.10由
108-view calibration选择；在111-view独立check上，foundation IoU/precision分别+25.81/
+26.29pp，recall为98.60%，总mIoU +5.35pp，background→foundation减少66.84%。

`SEM-151-FOUNDATION-GATE-FINAL`随后将冻结0.10应用于SEM-141的74份已物化开发预测，并在
所有候选及manifest落盘后才读取标签。含背景mIoU为0.830934（+5.60pp），不含背景mIoU为
0.792639（+6.97pp）；foundation IoU/precision/recall为0.727703/0.782758/0.911866，
相对SEM-141分别+27.90/+31.54/−0.63pp。background→foundation从156,852降至37,460
（−76.12%），worst-10%及140–146不含背景mIoU分别+4.16/+1.75pp。74视角中foundation
IoU为70升、4平、0降，deck/`stay_cable`/tower逐像素完全一致；RGB字节、checkpoint重载
和官方外部相机门全部通过。

该结果通过预注册开发门并推荐隐藏评测，但74-view已明确复用，不能视为新盲测。0.10、父模型和
4→0规则冻结，不再据此开发集调参；SEM-141/111/001继续作为回退。

### SEM-160/161 deck单向恢复

`SEM-160-DECK-BINARY-OOF`在冻结渲染RGB上使用原生960×960 crop、960/480滑窗和
SegFormer-B2学习non-deck/deck，只允许foundation审核前后均为background的位置0→deck。
108-view calibration选择0.99后，111-view独立check相对SEM-150使deck IoU/recall
+0.437/+0.841pp、总mIoU +0.093pp，precision保持95.25%，全部保护预测逐位一致。

`SEM-161-DECK-BINARY-FINAL`随后用相同配方在219图训练唯一final-3k。checkpoint严格重载并
冻结hash后，74份概率与预测全部写入manifest，再读取已复用开发标签。相对SEM-151，含背景
mIoU为0.833870（+0.294pp），不含背景mIoU为0.796056（+0.342pp）；deck
IoU/precision/recall为0.914667/0.950392/0.960525，其中IoU/recall +1.367/+2.190pp。
158–161平均deck IoU从0.769695升至0.796598（+2.690pp），四帧均升；标准worst-10%和
140–146不含背景mIoU分别+0.053/+0.963pp，deck专属worst-10%均值+1.610pp。

74帧中deck IoU为63升、3平、8降。新增128,641个deck像素包含93,060真deck、18,646
background和16,935真`stay_cable`，没有tower/foundation；background→deck从74,985升至
93,631（1.24866倍，低于冻结1.25门）。079–082单帧deck IoU下降3.54–5.31pp，是保留的主要
风险。所有非eligible像素、已有deck及class 2/3/4预测完全不变；RGB字节、checkpoint重载与
官方外部相机1320×989 RGB/五logits/PNG均通过。SEM-161通过预注册门并推荐隐藏评测，但不得
据复用开发集继续调参；SEM-151/141/111/001依次保留回退。

### SEM-310–329 评估范围与最近结果

这批实验的基点为Git `49cdcdb1c9915a9544d568754af95ea20a85ec9c`加各阶段封存工作树源码，seed20260805，数据manifest为`data/manifests/contest_dataset_v1.json`。各ID的独立配置、源版本、访问清单和报告由[registry](experiments/registry.yaml)关联；详细路径索引见[阶段总结](docs/experiments/research_summary_20260908.md)。

- SEM310只有架构/接口工程证据，预训练质量受权重授权阻塞；没有DINOv3质量分数。
- SEM311在`sem_005_oof_spatial_v1`的复用108 calibration上比较DINOv2与冻结MiT：mIoU/foundation IoU对MiT +2.038/+9.878pp，对SEM160 −2.653/−12.947pp。SEM312复用这些calibration输出进行方向诊断。SEM313固定只审核旧foundation，在同111 check上对SEM160 +0.811/+3.980pp，recall−1.003pp。
- SEM314是另立全219训练后的74开发比较，mIoU为83.474097%、foundation IoU73.207788%、recall87.926293%；相对SEM161 +0.087080/+0.437443/−3.260272pp。它与前项OOF使用不同目标集合，不直接拼接分数。
- SEM320/321为3对匹配工程/几何预检，SEM322为同3目标pose-top3可达性；SEM323扩到105 calibration目标。SEM324的标签一致94.086%和4→0纠正23474/误删3707来自512网格采样，不是完整1320×989图的mIoU。
- SEM325实际105 calibration/111 check原生预测，check mIoU/foundation IoU/recall为87.643590%/69.461184%/96.798472%，相对SEM160 +0.812045/+3.969178/−1.806420pp。SEM326另用全219参考库进入最终SEM161/74开发比较。
- SEM327是105 calibration上的2px边界保护诊断，较dense的foundation IoU−1.451937pp、recall+1.560494pp。SEM328实际第二源否决在111 check较SEM325的mIoU/foundation IoU/recall +0.009221/+0.047017/+0.423304pp，precision−0.177683pp。SEM329固定该规则进行最终三臂比较。

### SEM-329 固定74开发图三臂比较

来源：[SEM329报告](experiments/runs/SEM-329-SECOND-SOURCE-VETO-FINAL/report.md)及[config](experiments/runs/SEM-329-SECOND-SOURCE-VETO-FINAL/config.yaml)。划分`sem_001_scaffold_static_v1`为219参考源/74开发/7隔离/100无标注。74张1320×989 PINHOLE语义图先封存预测，完成原生重构和4次fresh外相机检查后再评分。相同GT、相同累计混淆矩阵口径，两父方案的逐图及整体矩阵精确复现旧记录。

| 方案 | 含背景全局mIoU（%） | foundation IoU（%） | foundation precision（%） | foundation recall（%） |
|---|---:|---:|---:|---:|
| SEM161 | 83.387017 | 72.770345 | 78.275843 | 91.186564 |
| SEM326 | 83.682675 | 74.242876 | 81.475684 | 89.319965 |
| SEM329 | 83.705003 | 74.354428 | 81.244656 | 89.761802 |

SEM329对SEM161的mIoU/foundation IoU/precision/recall为+0.317985/+1.584083/+2.968813/−1.424762pp；对SEM326为+0.022328/+0.111552/−0.231028/+0.441838pp。class1/2/3像素及指标完全不变。

1431个恢复像素按GT 0/1/2/3/4计数为672/14/1/5/739；剩余相对SEM161的删除计数为7590/20/9/51/2383。前者意味着救回739真foundation、放弃672正确背景纠错、20仍错；净正确像素对SEM326 +67，对SEM161 +5207。foundation FP降低与五类像素完全纠正分别记录。

空间组为SEM326既有trajectory segment与74开发图的交集，不重建OOF折。四组对SEM161的mIoU/foundation IoU均升，对SEM326三升一微降。固定140–146报告逐视角含背景mIoU均值，SEM161/326/329分别83.863630%/84.814621%/84.816663%。最差10%固定为按SEM161逐视角含背景mIoU排名的8张061/060/059/062/065/063/057/058，分别58.733667%/58.751901%/58.751901%；不以新候选重排，也不与旧不含背景的worst10指标混用。

这些均为历史复用开发证据；独立复算证明算术及输出一致，不等于新盲测或统计显著。ADR-078按连续收益与代价保留研究候选；旧+0.30/+1.50pp及90%recall门只作历史对照，13/14满足。工程通过、研究收益和正式包是否替换是分别记录的结论。

## 系统与复现

- 单视角RGB和语义渲染时间；
- 峰值GPU显存；
- checkpoint和配置总大小；
- Gaussian或anchor数量；
- 训练总时长和GPU型号；
- 全新环境从checkpoint运行成功率；
- 输出分辨率、语义dtype和类别值域检查。

## RGB语义解耦

固定同一相机，比较增加语义模块前后的RGB：

- 最大绝对像素差；
- 平均绝对像素差；
- 图像SHA-256；
- RGB参数是否发生变化。

默认要求最大绝对差为0；若实现使用非确定性算子，必须预先登记可接受误差及依据。

## 坐标对齐

至少记录：

- 官方相机中心经Sim(3)变换后的误差；
- 官方视线方向和训练坐标视线方向的夹角；
- 已发布相机经提交入口渲染与本地入口渲染的像素差；
- 畸变前后相机模型、分辨率和有效区域。

## 模块保留门槛

每个实验的事前条件以其不可变config和对应ADR为准；[ROADMAP.md](ROADMAP.md)记录当前推进顺序，原阶段门保存在[历史路线](docs/history/roadmap_pre_summary_20260908.md)。空间划分、开发复用范围和直接父控制需明确记录，不能用相邻帧随机划分替代。新SEM329依ADR-078报告连续正向收益及precision/recall/局部代价；旧实验的门与完成状态保持原记录。
