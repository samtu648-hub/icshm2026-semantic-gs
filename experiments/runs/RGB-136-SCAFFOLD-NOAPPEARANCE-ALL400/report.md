# RGB-136-SCAFFOLD-NOAPPEARANCE-ALL400

2026-09-15 用户“开始做吧”授权全400/30k独立重训。完整训练、全量评分与独立复现已完成；保留为部署一致性RGB候选，未替换旧系统。父RGB120、同303来源初始化、seed0、commit 49cdcdb1c9915a9544d568754af95ea20a85ec9c，manifest/split见config。

唯一主要变量appearance_dim32→0，使用上游原生无外观分支，从头训练；不是只把旧embedding设零，也不改旧模型。颜色MLP输入维数相应变化，初始化随机数消耗及之后拓扑可能变化；同seed不代表逐参数初值相同。其余模型配置/优化/损失/增密保持RGB120。单seed全400拟合不构成泛化排名。

训练/新视角均不查询逐照片UID。先32步原生工程预检及保存重载UID恒等，再一次30k正式运行。预计RGB训练约30分钟，完整导出评分另计，约2–4GiB新增；GPU共用锁串行，空间实测监测，不自动减步。旧三模型、SEM370/371/372和冻结包保持，独立SEM373任务不受覆盖。

终点全400 PSNR/SSIM/LPIPS，与旧Scaffold UID0及训练UID附表、Standard/Mip分列；固定九图和144帧官方轨迹、159/160岸线/174天空与真索/桥面保护区域复核。此轮先完成RGB选型对照，不冒充已有新语义结果。

实际批次 outputs/quality/scaffold_noappearance/smoke-r1：failed_preserved，0.528秒，smoke=True。质量验收另记。

smoke-r1为零训练准备失败：通用哈希校验禁止图片指向历史outputs的既有符号链接。改为显式限制解析目标在项目outputs并逐图核对原登记SHA256，不改变图像；失败目录保持，smoke-r2使用修正校验。

实际批次 outputs/quality/scaffold_noappearance/smoke-r2：failed_preserved，0.532秒，smoke=True。质量验收另记。

smoke-r2仍为零训练校验失败：部分历史图片解析到已配置artifact-root，而非项目outputs；400张哈希实际全部相同。smoke-r3显式允许这两个既有产物根，继续完整哈希核验。两次失败不删除。

实际批次 outputs/quality/scaffold_noappearance/smoke-r3：completed_smoke，21.973秒，smoke=True。质量验收另记。

32步smoke-r3通过，三相机×四UID原生重载渲染逐像素恒等（含不存在的999号UID），嵌入模块确实不存在，模型signature不变。正式full-r1已从头开始30000步；两次前置失败均0训练，未挑选质量端点。

实际批次 outputs/quality/scaffold_noappearance/full-r1：completed_all400，1734.463秒，smoke=False。质量验收另记。

full-r1原生30k训练完成：1734.451秒（28.908分钟，包含原生保存/末尾流程），GPU锁等待0.012秒；峰值allocated937.488MiB、reserved3522MiB。全400实际训练相机、评价集合为空；上游末尾空test集合显示Test FPS nan是无测试相机的统计，不是训练损失或模型非有限。完整项目评分另行执行。最终UID恒等验证通过。源代码与原RGB120/18冻结哈希保持。

## 全量结果（同一训练内评价口径）

所有分数绑定本实验config、seed0、Git commit及released_all400_rgb_all300_sem_v1；PSNR/SSIM/LPIPS采用QUAL007相同原生1320×989 PINHOLE、uint8 PNG、全400逐图等权平均及精度设置。原参数/源代码完整展开核对，optimization和pipeline与RGB120完全一致；模型差异只有appearance_dim32→0及独立路径。各算法保留自身原生配方，同30k不是同参数量/计算预算；本实验不声称隔离整个架构的因果优势。

- 旧Scaffold固定UID0：PSNR 22.331303918 dB，SSIM 0.884667411，LPIPS 0.260564623。
- 旧Scaffold训练UID附表：PSNR 23.340185562 dB，SSIM 0.894768360，LPIPS 0.248750016。
- Standard RGB130：PSNR 23.217672855 dB，SSIM 0.893217550，LPIPS 0.252781716。
- Mip RGB131：PSNR 23.224402798 dB，SSIM 0.893872859，LPIPS 0.251057966。
- 新Scaffold RGB136，无逐照片外观：PSNR 23.203860168 dB，SSIM 0.892379803，LPIPS 0.250841964。

新模型相对旧UID0：PSNR +0.872556250 dB，391张改善、9张退化；全图MAE 8.539745773→5.345892980（0–255）。相对Standard：PSNR −0.013812687 dB（183改善/217退化），相对Mip −0.020542630 dB（180/220）；LPIPS略优于两者但SSIM略低。相对旧训练UID附表PSNR −0.136325394 dB（87/313）。这不是绝对梯度实验的续训，也不能把不依赖UID等同于质量必然最好。

## 固定区域和图像核查

以下复用RGB132已登记区域，与QUAL007另一个大岸线框不同，不能跨框直接比较旧报告数值。MAE单位0–255，越低越好，区域只用于评价。

- 159-shore：scaffold_uid0 16.694444；standard 6.972376；mip 7.520368；scaffold_noappearance 9.241647；scaffold_native_uid 7.960158。
- 159-near：scaffold_uid0 12.917815；standard 9.163780；mip 8.930335；scaffold_noappearance 9.873445；scaffold_native_uid 9.623589。
- 159-far-occlusion：scaffold_uid0 11.000960；standard 4.949303；mip 9.797207；scaffold_noappearance 4.916879；scaffold_native_uid 4.309973。
- 159-water-ghost：scaffold_uid0 5.630432；standard 3.063873；mip 3.037994；scaffold_noappearance 3.215895；scaffold_native_uid 3.091142。
- 160-shore：scaffold_uid0 16.238499；standard 7.305620；mip 8.076516；scaffold_noappearance 8.595527；scaffold_native_uid 7.873053。
- 160-near：scaffold_uid0 12.194583；standard 9.790456；mip 8.748016；scaffold_noappearance 10.428274；scaffold_native_uid 9.445893。
- 174-sky：scaffold_uid0 6.788352；standard 5.230018；mip 5.200496；scaffold_noappearance 3.979185；scaffold_native_uid 3.847980。

159/160原生裁剪显示新模型岸线颜色与清晰度明显好于旧UID0，但田地/道路的拖长纹理仍在，Standard更清楚。174天空中Standard/Mip可见的斜向细条在新Scaffold图中不明显，天空区域误差低于两者；这只针对已检查的174，146背景仍有细线，不能宣称天空伪影全部修复。桥面与主缆主体在固定图保留，细索断续和动态车辆/水纹缺失仍有，不能仅凭整体MAE或参数冻结断言结构无损。

300张标签区域的RGB MAE：deck旧UID0/Standard/Mip/新模型为13.459905/10.410776/10.668769/10.568612；stay_cable为8.832236/5.705547/5.612663/5.944924；tower为9.523890/5.187677/4.928064/5.061625；foundation为7.708545/5.074221/5.083265/5.287517。新模型大幅好于旧UID0，但不是所有结构区域优于Standard/Mip。以上是RGB照片误差，绝不是语义IoU或删除结构数。

统一144相机、12fps同步四方视频已生成，FFprobe确认144帧与尺寸1320×1038。离线查看0/48/71/72/73/96/143接触表及固定159/160/174/146原图：新模型未出现Mip在71的远端遮挡雾块，远塔/桥面保留；仍有水面亮块和近岸拉丝。稀疏抽帧不证明整段无闪烁，轨迹无照片真值，不评分。旧训练UID没有新视角专属嵌入，不伪造对应轨迹。

## 验证、范围与回退

32步工程预检与30k终点各三相机×四UID（0/158/399/999）完全恒等，嵌入模块不存在。7k、15k、30k各67个模型/优化器张量全部有限；最终189887 anchors。三官方/作者原生相机投影和RGB最大误差均为0。544张导出图在独立新进程重载后逐像素完全一致；1609个保存图PSNR独立公式复算通过（四完整400图+旧训练UID固定九图），全400区域统计另存audit.json，旧训练UID其余391张的指标沿用已登记附表而非声称重新渲染审计。三旧模型31份登记产物及18正式冻结哈希均保持；所有失败保留。

原生训练1734.451秒，峰值allocated937.488MiB；完整导出74.379秒、400评分71.680秒、544重载17.010秒、CPU比较136.712秒、视频12.712秒。本次新模型渲染中位8.249ms、部署端点57.280MiB；原Standard约11.487ms/154.277MiB、Mip11.776ms/196.369MiB是QUAL007同机旧记录，不冒充本轮严格同时资源基准。总输出收尾前约1.169GiB，剩余约77GiB。

浏览器工具URL安全策略拒绝打开file本地HTML，未绕过。页面已完成Node语法与本地资源检查，原生PNG/接触表已离线查看，视频完成FFprobe核验；没有声称浏览器交互或视频播放已通过。页面、视频及原生图片保留供用户打开。

结论：部署一致性问题已通过原生无外观分支解决，新Scaffold全量RGB接近Standard/Mip，并在天空/部分远端结构上有优势，但岸边任务仍不能宣布它最好。保留新模型候选、保留原三模型，不自动晋级或追加seed/步数/参数网格。下一步若要比较完整RGB+语义系统，需要为RGB136的实际输出训练同配方语义模型；本轮没有执行，不把SEM370/371/372任何旧分数挂到RGB136名下。回退使用原RGB120/130/131路径即可，无需删除或改写候选。

收尾：`bash scripts/repo_health.sh`通过（520 tests，72 skipped）；`git diff --check`通过。可选默认dataset路径跳过不影响本实验实际读取已登记400照片。没有Git提交/推送、上传或旧模型删除。
