# RGB-115：数值适配后的独立PGSR完整对照

当前状态：2026-09-09从头30k及独立RGB/有限性/原生探针验收已完成，PSNR12.0275916/SSIM0.75575694/LPIPS0.58049183；详见文末“最新终态”。以下运行中段落为先前记录，不覆盖实际完成事实。

父RGB111/RGB114/RGB110；seed0、同303/74、1320×989、72292训练DLT点、完整30k协议不变。唯一主要变量为多视角循环的可求导域：先选原本有效的对应，再对有效对应建立梯度，不关闭几何损失或清零坏参数。来源副本`outputs/quality/source/pgsr-safe-cycle`，作者commit及许可证见config/third_party。

先决条件是RGB114 safe-cycle完整3000步窗口通过，然后本ID从共同初始化重新做smoke和30k，不从坏30k或诊断恢复模型继续。新`execution_config.yaml`精确绑定该诊断result、补丁及helper SHA；runner核验实际source与窗口完全一致，full必须绑定同配置/代码/源/辅助依赖的有限smoke。主config原预注册hash不变。

`outputs/quality/runs/RGB-115-PGSR-SAFE-CYCLE-smoke-fresh-r1`已通过300步，31.50856s、72292共同点、18冻结一致，result SHA `f280929e4bf94a067535cb84b267335435d6f78398c5d6a33e4a15f6fb461c35`。smoke几何项尚未启用，不能代替原3k窗口或完整30k。独立真实探针通过，result SHA `8ca0321ec8e8685431c674805a2f79faa6288cfbcec5b4a61f2415737c121413`。

一次性队列`outputs/quality/queues/pgsr-safe-serial-20260909-r1`已在unix1788919862.7597756开始本ID从头完整训练，运行目录`outputs/quality/runs/RGB-115-PGSR-SAFE-CYCLE-full-pgsr-safe-serial-20260909-r1`。后续自动进行有限性门、74双重载/固定RGB评分及完整模型探针；任何失败停止保留，不降配/重试/晋级。实际终态以后以event/result为准；同款静态/二维语义还需另做。

预期执行在独立SOF两臂之后，服从单GPU/100GiB磁盘门；新GPU任务只在前任务已退出时启动。历史冻结RGB030+SEM161和RGB111失败证据保留，不提交/推送/上传。
# 最新终态：2026-09-09已完成从头30k与独立验收

`pgsr-safe-serial-20260909-r1`于11:08正常完成，status为`completed_repair_rgb_only_semantic_comparison_pending`。下方先前“运行中/待验收”是历史记录，不覆盖本节。

RGB115从共同训练DLT点云从头30000步，seed0、303/74、1320×989；7k/15k/30k保存PLY全字段有限，74双fresh RGB重载及原生适配器探针通过，18冻结不变。不是从损坏模型续训。

开发PSNR12.0275916 dB、SSIM0.75575694、LPIPS_Alex0.58049183，明显弱于RGB110；数值修复有效不等于质量提升。运行3349.9651s，allocated3066.417MiB/reserved6986MiB。全部训练相机存在合法多视图邻居，但这不是逐步/逐像素的有效覆盖率。

完整run SHA `9f183bb25d1d2e4f14182e688c0fabe7c84bf6e6ae2960706f11f787cbade5c4`，独立audit SHA `7a228b038051167e39f8937d67d28d3fc20df7550412d4e257610ba0e9c6626c`，full probe SHA `ab9ee3aae9e08e8d2d1a392d9738eb984cf8bed89cf64ad6f919b5cb1df27209`。三者均在`outputs/quality/{runs,audits,probes}/RGB-115-PGSR-SAFE-CYCLE-full-pgsr-safe-serial-20260909-r1`。

PGSR静态/二维专家尚未训练，无方法晋级；用户现授权优先RGB117相机一致性诊断。原PGSR及失败证据全部保留。
