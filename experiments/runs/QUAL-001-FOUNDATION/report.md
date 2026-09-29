# QUAL-001 质量路线基础实施

状态：implementation_in_progress，更新2026-09-09。父实验、seed、manifest及split见config；仅Scaffold对照完成有效RGB验收，PGSR原完整运行数值失败，未证明新系统收益。

最新覆盖上述旧时态：SOF112/113两臂30k/数值/74双重载/原生探针已经完成，当前画质均明显落后新Scaffold；RGB116完成只读检查点和作者接口诊断。SEM330 Scaffold10k与两fresh74独立语义评分完成（整体mIoU60.68385%，非完整专家系统），RGB115新的300步/探针通过、从头30k正在运行，SOF同款静态批次已启动等待GPU。quality62测试、repo_health335测试/63可选跳过通过，日志`quality-tests-audit-final.log`、`repo-health-audit-r1.log`。旧冻结成果不变，尚未证明新完整系统更优；各ID report/registry为具体证据。

- 用户授权完整质量路线；ADR-079明确独立副本和冻结边界。
- 沙箱GPU失败，主机确认RTX5070/12227MiB/580.173.02；未改驱动。
- preflight-r1在18冻结hash通过后因primary_fold读取类型错误失败，原文件保留。修正解析后r2通过，不改输入。
- 18/18冻结一致；三折RGB训练196/201/240，语义训练114/128/177，评测95/84/40，RGB缓冲12/18/23。只生成协议，未重建三折。
- 四臂已登记，PGSR/SOF官方源码、原生扩展及隔离依赖已就绪；两者已通过主机GPU环境的renderer/train导入。固定来源、许可证和兼容补丁见third_party/README及两个quality补丁。
- 原始400图/300标注完整性复查完成，生成的指纹与比赛manifest一致。该完整性检查读取了原始标注以验证格式/类别计数，不用于训练、选择阈值或证明泛化。预检本身只读元数据。
- 新增官方相机/语义附件绑定、三种骨干原生多通道接口、两训练来源桥面补回复核核、逐级语义修改归因、双进程RGB重载评测工具。Scaffold完整模型、PGSR修正共同初始化模型、SOF-NVS模型均通过真实GPU的16维特征梯度和RGB不变性探针；SOF-GEO单独探针待队列执行，不冒用NVS证明。探针限固定视角及合成畸变/偏心内参，非未知相机/几何质量验收。
- 静态五分类字段及16→128→5特征分类头已实现、CPU训练梯度/存取测试通过；它们不是已训练专家，也未实现GAGS粒度教师和蒸馏目标。PINHOLE专家RGB导出入口已实现但尚未GPU运行；RoMa重建未接入复核核。
- 实际隔离环境中37项新测试全部通过；repo-health-r3通过，310测试、56项可选依赖跳过。默认数据目录检查被跳过，但真实外置数据已单独通过上述指纹复查。preflight-r3再次通过18冻结hash及元数据门。
- Scaffold已完成30k、74双进程独立重载、固定PSNR/SSIM/LPIPS和模型接口探针。PGSR原30k已经结束但6080高斯含NaN，原串行队列停止；RGB114已定位多视角geo_loss/pixel_noise norm反传异常，独立safe-cycle候选窗口正在验证。SOF两臂完成300步工程，完整30k待新独立队列。后续专家/GAGS/匹配重建与最终12单元验证未完成。
- 本次续推新增评分前模型参数有限性门、精确字段错误、独立臂选择与实际依赖代码锁。隔离环境46项quality测试通过；最新repo_health为319测试/59可选跳过。QUAL002已完成六真实配对图、7级语义、3专家概率的实际归因，发现部分raw专家deck证据未被融合利用；不是完整几何真值归因或改规则授权。

证据：Git忽略的`outputs/quality/preflight-r1.json`至`preflight-r3.json`、`raw-data-manifest.json`、`raw-data-audit.log`、`preparation/manifest-r1/result.json`、各runs/audits/probes及diagnostics目录。原队列`outputs/quality/queues/serial-20260909-r1/result.json`已经记录停止。续推测试见`runtime-tests-numeric-r1.log`及`repo-health-numeric-r2.log`。旧模型和提交包保留。basecommit49cdcdb1c9915a9544d568754af95ea20a85ec9c不代表完整工作树；每run单独记录实际code/source/config/input哈希。没有commit/push/上传或费用变更。

队列只处理阶段A的RGB训练/评测/接口，不自动选型、不自动开始专家/GAGS/全链验证。任何失败或代码/配置变化会停队列并留存原因；不把停止记为方法质量失败。不保证本机可完成后期密集模型或更大专家配置，必要时另行确认资源。

续推终态：RGB114 safe-cycle独立3000步数值窗口已经通过，18冻结hash不变；RGB115完整训练仍未执行。当前新队列为`sof-serial-20260909-r2`，RGB112完整30k已实际开始，RGB113随后。此前“窗口正在验证/等待新队列”由此更新。

## 保留的工程失败

2026-09-09新增[SEM330](../SEM-330-QUALITY-STATIC/report.md)：原生静态10k训练入口、逐折标签计数/隔离、坏梯度停止、冻结RGB及严格smoke绑定已实现；四组真实CPU预检完成，原生GPU训练仍待执行。新10项测试使quality套件达56项。旧Scaffold探针代码指纹过期被正确拦住，另行刷新，不撤销旧RGB成绩。一次性后续队列定义仅等待SOF成功后执行Scaffold静态拟合；不把排队写成完成，详情以SEM330报告/事件为准。

PGSR：rasterizer缺cstdint、simple-knn缺cfloat；SOF：Python3.12 dataclass默认值、simple_knn包目录、appearance关闭时checkpoint保存None。另有扩展编译成功却未包含SM120的问题：PyTorch的路径字符串判断被含research的源码路径触发，现显式指定SM120。修复不改变分辨率/容量/训练损失。

PGSR/SOF导入初次失败分别涉及未就绪扩展、沙箱socket/CUDA访问和缺trimesh；对应日志均保留，主机最终导入已通过。Scaffold首次独立重载缺simple_knn路径，补足隔离PYTHONPATH后r2通过。以上均属工程失败，不是充分训练后的质量否决。

PGSR smoke-r1虽通过训练/重载/特征工程验收，但发现TXT加载器隐式>2px过滤，初始化70,531点而非共同72,292点。该attempt不进入同初始化对照。原数据未变；worker明确加载共同DLT PLY并记录原生加载器与实际输入点数，另开attempt复验。SOF/Scaffold本来读取共同PLY；Scaffold之后的锚点体素化属于其原生表示机制。
