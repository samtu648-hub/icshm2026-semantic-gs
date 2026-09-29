# RGB-111 PGSR

状态：full_30k_finished_but_numeric_integrity_failed，2026-09-09。原生训练退出成功不等于模型验收通过；本次不可用于正常方法质量选型。

父实验RGB110/QUAL-001；seed0，303训练/74开发、1320×989。源码commit de24f1a38b350387e8d8fe381b2cd70c1ae946e7，隔离CUDA/依赖及补丁见third_party记录。basecommit49cdcdb1c9915a9544d568754af95ea20a85ec9c加逐run code/source/config/input哈希。

## 工程尝试与共同初始化修正

- smoke-r1：原生300步通过，30.792001s，allocated8030.918457MiB；两fresh作者render.py重载74视角逐字节相同，67.636804s。16维特征梯度/RGB不变性通过。但作者TXT reader隐式>2px过滤掉1761点，输入70531而非共同72292；不纳入同初始化质量对照，不删除。
- smoke-common-init-r2：worker明确使用共同DLT PLY，72292点，31.564415s；model/input.ply的辅助复制路径仍指向原生过滤副本，实际训练point_cloud已经是共同输入。该辅助路径在r3修正；r2保留作工程过渡证据，不用于最终复现。
- smoke-common-init-cpu-r3：共同PLY及model/input.ply字节一致；图像仅改为CPU缓存，取样后按作者get_image转GPU，分辨率/容量/损失不变。300步30.792144s，allocated520.532715MiB/reserved674MiB。训练相机303/303有合格训练邻居，无开发来源；这是相机资格覆盖，不冒称像素/训练步覆盖。
- r3实际检查点重载的原生16通道、有限非零特征梯度、冻结RGB参数及固定视角RGB逐像素一致通过。原生相机姿态和合成非零畸变/偏心内参接口通过；不是未知相机的几何精度验证，也没有语义质量成绩。

## 完整运行终态与数值失败

`runs/RGB-111-PGSR-full-serial-20260909-r1`已完成30k，3321.285633秒，allocated3103.736816MiB/reserved5658MiB。74开发两fresh作者渲染字节相同，PSNR12.797771638、SSIM0.765335668、LPIPS0.562081778；这些是损坏运行的观测分数，不是PGSR质量上限。完整原生接口检查拒绝非有限模型参数，原队列`serial-20260909-r1`已停止，未启动SOF完整运行。旧报告的“正在运行”由此终态取代，原日志/result不改。

RGB114独立只读检查发现7k PLY399222点全部有限，15k PLY573936点中651点含NaN，30k同总数中6080点含NaN。字段包括xyz、opacity、scale和rotation。RGB114-r6在有限7k训练checkpoint的重启诊断中，于7121步分离各损失梯度：仅多视角geo_loss出现非有限xyz240/opacity80/scaling240/rotation320元素，RGB、normal、NCC、scale分量均有限。属于数值训练失败，不是语义头误报，也尚不证明低RGB分全部由这一原因造成。

PGSR单视角/多视角几何项在7000步后启用；300步没有覆盖该阶段。修复需另独立source/attempt验证，不过滤坏高斯后冒称原模型通过，不用恢复训练的诊断结果替代新的完整30k。详见RGB114报告。

## 证据

- 共同初始化SHA256 `10a88fb018c27530b3bbfa0f7000408693ba75228067f5311e7775e4755e68f2`。
- r3训练result：`39dd431b1585bd0e15e61532f8820ef8af8cd06d4996d2d357411674563a1c5d`。
- `outputs/quality/probes/PGSR-common-init-cpu-r3/result.json`：`a8d4e2d51fedf47701c20a9bb25467e1e4f674cf049d56f75dd4cc5b27c7328a`。
- 每run记录源码/配置/运行代码/实际名单、初始化点数、邻居清单和输入/模型hash；18冻结产物检查保持一致。旧RGB030+SEM161是回退，不覆盖模型或删除失败证据。
