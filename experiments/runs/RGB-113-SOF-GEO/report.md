# RGB-113 SOF-GEO

状态：completed_30k_rgb_and_independent_reload_audit，2026-09-09。相对RGB112只改变几何正则组，不启用每图外观拟合；不是完整TNT复现。RGB好于当前NVS但仍明显落后Scaffold，未晋级完整系统。

完整运行`outputs/quality/runs/RGB-113-SOF-GEO-full-sof-serial-20260909-r2`用时2912.1272s；7k/15k/30k保存PLY有限，74双fresh作者渲染精确、最终独立原生字段/RGB探针通过。74开发PSNR14.00705722、SSIM0.788695816、LPIPS0.503669645；同seed0新Scaffold为20.65404981/0.850192733/0.319108838。训练result SHA `80aa9fa23e261fc128b99e290415b39fec84b088fac9fbb9c495e3395a6c060e`；audit/probe位于同名目录。

[RGB116](../RGB-116-SOF-CHECKPOINT-DIAG/report.md)进一步完成7k/15k/30k及作者接口对照；不是适配器额外造成低分，训练样本略升而开发后半程下降。尚未隔离任何单损失的因果，也不按开发指标把15k替换为最终模型。

父实验RGB112/QUAL-001；seed0、303/74、1320×989；共同训练DLT72292点/PLY hash10a88fb018c27530b3bbfa0f7000408693ba75228067f5311e7775e4755e68f2。basecommit49cdcdb1c9915a9544d568754af95ea20a85ec9c及实际run源码、配置、上游补丁、输入hash完整登记。

smoke-r1原生300步通过，58.455369s，allocated6275.719238MiB/reserved6922MiB。主要几何正则尚未达到启用步数，不能据此推断几何配置优劣。模型独立保存，18冻结文件一致。

原队列`serial-20260909-r1`在PGSR门停止，未运行本臂。后来的`sof-serial-20260909-r2`已完成本臂独立smoke探针、30k及全套RGB验收，不冒用NVS探针。静态/二维语义另登记，不把RGB结束等同全方案完成。

证据：`outputs/quality/runs/RGB-113-SOF-GEO-smoke-r1/result.json` SHA256 `bed002347b9cf827fba78bec7ec808c329cda84b8cbd5c68d348373c45bcdf73`；配置SHA256 `35e84cd4703169b2396ff3e6a2f15d382d921b3512d43bfcba9e68351065482e`。失败不删除，不自动更改几何组合或降低容量。旧RGB030+SEM161保留回退。
