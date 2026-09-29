# RGB-110 Scaffold匹配对照

状态：completed_30k_rgb_and_independent_reload_audit，2026-09-09。语义专家、外相机与全链路验证尚未完成；不宣布整体方案晋级。

## 实际执行

父实验QUAL-001/RGB030；seed0；303训练/74开发、原生1320×989、相同训练观测DLT初始化。独立目录，未覆盖历史RGB030。basecommit49cdcdb1c9915a9544d568754af95ea20a85ec9c加实际工作树快照；config/source/worker/input哈希保存在run/result.json，不能仅凭basecommit复现未提交代码。

- smoke-r1：300步原生训练、74输出、保存检查点通过；79.043440s，峰值allocated6287.842773MiB/reserved6800MiB。独立检查点重载r2通过，70530anchors；首次路径缺simple_knn失败不删。
- full-r1：30,000步完整训练，7k/15k/30k检查点保存；含原生导出1762.581108s，worker1760.509744s，峰值allocated6578.979004MiB/reserved10312MiB。实际303训练/74开发名单审计通过。
- audit-full-r1：两个全新进程分别加载复制检查点，74张PNG逐字节相同；不覆盖原训练目录。独立重载及固定RGB评分83.368667s，前后18冻结hash均通过。
- 原生多通道/接口探针：完整模型16维有符号字段能取得有限非零梯度，更新字段后RGB张量及固定054视角的RGB逐像素不变；最大线性误差6.854534149e-7。已知姿态加合成非零畸变/偏心内参检查通过，不能泛化成任意相机精度结论。`outputs/quality/probes/Scaffold-full-r1/result.json` SHA256 `75e2ad22a2606e878c4810738dd46727df0c42aa3bf1786ef02df264b294ee2c`。

## 固定74开发集结果

PSNR均值20.654049810932dB；SSIM均值0.850192733256；LPIPS-Alex均值0.319108837561。最差10%均值依次17.757787746797dB、0.751702160816、0.435469646007。原生PNG经同一评测实现，未裁剪/降采样/遮罩。

这是新协议Scaffold对照，不是新骨干收益，也不是新盲测或三seed稳定性证据。历史RGB030仍为独立冻结参照。尚未评测新协议语义、固定结构ROI、外相机语义和连续轨迹；不能用这一次完整RGB运行及接口探针替代整套质量升级结论。

## 证据

- config SHA256 `1ac3ddb4dbf22ece779d93d620e22068d1bbc0dd5a47e6da1f4e6c63b278c8f2`。
- `outputs/quality/runs/RGB-110-SCAFFOLD-CTRL-full-r1/result.json`：`46bd0fc62d97c7c6088134747ed2ab60e7e6cff2c968b07c43d712edebff18d5`。
- `outputs/quality/audits/RGB-110-full-r1/result.json`：`1cb9ef82743e4c836a32ac56658d157b4cbebe14a4e6cc537021c06dabfed521`。
- 同目录`metrics.json`：`80b854166629bf3787b9664011653e87438dc6464628ddd55b875a94696abfec`；逐视图metrics.csv、两次worker/日志/渲染哈希及输入映射保留。

训练耗时可用于本机Scaffold排期，不能套用为PGSR/SOF/GAGS时长。回退为既有冻结RGB030+SEM161，不需要删除任何新成果。
