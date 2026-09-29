# SEM-370-ALL300

2026-09-15：用户授权全400 RGB→全300匹配语义→再研究拉丝/幽灵线。当前准备中，无完成结果。

父实验：RGB-120-ALL400;SEM-330;SEM-331;SEM-349;SEM-362;SEM-363；seed 20260805；commit 49cdcdb1c9915a9544d568754af95ea20a85ec9c；manifest data/manifests/contest_dataset_v1.json；split released_all400_rgb_all300_sem_v1。

旧303初始化保持，只扩展光度训练覆盖。语义沿用static10k、M2F768/6k、B2配对3k/6k、deck及stay_cable各3k；SEM363融合头固定迁移。全300训练后诊断是训练内拟合，历史74不再是新版本独立留出。预计训练1–2小时、新增约10–20GiB（含检查点/渲染）；无机械质量/时间门。保留旧模型、失败和用户修改，不上传推送。

准备纠正：旧缓存为293标签，已复用293并按同方法补齐7个guard标签；全400图补齐23张，其余377哈希复用。r1零训练准备失败留存在outputs/quality/all_data/RGB-120-r1/failure.json；实际RGB批次r2。未重跑已完成训练。

自动队列实际结果：全数据队列中断，失败和已完成产物保留；参见queue/result.json，不代表方法质量被否定。

{
  "status": "failed_preserved",
  "stages": [],
  "code_sha256": {
    "scripts/run_quality_all300.py": "1b4ce333fc30f526053183c36995f0bec9b34474dfbaf152161edd6b996a2b01",
    "scripts/diagnose_quality_all_data.py": "ae03e306ac1ac531b7516eb1e125b73253e22db4fa403ee4b53c2691183296d8",
    "scripts/queue_quality_all_data.py": "7582aacada1f0acc66989c8112a7a8a07e3b8e07a872b174f5aa92508e564454"
  },
  "started_unix": 1789406125.6691976,
  "error": "KeyboardInterrupt: ",
  "seconds": 173.94960130800246
}

队列准备修复：旧标签读取器要求文件在本根目录，293复用标签改为逐字节复制（hash不变）；新M2F保存改用原有save_checkpoint以保留版本元数据和RNG。queue第一次仅等待RGB、零语义训练，主动中断并保留；使用queue-r2重新绑定源码，不涉及重复训练。400官方姿态、377旧RGB、300标签计数、原初始化全部核对通过，见input_check.json。

当前执行：RGB120-r2已超过15k/30k，7k和15k产物存在；SEM370 queue-r2等待最终RGB。实际输入核对400姿态/377复用RGB/300标签通过；CPU运行时导入与M2F五类配置/学习率核对通过。repo_health：518测试（72可选跳过）、199配置及链接通过；git diff --check通过。新语义尚未开始，未报告质量提升。旧模型/用户修改保持。回退只需继续使用原RGB110+SEM363或冻结RGB030+SEM161；本次模型在all_data独立目录。

2026-09-15 基线定位与即时RGB检查：RGB120定位为全数据重建基线，30k已完成（400实际训练相机，1745.69秒）；完成后立即复用SEM370原生导出做固定九图RGB检查，SEM370继续作为对应全300语义基线。159/160桥面及岸线清晰度、146/081/057/064塔索轮廓明显改善，174天空原有明显线条减弱；159/160岸边及近景仍有拉丝、桥面细线与动态车辆细节仍模糊，水面纹理过平滑。当前图不能判断连续视角稳定性，亦不证明几何根因或独立泛化。完整语义后只选择一个有证据支持的重建变量作匹配优化，不启动新重建训练。

对照页：outputs/quality/all_data/RGB-120-r2/inspection/index.html；输入/端点/代码哈希及逐图PSNR在inspection/result.json。九图静态视觉检查完成；没有新增GPU渲染或质量门。SEM370导出300张和3图fresh RGB精确复现已完成，当前static阶段运行，旧基线不变。

自动队列实际结果：全400 RGB／全300既有语义训练及固定9图诊断完成；融合复用SEM363，非独立盲测。待人工查看新对照后研究表示/优化伪影。

{
  "status": "completed_full_data_training_and_fixed9_diagnostics",
  "stages": [
    {
      "stage": "export",
      "result_sha256": "6a5c387c3a8c0804d8dba839bae973d9df1057acb55d8686cb7323d66595b3ef",
      "seconds": 41.934800619997986,
      "trained_steps": 0
    },
    {
      "stage": "export-replay",
      "result_sha256": "c71b5a3a710c45681e868558e00d0dc4fee5279f436391bfda615142def46af8",
      "seconds": 0.9785583339980803,
      "trained_steps": 0
    },
    {
      "stage": "static",
      "result_sha256": "ba17010da16b38de5722a157d514786c44a0d838a3926637fd0e54ce77a68d1b",
      "seconds": 967.285296066002,
      "trained_steps": 10000
    },
    {
      "stage": "static-replay",
      "result_sha256": "4dc0f7f727f4a93dd57e66f77034872b7ceb925ba69a3948618e2d78173f92c5",
      "seconds": 1.267710903997795,
      "trained_steps": 0
    },
    {
      "stage": "mask2former",
      "result_sha256": "5e5c7ad0bf9bfa1f30e521422e538c52e962eadbe6d5024823db1c8959ef75cb",
      "seconds": 1726.4771529120007,
      "trained_steps": 6000
    },
    {
      "stage": "mask2former-replay",
      "result_sha256": "f6c2dbe16488e70186ac6f67c5c3bda09bbd8bb614e9e5bbb4c80d31286f28ba",
      "seconds": 6.497366285002499,
      "trained_steps": 0
    },
    {
      "stage": "b2",
      "result_sha256": "f178f98dd14606b9f76670397d5f97823d367ce918b3e49b41a71629e7f6fa46",
      "seconds": 674.1760591990023,
      "trained_steps": 6000
    },
    {
      "stage": "b2-replay",
      "result_sha256": "e1012f7c7b9f8e04ed6891f5e7936840d2ff99914873689042f4168fb3d899c9",
      "seconds": 4.648002510999504,
      "trained_steps": 0
    },
    {
      "stage": "deck",
      "result_sha256": "eced7866e46886e235e0ebae4af36044c4d80f806031866c170e30699bacb980",
      "seconds": 528.6980054669984,
      "trained_steps": 3000
    },
    {
      "stage": "deck-replay",
      "result_sha256": "e2daef39b7c5f4ddb762a1b754fcf2d4b60fa3940d44d0ac8dfc6e7fa173ead2",
      "seconds": 3.4253122990012344,
      "trained_steps": 0
    },
    {
      "stage": "stay_cable",
      "result_sha256": "6896384a9b89b28457a157193d3c1e0598400370f16f9508d6ad2439f321581b",
      "seconds": 528.547709945,
      "trained_steps": 3000
    },
    {
      "stage": "stay_cable-replay",
      "result_sha256": "3966c20ac370f7a3ae4f3a17c48d7f566b2fa8979972a9a930ac8a00bd6d8644",
      "seconds": 3.4353234660011367,
      "trained_steps": 0
    }
  ],
  "code_sha256": {
    "scripts/run_quality_all300.py": "3610df10f369d75298a0c5ad7c548b02b427909f327addf1f0c3e2e9f0aebcd9",
    "scripts/diagnose_quality_all_data.py": "ae03e306ac1ac531b7516eb1e125b73253e22db4fa403ee4b53c2691183296d8",
    "scripts/queue_quality_all_data.py": "687ea424934796abe0c338e27ed2c55935947748faf5258d5a28f113b70eb535"
  },
  "started_unix": 1789406360.5287795,
  "frozen_artifacts_unchanged": 18,
  "diagnostic_result_sha256": "f611aaed88e698435e595f67909845abeebd13892aacc0bce0f301952f67d0ea",
  "seconds": 5459.334249209001
}
