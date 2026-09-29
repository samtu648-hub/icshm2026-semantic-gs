# RGB-175：ImprovedGS空间留出100万预算对照

## 结论

RGB175完成353张训练、30张严格训练排除holdout、17张buffer排除的50k训练，作为RGB176的匹配100万预算控制。它在真正空间留出上显著优于150万预算候选，因此选择100万预算作为后续未知视角重建的较安全配方。该结论支持全数据对应模型RGB166，而不是把本空间模型当成全400最终模型。

## 协议与完整性

- 实验ID：`RGB-175-IMPROVEDGS-SPATIAL-BUDGET10M`
- split：`spatial_confirmation_v1`，353 train / 30 holdout / 17 buffer。
- 初始化：仅由训练视角建立的78,045点，SHA256 `b7bddeda3759a11e833bd3d60ee554b8e005688ab14f9d41b99d99e2ac59521a`。
- 固定ImprovedGS的EAS/LAS/RAP/MU、seed 0、50k、15k增密截止和相机采样；与RGB176唯一主要变量为预算100万对150万。
- 训练/初始化/access中均排除holdout和buffer。RGB175与RGB176前1,000个相机采样完全一致。
- 首次预检在创建训练输出前因沙盒内`nvidia-smi`不可用退出，失败摘要完整保存在`outputs/quality/RGB-175-RGB-176-spatial-pair-summary-preflight-r1.json`；授权GPU环境的r2通过。

## 训练产物

- 最终Gaussian：920,177。
- 训练时间：1,378.448秒；runner wall：1,383.119秒。
- 峰值reserved显存：6,018 MiB。
- PLY大小：228,205,427字节。
- 最终PLY SHA256：`73add937cbe90f13eae90fe330b2a316a1036836d11d56d7a52e53f251162896`。
- result SHA256：`2ab402ff8ef9988cea2386a2bf68fabba4292b30c5a4be47a1703e6d8308dd8e`。

## 匹配比较

相对RGB175，RGB176在353张训练图提高0.016242 dB，但在30张holdout下降0.867008 dB、仅4/30张PSNR更高；17张buffer下降0.236195 dB。严格holdout桥梁union下降0.695295 dB，`stay_cable`下降1.502565 dB，可见支持下降2.9970个百分点，骨架支持下降4.8733个百分点。

RGB175因此不是“画面最好”的全量展示模型，而是空间确认中更可靠的100万预算控制。`spatial_confirmation_v1`是历史已暴露开发split，不是盲测或隐藏测试，不能据此宣称未知测试集精度。

## 决定

选择100万预算配方，不选择RGB176。全400对应角色由RGB166承担；RGB170仍可保留为全400已见视角展示/拟合候选。旧模型、日志和失败现场均未覆盖，不启动语义训练。

