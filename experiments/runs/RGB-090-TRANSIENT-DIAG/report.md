# RGB-090-TRANSIENT-DIAG — Training-only deck error stratification

- 状态：completed_training_only_diagnostic
- 父实验：RGB-030-FULL
- 配置：本目录 config.yaml；seed 0；split rgb_r1_spatial_v1。
- Git HEAD：`49cdcdb1c9915a9544d568754af95ea20a85ec9c`加保留工作树；配置SHA256 `2f171f327304eebb29f864ccd4265ed247b42e895ac5e4dd059f0c5ce0d5ed50`；逐输入哈希见result.json。
- 协议 ADR-048；只读取219训练图、训练GT和冻结SEM-160 OOF预测，不覆盖任何模型或按诊断生成训练mask。

## 结果与解释

训练侧真deck被预测为背景共134,075像素，RGB MAE均值0.03642，低于正确deck的0.05138；其中79.26%的MAE低于预注册0.05描述线。它不支持“桥面漏检普遍是RGB高误差造成”的强假设。腐蚀8px后deck内部漏检仅2,509像素，其MAE 0.06385，说明局部内部缺口有不同的误差特征，但不能据此作因果结论。

预测deck落在真实背景/真实stay_cable上分别399,290/284,212像素，MAE分别0.13176/0.04669。因此背景误补与RGB高误差更相关，而索穿越不能凭颜色残差当作待填满的洞。保留两条独立验收：RGB伪影改善不等于deck召回改善。

六张原生分辨率参考RGB/冻结渲染/GT/OOF预测面板按训练侧deck漏检数排序，未读取74-view或按面板调整训练参数。路径为外部输出`r2/rgb_090/diagnostic/full-20260906-r1`。result SHA256：`2b0d75b52872e3a4ff0afe0542344e62f12ea244dff3343c2af2cce1377af6fe`。
