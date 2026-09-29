# 已核对结果表

伙伴交接版 · 证据截止 2026-09-23 · 整理 2026-09-29

每张表保留独立评价角色。数值来自保存记录，不进行新模型评分。

## 历史系统演进

证据角色：historical74

同74开发视角；不同配方及seed，不是单变量消融。

| 实验 | 五类mIoU (%) | seed |
| --- | --- | --- |
| SEM001 | 61.610304 | 20260731 |
| SEM111 | 76.063477 | 20260902 |
| SEM141 | 77.489877 | 20260805 |
| SEM151 | 83.093383 | not_applicable_deterministic_frozen_probability_gate |
| SEM161 | 83.387017 | 20260805 |
| SEM329 | 83.705003 | 20260805 |

来源：[outputs/research_report_20260908/data/content.json](https://samtu648-hub.github.io/icshm2026-semantic-gs/evidence/results/outputs__research_report_20260908__data__content.json)

## SEM350固定完整系统比较

证据角色：historical74

new_b2是融合版；raw B2 75.324650%另见REPORT002重算。RGB030/RGB110、B2 3k/M2F 6k不能作为纯容量消融。

| 系统 | 五类mIoU (%) | 前景mIoU (%) | background IoU (%) | deck IoU (%) | stay_cable IoU (%) | tower IoU (%) | foundation IoU (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| old_sem161 | 83.387017 | 79.605572 | 98.512797 | 91.466661 | 70.698321 | 83.486962 | 72.770345 |
| new_b2 | 80.917724 | 76.514911 | 98.528975 | 91.630163 | 69.717617 | 85.731904 | 58.979960 |
| new_raw | 86.513086 | 83.381702 | 99.038622 | 93.387560 | 77.400846 | 85.314116 | 77.424285 |
| new_fused | 86.542946 | 83.462326 | 98.865427 | 91.630477 | 77.137315 | 85.731904 | 79.349609 |
| new_selected | 86.515654 | 83.384912 | 99.038622 | 93.387560 | 77.400846 | 85.315578 | 77.435664 |

来源：[outputs/quality/final_comparison/SEM-350-evaluation-r1/result.json](https://samtu648-hub.github.io/icshm2026-semantic-gs/evidence/results/outputs__quality__final_comparison__SEM-350-evaluation-r1__result.json)

## 空间v1：分seed、分区域

证据角色：spatial_development

每行来自该seed对应区域的汇总分数。RGB seed仅0；历史暴露开发fold，不作显著性声明。

| 语义seed | 骨干 | 区域 | five_class_miou (%) | foreground_miou (%) | stay_cable_iou (%) | boundary_f1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 20260805 | absgs | holdout | 89.204606 | 86.667377 | 89.355622 | 79.519325 |
| 20260805 | absgs | region-1 | 85.003155 | 81.554443 | 76.743439 | 77.050142 |
| 20260805 | absgs | region-2 | 90.909214 | 88.762301 | 92.961048 | 73.171793 |
| 20260805 | absgs | region-3 | 91.425195 | 89.340557 | 92.910744 | 92.736153 |
| 20260805 | standard | holdout | 71.980289 | 65.573421 | 73.667201 | 61.965084 |
| 20260805 | standard | region-1 | 45.958714 | 34.013837 | 26.327263 | 30.916832 |
| 20260805 | standard | region-2 | 90.881891 | 88.724010 | 92.971131 | 73.207893 |
| 20260805 | standard | region-3 | 91.446060 | 89.365197 | 92.791097 | 92.105426 |
| 20260919 | absgs | holdout | 89.409963 | 86.926324 | 88.897017 | 78.185904 |
| 20260919 | absgs | region-1 | 85.902002 | 82.653560 | 76.335758 | 76.922595 |
| 20260919 | absgs | region-2 | 90.200062 | 87.907708 | 92.312661 | 69.750025 |
| 20260919 | absgs | region-3 | 91.444097 | 89.364440 | 92.889340 | 92.592448 |
| 20260919 | standard | holdout | 73.740869 | 67.741440 | 80.978347 | 68.301287 |
| 20260919 | standard | region-1 | 48.800747 | 37.450775 | 37.440752 | 37.863744 |
| 20260919 | standard | region-2 | 90.701027 | 88.517324 | 93.485861 | 76.376295 |
| 20260919 | standard | region-3 | 91.715468 | 89.701708 | 93.016017 | 92.270308 |

来源：[outputs/quality/QUAL-011/r1/result.json](https://samtu648-hub.github.io/icshm2026-semantic-gs/evidence/results/outputs__quality__QUAL-011__r1__result.json)

## 空间v2：分seed、分区域

证据角色：spatial_development

每行来自该seed对应区域的汇总分数。RGB seed仅0；历史暴露开发fold，不作显著性声明。

| 语义seed | 骨干 | 区域 | five_class_miou (%) | foreground_miou (%) | stay_cable_iou (%) | boundary_f1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 20260805 | absgs | holdout | 89.944096 | 87.600675 | 83.615702 | 75.467930 |
| 20260805 | absgs | region-1 | 79.537956 | 74.528533 | 58.425038 | 73.085033 |
| 20260805 | absgs | region-2 | 92.893267 | 91.189437 | 92.640820 | 89.126859 |
| 20260805 | absgs | region-3 | 89.055970 | 86.661011 | 83.043737 | 68.137106 |
| 20260805 | standard | holdout | 89.443071 | 87.005705 | 80.698006 | 70.061941 |
| 20260805 | standard | region-1 | 80.647941 | 75.915001 | 56.341405 | 69.210688 |
| 20260805 | standard | region-2 | 92.885166 | 91.179503 | 92.603639 | 87.416118 |
| 20260805 | standard | region-3 | 87.552620 | 84.880191 | 78.659766 | 59.240964 |
| 20260919 | absgs | holdout | 89.767499 | 87.404296 | 81.774993 | 74.173220 |
| 20260919 | absgs | region-1 | 80.657069 | 75.927832 | 58.452860 | 73.391070 |
| 20260919 | absgs | region-2 | 93.259032 | 91.645449 | 93.477449 | 91.297811 |
| 20260919 | absgs | region-3 | 88.373915 | 85.885176 | 79.602430 | 63.324699 |
| 20260919 | standard | holdout | 89.561816 | 87.157925 | 80.834737 | 72.706601 |
| 20260919 | standard | region-1 | 80.961721 | 76.305586 | 59.278683 | 72.586022 |
| 20260919 | standard | region-2 | 93.352175 | 91.761430 | 93.536262 | 90.557236 |
| 20260919 | standard | region-3 | 87.535877 | 84.874771 | 77.949304 | 60.862470 |

来源：[outputs/quality/QUAL-013/r1/result.json](https://samtu648-hub.github.io/icshm2026-semantic-gs/evidence/results/outputs__quality__QUAL-013__r1__result.json)

## 全量原三骨干及计算成本

证据角色：all400_training_fit

Scaffold为固定UID0；模型primitive的anchor/Gaussian单位不同，不能直接比较数量。

| 骨干 | PSNR (dB) | SSIM | LPIPS | 训练 (s) | 渲染中位 (ms) | RGB部署 (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| scaffold | 22.331304 | 0.884667 | 0.260565 | 1744.253791 | 8.842815 | 58.619248 |
| standard | 23.217673 | 0.893218 | 0.252782 | 1905.192321 | 11.486743 | 154.277263 |
| mip | 23.224403 | 0.893873 | 0.251058 | 2225.481217 | 11.775999 | 196.368675 |

来源：[outputs/quality/backbone_comparison/browser-r1/data.json](https://samtu648-hub.github.io/icshm2026-semantic-gs/evidence/results/outputs__quality__backbone_comparison__browser-r1__data.json)

## RGB166同轨迹训练长度

证据角色：all400_training_fit

排除历史RGB165 parent_30k；本表只比较同一次RGB166训练端点。

| step | PSNR (dB) | SSIM | LPIPS |
| --- | --- | --- | --- |
| 30000 | 23.428528 | 0.902798 | 0.227520 |
| 35000 | 23.444014 | 0.903429 | 0.225576 |
| 40000 | 23.461913 | 0.904003 | 0.223960 |
| 45000 | 23.479099 | 0.904393 | 0.222851 |
| 50000 | 23.490679 | 0.904821 | 0.221597 |

来源：[outputs/quality/RGB-166/full-r1/evaluation-r1/result.json](https://samtu648-hub.github.io/icshm2026-semantic-gs/evidence/results/outputs__quality__RGB-166__full-r1__evaluation-r1__result.json)

## SEM436/437完整训练曲线

证据角色：all300_training_fit

全300标签参与训练；12k选择仅为全量开发端点，不能外推空间最优。

| 实验 | step | 五类mIoU (%) |
| --- | --- | --- |
| SEM436 | 6000 | 95.361076 |
| SEM436 | 8000 | 95.448555 |
| SEM436 | 10000 | 95.430798 |
| SEM436 | 12000 | 95.595068 |
| SEM437 | 6000 | 95.497317 |
| SEM437 | 8000 | 95.557309 |
| SEM437 | 10000 | 95.545455 |
| SEM437 | 12000 | 95.678649 |

来源：[experiments/registry.yaml](https://samtu648-hub.github.io/icshm2026-semantic-gs/data/registry_snapshot.json)

## 后期12k端点逐类质量

证据角色：all300_training_fit

SEM436/437分别取各自封存inference/result.json；文件均随包。逐类取舍不可由总体均值代替。

| 实验 | 类别 | IoU (%) | precision (%) | recall (%) |
| --- | --- | --- | --- | --- |
| SEM436 | background | 99.482164 | 99.780123 | 99.700728 |
| SEM436 | deck | 97.224927 | 98.355287 | 98.831745 |
| SEM436 | stay_cable | 95.617738 | 97.556671 | 97.963741 |
| SEM436 | tower | 94.612358 | 96.926656 | 97.538475 |
| SEM436 | foundation | 91.038154 | 95.053881 | 95.565233 |
| SEM437 | background | 99.490672 | 99.771390 | 99.717995 |
| SEM437 | deck | 97.267769 | 98.465594 | 98.764787 |
| SEM437 | stay_cable | 95.651719 | 97.647155 | 97.908274 |
| SEM437 | tower | 94.555944 | 96.987426 | 97.417132 |
| SEM437 | foundation | 91.427140 | 95.216451 | 95.828726 |

来源：[outputs/quality/SEM-437/r2/inference/result.json](https://samtu648-hub.github.io/icshm2026-semantic-gs/evidence/results/outputs__quality__SEM-437__r2__inference__result.json)

## QUAL026容量增量

证据角色：spatial_development

RGB176(1.5M)减RGB175(1.0M)；RGB170不参加此因果比较。

| 划分 | PSNR差 (dB) | SSIM差 | LPIPS差 |
| --- | --- | --- | --- |
| train | 0.016242 | 0.000335 | -0.000561 |
| holdout | -0.867008 | -0.003175 | 0.002238 |
| buffer | -0.236195 | -0.001748 | 0.001169 |

来源：[outputs/quality/QUAL-026/r1/result.json](https://samtu648-hub.github.io/icshm2026-semantic-gs/evidence/results/outputs__quality__QUAL-026__r1__result.json)
